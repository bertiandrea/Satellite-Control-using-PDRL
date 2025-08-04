# agent_wrapper_CAPS.py

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Mapping, Any

from skrl import config
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR

class PPOWrapperCAPS(PPO):
    def __init__(self, models, memory=None, observation_space=None, action_space=None,
                 device=None, cfg=None):
        super().__init__(models, memory, observation_space, action_space, device, cfg)

        self._lambda_t = cfg.get("lambda_temporal_smoothness", 0.0)
        self._lambda_s = cfg.get("lambda_spatial_smoothness", 0.0)
        self._noise_std = cfg.get("noise_std", 0.00)

        self._gauss = torch.distributions.Normal(0, 1)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)

        self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
        self._tensors_names.append("next_states")
    
    def _update(self, timestep: int, timesteps: int) -> None:
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(self._learning_epochs):
            kl_divergences = []
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_next_states,  # <-- !!!
            ) in sampled_batches:
                #####################################################################################
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
                    
                #####################################################################################
                #torch.no_grad() :
                # Disabilita l’accumulazione del grafo computazionale, risparmiando memoria e tempo di calcolo quando non serve fare back-prop.
                #torch.autocast(device_type=…, enabled=…) :
                # Abilita (condizionatamente, a seconda di _mixed_precision) la precisione mista: le operazioni in virgola mobile verranno 
                # automaticamente fatte in FP16 (dove sicuro) per velocizzare i calcoli e ridurre l’uso di memoria, ma torneranno in FP32 quando serve.
                with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # μ(s_t), μ(s_{t+1}), μ(near s_t)
                    next_proc = self._state_preprocessor(sampled_next_states, train=False)
                    mu_cur,  _, _ = self.policy.compute({"states": sampled_states}, role="policy")
                    mu_next, _, _ = self.policy.compute({"states": next_proc},    role="policy")
                    noise = self._gauss.sample(sample_shape=sampled_states.shape).to(sampled_states) * self._noise_std
                    near_states = self._state_preprocessor(sampled_states + noise, train=False)
                    mu_near, _, _ = self.policy.compute({"states": near_states},  role="policy")
                    
                    # Temporal Smoothness Lt
                    Lt = self._lambda_t * F.mse_loss(mu_next, mu_cur)
                    # Spatial Smoothness Ls
                    Ls = self._lambda_s * F.mse_loss(mu_near, mu_cur)

                    CAPS_loss = Lt + Ls

                #####################################################################################
                
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                ######################################################################################

                self.optimizer.zero_grad()
                self.scaler.scale(CAPS_loss + policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss  += policy_loss.item()
                cumulative_value_loss   += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        
        self.track_data("CAPS / Lt", Lt.item())
        self.track_data("CAPS / Ls", Ls.item())
