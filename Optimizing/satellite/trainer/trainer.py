# trainer.py

from typing import Optional
import atexit
import sys
import tqdm
import copy

import torch

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper

SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}

class Trainer:
    def __init__(
        self,
        env: Wrapper,
        agent: Agent,
        cfg: Optional[dict] = None,
    ) -> None:
        self.cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        self.cfg.update(cfg if cfg is not None else {})
        print(f"Trainer configuration: {self.cfg}")
        self.env = env
        self.agent = agent

        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")
        self.stochastic_evaluation = self.cfg.get("stochastic_evaluation", False)

        self.initial_timestep = 0

        # register environment closing if configured
        if self.close_environment_at_exit:
            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        self.agent.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        self.agent.set_running_mode("train")

        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            self.agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                if not self.headless:
                    self.env.render()

                self.agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agent.track_data(f"Info / {k}", v.item())

            self.agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def init_step_train(self) -> None:
        self.agent.set_running_mode("train")

        states, infos = self.env.reset()
    
        return states, infos
    
    def step_train(self, states, infos, timestep) -> None:
        self.agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

        with torch.no_grad():
            actions = self.agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]

            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            if not self.headless:
                self.env.render()

            self.agent.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=self.timesteps,
            )

            if self.environment_info in infos:
                for k, v in infos[self.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        self.agent.track_data(f"Info / {k}", v.item())

        self.agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

        if self.env.num_envs > 1:
            states = next_states
        else:
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states
        
        return states, infos, rewards
    

    def eval(self) -> None:
        self.agent.set_running_mode("eval")

        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            self.agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                outputs = self.agent.act(states, timestep=timestep, timesteps=self.timesteps)
                actions = outputs[0] if self.stochastic_evaluation else outputs[-1].get("mean_actions", outputs[0])

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                if not self.headless:
                    self.env.render()

                self.agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agent.track_data(f"Info / {k}", v.item())

            super(type(self.agent), self.agent).post_interaction(timestep=timestep, timesteps=self.timesteps)

            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

