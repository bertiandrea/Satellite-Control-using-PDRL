# satellite_reward.py

from code.utils.satellite_util import quat_diff_rad

import isaacgym #BugFix
import torch

from abc import ABC, abstractmethod
import math
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

# ============================================================================ #
#                               Base Class                                     #
# ============================================================================ #

class RewardFunction(ABC):
    """Abstract base class for all reward functions."""

    def __init__(self, log_reward: bool = True, log_reward_interval: int = 100):
        self.global_step: int = 0
        self.log_reward: bool = log_reward
        self.log_reward_interval: int = log_reward_interval

        self.writer: Optional[SummaryWriter] = None
        if self.log_reward:
            self.writer = SummaryWriter(comment="_satellite_reward")

    @abstractmethod
    def compute(
        self,
        quats: torch.Tensor,
        ang_vels: torch.Tensor,
        ang_accs: torch.Tensor,
        goal_quat: torch.Tensor,
        goal_ang_vel: torch.Tensor,
        goal_ang_acc: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the reward value."""
        pass

    def _log_scalar(self, tag: str, value: float):
        """Helper to log scalar values with TensorBoard."""
        if self.log_reward and self.writer:
            if self.global_step % self.log_reward_interval == 0:
                self.writer.add_scalar(tag, value, global_step=self.global_step)

    @staticmethod
    def _assert_valid_tensor(tensor: torch.Tensor, name: str):
        """Ensure tensors have no NaN or Inf values."""
        assert not torch.isnan(tensor).any(), f"{name} has NaN values"
        assert not torch.isinf(tensor).any(), f"{name} has Inf values"


# ============================================================================ #
#                             SimpleReward                                     #
# ============================================================================ #

class SimpleReward(RewardFunction):
    """Gaussian-shaped reward based on quaternion, angular velocity, and acceleration errors."""

    def __init__(self, log_reward: bool = True, log_reward_interval: int = 100):
        super().__init__(log_reward, log_reward_interval)
        self.alpha_q = 1.0
        self.alpha_omega = 0.0
        self.alpha_acc = 0.0

    def compute(
        self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions
    ):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        acc_err = torch.norm(ang_accs - goal_ang_acc, dim=1)

        self._assert_valid_tensor(phi, "phi")
        self._assert_valid_tensor(omega_err, "omega_err")
        self._assert_valid_tensor(acc_err, "acc_err")

        r_q = self.alpha_q * (1.0 / (1.0 + phi**2))
        r_omega = r_q * self.alpha_omega * (1.0 / (1.0 + omega_err**2))
        r_acc = r_q * self.alpha_acc * (1.0 / (1.0 + acc_err**2))

        reward = r_q + r_omega + r_acc
        self._assert_valid_tensor(reward, "reward")

        self._log_scalar("Reward_policy/q", r_q.mean().item())
        self._log_scalar("Reward_policy/omega", r_omega.mean().item())
        self._log_scalar("Reward_policy/acc", r_acc.mean().item())
        self._log_scalar("Reward_policy/total", reward.mean().item())

        self.global_step += 1
        return reward


# ============================================================================ #
#                      Exponential Stabilization Reward                        #
# ============================================================================ #

class ReductionReward(RewardFunction):
    """Encourages stabilization by rewarding reduction in attitude error."""

    def __init__(self, log_reward: bool = True, log_reward_interval: int = 100):
        super().__init__(log_reward, log_reward_interval)
        self.prev_phi: Optional[torch.Tensor] = None
        self.sigma = 0.14 * 2 * math.pi
        self.th_ang_goal = 0.1 * (math.pi / 180)
        self.th_vel_goal = 1.0 * (math.pi / 180)
        self.bonus = 10.0
        self.lambda_u = 1e-5

    def compute(
        self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions
    ):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)

        r_q = torch.exp(-phi / self.sigma)

        if self.prev_phi is None:
            reward = torch.zeros_like(phi)
        else:
            delta = phi - self.prev_phi
            reward = torch.where(delta > 0.0, r_q - 1.0, r_q)

        in_goal = ((phi <= self.th_ang_goal) & (omega_err <= self.th_vel_goal))
        r_bonus = self.bonus * in_goal.float()

        u_norm_sq = torch.sum(actions ** 2, dim=-1)
        r_effort = self.lambda_u * u_norm_sq

        final_reward = reward + r_bonus - r_effort
        self._assert_valid_tensor(reward, "reward")

        self.prev_phi = phi.clone()

        self._log_scalar("Reward_policy/phi", phi.mean().item() * (180 / torch.pi))
        self._log_scalar("Reward_policy/omega_err", omega_err.mean().item())
        self._log_scalar("Reward_policy/in_goal", in_goal.float().mean().item())   
        self._log_scalar("Reward_policy/energy", u_norm_sq.mean().item())
        self._log_scalar("Reward_policy/max_torque", actions.abs().max().item())

        self._log_scalar("Reward_policy/reward", reward.mean().item())
        self._log_scalar("Reward_policy/bonus", r_bonus.mean().item())
        self._log_scalar("Reward_policy/effort", r_effort.mean().item())
        self._log_scalar("Reward_policy/total", final_reward.mean().item())

        self.global_step += 1
        return final_reward
