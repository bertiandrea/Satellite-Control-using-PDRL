# satellite_reward.py

from code.utils.satellite_util import quat_diff_rad

import isaacgym #BugFix
import torch

from abc import ABC, abstractmethod
import math

from torch.utils.tensorboard import SummaryWriter

class RewardFunction(ABC):
    def __init__(self, log_reward=True, log_reward_interval=100):
        """
        Base class for reward functions.
        Subclasses must implement the compute method.
        """
        self.global_step = 0
        self.log_reward = log_reward
        if self.log_reward:
            self.writer = SummaryWriter(comment="_satellite_reward")
            self.log_reward_interval = log_reward_interval
                
    @abstractmethod
    def compute(self,
                 quats: torch.Tensor,
                 ang_vels: torch.Tensor,
                 ang_accs: torch.Tensor,
                 goal_quat: torch.Tensor,
                 goal_ang_vel: torch.Tensor,
                 goal_ang_acc: torch.Tensor,
                 actions: torch.Tensor
                 ) -> torch.Tensor:
        """
        Compute reward given state and actions.
        Must be implemented by subclasses.
        """
        pass

class SimpleReward(RewardFunction):
    """
    Simple test reward: weighted inverse errors with dynamic scaling.
    """
    def __init__(self, log_reward, log_reward_interval, alpha_q=100.0, alpha_omega=0.0, alpha_acc=0.0):
        super().__init__(log_reward, log_reward_interval)
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        # attitude error [0-pi] (radians)
        phi_raw = quat_diff_rad(quats, goal_quat)
        # attitude error [0-infinity] (radians)
        phi = torch.tan(torch.div(phi_raw, 2.0)) # tan(phi/2)
        # angular velocity error [0-infinity] (rad/s)
        omega_err = torch.norm(torch.sub(ang_vels, goal_ang_vel), dim=1)
        # angular acceleration error [0-infinity] (rad/s^2)
        acc_err   = torch.norm(torch.sub(ang_accs, goal_ang_acc), dim=1)

        assert not torch.isnan(phi).any(), "phi has NaN"
        assert not torch.isinf(phi).any(), "phi has Inf"
        assert not torch.isnan(omega_err).any(), "omega_err has NaN"
        assert not torch.isinf(omega_err).any(), "omega_err has Inf"
        assert not torch.isnan(acc_err).any(), "acc_err has NaN"
        assert not torch.isinf(acc_err).any(), "acc_err has Inf"

        r_q      = torch.mul(
            self.alpha_q,
            torch.exp(-torch.square(phi))
        )

        r_omega  = torch.mul(
            r_q,
            torch.mul(
                self.alpha_omega,
                torch.exp(-torch.square(omega_err))
            )
        )

        r_acc    = torch.mul(
            r_q,
            torch.mul(
                self.alpha_acc,  
                torch.exp(-torch.square(acc_err))
            )
        )

        reward   = torch.add(torch.add(r_q, r_omega), r_acc)
        
        assert not torch.isnan(reward).any(), "reward has NaN"
        assert not torch.isinf(reward).any(), "reward has Inf"

        if self.log_reward:
            if self.global_step % self.log_reward_interval == 0:
                self.writer.add_scalar('Reward_policy/q', r_q.mean().item(), global_step=self.global_step)
                self.writer.add_scalar('Reward_policy/omega', r_omega.mean().item(), global_step=self.global_step)
                self.writer.add_scalar('Reward_policy/acc', r_acc.mean().item(), global_step=self.global_step)
                self.writer.add_scalar('Reward_policy/total', reward.mean().item(), global_step=self.global_step)
        
        self.global_step += 1

        return reward

class CurriculumReward(RewardFunction):
    """
    Curriculum reward: weighted inverse errors with dynamic scaling based on global step.
    """
    def __init__(self, log_reward, log_reward_interval, alpha_q=100.0, alpha_omega=0.0, alpha_acc=0.0):
        super().__init__(log_reward, log_reward_interval)
        self.changing_steps = [10000, 20000, 30000, 40000, 50000]
        self.k = [1.0, 2.0, 4.0, 8.0, 16.0]
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc
        self.prev_actions = None

    def get_current_k(self):
        for i, step in enumerate(self.changing_steps):
            if self.global_step < step:
                return self.k[i]
        return self.k[-1]
    
    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        # attitude error [0-pi] (radians)
        phi_raw = quat_diff_rad(quats, goal_quat)
        # attitude error [0-infinity] (radians)
        phi = torch.tan(torch.div(phi_raw, 2.0)) # tan(phi/2)
        # angular velocity error [0-infinity] (rad/s)
        omega_err = torch.norm(torch.sub(ang_vels, goal_ang_vel), dim=1)
        # angular acceleration error [0-infinity] (rad/s^2)
        acc_err   = torch.norm(torch.sub(ang_accs, goal_ang_acc), dim=1)

        assert not torch.isnan(phi).any(), "phi has NaN"
        assert not torch.isinf(phi).any(), "phi has Inf"
        assert not torch.isnan(omega_err).any(), "omega_err has NaN"
        assert not torch.isinf(omega_err).any(), "omega_err has Inf"
        assert not torch.isnan(acc_err).any(), "acc_err has NaN"
        assert not torch.isinf(acc_err).any(), "acc_err has Inf"

        k = self.get_current_k()

        r_q      = torch.mul(
            self.alpha_q,
            torch.exp(-torch.square(phi) * k)
        )

        r_omega  = torch.mul(
            r_q,
            torch.mul(
                self.alpha_omega,
                torch.exp(-torch.square(omega_err) * k)
            )
        )

        r_acc    = torch.mul(
            r_q,
            torch.mul(
                self.alpha_acc,  
                torch.exp(-torch.square(acc_err) * k)
            )
        )

        reward = torch.add(torch.add(r_q, r_omega), r_acc)

        assert not torch.isnan(reward).any(), "reward has NaN"
        assert not torch.isinf(reward).any(), "reward has Inf"

        if self.log_reward:
            if self.global_step % self.log_reward_interval == 0:
                self.writer.add_scalar('Reward_policy/q', r_q.mean().item(), global_step=self.global_step)
                self.writer.add_scalar('Reward_policy/omega', r_omega.mean().item(), global_step=self.global_step)
                self.writer.add_scalar('Reward_policy/acc', r_acc.mean().item(), global_step=self.global_step)
                self.writer.add_scalar('Reward_policy/total', reward.mean().item(), global_step=self.global_step)
        
        self.global_step += 1
        
        return reward

class FineCurriculumReward(RewardFunction):
    """
    Gaussian reward shaping with curriculum.
    Sensitive to ~0.1° and gradually sharpens as training progresses.
    """
    def __init__(self, log_reward, log_reward_interval, alpha_q=100.0, alpha_omega=0.0, alpha_acc=0.0):
        super().__init__(log_reward, log_reward_interval)

        self.changing_steps = [5000, 10000, 15000, 25000, 35000, 45000, 60000, 80000]
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc
        self.prev_actions = None

        self.target_deg = 10 # target angle in degrees
        self.final_target_deg = 0.1 # final target angle in degrees
        self.r_at_target = 0.99 # reward at target angle
        self.base_sigma = self.target_deg * (math.pi / 180) / math.sqrt(-math.log(self.r_at_target))
        self.final_sigma = self.final_target_deg * (math.pi / 180) / math.sqrt(-math.log(self.r_at_target))
        # initialize sigma list by interpolating between base_sigma and final_sigma over the curriculum steps
        self.sigma = [
            self.base_sigma + (self.final_sigma - self.base_sigma) * (i / (len(self.changing_steps) - 1))
            for i in range(len(self.changing_steps))
        ]

    def get_current_sigma(self):
        for i, step in enumerate(self.changing_steps):
            if self.global_step < step:
                return self.sigma[i]
        return self.sigma[-1]

    def gaussian_reward(self, err, sigma):
        return torch.exp(torch.div(-torch.square(err), torch.square(sigma)))

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        # attitude error [0-pi] (radians)
        phi_raw = quat_diff_rad(quats, goal_quat)
        # attitude error [0-infinity] (radians)
        phi = torch.tan(torch.div(phi_raw, 2.0)) # tan(phi/2)
        # angular velocity error [0-infinity] (rad/s)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        # angular acceleration error [0-infinity] (rad/s^2)
        acc_err   = torch.norm(ang_accs - goal_ang_acc, dim=1)

        assert not torch.isnan(phi).any(), "phi has NaN"
        assert not torch.isinf(phi).any(), "phi has Inf"
        assert not torch.isnan(omega_err).any(), "omega_err has NaN"
        assert not torch.isinf(omega_err).any(), "omega_err has Inf"
        assert not torch.isnan(acc_err).any(), "acc_err has NaN"
        assert not torch.isinf(acc_err).any(), "acc_err has Inf"
        
        sigma = torch.tensor(self.get_current_sigma(), dtype=torch.float32, device=quats.device)

        r_q     = torch.mul(self.alpha_q, self.gaussian_reward(phi, sigma))
        r_omega = torch.mul(self.alpha_omega, self.gaussian_reward(omega_err, sigma))
        r_acc   = torch.mul(self.alpha_acc, self.gaussian_reward(acc_err, sigma))

        reward = torch.add(torch.add(r_q, r_omega), r_acc)

        assert not torch.isnan(reward).any(), "reward has NaN"
        assert not torch.isinf(reward).any(), "reward has Inf"
        
        if self.log_reward:
            if self.global_step % self.log_reward_interval == 0:
                self.writer.add_scalar('Reward/q',     r_q.mean().item(),     self.global_step)
                self.writer.add_scalar('Reward/omega', r_omega.mean().item(), self.global_step)
                self.writer.add_scalar('Reward/acc',   r_acc.mean().item(),   self.global_step)
                self.writer.add_scalar('Reward/total', reward.mean().item(),  self.global_step)

        self.global_step += 1

        return reward

class WeightedSumReward(RewardFunction):
    """
    Weighted sum of inverse errors with bonuses and penalties.
    """
    def __init__(self,
                 alpha_q=1.0, alpha_omega=0.3, alpha_acc=0.1,
                 q_threshE=1e-2, omega_threshE=1e-2,
                 q_threshL=1e-2, omega_threshL=1e-2,
                 bonus_q=200.0, bonus_stable=1000.0,
                 penalty_lvl1=-10.0, penalty_lvl2=-50.0,
                 action_saturation_thresh=None, penalty_saturation=-10.0):
        super().__init__()
        self.alpha_q = alpha_q
        self.alpha_omega = alpha_omega
        self.alpha_acc = alpha_acc
        self.q_threshE = q_threshE
        self.omega_threshE = omega_threshE
        self.q_threshL = q_threshL
        self.omega_threshL = omega_threshL
        self.bonus_q = bonus_q
        self.bonus_stable = bonus_stable
        self.penalty_lvl1 = penalty_lvl1
        self.penalty_lvl2 = penalty_lvl2
        self.action_saturation_thresh = action_saturation_thresh
        self.penalty_saturation = penalty_saturation

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        acc_err = torch.norm(ang_accs - goal_ang_acc, dim=1)

        base = (
            self.alpha_q * (1.0 / (1.0 + phi)) +
            self.alpha_omega * (1.0 / (1.0 + omega_err)) +
            self.alpha_acc * (1.0 / (1.0 + acc_err))
        )
        bonus = torch.zeros_like(base)
        # early success bonus
        bonus = torch.where(phi <= self.q_threshE, bonus + self.bonus_q, bonus)
        bonus = torch.where((phi <= self.q_threshE) & (omega_err <= self.omega_threshE),
                            bonus + self.bonus_stable, bonus)
        # level 1 penalty
        bonus = torch.where((phi >= self.q_threshL) | (omega_err >= self.omega_threshL),
                            bonus + self.penalty_lvl1, bonus)
        # level 2 penalty
        bonus = torch.where((phi >= 2.0 * self.q_threshL) | (omega_err >= 2.0 * self.omega_threshL),
                            bonus + self.penalty_lvl2, bonus)
        # action saturation penalty
        if self.action_saturation_thresh is not None:
            saturated = torch.any(actions.abs() >= self.action_saturation_thresh, dim=1)
            bonus = torch.where(saturated, bonus + self.penalty_saturation, bonus)

        return base + bonus

class TwoPhaseReward(RewardFunction):
    """
    Phase 1: reward based on improvement until phi < threshold.
    Phase 2: exponential decay once within threshold.
    """
    def __init__(self,
                 threshold=math.radians(1.0),
                 r1_pos=0.1, r1_neg=-0.1,
                 alpha=1.0, beta=0.5):
        super().__init__()
        self.threshold = threshold
        self.r1_pos = r1_pos
        self.r1_neg = r1_neg
        self.alpha = alpha
        self.beta = beta
        self.prev_phi = None

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)

        if self.prev_phi is None:
            r1 = torch.zeros_like(phi)
        else:
            delta = phi - self.prev_phi
            r1 = torch.where(delta < 0.0, self.r1_pos, self.r1_neg)
        r2 = self.alpha * torch.exp(-phi / self.beta)

        self.prev_phi = phi.clone()

        return torch.where(phi >= self.threshold, r2, r1)

class ExponentialStabilizationReward(RewardFunction):
    """
    Exponential stabilization reward with bonus when within goal radius.
    """
    def __init__(self,
                 scale=0.14 * 2.0 * math.pi,
                 bonus=9.0,
                 goal_deg=0.25):
        super().__init__()
        self.scale = scale
        self.bonus = bonus
        self.goal_rad = math.radians(goal_deg)
        self.prev_phi = None

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)
        
        exp_term = torch.exp(-phi / self.scale)
        if self.prev_phi is None:
            r = exp_term
        else:
            delta = phi - self.prev_phi
            r = torch.where(delta > 0.0, exp_term, exp_term - 1.0)
        bonus = (phi <= self.goal_rad).float() * self.bonus

        self.prev_phi = phi.clone()

        return r + bonus

class ContinuousDiscreteEffortReward(RewardFunction):
    """
    Combines error penalty, effort penalty, success bonus, and failure penalty.
    """
    def __init__(
        self,
        error_thresh=1e-2,
        bonus=5.0,
        effort_penalty=0.1,
        fail_thresh=4.0,
        fail_penalty=-100.0
    ):
        super().__init__()
        self.error_thresh = error_thresh
        self.bonus = bonus
        self.effort_penalty = effort_penalty
        self.fail_thresh = fail_thresh
        self.fail_penalty = fail_penalty

    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        
        u_norm_sq = torch.sum(actions.pow(2), dim=1)
        sup_err = torch.max(phi, omega_err)
        r1 = -(phi + omega_err + self.effort_penalty * u_norm_sq)
        r2 = torch.where(sup_err <= self.error_thresh, self.bonus, torch.zeros_like(phi))
        r3 = torch.where(sup_err >= self.fail_thresh, self.fail_penalty, torch.zeros_like(phi))

        return r1 + r2 + r3

class ShapingReward(RewardFunction):
    """
    Reward shaping variants R1-R4 with custom beta and tau functions.
    """
    def __init__(self, mode='R4'):
        super().__init__()
        assert mode in ['R1', 'R2', 'R3', 'R4'], "Unsupported mode"
        self.mode = mode
        self._prev_phi = None
    
    @staticmethod
    def beta_fn(delta, mode):
        if mode in ['R1', 'R2']:
            return torch.where(delta > 0.0, 0.5, 1.0)
        return torch.exp(-0.5 * (math.pi + delta))

    @staticmethod
    def tau_fn(phi, mode):
        if mode in ['R1', 'R3']:
            return torch.exp(2.0 - phi.abs())
        return 14.0 / (1.0 + torch.exp(2.0 * phi.abs()))
    
    def compute(self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions):
        phi = quat_diff_rad(quats, goal_quat)

        if self._prev_phi is None:
            delta = torch.zeros_like(phi)
        else:
            delta = phi - self._prev_phi
        beta = self.beta_fn(delta, self.mode)
        tau = self.tau_fn(phi, self.mode)

        self._prev_phi = phi.clone()
        
        return beta * tau
