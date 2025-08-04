# controller.py

import isaacgym #BugFix
import torch

class SatelliteAttitudeController:
    def __init__(self, torque_tau, pid, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.torque_tau = torque_tau
        self.pid = pid
        self.prev_torque = torch.zeros((num_envs, 3), dtype=torch.float, device=device)

    def compute_control(self, measure: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.sub(measure, target)
        raw_torque = self.pid.update(error)
        
        # Apply low-pass filter to the torque command
        torque_cmd = self.torque_tau * raw_torque + (1 - self.torque_tau) * self.prev_torque
        
        self.prev_torque = torque_cmd

        return torque_cmd

    def reset(self, env_ids):
        self.prev_torque[env_ids] = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
        self.pid.reset(env_ids)
