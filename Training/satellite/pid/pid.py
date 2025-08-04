# pid.py

import isaacgym #BugFix
import torch

import numpy as np

class PID():
    def __init__(self, num_envs: int, device: torch.device, dt: float, Kp: float, Ki: float, Kd: float,
                 clamp_p: float = np.inf, clamp_d: float = np.inf, clamp_i: float = np.inf, clamp_u: float = np.inf) -> None:
        self.dt = dt
        self.device = device

        self.Kp = torch.full((num_envs, 3), Kp, device=device)
        self.Ki = torch.full((num_envs, 3), Ki, device=device)
        self.Kd = torch.full((num_envs, 3), Kd, device=device)

        self.integral = torch.zeros((num_envs, 3), dtype=torch.float, device=device)
        self.prev_error = torch.zeros((num_envs, 3), dtype=torch.float, device=device)

        self.clamp_p = clamp_p
        self.clamp_i = clamp_i
        self.clamp_d = clamp_d
        self.clamp_u = clamp_u
    
    def update(self, error: torch.Tensor) -> torch.Tensor:
        # Proportional action
        p_term = torch.matmul(self.Kp, error)
        
        # Derivative action
        d_term = torch.matmul(self.Kd, (error - self.prev_error) / self.dt) 
        d_term = torch.clamp(d_term, -self.clamp_d, self.clamp_d)

        # Integral action
        self.integral += error * self.dt
        i_term = torch.matmul(self.Ki, self.integral)
        i_term = torch.clamp(i_term, -self.clamp_i, self.clamp_i)

        u = p_term + i_term + d_term
        u = torch.clamp(u, -self.clamp_u, self.clamp_u)
   
        self.prev_error[:] = error

        return u
    
    def reset(self, env_ids):
        self.prev_error[env_ids] = 0.0
        self.integral[env_ids] = 0.0