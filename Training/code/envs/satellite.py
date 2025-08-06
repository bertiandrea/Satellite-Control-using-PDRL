# satellite.py

from code.utils.satellite_util import get_euler_xyz, quat_from_euler_xyz, sample_random_quaternion_batch, quat_diff, quat_diff_rad, quat_axis, quat_mul
from code.envs.vec_task import VecTask, ADRVecTask
from code.rewards.satellite_reward import (
    TestReward,
    RewardFunction
)

import isaacgym #BugFix
import torch
from isaacgym import gymutil, gymtorch, gymapi

from pathlib import Path
import numpy as np

BASE_COLORS_SAT  = torch.tensor([[1,0,1], [0,1,1], [1,1,0]], dtype=torch.float)
BASE_COLORS_GOAL = torch.tensor([[0,0,1], [0,1,0], [1,0,0]], dtype=torch.float)

class Satellite(ADRVecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, reward_fn: RewardFunction = None):
        self.dt =                   cfg["sim"].get('dt', 1 / 60.0)                          # seconds
        self.max_episode_length =   int(cfg["env"].get('episode_length_s', 120) / self.dt)  # seconds

        self.env_spacing =          cfg["env"].get('envSpacing', 0.0)                       # meters
        self.asset_name =           cfg["env"]["asset"].get('assetName', 'satellite')
        self.asset_root =           cfg["env"]["asset"].get('assetRoot', str(Path(__file__).resolve().parent.parent))
        self.asset_file =           cfg["env"]["asset"].get('assetFileName', 'satellite.urdf')
        self.asset_init_pos_p =     cfg["env"]["asset"].get('init_pos_p', [0.0, 0.0, 0.0])
        self.asset_init_pos_r =     cfg["env"]["asset"].get('init_pos_r', [0.0, 0.0, 0.0, 1.0])
        self.torque_scale =         cfg["env"].get('torque_scale', 1.0)
        self.threshold_ang_goal =   cfg["env"].get('threshold_ang_goal', 0.01745)           # radians
        self.threshold_vel_goal =   cfg["env"].get('threshold_vel_goal', 0.01745)           # radians/sec
        self.goal_time =            cfg["env"].get('goal_time', 10) / self.dt               # seconds
        self.sparse_reward =        cfg["env"].get('sparse_reward', 100.0)
        self.overspeed_ang_vel =    cfg["env"].get('overspeed_ang_vel', 0.78540)            # radians/sec
        self.debug_arrows =         cfg["env"].get('debug_arrows', False)
        self.debug_prints =         cfg["env"].get('debug_prints', False)
        self.heartbeat =            cfg.get('heartbeat', False)

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        ################# SETUP SIM #################
        self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.actor_root_state).view(self.num_envs, 13)
        self.satellite_pos     = self.root_states[:, 0:3]
        self.satellite_quats   = self.root_states[:, 3:7]
        self.satellite_linvels = self.root_states[:, 7:10]
        self.satellite_angvels = self.root_states[:, 10:13]
        #############################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_root_states = self.root_states.clone()
        print(f"Initial root states: {self.initial_root_states[0]}")
        ########################################

        self.prev_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.goal_quat = sample_random_quaternion_batch(self.device, self.num_envs)
        self.goal_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_ang_acc = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.torque_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        self.force_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        self.root_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int) * self.num_bodies

        if reward_fn is None:
            self.reward_fn: RewardFunction = TestReward()
        else:
            self.reward_fn = reward_fn
        
        ###################################################
        self.in_goal_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ###################################################

    def create_sim(self) -> None:
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params) # Acquires the sim pointer
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))
        ###################################################
        if self.randomize:
            print("Applying randomizations...")
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int)
            adr_objective = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.apply_randomizations(ids, self.dr_params, adr_objective)
        ###################################################

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actor_handles = []
        self.sat_glob_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            origin = self.gym.get_env_origin(env)
            self.sat_glob_pos[i] = torch.tensor([origin.x, origin.y, origin.z],
                                                dtype=torch.float,
                                                device=self.device)
            ###################################################
            actor_handle = self.create_actor(i, env, self.asset, self.asset_init_pos_p, self.asset_init_pos_r, 1, self.asset_name)
            ###################################################
            self.actor_handles.append(actor_handle)
            self.envs.append(env)

    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        return asset
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        actor_handle =  self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)
        return actor_handle

    def draw_arrows(self):
        x_goal = quat_axis(self.goal_quat, 0)
        y_goal = quat_axis(self.goal_quat, 1)
        z_goal = quat_axis(self.goal_quat, 2)
        x_sat  = quat_axis(self.satellite_quats, 0)
        y_sat  = quat_axis(self.satellite_quats, 1)
        z_sat  = quat_axis(self.satellite_quats, 2)

        sat_lines = torch.cat([
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + x_sat * 1.5], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + y_sat * 1.5], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + z_sat * 1.5], dim=1),
        ], dim=0)  # → (3N,2,3)
        goal_lines = torch.cat([
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + x_goal * 2.0], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + y_goal * 2.0], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + z_goal * 2.0], dim=1),
        ], dim=0)  # → (3N,2,3)
        all_lines = torch.cat([sat_lines, goal_lines], dim=0)  # → (6N,2,3)

        colors_sat  = BASE_COLORS_SAT.repeat_interleave(self.num_envs, dim=0)   # (3N,3)
        colors_goal = BASE_COLORS_GOAL.repeat_interleave(self.num_envs, dim=0)  # (3N,3)
        all_colors = torch.cat([colors_sat, colors_goal], dim=0)  # (6N,3)

        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(
            self.viewer,
            None,
            6 * self.num_envs,
            all_lines.cpu().numpy(),
            all_colors.cpu().numpy()
        )

    ################################################################################################################################
           
    def reset_idx(self, ids: torch.Tensor) -> None:
        ###################################################
        if self.randomize:
            self.apply_randomizations(ids, self.dr_params, self.rew_buf)
        ###################################################

        ################# SIM #################
        self.root_states[ids] = torch.zeros((len(ids), 13), dtype=torch.float32, device=self.device)
        self.root_states[ids, 3:7] = sample_random_quaternion_batch(self.device, len(ids))
        
        idx32 = ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.actor_root_state, gymtorch.unwrap_tensor(idx32), len(idx32)
        )
        #######################################

        self.prev_angvel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

        self.goal_quat[ids] = sample_random_quaternion_batch(self.device, len(ids))
        self.goal_ang_vel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)
        self.goal_ang_acc[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

        self.progress_buf[ids] = 0
        self.reset_buf[ids] = False
        self.timeout_buf[ids] = False

        self.rew_buf[ids] = 0.0
        self.episode_rew_buf[ids] = 0.0

        self.in_goal_buf[ids] = 0
        self.goal_reached[ids] = False

    ################################################################################################################################
                
    def termination(self) -> None:
        self.reset_ids  = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
        if len(self.reset_ids) > 0:
            self.reset_idx(self.reset_ids)
    
    def apply_torque(self) -> None:        
        self.actions = torch.mul(self.actions, self.torque_scale)

        #########################################
        
        self.actions[self.reset_ids] = torch.zeros((len(self.reset_ids), 3), dtype=torch.float, device=self.device)
        
        #########################################
        self.writer.add_scalar('Actions/action_X', self.actions[0, 0].item(), global_step=self.control_steps)
        self.writer.add_scalar('Actions/action_Y', self.actions[0, 1].item(), global_step=self.control_steps)
        self.writer.add_scalar('Actions/action_Z', self.actions[0, 2].item(), global_step=self.control_steps)

        assert not torch.isnan(self.actions).any(), f"actions has NaN: {self.actions, self.states_buf}"
        assert not torch.isinf(self.actions).any(), f"actions has Inf: {self.actions, self.states_buf}"
        #########################################

        ################## SIM ##################
        self.torque_tensor[self.root_indices] = self.actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.force_tensor),  
            gymtorch.unwrap_tensor(self.torque_tensor), 
            gymapi.LOCAL_SPACE,
        )
        #########################################
                
    def compute_observations(self) -> None:
        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.satellite_angacc = torch.div(
            torch.sub(self.satellite_angvels, self.prev_angvel),
            self.dt
        )

        self.prev_angvel = self.satellite_angvels.clone()
        self.obs_buf = torch.cat(
            (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), quat_diff_rad(self.satellite_quats, self.goal_quat).unsqueeze(-1), 
                self.satellite_angacc, self.actions), dim=-1)
        self.states_buf = torch.cat(
            (self.obs_buf, self.satellite_angvels), dim=-1)
        ########################################

        ########################################
        assert not torch.isnan(self.obs_buf).any(), f"self.obs_buf has NaN: {self.actions, self.obs_buf}"
        assert not torch.isinf(self.obs_buf).any(), f"self.obs_buf has Inf: {self.actions, self.obs_buf}"
        assert not torch.isnan(self.states_buf).any(), f"self.states_buf has NaN: {self.actions, self.states_buf}"
        assert not torch.isinf(self.states_buf).any(), f"self.states_buf has Inf: {self.actions, self.states_buf}"
        ########################################

    def compute_reward(self) -> None:
        self.rew_buf = self.reward_fn.compute(
            self.satellite_quats, self.satellite_angvels, self.satellite_angacc,
            self.goal_quat, self.goal_ang_vel, self.goal_ang_acc,
            self.actions
        )
        self.rew_buf = torch.where(
            self.goal_reached,
            torch.add(self.rew_buf, self.sparse_reward),
            self.rew_buf
        )
        self.episode_rew_buf += self.rew_buf
        self.writer.add_scalar('Reward_policy/total_episode', self.episode_rew_buf.mean().item(), global_step=self.control_steps)

    def check_termination(self) -> None:
        #########################################
        angle_diff = quat_diff_rad(self.satellite_quats, self.goal_quat)
        ang_vel_diff = torch.norm(
            torch.sub(self.satellite_angvels, self.goal_ang_vel),
            dim=1
        )
        goal = torch.logical_and(
            torch.lt(angle_diff, self.threshold_ang_goal),
            torch.lt(ang_vel_diff, self.threshold_vel_goal)
        )
        #########################################

        #########################################
        self.in_goal_buf = torch.add(self.in_goal_buf, goal.to(torch.long))
        self.goal_reached = torch.ge(self.in_goal_buf, self.goal_time)
        #########################################

        #########################################
        self.writer.add_scalar('Goal/angle_diff', angle_diff.mean().item(), global_step=self.control_steps)
        self.writer.add_scalar('Goal/goal', goal.sum(dim=0).item(), global_step=self.control_steps)
        self.writer.add_scalar('Goal/goal_reached', self.goal_reached.sum(dim=0).item(), global_step=self.control_steps)
        #########################################

        #########################################
        timeout = torch.ge(self.progress_buf, self.max_episode_length)
        overspeed = torch.ge(
            torch.norm(self.satellite_angvels, dim=1),
            self.overspeed_ang_vel
        )

        self.timeout_buf = timeout
        self.reset_buf = torch.logical_or(timeout, overspeed)
        #########################################
    
    def pre_physics_step(self, actions):
        if self.heartbeat:
            return

        self.actions = actions.to(self.device)

        self.termination()

        self.apply_torque()

    def post_physics_step(self):
        self.progress_buf += 1
        
        if self.heartbeat:
            return
        
        self.compute_observations()

        self.compute_reward()

        self.check_termination()
        
        if self.debug_arrows:
            self.draw_arrows()
