import math
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np

import isaacgym #BugFix
from isaacgym import gymapi, gymtorch, gymutil
import torch

from .vec_task import VecTask
from satellite.utils.satellite_util import quat_axis

class Quadcopter(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = 500

        dofs_per_env = 8
        bodies_per_env = 9

        # Observations:
        # 0:13 - root state
        # 13:29 - DOF states
        # Actions:
        # 0:8 - rotor DOF position targets
        # 8:12 - rotor thrust magnitudes

        self.cfg["env"]["numStates"] = 21
        self.cfg["env"]["numObservations"] = 21 # Not used, but kept for compatibility
        self.cfg["env"]["numActions"] = 12
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_quats = vec_root_tensor[..., 3:7]
        self.root_linvels = vec_root_tensor[..., 7:10]
        self.root_angvels = vec_root_tensor[..., 10:13]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = vec_root_tensor.clone()
        self.initial_dof_states = vec_dof_tensor.clone()

        max_thrust = 2
        self.thrust_lower_limits = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self.device, dtype=torch.float32)

        self.dof_position_targets = torch.zeros((self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, bodies_per_env, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "./satellite"
        asset_file = "quadcopter.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dofs = self.gym.get_asset_dof_count(asset)

        dof_props = self.gym.get_asset_dof_properties(asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []
        for i in range(self.num_dofs):
            self.dof_lower_limits.append(dof_props['lower'][i])
            self.dof_upper_limits.append(dof_props['upper'][i])

        self.dof_lower_limits = torch.tensor(self.dof_lower_limits, dtype=torch.float, device=self.device)
        self.dof_upper_limits = torch.tensor(self.dof_upper_limits, dtype=torch.float, device=self.device)
        self.dof_ranges = self.dof_upper_limits - self.dof_lower_limits

        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        self.envs = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(env, asset, default_pose, "quadcopter", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            dof_props['stiffness'].fill(1000.0)
            dof_props['damping'].fill(0.0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            chassis_color = gymapi.Vec3(0.8, 0.6, 0.2)
            rotor_color = gymapi.Vec3(0.1, 0.2, 0.6)
            arm_color = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, chassis_color)
            self.gym.set_rigid_body_color(env, actor_handle, 1, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 3, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 5, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 7, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 2, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)
            self.gym.set_rigid_body_color(env, actor_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)
            self.gym.set_rigid_body_color(env, actor_handle, 6, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)
            self.gym.set_rigid_body_color(env, actor_handle, 8, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)

            self.envs.append(env)

    ######################################################################################################

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_states[env_ids] = self.initial_dof_states[env_ids]

        actor_indices = self.all_actor_indices[env_ids].flatten()

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] += torch.zeros((num_resets, 1), device=self.device).flatten() 
        self.root_states[env_ids, 1] += torch.zeros((num_resets, 1), device=self.device).flatten()
        self.root_states[env_ids, 2] += torch.zeros((num_resets, 1), device=self.device).flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.dof_positions[env_ids] = torch.zeros((num_resets, self.num_dofs), dtype=torch.float32, device=self.device)
        self.dof_velocities[env_ids] = 0.0
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    ######################################################################################################

    def pre_physics_step(self, _actions):
        actions = _actions.to(self.device)

        #####################################################################################
    
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        ######################################################################################
        
        thrust_action_speed_scale = 200
        self.thrusts += self.dt * thrust_action_speed_scale * actions[:, 8:12]
        self.thrusts[:] = torch.clamp(self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits)

        self.forces[:, 2, 2] = self.thrusts[:, 0]
        self.forces[:, 4, 2] = self.thrusts[:, 1]
        self.forces[:, 6, 2] = self.thrusts[:, 2]
        self.forces[:, 8, 2] = self.thrusts[:, 3]

        dof_action_speed_scale = 8 * math.pi
        self.dof_position_targets += self.dt * dof_action_speed_scale * actions[:, 0:8]
        self.dof_position_targets[:] = torch.clamp(self.dof_position_targets, self.dof_lower_limits, self.dof_upper_limits)

        ####################################################################################

        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0
        self.dof_position_targets[reset_env_ids] = self.dof_positions[reset_env_ids]

        ####################################################################################

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    ######################################################################################################

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    ######################################################################################################

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
    
        target_x = 0.0
        target_y = 0.0
        target_z = 1.0
        self.states_buf[..., 0] = (target_x - self.root_positions[..., 0]) / 3
        self.states_buf[..., 1] = (target_y - self.root_positions[..., 1]) / 3
        self.states_buf[..., 2] = (target_z - self.root_positions[..., 2]) / 3
        self.states_buf[..., 3:7] = self.root_quats
        self.states_buf[..., 7:10] = self.root_linvels / 2
        self.states_buf[..., 10:13] = self.root_angvels / math.pi
        self.states_buf[..., 13:21] = self.dof_positions
        return self.states_buf
    
    ######################################################################################################

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward()

    ######################################################################################################

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_quadcopter_reward(root_positions: torch.Tensor, root_quats: torch.Tensor, root_linvels: torch.Tensor, root_angvels: torch.Tensor,
                              reset_buf: torch.Tensor, progress_buf: torch.Tensor, max_episode_length: float) -> Tuple[torch.Tensor, torch.Tensor]:
    target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
                             root_positions[..., 1] * root_positions[..., 1] +
                             (1 - root_positions[..., 2]) * (1 - root_positions[..., 2]))
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 3.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.3, ones, die)

    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
