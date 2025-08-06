# vec_task.py

import isaacgym #BugFix
import torch
from isaacgym import gymtorch, gymapi

import os
import time
import sys
import numpy as np
from datetime import datetime
from os.path import join
from typing import Dict, Any, Tuple
from abc import ABC

from gym import spaces

from torch.profiler import record_function
from torch.utils.tensorboard import SummaryWriter

import copy
import random
import operator, random
from copy import deepcopy
from collections import deque
from code.utils.dr_utils import get_property_setter_map, get_property_getter_map, get_default_setter_args, \
                apply_random_samples, modify_adr_param, set_value_by_path

class WorkerModes:
    ADR_ROLLOUT = 0  # rollout with current ADR params
    ADR_BOUNDARY = 1 # rollout with params on boundaries of ADR, used to decide whether to expand ranges

class BoundaryWorkerModes:
    LOW = 0
    HIGH = 1

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        print("Using EXISTING Sim Instance")
        return EXISTING_SIM
    else:
        print("Creating NEW Sim Instance")
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class Env(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool):
        self.cfg = config

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        self.headless = headless

        enable_camera_sensors = config["env"].get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments
        
        self.num_observations = config["env"].get("numObservations", 0)
        self.num_states = config["env"].get("numStates", 0)

        self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf, dtype=np.float64)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf, dtype=np.float64)

        self.num_actions = config["env"]["numActions"]
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1., dtype=np.float64)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        self.control_steps: int = 0

        self.render_fps: int = config["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = join("recorded_frames", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.writer = SummaryWriter(comment="_satellite")

class VecTask(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False): 
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)

        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        self.dt: float = self.sim_params.dt

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

        self.obs_states_dict = {}

    def set_viewer(self):
        self.enable_viewer_sync = True
        self.viewer = None

        if self.headless == False:
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.episode_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        #sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        # WORKAROUND: BugFix for IsaacGym not handling multiple Gym instances correctly in the same process (Needed for Hyperparameter Optimization)
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        with record_function("#VecTask__STEP"):
            if self.randomize:
                if self.debug_prints:
                    print("Action BEFORE randomization:")
                    print(f"actions[0]: {', '.join(f'{v:.2f}' for v in actions[0].tolist())}")
                actions = self.dr_randomizations['actions']['noise_lambda'](actions)
                if self.debug_prints:
                    print("Action AFTER randomization:")
                    print(f"actions[0]: {', '.join(f'{v:.2f}' for v in actions[0].tolist())}")

            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            
            if self.debug_prints:
                print("#" * 50)
                print(f"Actions         MAX: {actions.max().item():.2f} MIN: {actions.min().item():.2f} MEAN: {actions.mean().item():.2f} STD: {actions.std().item():.2f}")  # Debugging output

            with record_function("$VecTask__step__pre_physics_step"):
                self.pre_physics_step(actions)

            for i in range(self.control_freq_inv):
                if self.force_render:
                    with record_function("#VecTask__step__RENDER"):
                        self.render()
                with record_function("#VecTask__step__SIM"):
                    self.gym.simulate(self.sim)

            if self.device == 'cpu':
                with record_function("$VecTask__step__FETCH_RESULTS"):
                    self.gym.fetch_results(self.sim, True)

            with record_function("$VecTask__step__post_physics_step"):
                self.post_physics_step()

            if self.randomize:
                if self.debug_prints:
                    print("Observations BEFORE randomization:")
                    print(f"obs_buf[0]: {', '.join(f'{v:.2f}' for v in self.obs_buf[0].tolist())}")
                self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
                if self.debug_prints:
                    print("Observations AFTER randomization:")
                    print(f"obs_buf[0]: {', '.join(f'{v:.2f}' for v in self.obs_buf[0].tolist())}")

            self.obs_states_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_states_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            
            self.control_steps += 1
            self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

            if self.debug_prints:
                num_quats = 4; num_quat_diff = 4; num_quat_diff_rad = 1; num_angacc = 3; num_actions = 3; num_angvels = 3
                l_index = 0; h_index = num_quats
                print(f"Quats           MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats; h_index = num_quats + num_quat_diff
                print(f"QuatsDiff       MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff; h_index = num_quats + num_quat_diff + num_quat_diff_rad
                print(f"QuatsDiffRad    MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff + num_quat_diff_rad; h_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc
                print(f"AngAcc          MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc; h_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc + num_actions
                print(f"Act             MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc + num_actions; h_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc + num_actions + num_angvels
                print(f"AngVels         MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                print(f"Reward          MAX: {self.rew_buf.max().item():.2f} MIN: {self.rew_buf.min().item():.2f} MEAN: {self.rew_buf.mean().item():.2f} STD: {self.rew_buf.std().item():.2f}")  # Debugging output

                print(f"Timeouts:       {self.timeout_buf.sum().item()}")  # Debugging output
                print(f"Reset:          {self.reset_buf.sum().item()}")  # Debugging output
                print(f"Extras:         {self.extras}")  # Debugging output
                print(f"Steps:          {self.control_steps}")  # Debugging output

        return self.obs_states_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def reset(self):
        self.obs_states_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.obs_states_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_states_dict

    def render(self, mode="rgb_array"):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                self.gym.sync_frame_time(self.sim)

                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    render_dt = self.dt * self.control_freq_inv
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        sim_params = gymapi.SimParams()

        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        if physics_engine == "physx":
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        return sim_params
    

class ADRVecTask(VecTask):
    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        ###################################################
        self.randomize = config["dr_randomization"].get("enabled", False)
        self.use_adr = config["dr_randomization"].get("automatic", False)
        self.dr_params = config["dr_randomization"].get("dr_params", {})
        ###################################################

        if self.randomize: 
            self.first_randomization = True
            self.original_props = {}
            self.dr_randomizations = {}
            self.last_step = -1

        if self.randomize and self.use_adr:
            self.adr_cfg = config["dr_randomization"].get("adr", {})

            self.worker_adr_boundary_fraction = self.adr_cfg["worker_adr_boundary_fraction"]
            self.adr_queue_threshold_length = self.adr_cfg["adr_queue_threshold_length"]

            self.adr_objective_threshold_low = self.adr_cfg["adr_objective_threshold_low"]
            self.adr_objective_threshold_high = self.adr_cfg["adr_objective_threshold_high"]

            self.adr_rollout_perf_alpha = self.adr_cfg["adr_rollout_perf_alpha"]

            ################################################
            self.adr_param = self.adr_cfg["adr_param"]
            self.adr_param["range"] = self.adr_param["init_range"]
            ################################################
            self.adr_tensor_values = torch.zeros(config["env"]["numEnvs"], device=sim_device)
            self.worker_types = torch.zeros(config["env"]["numEnvs"], dtype=torch.long, device=sim_device)
            self.adr_modes = torch.zeros(config["env"]["numEnvs"], dtype=torch.long, device=sim_device)
            self.adr_objective_queues = [deque(maxlen=self.adr_queue_threshold_length) for _ in range(2)]
            
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    ######################################################################################################################
    
    def build_current_adr_params(self, dr_param):
        current_adr_params = copy.deepcopy(dr_param)
        set_value_by_path(current_adr_params, self.adr_param["range_path"], self.adr_param["range"])    
        return current_adr_params

    def build_current_adr_params_by_env_id(self, env_id, current_adr_params):       
        if self.worker_types[env_id] == WorkerModes.ADR_ROLLOUT:
            return current_adr_params
        
        elif self.worker_types[env_id] == WorkerModes.ADR_BOUNDARY:
            adr_mode = self.adr_modes[env_id]  # 0=low, 1=high
            env_adr_params = copy.deepcopy(current_adr_params)

            boundary_value = self.adr_param["range"][adr_mode]
            set_value_by_path(env_adr_params, self.adr_param["range_path"], [boundary_value, boundary_value])
            
            return env_adr_params
        else:
            raise NotImplementedError

    ######################################################################################################################

    def sample_adr_params_for_envs(self, env_ids):
        param_range = self.adr_param["range"]

        # Mask for which envs we sample
        sample_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sample_mask[env_ids] = True

        # Masks for boundary/rollout workers
        adr_workers_low_mask = (self.worker_types == WorkerModes.ADR_BOUNDARY) & (self.adr_modes == 0) & sample_mask
        adr_workers_high_mask = (self.worker_types == WorkerModes.ADR_BOUNDARY) & (self.adr_modes == 1) & sample_mask

        rollout_workers_mask = (~adr_workers_low_mask) & (~adr_workers_high_mask) & sample_mask
        rollout_workers_env_ids = torch.nonzero(rollout_workers_mask, as_tuple=False).squeeze(-1)

        result = torch.zeros((len(env_ids),), device=self.device, dtype=torch.float)
        # Rollout workers → random uniform
        result[rollout_workers_mask[env_ids]] = (
            torch.rand(len(rollout_workers_env_ids), device=self.device, dtype=torch.float) * (param_range[1] - param_range[0]) + param_range[0]
        )
        # Boundary workers → fixed values
        result[adr_workers_low_mask[env_ids]] = param_range[0]
        result[adr_workers_high_mask[env_ids]] = param_range[1]
        
        self.adr_tensor_values[env_ids] = result

        return result

    def assign_new_worker_modes(self, env_ids):
        rand_vals = torch.rand(len(env_ids), device=self.device, dtype=torch.float)

        # Rollout or boundary assignment
        new_worker_types = torch.where(
            rand_vals < self.worker_adr_boundary_fraction,
            torch.full_like(rand_vals, WorkerModes.ADR_ROLLOUT, dtype=torch.long),
            torch.full_like(rand_vals, WorkerModes.ADR_BOUNDARY, dtype=torch.long)
        )

        self.worker_types[env_ids] = new_worker_types

        # ADR modes: 0 = low bound, 1 = high bound
        self.adr_modes[env_ids] = torch.randint(0, 2, (len(env_ids),), dtype=torch.long, device=self.device)

    def adr_update(self, env_ids, adr_objective):
        """
        Performs ADR update step (implements algorithm 1 from https://arxiv.org/pdf/1910.07113.pdf).
        ADR (Automatic Domain Randomization) adjusts environment parameters to match a target objective.
        """
        # Create a boolean mask for environments selected for ADR update in this batch
        rand_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rand_env_mask[env_ids] = True

        ################################################################
        adr_workers_boundary = (self.worker_types == WorkerModes.ADR_BOUNDARY)
        adr_workers_rollout = (self.worker_types == WorkerModes.ADR_ROLLOUT)
        self.writer.add_scalar('ADR/workers_boundary', adr_workers_boundary.sum().item(), global_step=self.control_steps)
        self.writer.add_scalar('ADR/workers_rollout', adr_workers_rollout.sum().item(), global_step=self.control_steps)
        ################################################################
        
        # Identify workers on lower-bound exploration for this param (N)
        adr_workers_low = (self.worker_types == WorkerModes.ADR_BOUNDARY) & (self.adr_modes == BoundaryWorkerModes.LOW)
        # Identify workers on upper-bound exploration for this param (N)
        adr_workers_high = (self.worker_types == WorkerModes.ADR_BOUNDARY) & (self.adr_modes == BoundaryWorkerModes.HIGH)
        
        # Filter to those that were sampled in this batch
        adr_done_low = rand_env_mask & adr_workers_low
        adr_done_high = rand_env_mask & adr_workers_high

        # Collect performance objectives for boundary samples
        objective_low_bounds = adr_objective[adr_done_low]
        objective_high_bounds = adr_objective[adr_done_high]

        # Append observed objectives to respective queues for stats
        self.adr_objective_queues[BoundaryWorkerModes.LOW].extend(objective_low_bounds.cpu().numpy().tolist())
        self.adr_objective_queues[BoundaryWorkerModes.HIGH].extend(objective_high_bounds.cpu().numpy().tolist())
        
        # Compute mean performance at each boundary
        mean_low = np.mean(self.adr_objective_queues[BoundaryWorkerModes.LOW]) if self.adr_objective_queues[BoundaryWorkerModes.LOW] else 0.
        mean_high = np.mean(self.adr_objective_queues[BoundaryWorkerModes.HIGH]) if self.adr_objective_queues[BoundaryWorkerModes.HIGH] else 0.

        ################################################################
        self.writer.add_scalar('ADR/mean_low', mean_low, global_step=self.control_steps)
        self.writer.add_scalar('ADR/mean_high', mean_high, global_step=self.control_steps)
        ################################################################

        range_lower = self.adr_param["range"][0]
        range_upper = self.adr_param["range"][1]
        range_limits = self.adr_param["limits"]    # absolute allowed bounds
        init_range = self.adr_param["init_range"]  # initial default range

        changed_low, changed_high = False, False

        # Adjust lower bound if enough samples accumulated
        if len(self.adr_objective_queues[BoundaryWorkerModes.LOW]) >= self.adr_queue_threshold_length:
            # If performance too low, increase lower bound (make task easier)
            if mean_low < self.adr_objective_threshold_low:
                range_lower, changed_low = modify_adr_param(range_lower, 'up', self.adr_param, param_limit=init_range[0])
            # If performance too high, decrease lower bound (make task harder)
            elif mean_low > self.adr_objective_threshold_high:
                range_lower, changed_low = modify_adr_param(range_lower, 'down', self.adr_param, param_limit=range_limits[0])            
            if changed_low:
                self.adr_objective_queues[BoundaryWorkerModes.LOW].clear()
                self.worker_types[adr_workers_low] = WorkerModes.ADR_ROLLOUT
        
        # Adjust upper bound if enough samples accumulated
        if len(self.adr_objective_queues[BoundaryWorkerModes.HIGH]) >= self.adr_queue_threshold_length:
            # If performance too low, decrease upper bound (make task easier)
            if mean_high < self.adr_objective_threshold_low:
                range_upper, changed_high = modify_adr_param(range_upper, 'down', self.adr_param, param_limit=init_range[1])
            # If performance too high, increase upper bound (make task harder)
            elif mean_high > self.adr_objective_threshold_high:
                range_upper, changed_high = modify_adr_param(range_upper, 'up', self.adr_param, param_limit=range_limits[1])
            if changed_high:
                self.adr_objective_queues[BoundaryWorkerModes.HIGH].clear()
                self.worker_types[adr_workers_high] = WorkerModes.ADR_ROLLOUT

        self.adr_param["range"] = [range_lower, range_upper]

        ################################################################
        self.writer.add_scalar(f'ADR/range_lower', range_lower, global_step=self.control_steps)
        self.writer.add_scalar(f'ADR/range_upper', range_upper, global_step=self.control_steps)
        ################################################################

        self.assign_new_worker_modes(env_ids)

        # Sample ADR tensors (optionally with new ranges) for the environments
        self.sample_adr_params_for_envs(env_ids)

    def _randomize_non_physical_params(self, dr_params):
        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                if op_type == 'additive': op = operator.add
                else: op = operator.mul
                ####################################################################
                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1
                ####################################################################
                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])
                    ####################################################################
                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling
                        mu = mu * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        var_corr = var_corr * sched_scaling
                        mu_corr = mu_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                    ####################################################################
                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])
                    ####################################################################
                    self.dr_randomizations[nonphysical_param] = {
                        'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}
                    ####################################################################
                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])
                    ####################################################################
                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                    ####################################################################
                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])
                    ####################################################################
                    self.dr_randomizations[nonphysical_param] = {
                        'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}
                    ####################################################################

    def _randomize_sim_params(self, dr_params):
        if "sim_params" in dr_params:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)
            if self.first_randomization:
                self.original_props["sim_params"] = {attr: getattr(prop, attr) for attr in dir(prop)}
            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)
            self.gym.set_sim_params(self.sim, prop)
    
    def _randomize_actor_properties(self, dr_params, env_ids, param_maps, current_adr_params=None):
        ############################### ADR ################################
        for i_, env_id in enumerate(env_ids):
            if self.use_adr:
                env_dr_params = self.build_current_adr_params_by_env_id(env_id, current_adr_params)
            else:
                env_dr_params = dr_params
        ####################################################################    
            for actor, actor_properties in env_dr_params["actor_params"].items():
                ####################################################################
                for prop_name, prop_attrs in actor_properties.items():
                    ####################################################################
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(self.envs[env_id], self.gym.find_actor_handle(self.envs[env_id], actor))
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(self.envs[env_id], self.gym.find_actor_handle(self.envs[env_id], actor), n, gymapi.MESH_VISUAL, gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    ####################################################################
                    prop = param_maps["getters"][prop_name](self.envs[env_id], self.gym.find_actor_handle(self.envs[env_id], actor))
                    set_random_properties = True
                    ####################################################################
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [{attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        ####################################################################
                        for attr, attr_randomization_params in prop_attrs.items():
                            for idx, (p, og_p) in enumerate(zip(prop, self.original_props[prop_name])):
                                ####################################################################
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    ############################### ADR ################################
                                    original_randomization_params = env_dr_params['actor_params'][actor][prop_name][attr]
                                    ####################################################################
                                    apply_random_samples(p, og_p, attr, attr_randomization_params, self.last_step, None, bucketing_randomization_params=original_randomization_params)
                                else:
                                    set_random_properties = False
                    ####################################################################
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        ####################################################################
                        for attr, attr_randomization_params in prop_attrs.items():
                            p, og_p = prop, self.original_props[prop_name]
                            ####################################################################
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                ############################### ADR ################################
                                original_randomization_params = env_dr_params['actor_params'][actor][prop_name][attr]
                                ####################################################################
                                apply_random_samples(p, og_p, attr, attr_randomization_params, self.last_step, None, bucketing_randomization_params=original_randomization_params)
                            else:
                                set_random_properties = False
                    ####################################################################
                    if set_random_properties:
                        setter = param_maps["setters"][prop_name]
                        default_args = param_maps["defaults"][prop_name]
                        setter(self.envs[env_id], self.gym.find_actor_handle(self.envs[env_id], actor), prop, *default_args)
                    
    def apply_randomizations(self, env_ids, dr_params, adr_objective=None):
        self.last_step = self.gym.get_frame_count(self.sim)

        if self.first_randomization:
            self._randomize_non_physical_params(dr_params)

        current_adr_params = None
        if self.use_adr:
            self.adr_update(env_ids, adr_objective)
            current_adr_params = self.build_current_adr_params(dr_params)

        if self.debug_prints and self.use_adr:
            print(f"Current ADR params: {current_adr_params}")
        elif self.debug_prints:
            print(f"Current DR params: {dr_params}")
        
        param_maps = {
            "getters": get_property_getter_map(self.gym),
            "setters": get_property_setter_map(self.gym),
            "defaults": get_default_setter_args(self.gym),
        }
        
        #self._randomize_sim_params(dr_params)

        self._randomize_actor_properties(dr_params, env_ids, param_maps, current_adr_params)

        self.first_randomization = False

        if self.debug_prints:
            header = f"{'Env':<5} {'Ixx':>10} {'Iyy':>10} {'Izz':>10}"
            print("=" * len(header))
            print(header)
            print("-" * len(header))
            for env_id in env_ids:
                env = self.envs[env_id]
                actor_handle = self.actor_handles[env_id]
                rb_props = self.gym.get_actor_rigid_body_properties(env, actor_handle)
                I = rb_props[0].inertia
                print(f"{env_id:<5} {I.x.x:>10.4f} {I.y.y:>10.4f} {I.z.z:>10.4f}")
            print("=" * len(header) + "\n")