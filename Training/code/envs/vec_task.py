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
                apply_random_samples, check_buckets, nested_dict_set_attr, modify_adr_param

class RolloutWorkerModes:
    ADR_ROLLOUT = 0  # rollout with current ADR params
    ADR_BOUNDARY = 1 # rollout with params on boundaries of ADR, used to decide whether to expand ranges
    TEST_ENV = 2     # rollout with default DR params, used to measure overall success rate. (currently unused)

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

            self.adr_extended_boundary_sample = self.adr_cfg["adr_extended_boundary_sample"]

            self.adr_rollout_perf_alpha = self.adr_cfg["adr_rollout_perf_alpha"]

            self.update_adr_ranges = self.adr_cfg["update_adr_ranges"]

            self.adr_clear_other_queues = self.adr_cfg["clear_other_queues"]

            self.adr_load_from_checkpoint = self.adr_cfg["adr_load_from_checkpoint"]

            ################################################
            self.adr_params = self.adr_cfg["adr_params"]
            self.adr_params_keys = list(self.adr_params.keys())
            ################################################
            self.adr_rollout_perf_last = None
            self.adr_tensor_values = {}
            self.adr_params_builtin_keys = []

            self.worker_types = torch.zeros(config["env"]["numEnvs"], dtype=torch.long, device=sim_device)
            self.adr_modes = torch.zeros(config["env"]["numEnvs"], dtype=torch.long, device=sim_device)

            ################################################
            for k in self.adr_params:
                self.adr_params[k]["range"] = self.adr_params[k]["init_range"]
                ################################################
                if "limits" not in self.adr_params[k]:
                    self.adr_params[k]["limits"] = [None, None]
                ################################################
                if "delta_style" in self.adr_params[k]:
                    assert self.adr_params[k]["delta_style"] in ["additive", "multiplicative"]
                else:
                    self.adr_params[k]["delta_style"] = "additive"
                ################################################
                if "range_path" in self.adr_params[k]:
                    self.adr_params_builtin_keys.append(k)
                else:
                    param_type = self.adr_params[k].get("type", "uniform")
                    dtype = torch.long if param_type == "categorical" else torch.float
                    self.adr_tensor_values[k] = torch.zeros(self.cfg["env"]["numEnvs"], device=sim_device, dtype=dtype)
                ################################################
            
            self.num_adr_params = len(self.adr_params)
            self.adr_objective_queues = []
            for _ in range(2 * self.num_adr_params):
                self.adr_objective_queues.append(deque(maxlen=self.adr_queue_threshold_length))
            
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def get_current_adr_params(self, dr_params):
        current_adr_params = copy.deepcopy(dr_params)
        for k in self.adr_params_builtin_keys:
            nested_dict_set_attr(current_adr_params, self.adr_params[k]["range_path"], self.adr_params[k]["range"])    
        return current_adr_params

    def get_dr_params_by_env_id(self, env_id, default_dr_params, current_adr_params):
        env_type = self.worker_types[env_id]
        if env_type == RolloutWorkerModes.ADR_ROLLOUT:
            return current_adr_params
        elif env_type == RolloutWorkerModes.ADR_BOUNDARY:
            ####################################################
            adr_mode = int(self.adr_modes[env_id])

            env_adr_params = copy.deepcopy(current_adr_params)
            param_name = self.adr_params_keys[adr_mode // 2]
            
            if not param_name in self.adr_params_builtin_keys:
                return env_adr_params

            if self.adr_extended_boundary_sample:
                boundary_value = self.adr_params[param_name]["next_limits"][adr_mode % 2] 
            else:
                boundary_value = self.adr_params[param_name]["range"][adr_mode % 2]
            new_range = [boundary_value, boundary_value]

            nested_dict_set_attr(env_adr_params, self.adr_params[param_name]["range_path"], new_range)
            
            return env_adr_params
            ####################################################
        elif env_type == RolloutWorkerModes.TEST_ENV:
            return default_dr_params
        else:
            raise NotImplementedError

    def sample_adr_tensor(self, param_name, env_ids=None):
        sample_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sample_mask[env_ids] = True

        param_range = self.adr_params[param_name]["range"]
        next_limits = self.adr_params[param_name].get("next_limits", None)
        param_type = self.adr_params[param_name].get("type", "uniform")

        n = self.adr_params_keys.index(param_name)

        adr_workers_low_mask = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == (2 * n)) & sample_mask
        adr_workers_high_mask = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == (2 * n + 1)) & sample_mask
        
        rollout_workers_mask = (~adr_workers_low_mask) & (~adr_workers_high_mask) & sample_mask
        rollout_workers_env_ids = torch.nonzero(rollout_workers_mask, as_tuple=False).squeeze(-1)

        if param_type == "uniform":
            result = torch.zeros((len(env_ids),), device=self.device, dtype=torch.float)
            uniform_noise_rollout_workers = torch.rand((rollout_workers_env_ids.shape[0],), device=self.device, dtype=torch.float) * (param_range[1] - param_range[0]) + param_range[0]
            
            result[rollout_workers_mask[env_ids]] = uniform_noise_rollout_workers
            if self.adr_extended_boundary_sample:
                result[adr_workers_low_mask[env_ids]] = next_limits[0]
                result[adr_workers_high_mask[env_ids]] = next_limits[1]
            else:
                result[adr_workers_low_mask[env_ids]] = param_range[0]
                result[adr_workers_high_mask[env_ids]] = param_range[1]
        
        elif param_type == "categorical":
            result = torch.zeros((len(env_ids), ), device=self.device, dtype=torch.long)
            uniform_noise_rollout_workers = torch.randint(int(param_range[0]), int(param_range[1])+1, size=(rollout_workers_env_ids.shape[0]), device=self.device)
            
            result[rollout_workers_mask[env_ids]] = uniform_noise_rollout_workers
            result[adr_workers_low_mask[env_ids]] = int(next_limits[0] if self.adr_extended_boundary_sample else param_range[0])
            result[adr_workers_high_mask[env_ids]] = int(next_limits[1] if self.adr_extended_boundary_sample else param_range[1])
        
        else:
            raise NotImplementedError(f"Unknown distribution type {param_type}")
        
        self.adr_tensor_values[param_name][env_ids] = result

        return result

    def recycle_envs(self, recycle_envs):
        worker_types_rand = torch.rand(len(recycle_envs), device=self.device, dtype=torch.float)

        new_worker_types = torch.zeros(len(recycle_envs), device=self.device, dtype=torch.long)

        new_worker_types[(worker_types_rand < self.worker_adr_boundary_fraction)] = RolloutWorkerModes.ADR_ROLLOUT
        new_worker_types[(worker_types_rand >= self.worker_adr_boundary_fraction)] = RolloutWorkerModes.ADR_BOUNDARY

        self.worker_types[recycle_envs] = new_worker_types

        self.adr_modes[recycle_envs] = torch.randint(0, self.num_adr_params * 2, (len(recycle_envs),), dtype=torch.long, device=self.device)

    def adr_update(self, rand_envs, adr_objective):
        """
        Performs ADR update step (implements algorithm 1 from https://arxiv.org/pdf/1910.07113.pdf).
        ADR (Automatic Domain Randomization) adjusts environment parameters to match a target objective.
        """
        # Create a boolean mask for environments selected for ADR update in this batch
        rand_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rand_env_mask[rand_envs] = True
        total_nats = 0.0  # accumulate information gain measure (in nats) across parameters

        if self.update_adr_ranges:

            # Randomize order of ADR parameters for unbiased updates
            adr_params_iter = list(enumerate(self.adr_params))
            random.shuffle(adr_params_iter)
            already_recycled = False  # track if envs have been recycled after a parameter change

            # Loop through each ADR parameter in random order
            for n, adr_param_name in adr_params_iter:
                # Identify workers on lower-bound exploration for this param
                adr_workers_low = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == 2 * n)
                # Identify workers on upper-bound exploration for this param
                adr_workers_high = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == 2 * n + 1)
                # Identify workers that are currently rolling out (not in ADR mode)
                adr_workers_rollout = (self.worker_types == RolloutWorkerModes.ADR_ROLLOUT)

                ################################################################
                self.writer.add_scalar('ADR/workers_low', adr_workers_low.sum().item(), global_step=self.control_steps)
                self.writer.add_scalar('ADR/workers_high', adr_workers_high.sum().item(), global_step=self.control_steps)
                self.writer.add_scalar('ADR/workers_others', adr_workers_rollout.sum().item(), global_step=self.control_steps)
                self.writer.add_scalar('ADR/total_workers', (adr_workers_low | adr_workers_high | adr_workers_rollout).sum().item(), global_step=self.control_steps)
                ################################################################

                # Filter to those that were sampled in this batch
                adr_done_low = rand_env_mask & adr_workers_low
                adr_done_high = rand_env_mask & adr_workers_high

                # Collect performance objectives for boundary samples
                objective_low_bounds = adr_objective[adr_done_low]
                objective_high_bounds = adr_objective[adr_done_high]

                # Append observed objectives to respective queues for stats
                self.adr_objective_queues[2 * n].extend(objective_low_bounds.cpu().numpy().tolist())
                self.adr_objective_queues[2 * n + 1].extend(objective_high_bounds.cpu().numpy().tolist())
                
                low_queue = self.adr_objective_queues[2 * n]
                high_queue = self.adr_objective_queues[2 * n + 1]
                
                # Compute mean performance at each boundary
                mean_low = np.mean(low_queue) if len(low_queue) > 0 else 0.
                mean_high = np.mean(high_queue) if len(high_queue) > 0 else 0.

                ################################################################
                self.writer.add_scalar('ADR/mean_low', mean_low, global_step=self.control_steps)
                self.writer.add_scalar('ADR/mean_high', mean_high, global_step=self.control_steps)
                print(f'ADR Parameter: {adr_param_name} | Low Mean: {mean_low:.2f} | High Mean: {mean_high:.2f}')
                ################################################################

                # Fetch current range and limits for this ADR parameter
                current_range = self.adr_params[adr_param_name]["range"]
                range_lower = current_range[0]
                range_upper = current_range[1]
                range_limits = self.adr_params[adr_param_name]["limits"]    # absolute allowed bounds
                init_range = self.adr_params[adr_param_name]["init_range"]  # initial default range

                # Retrieve or initialize next step limits placeholders
                [next_limit_lower, next_limit_upper] = self.adr_params[adr_param_name].get("next_limits", [None, None])

                changed_low, changed_high = False, False  # track adjustments to the range

                # Adjust lower bound if enough samples accumulated
                if len(low_queue) >= self.adr_queue_threshold_length:
                    # If performance too low, increase lower bound (make task easier)
                    if mean_low < self.adr_objective_threshold_low:
                        range_lower, changed_low = modify_adr_param(range_lower, 'up', self.adr_params[adr_param_name], param_limit=init_range[0])
                        if not changed_low:
                            print('RAISING LOWER BOUND FAILED (EASIER): Init Range lower bound (', init_range[0] ,') already reached.')
                    # If performance too high, decrease lower bound (make task harder)
                    elif mean_low > self.adr_objective_threshold_high:
                        print(f'Low boundary performance HIGH: {mean_low} > {self.adr_objective_threshold_high}.')
                        range_lower, changed_low = modify_adr_param(range_lower, 'down', self.adr_params[adr_param_name], param_limit=range_limits[0])
                    
                    if changed_low:
                        print(f'Changing {adr_param_name} lower bound. Queue length {len(low_queue)}. Mean perf: {mean_low}. Old val: {current_range[0]}. New val: {range_lower}')
                        self.adr_objective_queues[2 * n].clear()
                        # Switch boundary workers back to rollout mode
                        self.worker_types[adr_workers_low] = RolloutWorkerModes.ADR_ROLLOUT
                
                # Adjust upper bound if enough samples accumulated
                if len(high_queue) >= self.adr_queue_threshold_length:
                    # If performance too low, decrease upper bound (make task easier)
                    if mean_high < self.adr_objective_threshold_low:
                        range_upper, changed_high = modify_adr_param(range_upper, 'down', self.adr_params[adr_param_name], param_limit=init_range[1])
                        if not changed_high:
                            print('LOWERING UPPER BOUND FAILED (EASIER): Init Range lower bound (', init_range[1] ,') already reached.')
                    # If performance too high, increase upper bound (make task harder)
                    elif mean_high > self.adr_objective_threshold_high:
                        print(f'Upper boundary performance HIGH: {mean_high} > {self.adr_objective_threshold_high}.')
                        range_upper, changed_high = modify_adr_param(range_upper, 'up', self.adr_params[adr_param_name], param_limit=range_limits[1])
                    
                    if changed_high:
                        print(f'Changing {adr_param_name} upper bound. Queue length {len(high_queue)}. Mean perf: {mean_high}. Old val: {current_range[1]}. New val: {range_upper}')
                        self.adr_objective_queues[2 * n + 1].clear()
                        # Switch boundary workers back to rollout mode
                        self.worker_types[adr_workers_high] = RolloutWorkerModes.ADR_ROLLOUT

                # Compute "next_limits" heuristically for sampling
                if changed_low or next_limit_lower is None:
                    next_limit_lower, _ = modify_adr_param(range_lower, 'down', self.adr_params[adr_param_name], param_limit=range_limits[0])
                if changed_high or next_limit_upper is None:
                    next_limit_upper, _ = modify_adr_param(range_upper, 'up', self.adr_params[adr_param_name], param_limit=range_limits[1])

                # Save updated range back into parameters
                self.adr_params[adr_param_name]["range"] = [range_lower, range_upper]

                ################################################################
                self.writer.add_scalar(f'ADR/range_lower', range_lower, global_step=self.control_steps)
                self.writer.add_scalar(f'ADR/range_upper', range_upper, global_step=self.control_steps)
                ################################################################

                # Accumulate information gain (entropy) for reporting
                if not self.adr_params[adr_param_name]["delta"] < 1e-9:
                    upper_lower_delta = range_upper - range_lower
                    if upper_lower_delta < 1e-3:
                        upper_lower_delta = 1e-3
                    nats = np.log(upper_lower_delta)
                    total_nats += nats
                self.adr_params[adr_param_name]["next_limits"] = [next_limit_lower, next_limit_upper]

                # Log telemetry if available (every 100 steps or on change)
                if hasattr(self, 'extras') and ((changed_high or changed_low) or self.last_step % 100 == 0):
                    self.extras[f'adr/params/{adr_param_name}/lower'] = range_lower
                    self.extras[f'adr/params/{adr_param_name}/upper'] = range_upper
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/lower/value'] = mean_low
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/lower/queue_len'] = len(low_queue)
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/upper/value'] = mean_high
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/upper/queue_len'] = len(high_queue)
                
                # If clearing other queues on change, recycle all environments once
                if self.adr_clear_other_queues and (changed_low or changed_high):
                    for q in self.adr_objective_queues:
                        q.clear()
                    recycle_envs = torch.nonzero((self.worker_types == RolloutWorkerModes.ADR_BOUNDARY), as_tuple=False).squeeze(-1)
                    self.recycle_envs(recycle_envs)
                    already_recycled = True  # skip additional recycling below
                    break

            # After looping params, optionally log rollout performance stats
            if hasattr(self, 'extras') and self.last_step % 100 == 0:
                mean_perf = adr_objective[rand_env_mask & (self.worker_types == RolloutWorkerModes.ADR_ROLLOUT)].mean()
                # Exponentially-weighted moving average of rollout performance
                if self.adr_rollout_perf_last is None:
                    self.adr_rollout_perf_last = mean_perf
                else:
                    self.adr_rollout_perf_last = self.adr_rollout_perf_last * self.adr_rollout_perf_alpha + mean_perf * (1 - self.adr_rollout_perf_alpha)
                self.extras[f'adr/objective_perf/rollouts'] = self.adr_rollout_perf_last
                self.extras[f'adr/npd'] = total_nats / len(self.adr_params)

            # Recycle environments not yet recycled in boundary changes
            if not already_recycled:
                self.recycle_envs(rand_envs)

        else:
            # If ADR range updates disabled, simply mark these envs for rollout
            self.worker_types[rand_envs] = RolloutWorkerModes.ADR_ROLLOUT

        # Always resample ADR tensors for the selected environments
        for k in self.adr_tensor_values:
            self.sample_adr_tensor(k, rand_envs)

    def _randomize_non_physical_params(self, dr_params):
        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul
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
                env_dr_params = self.get_dr_params_by_env_id(env_id, dr_params, current_adr_params)
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
                                    #if i_ % 1000 == 0:
                                    #    print(f'  [_randomize_actor_properties] List prop "{prop_name}"[{idx}].{attr}: {og_p[attr]} -> {getattr(p, attr)}')
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
                                #if i_ % 1000 == 0:
                                #    print(f'  [_randomize_actor_properties] Scalar prop "{prop_name}".{attr}: {getattr(og_p, attr)} -> {getattr(p, attr)}')
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
            current_adr_params = self.get_current_adr_params(dr_params)
        
        param_maps = {
            "getters": get_property_getter_map(self.gym),
            "setters": get_property_setter_map(self.gym),
            "defaults": get_default_setter_args(self.gym),
        }
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)
        
        #self._randomize_sim_params(dr_params)

        self._randomize_actor_properties(dr_params, env_ids, param_maps, current_adr_params)

        self.first_randomization = False