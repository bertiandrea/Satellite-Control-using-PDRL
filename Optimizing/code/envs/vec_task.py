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
    