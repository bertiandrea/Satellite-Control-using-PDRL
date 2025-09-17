# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

NUM_ENVS = 4096
N_EPOCHS = 100
HEADLESS = True
DEBUG_ARROWS = False

ROLLOUTS = 64
LEARNING_EPOCHS = 32
MINI_BATCHES = 8

CONFIG = {
    # --- seed & devices ----------------------------------------------------
    "set_seed": False,
    "seed": 42,

    "physics_engine": "physx",

    "rl_device": "cuda:0",
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": HEADLESS,
    "virtual_screen_capture": False,
    "force_render": False,

    "profile": False,

    "heartbeat": False,

    # --- env section -------------------------------------------------------
    "env": {
        "numEnvs": NUM_ENVS,

        "numObservations": 15, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3)

        "numStates": 18, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3) + satellite_angvels (3)

        "numActions": 3,

        "envSpacing": 3.0,

        "threshold_ang_goal": 0.01, # radians
        "threshold_vel_goal": 0.1, # radians/sec
        "overspeed_ang_vel":  0.4, # radians/sec

        "sparse_reward": 10.0, # reward for staying the goal
        
        "max_episode_length": 120.0, # seconds
        "min_episode_length": 20.0, # seconds
        "episode_length_scaling": 0.95, # scaling factor for episode length
        "episode_length_scaling_steps": 2000, # steps after which the episode length is scaled

        "clipActions": 1.0,
        "clipObservations": 10.0,

        "torque_scale": 10.0,

        "debug_arrows": DEBUG_ARROWS,
        
        "debug_prints": False,
        
        "discretize_starting_pos": False,

        "asset": {

            "assetRoot": str(Path(__file__).resolve().parent.parent),
            "assetFileName": "satellite.urdf",
            "assetName": "satellite",

            #"disable_gravity"
            #"collapse_fixed_joints"
            #"slices_per_cylinder"
            #"replace_cylinder_with_capsule"
            #"fix_base_link"
            #"default_dof_drive_mode"
            #"self_collisions"
            #"flip_visual_attachments"

            #"density"
            #"angular_damping"
            #"linear_damping"
            #"max_angular_velocity"
            #"max_linear_velocity"
            #"armature"
            #"thickness"
        },
    },

    # --- sim section -------------------------------------------------------
    "sim": {
        "dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, 0.0],
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "substeps": 2,

        #"num_client_threads"
        #"stress_visualization"
        #"stress_visualization_max"
        #"stress_visualization_min"

        "physx": {
            "use_gpu": True,
            #"solver_type" = 1
            #"num_threads" = 4
            #"num_position_iterations" = 4
            #"num_velocity_iterations" = 1
            #"contact_offset"
            #"rest_offset"
            #"bounce_threshold_velocity"
            #"contact_collection"
            #"default_buffer_size_multiplier"
            #"max_depenetration_velocity"
            #"max_gpu_contact_pairs"
            #"num_subscenes"
            #"always_use_articulations"
            #"friction_correlation_distance"
            #"friction_offset_threshold"
        },
        #"flex": {
            #"solver_type"
            #"num_outer_iterations"
            #"num_inner_iterations"
            #"relaxation"
            #"warm_start"
            #"contact_regularization"
            #"deterministic_mode"
            #"dynamic_friction"
            #"friction_mode"
            #"geometric_stiffness"
            #"max_rigid_contacts"
            #"max_soft_contacts"
            #"particle_friction"
            #"return_contacts"
            #"shape_collision_distance"
            #"shape_collision_margin"
            #"static_friction"
        #},
    },

    # --- RL / PPO hyper-params --------------------------------------------
    "rl": {
        "PPO": {
            "num_envs": NUM_ENVS,
            "rollouts": ROLLOUTS,
            "learning_epochs": LEARNING_EPOCHS,
            "mini_batches": MINI_BATCHES,
            
            "learning_rate_scheduler" : KLAdaptiveRL,
            "learning_rate_scheduler_kwargs" : {"kl_threshold": 0.01},
            "state_preprocessor" : RunningStandardScaler,
            "value_preprocessor" : RunningStandardScaler,
            "rewards_shaper" : None,

            "kl_threshold" : 0, #Optional early-stop threshold on KL divergence between old and new policies (0 disables).

            "random_timesteps" : 0, #Number of initial timesteps with random actions before learning or policy-driven sampling.
            "learning_starts" : 0, #Number of environment steps to collect before beginning any gradient updates.
            
            "experiment": {
                "write_interval": "auto",
                "checkpoint_interval": "auto",
                "directory": "./runs",
                "wandb": False,
            },
        },
        "trainer": {
            "rollouts": ROLLOUTS,
            "n_epochs": N_EPOCHS,
            "timesteps": ROLLOUTS * N_EPOCHS,
            "disable_progressbar": False,
            "headless": HEADLESS,
        },
        "memory": {
            "rollouts": ROLLOUTS,
        },
    },
    # --- logging -----------------------------------------------------------
    "log_reward": {
        "log_reward": True,
        "log_reward_interval": 100,  # steps
    },
}