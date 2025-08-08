# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

NUM_ENVS = 4096
N_EPOCHS = 1000
HEADLESS = False
DEBUG_ARROWS = True
ROLLOUTS = 16

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
        
        "sensor_noise_std": 0.0,
        "actuation_noise_std": 0.0,

        "threshold_ang_goal": 0.01, # radians
        "threshold_vel_goal": 0.01, # radians/sec
        "overspeed_ang_vel": 3.14,  # radians/sec
        "goal_time": 10, # seconds
        "sparse_reward": 100.0, # reward for reaching the goal
        "episode_length_s": 30.0, # seconds

        "clipActions": 1.0,
        "clipObservations": 10.0,

        "torque_scale": 1000.0,

        "debug_arrows": DEBUG_ARROWS,
        
        "debug_prints": False,
        
        "discretize_starting_pos": True,

        "asset": {

            "assetRoot": str(Path(__file__).resolve().parent.parent),
            "assetFileName": "satellite.urdf",
            "assetName": "satellite",

            "init_pos_p": [0, 0, 0],
            "init_pos_r": [0, 0, 0, 1],
            
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

            "learning_rate_scheduler" : KLAdaptiveRL,
            "learning_rate_scheduler_kwargs" : {"kl_threshold": 0.01},
            "state_preprocessor" : RunningStandardScaler,
            "value_preprocessor" : RunningStandardScaler,

            "discount_factor" : 0.99, #(γ) Future reward discount; balances immediate versus long-term return.
            "learning_rate" : 1e-3, #Step size for optimizer (e.g. Adam) when updating policy and value networks.
            "grad_norm_clip" : 0.5, #Maximum norm value to clip gradients, preventing exploding gradients.
            "ratio_clip" : 0.2, #(ϵ) PPO’s clipping threshold on the policy probability ratio to constrain updates.
            "value_clip" : 0.2, #Clipping range for value function targets to stabilize value updates.
            "clip_predicted_values" : False, #If enabled, clips the new value predictions to lie within the range defined by value_clip around the old predictions.
            "entropy_loss_scale" : 0.00, #Coefficient multiplying the entropy bonus; encourages exploration when > 0.
            "value_loss_scale" : 1.0, #Coefficient weighting the value function loss in the total loss.
            "kl_threshold" : 0, #Optional early-stop threshold on KL divergence between old and new policies (0 disables).
            "lambda" : 0.95, #(λ) GAE parameter for bias–variance trade-off in advantage estimation.

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
    # --- explosion ---------------------------------------------------------
    "explosion": {
        "enabled": False,
        "explosion_time": 3,  # seconds
    },
    # --- asteroid ----------------------------------------------------------
    "asteroid": {
        "enabled": False,
        "object_mass": 0.0,  # kg
        "object_mass_std": 0.0,  # kg
        "object_mass_time": 120,  # seconds
    },
    # --- randomize masses --------------------------------------------------
    "randomize_masses": {
        "enabled": False,
        "mass_std": 5,
    },
    # --- low-level controller --------------------------------------------
    "controller": {
        "controller_logic": False
    },
    "pid": {
        "rate": {"kp": 0.5, "ki": 0.0, "kd": 0.1},
    },
}