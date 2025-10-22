# satellite_config.py

from pathlib import Path
import numpy as np

import isaacgym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

NUM_ENVS = 4096
TIMESTEPS = 120000
HEADLESS = False
DEBUG_ARROWS = True
LOG_TRAJECTORIES = False

ROLLOUTS = 16
LEARNING_EPOCHS = 8
MINI_BATCHES = 2

CONFIG = {
    # --- seed & devices ----------------------------------------------------
    "set_seed": True,
    "seed": 42,

    "profile": False,

    "physics_engine": "physx",

    "rl_device": "cuda:0",
    "sim_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": HEADLESS,
    "virtual_screen_capture": False,
    "force_render": False,

    # --- env section -------------------------------------------------------
    "env": {
        "numEnvs": NUM_ENVS,
        "numObservations": 15, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3)
        "numStates": 18, # satellite_quats (4) + quat_diff (4) + quat_diff_rad (1) + satellite_angacc (3) + actions (3) + satellite_angvels (3)
        "numActions": 3,
       
        "clipActions": 1.0,
        "clipObservations": 10.0,

        "max_episode_length": 1000.0,

        "envSpacing": 3.0,
        "torque_scale": 200.0,
        "debug_arrows": DEBUG_ARROWS,
        "debug_prints": False,
        "discretize_starting_pos": True,
        "log_trajectories": LOG_TRAJECTORIES,

        "asset": {
            "assetRoot": str(Path(__file__).resolve().parent.parent),
            "assetFileName": "satellite.urdf",
            "assetName": "satellite",
        },
    },

    # --- sim section -------------------------------------------------------
    "sim": {
        "dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, 0.0],
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "substeps": 2,

        "physx": {
            "use_gpu": True,
        },
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

            "discount_factor" : 0.99, #(γ) Future reward discount; balances immediate versus long-term return.
            "learning_rate" : 1e-3, #Step size for optimizer (e.g. Adam) when updating policy and value networks.
            "grad_norm_clip" : 1.0, #Maximum norm value to clip gradients, preventing exploding gradients.
            "ratio_clip" : 0.2, #(ϵ) PPO’s clipping threshold on the policy probability ratio to constrain updates.
            "clip_predicted_values" : True, #If enabled, clips the new value predictions to lie within the range defined by value_clip around the old predictions.
            "value_clip" : 0.2, #Clipping range for value function targets to stabilize value updates.
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
            "timesteps": TIMESTEPS,
            "disable_progressbar": False,
            "headless": HEADLESS,
            "stochastic_evaluation": False,
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
    # --- CAPS --------------------------------------------------------------
    "CAPS": {
        "enabled": False,
        "lambda_temporal_smoothness": 0.1,  # λ_t
        "lambda_spatial_smoothness": 0.1,   # λ_s
        "noise_std": 0.5,                   # σ
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
    }
}