# train.py

from satellite.configs.satellite_config import CONFIG
from satellite.envs.satellite import Satellite
from satellite.models.custom_model import Policy, Value, Shared
from satellite.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper
from satellite.CAPS.agent_wrapper_CAPS import PPOWrapperCAPS
from satellite.rewards.satellite_reward import (
    TestReward,
    TestRewardSpin,
    TestRewardCurriculum,
    WeightedSumReward,
    TwoPhaseReward,
    ExponentialStabilizationReward,
    ContinuousDiscreteEffortReward,
    ShapingReward,
)

import isaacgym
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Profiler imports
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
)
import os
import pandas as pd
# ──────────────────────────────────────────────────────────────────────────────

REWARD_MAP = {
    "test": TestReward,
    "test_spin": TestRewardSpin,
    "test_curriculum": TestRewardCurriculum,
    "weighted_sum": WeightedSumReward,
    "two_phase": TwoPhaseReward,
    "exp_stabilization": ExponentialStabilizationReward,
    "continuous_discrete_effort": ContinuousDiscreteEffortReward,
    "shaping": ShapingReward,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training con reward function selezionabile")
    parser.add_argument(
        "--reward-fn",
        choices=list(REWARD_MAP.keys()),
        default="test",
        help="Which RewardFunction?"
    )
    return parser.parse_args()

def setup_profiler(log_dir = "/home/andreaberti"):
    dir_path = log_dir + "/profiler_logs/ISAAC_SKRL_Integration/satellite"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    #################################################################################
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(dir_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    return prof

def save_profiler_results(prof, log_dir="/home/andreaberti"):
    events = prof.key_averages()
    #################################################################################
    output_path = log_dir + "/profiler_text/ISAAC_SKRL_Integration/satellite/text_output.txt"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(events.table(sort_by="self_cuda_time_total", row_limit=500))
        f.write("\n\n\n")
        f.write(events.table(sort_by="self_cpu_time_total", row_limit=500))
        f.write("\n\n\n")
        f.write(events.table(sort_by="self_cuda_memory_usage", row_limit=500))
        f.write("\n\n\n")
        f.write(events.table(sort_by="self_cpu_memory_usage", row_limit=500))
    #################################################################################
    rows = []
    for e in events:
        rows.append({
            "name":               e.key[:50],  # Truncate to 50 characters
            "self_cpu_time_ms":   e.self_cpu_time_total / 1e3,
            "cpu_time_ms":        e.cpu_time_total / 1e3,
            "self_cuda_time_ms":  e.self_device_time_total / 1e3,
            "cuda_time_ms":       e.device_time_total / 1e3,
            "self_cpu_memory_bytes":   e.self_cpu_memory_usage,
            "self_cuda_memory_bytes":  e.self_device_memory_usage,
            "cpu_memory_bytes":   e.cpu_memory_usage,
            "cuda_memory_bytes":  e.device_memory_usage,
            "count":              e.count,
            "flops":              e.flops,
            "device_type":        str(e.device_type),
        })
    df = pd.DataFrame(rows)
    df['order'] = df['name'].str[0].map({'#': 0, '$': 1}).fillna(2).astype(int)
    df = df.sort_values(['order', 'name'], ascending=[True, True])
    df = df.drop(columns='order')
    #################################################################################
    print(df.head(40))
    #################################################################################
    csv_path = log_dir + "/profiler_text/ISAAC_SKRL_Integration/satellite/csv_output.csv"
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

def main():
    args = parse_args()

    if CONFIG["set_seed"]:
        set_seed(CONFIG["seed"])
    
    #################################################################################

    env = Satellite(
        cfg=CONFIG,
        rl_device=CONFIG["rl_device"],
        sim_device=CONFIG["sim_device"],
        graphics_device_id=CONFIG["graphics_device_id"],
        headless=CONFIG["headless"],
        virtual_screen_capture=CONFIG["virtual_screen_capture"],
        force_render= CONFIG["force_render"],
        reward_fn=REWARD_MAP[args.reward_fn](CONFIG["log_reward"]["log_reward"], CONFIG["log_reward"]["log_reward_interval"])
    )
    
    env = IsaacGymWrapper(env)

    memory = RandomMemory(memory_size=CONFIG["rl"]["memory"]["rollouts"], num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Shared(env.state_space, env.action_space, env.device)
    models["value"] = models["policy"]  # Shared model for policy and value
   
    CONFIG["rl"]["PPO"]["state_preprocessor_kwargs"] = {
        "size": env.state_space, "device": env.device
    }
    CONFIG["rl"]["PPO"]["value_preprocessor_kwargs"] = {
        "size": 1, "device": env.device
    }
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update(CONFIG["rl"]["PPO"])

    if CONFIG["CAPS"]["enabled"]:
        cfg_ppo.update(CONFIG["CAPS"])
        agent = PPOWrapperCAPS(models=models,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.state_space,
                action_space=env.action_space,
                device=env.device)
    else:
        agent = PPO(models=models,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.state_space,
                action_space=env.action_space,
                device=env.device)
    
    trainer = SequentialTrainer(cfg=CONFIG["rl"]["trainer"], env=env, agents=agent)

    if CONFIG["profile"]:
        prof = setup_profiler()
        prof.start()

    trainer.train()
    
    if CONFIG["profile"]:
        prof.stop()
        save_profiler_results(prof)
    
    #################################################################################

if __name__ == "__main__":
    main()