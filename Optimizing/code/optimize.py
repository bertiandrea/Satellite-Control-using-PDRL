# train.py

from code.configs.satellite_config_opt import CONFIG
from code.envs.satellite import Satellite
from code.models.custom_model import Policy, Value, Shared
from code.envs.wrappers.isaacgym_envs_wrapper import IsaacGymWrapper
from code.rewards.satellite_reward import (
    SimpleReward,
    CurriculumReward,
    WeightedSumReward,
    TwoPhaseReward,
    ExponentialStabilizationReward,
    ContinuousDiscreteEffortReward,
    ShapingReward
)
from code.trainer.trainer import Trainer

import isaacgym
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed

import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Optimization imports
import os
import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
TENSORBOARD_TAG = "Reward / Instantaneous reward (mean)"
N_TRIALS = 1000
# ──────────────────────────────────────────────────────────────────────────────

REWARD_MAP = {
    "simple": SimpleReward,
    "curriculum": CurriculumReward,
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

def sample_ppo_params(trial: optuna.Trial):
    return {
        "discount_factor": trial.suggest_float("discount_factor", 0.90, 0.999),
        "lambda":          trial.suggest_float("lambda", 0.90,   0.999),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-5, 1e-2),
        "grad_norm_clip": trial.suggest_float("grad_norm_clip", 0.1, 1.0),
        "ratio_clip":   trial.suggest_float("ratio_clip", 0.1, 0.3),
        "value_clip": trial.suggest_float("value_clip", 0.1, 0.3),
        "clip_predicted_values": trial.suggest_categorical("clip_predicted_values", [True, False]),
        "entropy_loss_scale": trial.suggest_float("entropy_loss_scale", 0.0, 0.05),
        "value_loss_scale": trial.suggest_float("value_loss_scale", 0.5, 2.0),
    }

def objective(trial: optuna.Trial) -> float:
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

    hp = sample_ppo_params(trial)
    cfg_ppo.update({
        "discount_factor":       hp["discount_factor"],
        "lambda":                hp["lambda"],
        "learning_rate":         hp["learning_rate"],
        "grad_norm_clip":        hp["grad_norm_clip"],
        "ratio_clip":            hp["ratio_clip"],
        "value_clip":            hp["value_clip"],
        "clip_predicted_values": hp["clip_predicted_values"],
        "entropy_loss_scale":    hp["entropy_loss_scale"],
        "value_loss_scale":      hp["value_loss_scale"],
    })

    agent = PPO(models=models,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.state_space,
            action_space=env.action_space,
            device=env.device)
    
    trainer = Trainer(cfg=CONFIG["rl"]["trainer"], env=env, agent=agent)
    
    try:
        best_mean_return = 0
        states, infos = trainer.init_step_train()
        for epoch in range(CONFIG["rl"]["trainer"]["n_epochs"]):
            #############################################################################
            for n in range(CONFIG["rl"]["trainer"]["rollouts"]):
                states, infos, rewards = trainer.step_train(states, infos, (epoch * CONFIG["rl"]["trainer"]["rollouts"]) + n)
            #############################################################################
            mean_return = torch.sum(rewards, dim=0).item()
            print(f"Epoch {epoch+1}/{CONFIG['rl']['trainer']['n_epochs']}, mean_return: {mean_return:.3f}")
            if mean_return > best_mean_return:
                best_mean_return = mean_return
            #############################################################################
            trial.report(mean_return, step=epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned() 
            #############################################################################
    finally:
        env.close() # Force environment close to avoid memory leaks
    
    return best_mean_return

def main():
    global args
    args = parse_args()

    if CONFIG["set_seed"]:
        set_seed(CONFIG["seed"])
    
    ##################################################################
    
    study = optuna.create_study(
        study_name=f"Satellite_{args.reward_fn}",
        storage="sqlite:///optuna_study.db",
        sampler=TPESampler(n_startup_trials=10, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        direction="maximize",
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    ##################################################################

    log_dir = "/home/andreaberti"
    out_path = log_dir + "/optimizer_results/satellite/best_hyperparams.json"
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"\n✅ Salvato in {out_path}")
    print(f"Numero di trials: {len(study.trials)}")
    print(f"➤ mean_return migliore: {study.best_value:.3f}")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    
if __name__ == "__main__":
    main()