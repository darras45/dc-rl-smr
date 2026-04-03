"""
train_smr.py — Co-train all four SustainDC agents (LS, DC, BAT, SMR) with HAPPO.

Framework
---------
This repo uses HARL (Heterogeneous Agent RL, local in harl/).
The closest standard algorithm is HAPPO — Heterogeneous-Agent PPO — which
assigns each agent its own actor network while sharing a single centralised
critic.  This is the correct choice for agents with different observation
sizes (26 / 14 / 13 / 11) and different reward functions.

Training strategy
-----------------
All four agents are co-trained simultaneously.  The three existing agents
(LS, DC, BAT) have their own actors that continue to learn.  If you want to
hold them fixed after pre-training, point algo_args['train']['model_dir'] at
a checkpoint — the runner will load those weights before the first update and
you can freeze individual actors by setting their learning rate to 0 via
--actor_lr_ls 0 style overrides.  For a fresh 4-agent run (this script's
default) joint training from scratch is the right starting point.

Shared-observation note
-----------------------
HARLSustainDCEnv._create_shared_observation() has a branch for
`nonoverlapping_shared_obs_space=True` that is hard-coded for exactly 3
agents (produces a 29-D vector indexing states[0..2]).  The 4-agent case
requires the `False` branch, which dynamically concatenates all padded
observations.  With 4 agents and max obs dim = 26 the shared state is
26 × 4 = 104-D.  This is set automatically below.

TensorBoard metrics (SMR-specific)
-----------------------------------
Launch with:  tensorboard --logdir ./results
SMR scalars appear under the smr/ and eval_smr/ namespaces:
  smr/avg_reward                 — per-episode mean SMR reward
  smr/avg_power_fraction         — mean P_SMR / P_max
  smr/avg_grid_export_fraction   — mean P_export / P_max
  smr/avg_core_temp_C            — mean core temperature (°C)
  eval_smr/*                     — same signals during evaluation

Usage
-----
  # Default: fresh 4-agent HAPPO run
  python train_smr.py

  # Override any algo or env param via CLI
  python train_smr.py --algo happo --exp_name smr_run1 \\
      --n_rollout_threads 8 --num_env_steps 10000000

  # Resume from checkpoint
  python train_smr.py --load_config ./results/sustaindc/ny/happo/smr_4agent/seed-00001-.../config.json
"""

import os
import sys
import warnings
import argparse
import json

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.envs.sustaindc.sustaindc_logger import SustainDCLogger


# ============================================================
#  Extended logger — adds SMR-specific TensorBoard scalars
# ============================================================

class SMRSustainDCLogger(SustainDCLogger):
    """SustainDCLogger extended with per-episode SMR telemetry.

    Adds four training scalars (smr/*) and three evaluation scalars
    (eval_smr/*) to TensorBoard without modifying the base logger or
    any existing metrics.

    Agent index 3 corresponds to agent_smr in the ordered list
    ['agent_ls', 'agent_dc', 'agent_bat', 'agent_smr'].
    """

    _SMR_AGENT_IDX  = 3    # position of agent_smr in the agents list
    _CI_THRESHOLD   = 0.4  # must match ci_threshold in default_smr_reward

    # ------------------------------------------------------------------
    # Init helpers — add SMR accumulators on top of the base dicts
    # ------------------------------------------------------------------

    def episode_init(self, episode):
        super().episode_init(episode)
        self.metrics.update({
            "smr_reward_sum":              0.0,
            "smr_power_fraction_sum":      0.0,
            "smr_grid_export_fraction_sum": 0.0,
            "smr_core_temp_sum":           0.0,
            "smr_export_dirty_kw_sum":     0.0,  # export when CI >= threshold
            "smr_export_clean_kw_sum":     0.0,  # export when CI <  threshold
            "smr_export_total_kw_sum":     0.0,
            "smr_revenue_sum":             0.0,  # norm_price × export_fraction
            "smr_temp_delta_sum":          0.0,  # core temp change per step
        })

    def eval_init(self):
        super().eval_init()
        self.eval_metrics.update({
            "smr_reward_sum":              0.0,
            "smr_power_fraction_sum":      0.0,
            "smr_grid_export_fraction_sum": 0.0,
            "smr_core_temp_sum":           0.0,
            "smr_export_dirty_kw_sum":     0.0,
            "smr_export_clean_kw_sum":     0.0,
            "smr_export_total_kw_sum":     0.0,
            "smr_revenue_sum":             0.0,
            "smr_temp_delta_sum":          0.0,
        })

    def eval_init_off_policy(self, total_num_steps):
        super().eval_init_off_policy(total_num_steps)
        self.eval_metrics.update({
            "smr_reward_sum":              0.0,
            "smr_power_fraction_sum":      0.0,
            "smr_grid_export_fraction_sum": 0.0,
            "smr_core_temp_sum":           0.0,
            "smr_export_dirty_kw_sum":     0.0,
            "smr_export_clean_kw_sum":     0.0,
            "smr_export_total_kw_sum":     0.0,
            "smr_revenue_sum":             0.0,
            "smr_temp_delta_sum":          0.0,
        })

    # ------------------------------------------------------------------
    # per_step — accumulate SMR signals from the training rollout
    # ------------------------------------------------------------------

    def per_step(self, data):
        super().per_step(data)
        obs, _, rewards, dones, infos, _, _, _, _, _, _ = data

        # rewards shape: (n_rollout_threads, n_agents, 1)
        # Guard against runs with fewer than 4 agents (back-compat)
        if rewards.shape[1] > self._SMR_AGENT_IDX:
            self.metrics["smr_reward_sum"] += float(
                np.mean(rewards[:, self._SMR_AGENT_IDX, 0])
            )

        # infos[env_idx][agent_idx] — all agents share the combined info dict
        for env_i in range(len(infos)):
            step_info = infos[env_i][0]  # same dict for all agents
            max_smr_kw = step_info.get("max_smr_capacity_mw", 50.0) * 1000.0
            self.metrics["smr_power_fraction_sum"] += step_info.get(
                "smr_power_fraction", 0.0
            )
            export_kw = step_info.get("smr_grid_export_kW", 0.0)
            self.metrics["smr_grid_export_fraction_sum"] += export_kw / max(
                max_smr_kw, 1e-9
            )
            self.metrics["smr_core_temp_sum"] += step_info.get("smr_core_temp", 0.0)
            # CI-bucketed export tracking
            norm_ci = step_info.get("norm_CI", 0.0)
            self.metrics["smr_export_total_kw_sum"] += export_kw
            if norm_ci >= self._CI_THRESHOLD:
                self.metrics["smr_export_dirty_kw_sum"] += export_kw
            else:
                self.metrics["smr_export_clean_kw_sum"] += export_kw
            # Multi-objective metrics
            norm_price = step_info.get("norm_price", norm_ci)
            self.metrics["smr_revenue_sum"] += (export_kw / max(max_smr_kw, 1e-9)) * (norm_price - 0.5)
            self.metrics["smr_temp_delta_sum"] += step_info.get("smr_temp_delta", 0.0)

    # ------------------------------------------------------------------
    # eval_per_step — same accumulation during evaluation rollouts
    # ------------------------------------------------------------------

    def eval_per_step(self, eval_data):
        super().eval_per_step(eval_data)
        _, _, eval_rewards, _, eval_infos, _ = eval_data

        if eval_rewards.shape[1] > self._SMR_AGENT_IDX:
            self.eval_metrics["smr_reward_sum"] += float(
                np.mean(eval_rewards[:, self._SMR_AGENT_IDX, 0])
            )

        for env_i in range(len(eval_infos)):
            step_info = eval_infos[env_i][0]
            max_smr_kw = step_info.get("max_smr_capacity_mw", 50.0) * 1000.0
            self.eval_metrics["smr_power_fraction_sum"] += step_info.get(
                "smr_power_fraction", 0.0
            )
            export_kw = step_info.get("smr_grid_export_kW", 0.0)
            self.eval_metrics["smr_grid_export_fraction_sum"] += export_kw / max(
                max_smr_kw, 1e-9
            )
            self.eval_metrics["smr_core_temp_sum"] += step_info.get(
                "smr_core_temp", 0.0
            )
            # CI-bucketed export tracking
            norm_ci = step_info.get("norm_CI", 0.0)
            self.eval_metrics["smr_export_total_kw_sum"] += export_kw
            if norm_ci >= self._CI_THRESHOLD:
                self.eval_metrics["smr_export_dirty_kw_sum"] += export_kw
            else:
                self.eval_metrics["smr_export_clean_kw_sum"] += export_kw
            # Multi-objective metrics
            norm_price = step_info.get("norm_price", norm_ci)
            self.eval_metrics["smr_revenue_sum"] += (export_kw / max(max_smr_kw, 1e-9)) * (norm_price - 0.5)
            self.eval_metrics["smr_temp_delta_sum"] += step_info.get("smr_temp_delta", 0.0)

    # ------------------------------------------------------------------
    # episode_log — write SMR scalars and print to console
    # ------------------------------------------------------------------

    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        # Parent writes the standard 3-agent scalars then resets self.metrics
        # without the SMR keys, so we capture our values first.
        n = max(self.metrics["step_count"], 1)
        avg_smr_reward    = self.metrics["smr_reward_sum"]              / n
        avg_smr_pwr       = self.metrics["smr_power_fraction_sum"]      / n
        avg_smr_export    = self.metrics["smr_grid_export_fraction_sum"] / n
        avg_smr_temp      = self.metrics["smr_core_temp_sum"]           / n
        avg_smr_revenue   = self.metrics["smr_revenue_sum"]             / n
        avg_smr_tdelta    = self.metrics["smr_temp_delta_sum"]          / n
        total_export      = max(self.metrics["smr_export_total_kw_sum"], 1e-9)
        dirty_export_frac = self.metrics["smr_export_dirty_kw_sum"] / total_export
        clean_export_frac = self.metrics["smr_export_clean_kw_sum"] / total_export

        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)

        # --- TensorBoard ---
        self.writter.add_scalar("smr/avg_reward",
                                avg_smr_reward,   self.total_num_steps)
        self.writter.add_scalar("smr/avg_power_fraction",
                                avg_smr_pwr,      self.total_num_steps)
        self.writter.add_scalar("smr/avg_grid_export_fraction",
                                avg_smr_export,   self.total_num_steps)
        self.writter.add_scalar("smr/avg_core_temp_C",
                                avg_smr_temp,     self.total_num_steps)
        self.writter.add_scalar("smr/export_dirty_ci_fraction",
                                dirty_export_frac, self.total_num_steps)
        self.writter.add_scalar("smr/export_clean_ci_fraction",
                                clean_export_frac, self.total_num_steps)
        self.writter.add_scalar("smr/avg_revenue_proxy",
                                avg_smr_revenue,  self.total_num_steps)
        self.writter.add_scalar("smr/avg_temp_delta_C",
                                avg_smr_tdelta,   self.total_num_steps)

        # --- Console ---
        print(
            f"  [SMR] reward={avg_smr_reward:+.4f} | "
            f"P_frac={avg_smr_pwr:.3f} | "
            f"export_frac={avg_smr_export:.3f} | "
            f"T_core={avg_smr_temp:.1f}°C | "
            f"ΔT={avg_smr_tdelta:.2f}°C | "
            f"revenue={avg_smr_revenue:.4f} | "
            f"dirty={dirty_export_frac:.1%}  clean={clean_export_frac:.1%}"
        )

        # Restore SMR accumulators (parent reset wiped them)
        self.metrics.update({
            "smr_reward_sum":              0.0,
            "smr_power_fraction_sum":      0.0,
            "smr_grid_export_fraction_sum": 0.0,
            "smr_core_temp_sum":           0.0,
            "smr_export_dirty_kw_sum":     0.0,
            "smr_export_clean_kw_sum":     0.0,
            "smr_export_total_kw_sum":     0.0,
            "smr_revenue_sum":             0.0,
            "smr_temp_delta_sum":          0.0,
        })

    # ------------------------------------------------------------------
    # eval_log — write eval SMR scalars
    # ------------------------------------------------------------------

    def eval_log(self, eval_episode):
        n = max(self.eval_metrics["step_count"], 1)
        avg_smr_reward    = self.eval_metrics["smr_reward_sum"]              / n
        avg_smr_pwr       = self.eval_metrics["smr_power_fraction_sum"]      / n
        avg_smr_export    = self.eval_metrics["smr_grid_export_fraction_sum"] / n
        avg_smr_temp      = self.eval_metrics["smr_core_temp_sum"]           / n
        avg_smr_revenue   = self.eval_metrics["smr_revenue_sum"]             / n
        avg_smr_tdelta    = self.eval_metrics["smr_temp_delta_sum"]          / n
        total_export      = max(self.eval_metrics["smr_export_total_kw_sum"], 1e-9)
        dirty_export_frac = self.eval_metrics["smr_export_dirty_kw_sum"] / total_export
        clean_export_frac = self.eval_metrics["smr_export_clean_kw_sum"] / total_export

        super().eval_log(eval_episode)

        self.writter.add_scalar("eval_smr/avg_reward",
                                avg_smr_reward,    self.total_num_steps)
        self.writter.add_scalar("eval_smr/avg_power_fraction",
                                avg_smr_pwr,       self.total_num_steps)
        self.writter.add_scalar("eval_smr/avg_grid_export_fraction",
                                avg_smr_export,    self.total_num_steps)
        self.writter.add_scalar("eval_smr/avg_core_temp_C",
                                avg_smr_temp,      self.total_num_steps)
        self.writter.add_scalar("eval_smr/export_dirty_ci_fraction",
                                dirty_export_frac, self.total_num_steps)
        self.writter.add_scalar("eval_smr/export_clean_ci_fraction",
                                clean_export_frac, self.total_num_steps)
        self.writter.add_scalar("eval_smr/avg_revenue_proxy",
                                avg_smr_revenue,   self.total_num_steps)
        self.writter.add_scalar("eval_smr/avg_temp_delta_C",
                                avg_smr_tdelta,    self.total_num_steps)

        self.eval_metrics.update({
            "smr_reward_sum":              0.0,
            "smr_power_fraction_sum":      0.0,
            "smr_grid_export_fraction_sum": 0.0,
            "smr_core_temp_sum":           0.0,
            "smr_export_dirty_kw_sum":     0.0,
            "smr_export_clean_kw_sum":     0.0,
            "smr_export_total_kw_sum":     0.0,
            "smr_revenue_sum":             0.0,
            "smr_temp_delta_sum":          0.0,
        })


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train all four SustainDC agents (LS, DC, BAT, SMR) with HAPPO.",
    )
    parser.add_argument(
        "--algo", type=str, default="happo",
        choices=["happo", "hatrpo", "haa2c", "mappo"],
        help="On-policy HARL algorithm. HAPPO is recommended for heterogeneous agents.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="smr_4agent",
        help="Experiment name (appended to the results directory path).",
    )
    parser.add_argument(
        "--load_config", type=str, default="",
        help="Path to a saved config.json to resume a previous training run.",
    )

    args, unparsed_args = parser.parse_known_args()

    def process(v):
        try:
            return eval(v)
        except Exception:
            return v

    keys   = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = dict(zip(keys, values))
    args = vars(args)
    args["env"] = "sustaindc"   # required by on_policy_base_runner.init_dir()

    # ------------------------------------------------------------------ #
    #  Build config                                                        #
    # ------------------------------------------------------------------ #

    if args["load_config"]:
        with open(args["load_config"], encoding="utf-8") as f:
            saved = json.load(f)
        args["algo"] = saved["main_args"]["algo"]
        algo_args    = saved["algo_args"]
        env_args     = saved["env_args"]

    else:
        algo_args, env_args = get_defaults_yaml_args(args["algo"], "sustaindc")

        # --- Environment: activate all four agents ---
        env_args["agents"] = ["agent_ls", "agent_dc", "agent_bat", "agent_smr"]
        env_args["datacenter_capacity_mw"]  = 5
        env_args["smr_reward"]             = "default_smr_reward_lmp_dispatch"
        env_args["max_smr_capacity_mw"]    = 6.0
        env_args["smr_min_power_fraction"] = 0.2
        env_args["month"]                  = 0          # January; runner will rotate months
        env_args["days_per_episode"]       = 7
        env_args["location"]               = "ny"
        # Merit-order dispatch weights
        env_args["alpha_carbon"]           = 0.4   # CI weight in dispatch_value
        env_args["beta_revenue"]           = 0.5   # price weight in dispatch_value
        env_args["gamma_longevity"]        = 0.1   # soft thermal safety weight
        env_args["price_threshold"]        = 0.5   # dispatch crossover (centre of norm_price)

        # REQUIRED: the nonoverlapping path hardcodes 3 agents (29-D shared state).
        # With 4 agents we use the concatenation path: max_obs_dim(26) × 4 = 104-D.
        env_args["nonoverlapping_shared_obs_space"] = False

        # --- Device: enable GPU; set torch CPU threads to 1 when CUDA is used
        # (excess CPU threads contend with rollout workers and waste cores).
        # On CPU-only machines the cuda flag is silently ignored by init_device().
        algo_args["device"]["cuda"]          = True
        algo_args["device"]["torch_threads"] = 1

        # --- Algorithm: VM-optimised PPO hyperparameters ---
        # n_rollout_threads: one env per physical CPU core is a good rule of thumb.
        # 32 saturates a 30-64-core Lambda VM; override with --n_rollout_threads N
        # if your VM has more cores.
        algo_args["train"]["n_rollout_threads"]      = 32
        # episode_length is the rollout horizon (steps before each PPO update),
        # not the env episode length (controlled by days_per_episode above).
        # 2048 × 32 threads = 65 536 samples/update — large enough to keep the
        # GPU busy without starving it of data between updates.
        algo_args["train"]["episode_length"]         = 2048
        algo_args["train"]["num_env_steps"]          = 5_000_000
        algo_args["train"]["log_interval"]           = 10  # log every 10 episodes
        algo_args["train"]["eval_interval"]          = 25  # eval every 25 episodes
        algo_args["eval"]["use_eval"]                = True
        algo_args["eval"]["n_eval_rollout_threads"]  = 4
        algo_args["eval"]["eval_episodes"]           = 2
        # 16 mini-batches over 65 536 samples → 4 096 samples/mini-batch;
        # keeps GPU utilisation high for the [64,64] MLP actors/critic.
        algo_args["algo"]["actor_num_mini_batch"]    = 16
        algo_args["algo"]["critic_num_mini_batch"]   = 16
        # Independent actor networks per agent (correct for heterogeneous obs sizes)
        algo_args["algo"]["share_param"]             = False

    # Apply any extra CLI overrides (e.g. --n_rollout_threads 8)
    update_args(unparsed_dict, algo_args, env_args)

    # ------------------------------------------------------------------ #
    #  Patch logger registry with SMR-aware logger                        #
    # ------------------------------------------------------------------ #
    import harl.envs as harl_envs
    _original_logger = harl_envs.LOGGER_REGISTRY["sustaindc"]
    harl_envs.LOGGER_REGISTRY["sustaindc"] = SMRSustainDCLogger

    print("=" * 65)
    print("  SustainDC × SMR — HAPPO Co-Training")
    print("=" * 65)
    print(f"  Algorithm  : {args['algo']}")
    print(f"  Agents     : {env_args['agents']}")
    print(f"  SMR cap.   : {env_args['max_smr_capacity_mw']} MW  |  "
          f"P_min frac: {env_args['smr_min_power_fraction']}")
    print(f"  Reward     : {env_args['smr_reward']}  "
          f"(α={env_args.get('alpha_carbon',0.4)}  "
          f"β={env_args.get('beta_revenue',0.5)}  "
          f"γ={env_args.get('gamma_longevity',0.1)}  "
          f"θ={env_args.get('price_threshold',0.5)})")
    print(f"  Device     : {'CUDA (GPU)' if algo_args['device']['cuda'] else 'CPU'}")
    print(f"  Rollout Ts : {algo_args['train']['n_rollout_threads']}  |  "
          f"episode_length: {algo_args['train']['episode_length']}  |  "
          f"total steps: {algo_args['train']['num_env_steps']:,}")
    print(f"  Mini-batch : actor={algo_args['algo']['actor_num_mini_batch']}  |  "
          f"critic={algo_args['algo']['critic_num_mini_batch']}")
    print(f"  Logger     : {SMRSustainDCLogger.__name__}")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #
    from harl.runners import RUNNER_REGISTRY
    try:
        runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
        runner.run()
        runner.close()
    finally:
        # Restore original logger so other imports in the same process are clean
        harl_envs.LOGGER_REGISTRY["sustaindc"] = _original_logger


if __name__ == "__main__":
    main()
