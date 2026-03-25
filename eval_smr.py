"""
eval_smr.py — Deterministic policy evaluation for the trained agent_smr.

How to use
----------
1. Point MODEL_CHECKPOINT_DIR at the `models/` subdirectory produced by
   train_smr.py.  That is the ONLY line you need to edit:

       MODEL_CHECKPOINT_DIR = "./results/sustaindc/ny/happo/smr_4agent/" \\
                              "seed-00001-<your-timestamp>/models"

   The script infers config.json from one directory above that path.

2. Run:  python eval_smr.py

3. Output: smr_eval_policy.png saved to the current working directory.

Evaluation strategy
-------------------
- agent_smr  : trained HAPPO actor loaded from .pt weights, deterministic.
- agent_ls   : BaseFallback — hold (action 1).
- agent_dc   : BaseFallback — hold setpoint (action 1).
- agent_bat  : BaseFallback — idle (action 2).

This isolates the SMR policy while keeping the rest of the environment
physically consistent (DC runs, battery idles, workload shifts nothing).

Plot layout
-----------
Pane 1 — SMR Power Output vs DC Demand; green fill = grid export surplus.
Pane 2 — Normalised Grid Carbon Intensity (0 = cleanest, 1 = dirtiest).
Pane 3 — SMR action scatter: Ramp Down / Hold / Ramp Up over time.
"""

import os
import sys
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harl.runners import RUNNER_REGISTRY
from harl.utils.trans_tools import _t2n

# ============================================================
#  >>>  UPDATE THIS PATH WHEN YOUR TRAINING RUN FINISHES  <<<
# ============================================================
MODEL_CHECKPOINT_DIR = (
    "./results/sustaindc/ny/happo/smr_4agent/"
    "seed-00001-2026-03-25-14-22-00/models"
)
# ============================================================

# Agent index constants (must match train_smr.py agents list order)
_IDX_LS  = 0
_IDX_DC  = 1
_IDX_BAT = 2
_IDX_SMR = 3

# Base fallback actions for the three non-SMR agents
BASE_ACTIONS = {
    _IDX_LS:  1,   # hold (do nothing)
    _IDX_DC:  1,   # hold setpoint
    _IDX_BAT: 2,   # idle
}

ACTION_LABELS = {0: "Ramp Down", 1: "Hold", 2: "Ramp Up"}
ACTION_COLORS = {0: "#e74c3c", 1: "#95a5a6", 2: "#2ecc71"}
OUTPUT_FILE   = "smr_eval_policy.png"


# ============================================================
#  Load config from the run directory
# ============================================================

run_dir    = os.path.normpath(os.path.join(MODEL_CHECKPOINT_DIR, ".."))
config_path = os.path.join(run_dir, "config.json")

if not os.path.isfile(config_path):
    raise FileNotFoundError(
        f"config.json not found at {config_path}\n"
        "Make sure MODEL_CHECKPOINT_DIR points to the `models/` subfolder "
        "of a completed train_smr.py run."
    )

with open(config_path, encoding="utf-8") as f:
    saved = json.load(f)

main_args = saved["main_args"]
algo_args  = saved["algo_args"]
env_args   = saved["env_args"]

# Override runtime settings for a clean single-episode evaluation
algo_args["train"]["n_rollout_threads"]      = 1
algo_args["eval"]["n_eval_rollout_threads"]  = 1
algo_args["eval"]["eval_episodes"]           = 1
algo_args["eval"]["use_eval"]                = True
algo_args["train"]["model_dir"]              = MODEL_CHECKPOINT_DIR
# Save logs to a separate eval subdirectory so we don't pollute training logs
algo_args["logger"]["log_dir"]               = os.path.join(run_dir, "eval")

env_args["days_per_episode"] = 7   # one week — 7 × 24 × 4 = 672 steps

# Ensure env is correct (guard for configs saved before agent_smr)
if "agent_smr" not in env_args.get("agents", []):
    env_args["agents"] = ["agent_ls", "agent_dc", "agent_bat", "agent_smr"]
if "max_smr_capacity_mw" not in env_args:
    env_args["max_smr_capacity_mw"] = 50.0

max_smr_mw = float(env_args["max_smr_capacity_mw"])

# ============================================================
#  Build runner — restore() is called inside __init__ and
#  loads actor_agent{0..3}.pt from MODEL_CHECKPOINT_DIR
# ============================================================

print(f"Loading checkpoint from:\n  {MODEL_CHECKPOINT_DIR}\n")
main_args["env"] = "sustaindc"  # required by init_dir()

# Patch logger registry so the runner does not crash on the SMR reward key
import harl.envs as _harl_envs
from harl.envs.sustaindc.sustaindc_logger import SustainDCLogger
_harl_envs.LOGGER_REGISTRY["sustaindc"] = SustainDCLogger  # plain logger for eval

runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
runner.prep_rollout()   # set all actors to eval mode

num_agents = runner.num_agents
assert num_agents == 4, (
    f"Expected 4 agents, runner found {num_agents}. "
    "Check that agent_smr is in env_args['agents']."
)

# ============================================================
#  Evaluation loop
# ============================================================

print("Running deterministic evaluation episode …")
print(f"  SMR nameplate capacity : {max_smr_mw:.1f} MW")
print(f"  Episode length         : {env_args['days_per_episode']} days\n")

eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset()

eval_rnn_states = np.zeros(
    (1, num_agents, runner.recurrent_n, runner.rnn_hidden_size),
    dtype=np.float32,
)
eval_masks = np.ones((1, num_agents, 1), dtype=np.float32)

# Telemetry buffers
tel_smr_power_mw   = []
tel_dc_demand_mw   = []
tel_ci_norm        = []
tel_smr_action_int = []   # 0 / 1 / 2

step = 0
while True:
    action_collector = []

    for agent_id in range(num_agents):
        if agent_id == _IDX_SMR:
            # Trained deterministic policy for the SMR agent
            actions_t, rnn_t = runner.actor[agent_id].act(
                eval_obs[:, agent_id],
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None else None,
                deterministic=True,
            )
            eval_rnn_states[:, agent_id] = _t2n(rnn_t)
            action_collector.append(_t2n(actions_t))          # shape (1, 1)
        else:
            # Fixed fallback for LS / DC / BAT
            action_collector.append(
                np.array([[BASE_ACTIONS[agent_id]]], dtype=np.int64)
            )

    # Assemble actions: list of n_agents × (1,1)  →  (1, n_agents, 1)
    eval_actions = np.array(action_collector).transpose(1, 0, 2)

    (
        eval_obs,
        eval_share_obs,
        eval_rewards,
        eval_dones,
        eval_infos,
        eval_available_actions,
    ) = runner.eval_envs.step(eval_actions)

    # ---- Collect telemetry from the combined info dict ----
    info = eval_infos[0][0]   # env 0, agent 0 — all agents share the same dict

    smr_power_mw = info.get("smr_power_output_kW", 0.0) / 1000.0
    dc_demand_mw = info.get("dc_total_power_kW",   0.0) / 1000.0
    ci_norm      = float(info.get("norm_CI", 0.0))
    smr_action   = int(action_collector[_IDX_SMR][0, 0])

    tel_smr_power_mw.append(smr_power_mw)
    tel_dc_demand_mw.append(dc_demand_mw)
    tel_ci_norm.append(ci_norm)
    tel_smr_action_int.append(smr_action)

    step += 1

    # Reset RNN state and masks if any environment episode ended
    eval_dones_env = np.all(eval_dones, axis=1)   # shape (1,)
    if eval_dones_env[0]:
        eval_rnn_states[:] = 0.0
        eval_masks = np.zeros((1, num_agents, 1), dtype=np.float32)
        break   # single episode evaluation

runner.close()

# ============================================================
#  Convert to arrays and build time axis (hours)
# ============================================================

T         = len(tel_smr_power_mw)
timesteps = np.arange(T) * 0.25          # 15-min steps → hours
smr_pw    = np.array(tel_smr_power_mw)
dc_pw     = np.array(tel_dc_demand_mw)
ci        = np.array(tel_ci_norm)
actions   = np.array(tel_smr_action_int)

export_mask = smr_pw > dc_pw             # True where SMR outpaces DC demand

print(f"Episode complete — {T} timesteps ({T * 0.25:.1f} hours)")
print(f"  Mean SMR output   : {smr_pw.mean():.3f} MW  "
      f"(range {smr_pw.min():.3f}–{smr_pw.max():.3f})")
print(f"  Mean DC demand    : {dc_pw.mean():.3f} MW")
print(f"  Grid export steps : {export_mask.sum()} / {T}  "
      f"({100*export_mask.mean():.1f} %)")
print(f"  Action counts     : "
      f"↓ {(actions==0).sum()}  · {(actions==1).sum()}  ↑ {(actions==2).sum()}")

# ============================================================
#  Plot
# ============================================================

fig, axes = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={"height_ratios": [3, 2, 1.5], "hspace": 0.08},
)
fig.suptitle(
    f"SMR Agent — Deterministic Policy Evaluation  "
    f"(capacity = {max_smr_mw:.0f} MW, {env_args['days_per_episode']} days)",
    fontsize=13, fontweight="bold", y=0.98,
)

# ── Pane 1: Power ────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(timesteps, smr_pw, color="#2980b9", linewidth=1.6,
         label="SMR Power Output")
ax1.plot(timesteps, dc_pw,  color="#e67e22", linewidth=1.6,
         linestyle="--", label="DC Power Demand")
ax1.fill_between(
    timesteps, dc_pw, smr_pw,
    where=export_mask,
    alpha=0.25, color="#27ae60", interpolate=True, label="Grid Export (SMR > DC)",
)
ax1.set_ylabel("Power (MW)", fontsize=11)
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.set_title("SMR Power Output vs Data Center Demand", fontsize=10,
              loc="left", pad=4)

# ── Pane 2: Carbon Intensity ──────────────────────────────────
ax2 = axes[1]
# Colour the CI line by its own value (green → red via a colormap)
from matplotlib.collections import LineCollection
points  = np.array([timesteps, ci]).T.reshape(-1, 1, 2)
segs    = np.concatenate([points[:-1], points[1:]], axis=1)
lc      = LineCollection(segs, cmap="RdYlGn_r", norm=plt.Normalize(0, 1),
                         linewidth=2.0)
lc.set_array(ci)
ax2.add_collection(lc)
ax2.set_xlim(timesteps[0], timesteps[-1])
ax2.set_ylim(-0.05, 1.05)
cbar = fig.colorbar(lc, ax=ax2, pad=0.01, fraction=0.015)
cbar.set_label("CI (norm)", fontsize=8)
ax2.set_ylabel("Norm. Carbon\nIntensity", fontsize=11)
ax2.grid(axis="y", linestyle=":", alpha=0.5)
ax2.set_title("Grid Carbon Intensity  (0 = cleanest, 1 = dirtiest)",
              fontsize=10, loc="left", pad=4)

# ── Pane 3: Action Scatter ────────────────────────────────────
ax3 = axes[2]
for a_val, label in ACTION_LABELS.items():
    mask = actions == a_val
    if mask.any():
        ax3.scatter(
            timesteps[mask],
            np.full(mask.sum(), a_val),
            c=ACTION_COLORS[a_val], s=12, alpha=0.8,
            label=label, zorder=3,
        )
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(["Ramp\nDown", "Hold", "Ramp\nUp"], fontsize=9)
ax3.set_xlabel("Time (hours)", fontsize=11)
ax3.set_ylabel("Action", fontsize=11)
ax3.grid(axis="x", linestyle=":", alpha=0.5)
ax3.set_ylim(-0.5, 2.5)
ax3.set_title("SMR Control Actions", fontsize=10, loc="left", pad=4)

# Day-boundary vertical lines across all panes
for d in range(1, env_args["days_per_episode"]):
    for ax in axes:
        ax.axvline(x=d * 24, color="black", linewidth=0.6,
                   linestyle="--", alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {os.path.abspath(OUTPUT_FILE)}")
