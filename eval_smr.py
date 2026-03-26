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
Run 1 (No-SMR Baseline):  SustainDC with use_smr=False and base agents for
                           all three agents (LS / DC / BAT).  Records the
                           full DC grid draw and CI so carbon cost can be
                           computed without any nuclear offset.

Run 2 (Trained RL Policy): SustainDC with use_smr=True.  Trained HAPPO actor
                           loaded for agent_smr; base agents for LS / DC / BAT.
                           Same random seed as Run 1 so both episodes start on
                           the same calendar day/hour — true apples-to-apples.

Plot layout (5 panes)
---------------------
Pane 1 — SMR Power Output vs DC Demand; green fill = grid export surplus.
Pane 2 — Normalised Grid Carbon Intensity (0 = cleanest, 1 = dirtiest).
Pane 3 — SMR action scatter: Ramp Down / Hold / Ramp Up over time.
Pane 4 — CI-bucketed grid export (dirty vs. clean grid).
Pane 5 — Carbon footprint overlay: No-SMR baseline (dashed grey) vs RL SMR
          (solid blue); green fill = carbon savings; title shows % reduction.
"""

import os
import sys
import json
import random
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
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

# Both runs are seeded identically so they start on the same calendar day/hour.
EVAL_SEED = 42

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

CI_THRESHOLD = 0.4  # must match ci_threshold in default_smr_reward

# ============================================================
#  Load config from the run directory
# ============================================================

run_dir     = os.path.normpath(os.path.join(MODEL_CHECKPOINT_DIR, ".."))
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
#  Run 1 — No-SMR Baseline
# ============================================================

print("=" * 60)
print("Run 1 / 2 — No-SMR Baseline")
print("=" * 60)

from sustaindc_env import SustainDC

no_smr_env_args = dict(env_args)
no_smr_env_args["use_smr"]    = False
no_smr_env_args["agents"]     = ["agent_ls", "agent_dc", "agent_bat"]
no_smr_env_args["evaluation"] = True
if "month" not in no_smr_env_args:
    no_smr_env_args["month"] = 0

baseline_env = SustainDC(no_smr_env_args)

random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)
baseline_env.reset()

tel_dc_b = []   # DC demand (MW) — equals full grid draw with no SMR
tel_ci_b = []   # normalised CI

_base_dict = {"agent_ls": 1, "agent_dc": 1, "agent_bat": 2}

while True:
    _, _, term_b, trunc_b, info_b = baseline_env.step(_base_dict)
    s = info_b["agent_dc"]
    tel_dc_b.append(s.get("dc_total_power_kW", 0.0) / 1000.0)
    tel_ci_b.append(float(s.get("norm_CI", 0.0)))
    if trunc_b.get("__all__", False) or term_b.get("__all__", False):
        break

dc_pw_b = np.array(tel_dc_b)
ci_b    = np.array(tel_ci_b)

print(f"  Steps            : {len(dc_pw_b)}  ({len(dc_pw_b) * 0.25:.1f} h)")
print(f"  Mean DC demand   : {dc_pw_b.mean():.3f} MW")
print(f"  Mean norm CI     : {ci_b.mean():.3f}")
print(f"  Carbon proxy sum : {(dc_pw_b * ci_b).sum():.1f}  (MW·CI·steps)\n")


# ============================================================
#  Run 2 — Trained HAPPO SMR Policy
# ============================================================

print("=" * 60)
print("Run 2 / 2 — Trained SMR RL Policy")
print("=" * 60)
print(f"  Loading checkpoint from:\n    {MODEL_CHECKPOINT_DIR}\n")

# Patch logger registry so the runner does not crash on the SMR reward key
import harl.envs as _harl_envs
from harl.envs.sustaindc.sustaindc_logger import SustainDCLogger
_harl_envs.LOGGER_REGISTRY["sustaindc"] = SustainDCLogger  # plain logger for eval

main_args["env"] = "sustaindc"  # required by init_dir()
runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
runner.prep_rollout()   # set all actors to eval mode

num_agents = runner.num_agents
assert num_agents == 4, (
    f"Expected 4 agents, runner found {num_agents}. "
    "Check that agent_smr is in env_args['agents']."
)

print(f"  SMR nameplate capacity : {max_smr_mw:.1f} MW")
print(f"  Episode length         : {env_args['days_per_episode']} days\n")

# Seed identically to baseline run so both episodes start on the same day/hour
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)
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
tel_smr_action_int = []
tel_smr_export_kw  = []

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

    smr_power_mw  = info.get("smr_power_output_kW", 0.0) / 1000.0
    dc_demand_mw  = info.get("dc_total_power_kW",   0.0) / 1000.0
    ci_norm       = float(info.get("norm_CI", 0.0))
    smr_action    = int(action_collector[_IDX_SMR][0, 0])
    smr_export_kw = float(info.get("smr_grid_export_kW", 0.0))

    tel_smr_power_mw.append(smr_power_mw)
    tel_dc_demand_mw.append(dc_demand_mw)
    tel_ci_norm.append(ci_norm)
    tel_smr_action_int.append(smr_action)
    tel_smr_export_kw.append(smr_export_kw)

    step += 1

    # Reset RNN state and masks if any environment episode ended
    eval_dones_env = np.all(eval_dones, axis=1)   # shape (1,)
    if eval_dones_env[0]:
        eval_rnn_states[:] = 0.0
        eval_masks = np.zeros((1, num_agents, 1), dtype=np.float32)
        break   # single episode evaluation

runner.close()


# ============================================================
#  Arrays and derived quantities
# ============================================================

T         = len(tel_smr_power_mw)
timesteps = np.arange(T) * 0.25          # 15-min steps → hours
smr_pw    = np.array(tel_smr_power_mw)
dc_pw     = np.array(tel_dc_demand_mw)
ci        = np.array(tel_ci_norm)
actions   = np.array(tel_smr_action_int)
export_kw = np.array(tel_smr_export_kw)

# Net grid draw with SMR (power still drawn from grid after reactor offset)
net_grid_mw = np.maximum(0.0, dc_pw - smr_pw)

export_mask = smr_pw > dc_pw
dirty_mask  = ci >= CI_THRESHOLD
clean_mask  = ~dirty_mask

# CI-bucketed export stats (RL run)
total_export = export_kw.sum()
dirty_export = export_kw[dirty_mask].sum()
clean_export = export_kw[clean_mask].sum()
dirty_frac   = dirty_export / max(total_export, 1e-9)
clean_frac   = clean_export / max(total_export, 1e-9)

# Carbon comparison — align episode lengths in case of any off-by-one
T_cmp        = min(T, len(dc_pw_b))
t_cmp        = np.arange(T_cmp) * 0.25
carbon_nosmr = dc_pw_b[:T_cmp] * ci_b[:T_cmp]      # full DC load on grid
carbon_smr   = net_grid_mw[:T_cmp] * ci[:T_cmp]    # only residual on grid
carbon_saved_pct = (
    100.0 * (carbon_nosmr.sum() - carbon_smr.sum())
    / max(carbon_nosmr.sum(), 1e-9)
)


# ============================================================
#  Print stats
# ============================================================

print(f"RL episode complete — {T} steps ({T * 0.25:.1f} h)")
print(f"  Mean SMR output   : {smr_pw.mean():.3f} MW  "
      f"(range {smr_pw.min():.3f}–{smr_pw.max():.3f})")
print(f"  Mean DC demand    : {dc_pw.mean():.3f} MW")
print(f"  Grid export steps : {export_mask.sum()} / {T}  "
      f"({100*export_mask.mean():.1f} %)")
print(f"  Total export      : {total_export/1e3:.2f} MWh")
print(f"  Export on dirty grid (CI≥{CI_THRESHOLD}) : "
      f"{dirty_export/1e3:.2f} MWh  ({100*dirty_frac:.1f} %)")
print(f"  Export on clean grid (CI<{CI_THRESHOLD})  : "
      f"{clean_export/1e3:.2f} MWh  ({100*clean_frac:.1f} %)")
print(f"  Action counts     : "
      f"↓ {(actions==0).sum()}  · {(actions==1).sum()}  ↑ {(actions==2).sum()}")
print()
print(f"  *** Carbon savings vs No-SMR baseline: {carbon_saved_pct:+.1f}% ***")
print()


# ============================================================
#  Plot — 5 panes
# ============================================================

fig, axes = plt.subplots(
    5, 1, figsize=(14, 18), sharex=True,
    gridspec_kw={"height_ratios": [3, 2, 1.5, 2, 2.5], "hspace": 0.08},
)
fig.suptitle(
    f"SMR Agent — Deterministic Policy Evaluation vs No-SMR Baseline  "
    f"(capacity = {max_smr_mw:.0f} MW, {env_args['days_per_episode']} days)",
    fontsize=13, fontweight="bold", y=0.99,
)

# ── Pane 1: Power ────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(timesteps, smr_pw, color="#2980b9", linewidth=1.6,
         label="SMR Power Output (RL)")
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
points = np.array([timesteps, ci]).T.reshape(-1, 1, 2)
segs   = np.concatenate([points[:-1], points[1:]], axis=1)
lc     = LineCollection(segs, cmap="RdYlGn_r", norm=plt.Normalize(0, 1),
                        linewidth=2.0)
lc.set_array(ci)
ax2.add_collection(lc)
ax2.set_xlim(timesteps[0], timesteps[-1])
ax2.set_ylim(-0.05, 1.05)
cbar = fig.colorbar(lc, ax=ax2, pad=0.01, fraction=0.015)
cbar.set_label("CI (norm)", fontsize=8)
ax2.axhline(CI_THRESHOLD, color="#7f8c8d", linewidth=0.8, linestyle=":",
            alpha=0.7, label=f"CI threshold = {CI_THRESHOLD}")
ax2.set_ylabel("Norm. Carbon\nIntensity", fontsize=11)
ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)
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
ax3.set_ylabel("Action", fontsize=11)
ax3.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax3.grid(axis="x", linestyle=":", alpha=0.5)
ax3.set_ylim(-0.5, 2.5)
ax3.set_title("SMR Control Actions (RL Policy)", fontsize=10, loc="left", pad=4)

# ── Pane 4: CI-bucketed grid export ──────────────────────────
ax4 = axes[3]
export_mw = export_kw / 1000.0
ax4.fill_between(timesteps, 0, export_mw,
                 where=dirty_mask, interpolate=True,
                 color="#e74c3c", alpha=0.75,
                 label=f"Export — dirty grid (CI≥{CI_THRESHOLD})")
ax4.fill_between(timesteps, 0, export_mw,
                 where=clean_mask, interpolate=True,
                 color="#1abc9c", alpha=0.75,
                 label=f"Export — clean grid (CI<{CI_THRESHOLD})")
ax4.axhline(0, color="black", linewidth=0.6)
ax4.set_ylabel("Export (MW)", fontsize=11)
ax4.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax4.grid(axis="y", linestyle=":", alpha=0.5)
ax4.set_title(
    f"Grid Export by CI Bucket — dirty {100*dirty_frac:.1f} %  /  "
    f"clean {100*clean_frac:.1f} %",
    fontsize=10, loc="left", pad=4,
)

# ── Pane 5: Carbon footprint comparison ──────────────────────
ax5 = axes[4]
ax5.plot(t_cmp, carbon_nosmr,
         color="#95a5a6", linewidth=1.6, linestyle="--",
         label="No-SMR Baseline  (full DC load × CI)", zorder=2)
ax5.plot(t_cmp, carbon_smr,
         color="#2980b9", linewidth=1.9, linestyle="-",
         label="Trained SMR (RL)  (net grid draw × CI)", zorder=3)
ax5.fill_between(
    t_cmp, carbon_smr, carbon_nosmr,
    where=carbon_nosmr >= carbon_smr, interpolate=True,
    alpha=0.22, color="#27ae60", label="Carbon Saved",
)
ax5.set_ylabel("Carbon Draw\n(MW × norm CI)", fontsize=11)
ax5.set_xlabel("Time (hours)", fontsize=11)
ax5.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax5.grid(axis="y", linestyle=":", alpha=0.5)
ax5.set_title(
    f"Carbon Footprint — RL SMR saves {carbon_saved_pct:.1f}% "
    f"vs No-SMR Baseline",
    fontsize=10, loc="left", pad=4,
)

# Day-boundary vertical lines across all panes
for d in range(1, env_args["days_per_episode"]):
    for ax in axes:
        ax.axvline(x=d * 24, color="black", linewidth=0.6,
                   linestyle="--", alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"Plot saved → {os.path.abspath(OUTPUT_FILE)}")
