"""
eval_smr.py — Four-way policy evaluation + multi-objective Pareto analysis.

How to use
----------
1. Point MODEL_CHECKPOINT_DIR at the `models/` subdirectory produced by
   train_smr.py.  That is the ONLY line you need to edit:

       MODEL_CHECKPOINT_DIR = "./results/sustaindc/ny/happo/smr_4agent/" \\
                              "seed-00001-<your-timestamp>/models"

   The script infers config.json from one directory above that path.

2. Run:  python eval_smr.py

3. Outputs:
     smr_eval_policy.png       — 5-pane per-timestep telemetry
     smr_eval_pareto.png       — 3-objective Pareto frontier analysis

Evaluation strategy
-------------------
All four runs use the same EVAL_SEED so every episode starts on the same
calendar day and hour — true apples-to-apples comparison.

Run 1 (No-SMR Baseline):   SustainDC with use_smr=False.  Records full DC
                            grid draw and CI.  Carbon reference point.

Run 2 (Always-On SMR):     SMR action fixed to ramp-up (2) every step.
                            Reactor runs at maximum power regardless of CI.
                            Lower bound on RL — shows what zero intelligence gets.

Run 3 (Rule-Based SMR):    SMR ramps up when norm_CI >= CI_THRESHOLD, ramps
                            down otherwise.  Simple reactive threshold policy.
                            Benchmark — RL must beat this to justify its cost.

Run 4 (Trained RL Policy): Trained HAPPO actor for agent_smr; base agents for
                            LS / DC / BAT.

Three objectives tracked for each policy
-----------------------------------------
1. Carbon displacement : mean(smr_power_fraction × norm_CI)           — higher = better
2. Revenue proxy       : mean(export_fraction × (norm_price − 0.5))   — higher = better
                         Centred at 0.5: negative when below-avg price, positive above.
                         Matches the training reward exactly.
3. Thermal stability   : −mean(|ΔT_core| / 40°C)                     — higher = better (less cycling)

Pareto frontier
---------------
The rule-based CI threshold is swept from 0.0 → 1.0 to generate a curve in
objective space.  The trained RL policy appears as a single point on the same
chart.  If RL dominates or lies on the frontier it demonstrates genuine value
over the hand-crafted rule.

Plot layout
-----------
smr_eval_policy.png  (5 panes) — unchanged time-series telemetry.
smr_eval_pareto.png  (3 panes):
  Pane A — Carbon displacement vs Revenue proxy (Pareto curve + policy points)
  Pane B — Carbon displacement vs Thermal stability
  Pane C — Radar / bar chart — all three objectives for all four policies
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
    "seed-00001-2026-03-30-04-55-02/models"
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
#  Helper — run one episode of SustainDC with a fixed SMR rule
# ============================================================

def _run_rule_episode(smr_env_args, rule, seed):
    """Run one episode with a rule-based SMR agent.

    rule: callable(prev_ci) -> int action (0 / 1 / 2)
    Returns arrays: smr_power_mw, export_kw, ci_norm, revenue, temp_delta
    """
    env = SustainDC(smr_env_args)
    random.seed(seed)
    np.random.seed(seed)
    env.reset()

    smr_pw_list, export_list, ci_list, revenue_list, tdelta_list = [], [], [], [], []
    prev_ci = 0.5   # neutral starting value before first step
    act_dict = {"agent_ls": 1, "agent_dc": 1, "agent_bat": 2, "agent_smr": 1}

    while True:
        act_dict["agent_smr"] = rule(prev_ci)
        _, _, term, trunc, info = env.step(act_dict)
        s = info["agent_dc"]
        prev_ci = float(s.get("norm_CI", 0.0))
        smr_pw_mw  = s.get("smr_power_output_kW", 0.0) / 1000.0
        export_kw  = float(s.get("smr_grid_export_kW", 0.0))
        norm_price = float(s.get("norm_price", prev_ci))
        temp_delta = float(s.get("smr_temp_delta", 0.0))
        max_smr_kw = float(smr_env_args.get("max_smr_capacity_mw", 6.0)) * 1000.0
        smr_pw_list.append(smr_pw_mw)
        export_list.append(export_kw)
        ci_list.append(prev_ci)
        revenue_list.append((export_kw / max(max_smr_kw, 1e-9)) * (norm_price - 0.5))
        tdelta_list.append(temp_delta)
        if trunc.get("__all__", False) or term.get("__all__", False):
            break

    return (np.array(smr_pw_list),
            np.array(export_list),
            np.array(ci_list),
            np.array(revenue_list),
            np.array(tdelta_list))


smr_env_args = dict(env_args)
smr_env_args["use_smr"]    = True
smr_env_args["agents"]     = ["agent_ls", "agent_dc", "agent_bat", "agent_smr"]
smr_env_args["evaluation"] = True
if "month" not in smr_env_args:
    smr_env_args["month"] = 0


# ============================================================
#  Run 2 — Always-On SMR
# ============================================================

print("=" * 60)
print("Run 2 / 4 — Always-On SMR  (action = ramp-up every step)")
print("=" * 60)

ao_smr_pw, ao_export_kw, ao_ci, ao_revenue, ao_tdelta = _run_rule_episode(
    smr_env_args,
    rule=lambda _ci: 2,   # always ramp up
    seed=EVAL_SEED,
)

ao_dirty_mask  = ao_ci >= CI_THRESHOLD
ao_dirty_frac  = ao_export_kw[ao_dirty_mask].sum() / max(ao_export_kw.sum(), 1e-9)
ao_net_grid_mw = np.maximum(0.0, dc_pw_b[:len(ao_smr_pw)] - ao_smr_pw)
carbon_ao      = ao_net_grid_mw * ao_ci
ao_saved_pct   = 100.0 * (1.0 - carbon_ao.sum() / max((dc_pw_b[:len(ao_ci)] * ao_ci).sum(), 1e-9))

print(f"  Mean SMR output  : {ao_smr_pw.mean():.3f} MW")
print(f"  Export dirty frac: {100*ao_dirty_frac:.1f}%")
print(f"  Carbon savings   : {ao_saved_pct:+.1f}% vs No-SMR\n")


# ============================================================
#  Run 3 — Rule-Based CI Threshold SMR
# ============================================================

print("=" * 60)
print(f"Run 3 / 4 — Rule-Based SMR  (ramp-up if CI >= {CI_THRESHOLD})")
print("=" * 60)

rb_smr_pw, rb_export_kw, rb_ci, rb_revenue, rb_tdelta = _run_rule_episode(
    smr_env_args,
    rule=lambda ci: 2 if ci >= CI_THRESHOLD else 0,
    seed=EVAL_SEED,
)

rb_dirty_mask  = rb_ci >= CI_THRESHOLD
rb_dirty_frac  = rb_export_kw[rb_dirty_mask].sum() / max(rb_export_kw.sum(), 1e-9)
rb_net_grid_mw = np.maximum(0.0, dc_pw_b[:len(rb_smr_pw)] - rb_smr_pw)
carbon_rb      = rb_net_grid_mw * rb_ci
rb_saved_pct   = 100.0 * (1.0 - carbon_rb.sum() / max((dc_pw_b[:len(rb_ci)] * rb_ci).sum(), 1e-9))

print(f"  Mean SMR output  : {rb_smr_pw.mean():.3f} MW")
print(f"  Export dirty frac: {100*rb_dirty_frac:.1f}%")
print(f"  Carbon savings   : {rb_saved_pct:+.1f}% vs No-SMR\n")


# ============================================================
#  Run 4 — Trained HAPPO SMR Policy
# ============================================================

print("=" * 60)
print("Run 4 / 4 — Trained SMR RL Policy")
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
tel_revenue        = []
tel_temp_delta     = []

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
    norm_price_v  = float(info.get("norm_price", ci_norm))
    temp_delta_v  = float(info.get("smr_temp_delta", 0.0))
    max_smr_kw_v  = float(env_args.get("max_smr_capacity_mw", 6.0)) * 1000.0

    tel_smr_power_mw.append(smr_power_mw)
    tel_dc_demand_mw.append(dc_demand_mw)
    tel_ci_norm.append(ci_norm)
    tel_smr_action_int.append(smr_action)
    tel_smr_export_kw.append(smr_export_kw)
    tel_revenue.append((smr_export_kw / max(max_smr_kw_v, 1e-9)) * (norm_price_v - 0.5))
    tel_temp_delta.append(temp_delta_v)

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
rl_revenue    = np.array(tel_revenue)
rl_temp_delta = np.array(tel_temp_delta)

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
print("=" * 60)
print("  Baseline Comparison — Export Dirty-Grid Fraction")
print("=" * 60)
print(f"  Always-On  : {100*ao_dirty_frac:.1f}%   carbon savings {ao_saved_pct:+.1f}%")
print(f"  Rule-Based : {100*rb_dirty_frac:.1f}%   carbon savings {rb_saved_pct:+.1f}%")
print(f"  RL Policy  : {100*dirty_frac:.1f}%   carbon savings {carbon_saved_pct:+.1f}%")
print()
print(f"  *** RL lifts dirty-export by {100*(dirty_frac - ao_dirty_frac):+.1f}pp vs Always-On "
      f"and {100*(dirty_frac - rb_dirty_frac):+.1f}pp vs Rule-Based ***")
print()

# ============================================================
#  Three-objective comparison
# ============================================================

def _obj_carbon(smr_pw_arr, ci_arr, max_mw):
    """Mean carbon displacement: mean(P_frac × norm_CI)."""
    return float(np.mean((smr_pw_arr / max(max_mw, 1e-9)) * ci_arr))

def _obj_revenue(rev_arr):
    """Mean revenue proxy: mean(export_fraction × (norm_price − 0.5)).
    Centred: negative when exporting below avg price, positive when above.
    Matches the training reward exactly."""
    return float(np.mean(rev_arr))

def _obj_longevity(tdelta_arr):
    """Thermal stability: −mean(ΔT / 40°C), higher = less cycling."""
    return float(-np.mean(np.clip(tdelta_arr / 40.0, 0.0, 1.0)))

T_ao_c = min(T, len(ao_smr_pw))
T_rb_c = min(T, len(rb_smr_pw))

ao_carbon    = _obj_carbon(ao_smr_pw[:T_ao_c],  ao_ci[:T_ao_c],    max_smr_mw)
ao_rev       = _obj_revenue(ao_revenue[:T_ao_c])
ao_longevity = _obj_longevity(ao_tdelta[:T_ao_c])

rb_carbon    = _obj_carbon(rb_smr_pw[:T_rb_c],  rb_ci[:T_rb_c],    max_smr_mw)
rb_rev       = _obj_revenue(rb_revenue[:T_rb_c])
rb_longevity = _obj_longevity(rb_tdelta[:T_rb_c])

rl_carbon    = _obj_carbon(smr_pw,   ci,   max_smr_mw)
rl_rev       = _obj_revenue(rl_revenue)
rl_longevity = _obj_longevity(rl_temp_delta)

print("=" * 65)
print("  Multi-Objective Comparison (3 objectives)")
print("=" * 65)
print(f"  {'Policy':<18} {'Carbon↑':>10} {'Revenue↑':>10} {'Longevity↑':>12}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*12}")
print(f"  {'Always-On':<18} {ao_carbon:>10.4f} {ao_rev:>10.4f} {ao_longevity:>12.4f}")
print(f"  {'Rule-Based':<18} {rb_carbon:>10.4f} {rb_rev:>10.4f} {rb_longevity:>12.4f}")
print(f"  {'RL Policy':<18} {rl_carbon:>10.4f} {rl_rev:>10.4f} {rl_longevity:>12.4f}")
print()
print("  Note: Carbon = mean(P_frac × norm_CI) | "
      "Revenue = mean(export_frac × (norm_price−0.5)) | "
      "Longevity = −mean(|ΔT|/40°C)")
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

# ── Pane 4: CI-bucketed grid export comparison ────────────────
ax4 = axes[3]
export_mw = export_kw / 1000.0
ax4.fill_between(timesteps, 0, export_mw,
                 where=dirty_mask, interpolate=True,
                 color="#e74c3c", alpha=0.75,
                 label=f"RL — dirty grid (CI≥{CI_THRESHOLD})  {100*dirty_frac:.1f}%")
ax4.fill_between(timesteps, 0, export_mw,
                 where=clean_mask, interpolate=True,
                 color="#1abc9c", alpha=0.75,
                 label=f"RL — clean grid (CI<{CI_THRESHOLD})  {100*clean_frac:.1f}%")
ax4.axhline(0, color="black", linewidth=0.6)
ax4.set_ylabel("Export (MW)", fontsize=11)
ax4.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax4.grid(axis="y", linestyle=":", alpha=0.5)
ax4.set_title(
    f"RL Grid Export by CI Bucket — "
    f"Always-On dirty: {100*ao_dirty_frac:.1f}%  "
    f"Rule-Based: {100*rb_dirty_frac:.1f}%  "
    f"RL: {100*dirty_frac:.1f}%",
    fontsize=10, loc="left", pad=4,
)

# ── Pane 5: Carbon footprint — all four policies ──────────────
ax5 = axes[4]
T_ao = min(T_cmp, len(ao_smr_pw))
T_rb = min(T_cmp, len(rb_smr_pw))

ax5.plot(t_cmp, carbon_nosmr,
         color="#95a5a6", linewidth=1.5, linestyle="--",
         label=f"No-SMR Baseline", zorder=2)
ax5.plot(t_cmp[:T_ao], carbon_ao[:T_ao],
         color="#e67e22", linewidth=1.4, linestyle="-.",
         label=f"Always-On SMR  ({ao_saved_pct:+.1f}%)", zorder=3)
ax5.plot(t_cmp[:T_rb], carbon_rb[:T_rb],
         color="#e74c3c", linewidth=1.4, linestyle=":",
         label=f"Rule-Based SMR  ({rb_saved_pct:+.1f}%)", zorder=4)
ax5.plot(t_cmp, carbon_smr,
         color="#2980b9", linewidth=2.0, linestyle="-",
         label=f"Trained RL SMR  ({carbon_saved_pct:+.1f}%)", zorder=5)
ax5.fill_between(
    t_cmp, carbon_smr, carbon_nosmr,
    where=carbon_nosmr >= carbon_smr, interpolate=True,
    alpha=0.18, color="#27ae60", label="RL Carbon Saved",
)
ax5.set_ylabel("Carbon Draw\n(MW × norm CI)", fontsize=11)
ax5.set_xlabel("Time (hours)", fontsize=11)
ax5.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax5.grid(axis="y", linestyle=":", alpha=0.5)
ax5.set_title(
    f"Carbon Footprint — RL vs Baselines  "
    f"(RL saves {carbon_saved_pct:.1f}% vs No-SMR, "
    f"{carbon_saved_pct - rb_saved_pct:+.1f}pp vs Rule-Based)",
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


# ============================================================
#  Pareto Frontier Plot (smr_eval_pareto.png)
# ============================================================

print("\nGenerating Pareto frontier (CI-threshold sweep)…")

# Sweep CI threshold from 0.0 to 1.0 (step 0.05)
pareto_thresholds = np.arange(0.0, 1.05, 0.05)
pareto_carbon, pareto_rev, pareto_lon = [], [], []

for thr in pareto_thresholds:
    thr_ = float(thr)
    pw, xkw, ci_arr, rev_arr, td_arr = _run_rule_episode(
        smr_env_args,
        rule=lambda ci, t=thr_: 2 if ci >= t else 0,
        seed=EVAL_SEED,
    )
    T_p = len(pw)
    pareto_carbon.append(_obj_carbon(pw, ci_arr, max_smr_mw))
    pareto_rev.append(_obj_revenue(rev_arr))
    pareto_lon.append(_obj_longevity(td_arr))

pareto_carbon = np.array(pareto_carbon)
pareto_rev    = np.array(pareto_rev)
pareto_lon    = np.array(pareto_lon)

print(f"  Sweep complete ({len(pareto_thresholds)} threshold points).")

# ── Pareto figure layout ─────────────────────────────────────
PARETO_FILE = "smr_eval_pareto.png"

fig_p, axes_p = plt.subplots(1, 3, figsize=(18, 6))
fig_p.suptitle(
    "Multi-Objective SMR Dispatch: Pareto Frontier Analysis\n"
    "(Rule-Based CI-threshold sweep — RL policy shown as ★)",
    fontsize=13, fontweight="bold",
)

policy_points = {
    "Always-On":  (ao_carbon,  ao_rev,  ao_longevity,  "#e67e22", "o"),
    "Rule-Based": (rb_carbon,  rb_rev,  rb_longevity,  "#e74c3c", "s"),
    "RL Policy":  (rl_carbon,  rl_rev,  rl_longevity,  "#2980b9", "*"),
}

# ── Pane A: Carbon vs Revenue ──────────────────────────────
ax_a = axes_p[0]
sc_a = ax_a.scatter(
    pareto_rev, pareto_carbon,
    c=pareto_thresholds, cmap="viridis", s=40, zorder=3,
    label="Rule-Based (threshold sweep)",
)
ax_a.plot(pareto_rev, pareto_carbon, "k-", linewidth=0.8, alpha=0.4, zorder=2)
cb_a = fig_p.colorbar(sc_a, ax=ax_a)
cb_a.set_label("CI threshold", fontsize=8)
for name, (c_v, r_v, l_v, col, mk) in policy_points.items():
    ax_a.scatter(r_v, c_v, color=col, marker=mk,
                 s=200 if mk == "*" else 120, zorder=5,
                 edgecolors="black", linewidths=0.8, label=name)
ax_a.set_xlabel("Revenue proxy  (export_frac × norm_price) ↑", fontsize=10)
ax_a.set_ylabel("Carbon displacement  (P_frac × norm_CI) ↑", fontsize=10)
ax_a.set_title("Objective Space: Carbon vs Revenue", fontsize=11)
ax_a.legend(fontsize=8, framealpha=0.85)
ax_a.grid(True, linestyle=":", alpha=0.5)

# ── Pane B: Carbon vs Thermal Stability ───────────────────
ax_b = axes_p[1]
sc_b = ax_b.scatter(
    pareto_lon, pareto_carbon,
    c=pareto_thresholds, cmap="viridis", s=40, zorder=3,
)
ax_b.plot(pareto_lon, pareto_carbon, "k-", linewidth=0.8, alpha=0.4, zorder=2)
for name, (c_v, r_v, l_v, col, mk) in policy_points.items():
    ax_b.scatter(l_v, c_v, color=col, marker=mk,
                 s=200 if mk == "*" else 120, zorder=5,
                 edgecolors="black", linewidths=0.8, label=name)
ax_b.set_xlabel("Thermal stability  (−mean|ΔT|/40°C) ↑", fontsize=10)
ax_b.set_ylabel("Carbon displacement  (P_frac × norm_CI) ↑", fontsize=10)
ax_b.set_title("Objective Space: Carbon vs Longevity", fontsize=11)
ax_b.legend(fontsize=8, framealpha=0.85)
ax_b.grid(True, linestyle=":", alpha=0.5)

# ── Pane C: Bar chart — all three objectives for 4 policies ──
ax_c = axes_p[2]
policies      = ["Always-On", "Rule-Based", "RL Policy"]
colors_bar    = ["#e67e22",   "#e74c3c",    "#2980b9"]
obj_names     = ["Carbon\ndisplacement", "Revenue\nproxy\n(centred)", "Thermal\nstability"]
# Use raw values — do NOT min-max normalise across only 3 policies.
# That arithmetic guarantees one policy scores 1.0 on every metric by
# construction and hides the actual performance gap between policies.
raw = np.array([
    [ao_carbon, ao_rev, ao_longevity],
    [rb_carbon, rb_rev, rb_longevity],
    [rl_carbon, rl_rev, rl_longevity],
])

x = np.arange(len(obj_names))
width = 0.22
for i, (pol, col) in enumerate(zip(policies, colors_bar)):
    bars = ax_c.bar(x + (i - 1) * width, raw[i], width,
                    label=pol, color=col, alpha=0.85, edgecolor="white")
ax_c.set_xticks(x)
ax_c.set_xticklabels(obj_names, fontsize=9)
ax_c.set_ylabel("Raw objective score", fontsize=10)
ax_c.set_title("Objective Comparison — All Policies\n(raw scores — higher is better on all axes)", fontsize=11)
ax_c.axhline(0, color="black", linewidth=0.7, linestyle="-")
ax_c.legend(fontsize=9, framealpha=0.85)
ax_c.grid(axis="y", linestyle=":", alpha=0.5)
for i, (pol, col) in enumerate(zip(policies, colors_bar)):
    for j, v in enumerate(raw[i]):
        ax_c.text(j + (i - 1) * width, v + (0.002 if v >= 0 else -0.008),
                  f"{v:.3f}", ha="center",
                  va="bottom" if v >= 0 else "top",
                  fontsize=7, color="black")

plt.tight_layout()
fig_p.savefig(PARETO_FILE, dpi=150, bbox_inches="tight")
print(f"Pareto plot saved → {os.path.abspath(PARETO_FILE)}")
