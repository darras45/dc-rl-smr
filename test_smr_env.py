"""
test_smr_env.py — End-to-end sanity check for the agent_smr integration.

Runs 10 steps with:
  - agent_smr : random action from {0, 1, 2}
  - agent_ls  : base fallback (hold, action=1)
  - agent_dc  : base fallback (hold, action=1)
  - agent_bat : base fallback (idle, action=2)
"""

import random
import sys
import os

# Make sure the repo root is on the path regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sustaindc_env import SustainDC

ACTION_NAMES = {0: "RAMP DOWN", 1: "HOLD    ", 2: "RAMP UP "}

env_config = {
    "agents": ["agent_ls", "agent_dc", "agent_bat", "agent_smr"],
    "location": "ny",
    "cintensity_file": "NYIS_NG_&_avgCI.csv",
    "weather_file": "USA_NY_New.York-Kennedy.epw",
    "workload_file": "Alibaba_CPU_Data_Hourly_1.csv",
    "datacenter_capacity_mw": 5,
    "max_bat_cap_Mw": 2,
    "days_per_episode": 7,
    "max_smr_capacity_mw": 6.0,
    "smr_min_power_fraction": 0.2,
    "month": 0,
}

print("=" * 65)
print("  SustainDC + SMR — end-to-end sanity check")
print("=" * 65)

env = SustainDC(env_config)
states = env.reset()

print(f"\nEnvironment reset OK.")
print(f"Active agents   : {env.agents}")
print(f"Obs space sizes : {[s.shape for s in env.observation_space]}")
print(f"SMR obs shape   : {states['agent_smr'].shape}")
print()
print(f"{'Step':>4}  {'SMR Action':<12}  {'P_SMR (MW)':>10}  {'P_DC (MW)':>10}  {'R_SMR':>8}")
print("-" * 55)

for step in range(1, 11):
    action_dict = {
        "agent_ls":  1,                       # base: hold (do nothing)
        "agent_dc":  1,                       # base: hold setpoint
        "agent_bat": 2,                       # base: idle
        "agent_smr": random.choice([0, 1, 2]),
    }

    obs, rewards, terminateds, truncateds, info = env.step(action_dict)

    smr_action      = action_dict["agent_smr"]
    smr_power_mw    = env.smr_info["smr_power_output_kW"] / 1000.0
    dc_demand_mw    = env.dc_info.get("dc_total_power_kW", 0.0) / 1000.0
    smr_reward      = rewards.get("agent_smr", float("nan"))

    print(
        f"{step:>4}  {ACTION_NAMES[smr_action]:<12}  "
        f"{smr_power_mw:>10.3f}  "
        f"{dc_demand_mw:>10.3f}  "
        f"{smr_reward:>8.4f}"
    )

    if terminateds.get("__all__") or truncateds.get("__all__"):
        print("\n[Episode ended early]")
        break

print("-" * 55)
print("\nSanity check complete — no errors.\n")
