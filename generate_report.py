from fpdf import FPDF, XPos, YPos

pdf = FPDF()
pdf.set_margins(20, 20, 20)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

LM = 20   # left margin
RM = 20   # right margin
PW = 210  # page width
W  = PW - LM - RM   # 170 mm usable


def title(text):
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_x(LM)
    pdf.multi_cell(W, 8, text)
    pdf.ln(2)


def heading(text):
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(LM)
    pdf.multi_cell(W, 7, text)
    pdf.ln(1)


def subheading(text):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(LM)
    pdf.multi_cell(W, 6, text)


def body(text):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(LM)
    pdf.multi_cell(W, 6, text)
    pdf.ln(1)


def bullet(text):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(LM + 4)
    pdf.multi_cell(W - 4, 6, "- " + text)


# ── Title ─────────────────────────────────────────────────────
title("DC-RL + SMR Co-location - Weekly Progress Report")
body("March 2026")
pdf.ln(4)

body(
    "This project extends dc-rl, an open-source multi-agent RL framework "
    "from Hewlett Packard Enterprise. The base framework trains three agents "
    "to manage a data centre: load-shifting (LS), HVAC cooling (DC), and "
    "battery storage (BAT). Everything described below adds a fourth agent "
    "that controls a Small Modular Reactor (SMR) co-located with the data "
    "centre, with the goal of displacing carbon from the electricity grid."
)

# ── 1 ─────────────────────────────────────────────────────────
heading("1. New SMR Environment  (envs/smr_env.py)")

body(
    "A new environment was created to simulate the physical behaviour of a "
    "Small Modular Reactor. At every 15-minute timestep the agent picks one "
    "of three actions: ramp power up, hold, or ramp power down."
)

subheading("Power constraints")
body(
    "Maximum capacity is 6 MW. The reactor cannot drop below 20% of that "
    "(1.2 MW). This floor exists because real reactors cannot simply be "
    "switched off like a gas turbine. Going below roughly 20% requires a "
    "full cold shutdown, which takes days to recover from, so 20% is the "
    "accepted operating minimum for load-following SMRs."
)
body(
    "The ramp rate is capped at 5% of max capacity per step (0.3 MW per "
    "15-minute step). This reflects the physical speed limit of control-rod "
    "movement. Demanding instant large power swings would be unrealisable "
    "on a real system."
)

subheading("Thermal model")
body(
    "Reactors do not respond to power changes instantly. The core heats and "
    "cools on roughly a one-hour timescale. Without modelling this, the "
    "agent would learn a policy that looks good in simulation but could not "
    "physically be executed. A first-order thermal lag was implemented:"
)
body("    T(t) = T(t-1) + [T_target(P) - T(t-1)] x (1 - exp(-1 / tau))")
body(
    "The constants were chosen as follows:"
)
bullet(
    "T_coolant_base = 150 C. This is the coolant inlet temperature and the "
    "minimum the core can reach. It matches a typical pressurised water "
    "reactor (PWR) inlet. The core cannot go below this even at minimum power."
)
bullet(
    "Delta_T_max = 150 C. The additional temperature rise going from minimum "
    "to full power, giving a maximum core temperature of 300 C. This was "
    "chosen to leave a 20 C safety margin below the soft limit of 320 C, "
    "so the reward signal penalises the agent before it reaches a "
    "genuinely dangerous state."
)
bullet(
    "T_safety_limit = 320 C. The soft safety threshold in the reward. Set "
    "below the typical PWR hard cutoff of around 350 C, giving the agent "
    "room to correct course before hitting a real limit."
)
bullet(
    "tau = 4 timesteps (1 hour). The thermal time constant. Means that if "
    "the agent ramps up now, the core temperature keeps rising for roughly "
    "the next hour even if the agent immediately backs off. This is "
    "consistent with published SMR thermal response data and is the key "
    "reason RL is useful here: the agent must plan ahead rather than just "
    "react to the current carbon reading."
)
pdf.ln(2)
body(
    "Any power generated beyond the data centre's current demand is logged "
    "as grid export. This surplus goes back to the wider electricity grid "
    "and is the mechanism through which the SMR displaces fossil generation."
)

# ── 2 ─────────────────────────────────────────────────────────
heading("2. System Sizing  (sustaindc_env.py)")

body(
    "The data centre was set to 5 MW and the SMR to 6 MW, giving a "
    "co-location ratio of 1.2x. A 1:1 ratio would leave no surplus to "
    "export, removing the carbon-displacement mechanism entirely. A ratio "
    "much above 1.2x (e.g. 2x) would make the problem trivial because the "
    "agent could run flat-out at all times and still have surplus. The 1.2x "
    "ratio forces a real decision: when to ramp up and export versus when "
    "to back off, depending on live grid carbon intensity."
)

# ── 3 ─────────────────────────────────────────────────────────
heading("3. Reward Function  (utils/reward_creator.py)")

body(
    "The reward was designed to teach the agent to run at high power when "
    "the grid is dirty and back off when the grid is clean. It has five parts:"
)
bullet("R_carbon = SMR_fraction x norm_CI. Rewards high output proportional to how dirty the grid is.")
bullet("R_op_cost = -0.4 x SMR_fraction. A running cost that scales with power. Together with R_carbon the net signal is SMR_fraction x (norm_CI - 0.4), which is negative when the grid is clean and positive when dirty. The agent naturally learns to ramp down below the 0.4 threshold and ramp up above it.")
bullet("R_econ = export_fraction x (norm_CI - 0.4) x 0.5. Rewards exporting to a dirty grid and penalises exporting to a clean one. This closed a loophole where an early version of the reward caused the agent to simply run at maximum power all the time regardless of carbon intensity.")
bullet("R_stability = -0.02 x |ramp_direction|. A small penalty per control-rod movement to discourage unnecessary ramping.")
bullet("R_boundary = -5.0 if the agent attempts a physically illegal ramp. Hard stop at the power limits.")

# ── 4 ─────────────────────────────────────────────────────────
heading("4. Training Script with SMR Logging  (train_smr.py)")

body(
    "A full training script co-trains all four agents simultaneously using "
    "HAPPO (Heterogeneous-Agent PPO). HAPPO was chosen because each agent "
    "has a different observation size and reward function. It gives each "
    "agent its own actor network while sharing one centralised critic."
)
body(
    "A custom logger was added that tracks SMR-specific metrics in "
    "TensorBoard each episode: average power fraction, average grid export "
    "fraction, core temperature, and most importantly the fraction of all "
    "grid exports that happened during high-carbon (dirty grid) periods. "
    "This last metric is the key convergence signal. A well-trained agent "
    "should push it toward 80-100%."
)

# ── 5 ─────────────────────────────────────────────────────────
heading("5. No-SMR Baseline and Evaluation  (eval_smr.py, sustaindc_env.py)")

body(
    "A use_smr flag was added to the environment. When False, the SMR "
    "physics step is completely bypassed: the reactor contributes zero power "
    "and the battery and grid see the full data centre load as if no reactor "
    "exists. All other agents behave identically in both modes."
)
body(
    "The evaluation script now runs two episodes back-to-back using the "
    "same random seed, so both start on the same calendar day and hour. "
    "Run 1 is the No-SMR baseline. Run 2 is the trained RL policy. The "
    "output is a 5-pane figure:"
)
bullet("Pane 1: SMR power output vs DC demand, with green fill showing grid export.")
bullet("Pane 2: Grid carbon intensity over the 7-day episode.")
bullet("Pane 3: SMR control actions at each timestep (ramp up / hold / ramp down).")
bullet("Pane 4: Grid export split into dirty-grid vs clean-grid periods.")
bullet("Pane 5: Carbon footprint comparison. No-SMR baseline (dashed grey) overlaid with RL SMR (solid blue), green fill showing carbon saved, and a headline percentage reduction in the title.")
pdf.ln(2)
body(
    "Without this baseline the only available claim is that the agent "
    "learned something. With it the claim becomes: co-locating an SMR and "
    "controlling it with RL reduces the carbon footprint of the data "
    "centre grid draw by X%. That is the core research result."
)

# ── 6 ─────────────────────────────────────────────────────────
heading("6. Next Steps - Full 5M-Step Training Run")

body(
    "The local demo run uses around 50,000 steps, which is not enough for "
    "the agent to converge. The full run on the Lambda GPU VM uses "
    "5,000,000 steps across 32 parallel environments, equivalent to roughly "
    "7,400 full 7-day simulation episodes. At that scale we expect the "
    "agent to:"
)
bullet("Hold near minimum power during clean-grid periods when running costs outweigh benefits.")
bullet("Ramp up when carbon intensity spikes above the 0.4 threshold.")
bullet("Export surplus primarily during dirty-grid peaks, pushing the export_dirty_ci_fraction metric toward 80-100%.")
pdf.ln(2)
body(
    "Once training finishes, one path is updated in eval_smr.py and the "
    "evaluation script is run. Pane 5 gives the headline result and pane 4 "
    "gives the supporting evidence."
)
body(
    "The next comparison to implement after that is a rule-based CI "
    "threshold agent that ramps up whenever norm_CI exceeds 0.4 and ramps "
    "down otherwise, with no learning at all. If the RL agent's carbon "
    "savings exceed the rule-based savings, it shows that the agent is "
    "exploiting the CI forecast in its observation to anticipate dirty-grid "
    "periods rather than just reacting to them. That is the central "
    "publishable finding."
)

pdf.output("weekly_progress_report.pdf")
print("Saved -> weekly_progress_report.pdf")
