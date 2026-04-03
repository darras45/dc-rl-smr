# File where the rewards are defined
import numpy as np
from collections import deque

energy_history = deque(maxlen=10000)

def update_energy_history(value: float):
    """
    Update the global energy history deque with a new value.

    Args:
        value (float): The new energy value to add.
    """
    energy_history.append(value)

def normalize_energy(value: float):
    """
    Normalize the energy value based on the energy history, clipping outliers.

    Args:
        value (float): The energy value to normalize.

    Returns:
        float: The normalized energy value.
    """
    if len(energy_history) < 2:  # Avoid division by zero with too few samples
        return 0.0  # Default normalized value when history is insufficient

    # Convert history to a numpy array for processing
    history_array = np.array(energy_history)

    # Clip outliers (e.g., values beyond 1.5 times the interquartile range)
    q1 = np.percentile(history_array, 25)
    q3 = np.percentile(history_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Apply clipping
    clipped_history = np.clip(history_array, lower_bound, upper_bound)

    # Calculate mean and std from clipped values
    mean = np.mean(clipped_history)
    std = np.std(clipped_history)
    return (value - mean) / (std if std > 0 else 1)  # Prevent division by zero


def default_ls_reward(params: dict) -> float:
    """
    Calculates a reward value based on normalized load shifting.

    Args:
        params (dict): Dictionary containing parameters:
            norm_load_left (float): Normalized load left.
            out_of_time (bool): Indicator (alarm) whether the agent is in the last hour of the day.
            penalty (float): Penalty value.

    Returns:
        float: Reward value.
    """
    # Energy part of the reward
    total_energy = params['bat_total_energy_with_battery_KWh']
    update_energy_history(total_energy)  # Update the energy history deque
    norm_total_energy = normalize_energy(total_energy)  # Normalize using the deque
    norm_ci = params['norm_CI']
    
    footprint_reward = -1.0 * (norm_ci * norm_total_energy / 0.50)  # Mean and std reward. Negate to maximize reward and minimize energy consumption
    
    # Overdue Tasks Penalty (scaled)
    overdue_penalty_scale = 0.3  # Adjust this scaling factor as needed
    overdue_penalty_bias = 0.3
    # tasks_overdue_penalty = -overdue_penalty_scale * np.log(params['ls_overdue_penalty'] + 1) # +1 to avoid log(0) and be always negative
    tasks_overdue_penalty = -overdue_penalty_scale * np.sqrt(params['ls_overdue_penalty']) + overdue_penalty_bias # To have a +1 if the number of overdue tasks is 0, and a negative value otherwise
    # Oldest Task Age Penalty
    age_penalty_scale = 0.1  # Adjust this scaling factor as needed
    tasks_age_penalty = -age_penalty_scale * params['ls_oldest_task_age']  # Assume normalized between 0 and 1

    # Total Reward
    total_reward = footprint_reward + tasks_overdue_penalty + tasks_age_penalty
    clipped_reward = np.clip(total_reward, -10, 10)

    return clipped_reward


def default_dc_reward(params: dict) -> float:
    """
    Calculates a reward value based on the data center's total ITE Load and CT Cooling load.

    Args:
        params (dict): Dictionary containing parameters:
            data_center_total_ITE_Load (float): Total ITE Load of the data center.
            CT_Cooling_load (float): CT Cooling load of the data center.
            energy_lb (float): Lower bound of the energy.
            energy_ub (float): Upper bound of the energy.

    Returns:
        float: Reward value.
    """
    # Energy part of the reward
    total_energy = params['bat_total_energy_with_battery_KWh']
    norm_total_energy = normalize_energy(total_energy)  # Normalize using the deque
    norm_ci = params['norm_CI']
    
    footprint_reward = -1.0 * (norm_ci * norm_total_energy / 0.50)  # Mean and std reward. Negate to maximize reward and minimize energy consumption
    
    return footprint_reward


def default_bat_reward(params: dict) -> float:
    """
    Calculates a reward value based on the battery usage.

    Args:
        params (dict): Dictionary containing parameters:
            total_energy_with_battery (float): Total energy with battery.
            norm_CI (float): Normalized Carbon Intensity.
            dcload_min (float): Minimum DC load.
            dcload_max (float): Maximum DC load.

    Returns:
        float: Reward value.
    """
    # Energy part of the reward
    total_energy = params['bat_total_energy_with_battery_KWh']
    norm_total_energy = normalize_energy(total_energy)  # Normalize using the deque
    norm_ci = params['norm_CI']
    
    footprint_reward = -1.0 * (norm_ci * norm_total_energy / 0.50)  # Mean and std reward. Negate to maximize reward and minimize energy consumption
    
    return footprint_reward


def custom_agent_reward(params: dict) -> float:
    """
    A template for creating a custom agent reward function.

    Args:
        params (dict): Dictionary containing custom parameters for reward calculation.

    Returns:
        float: Custom reward value. Currently returns 0.0 as a placeholder.
    """
    # read reward input parameters from dict object
    # custom reward calculations 
    custom_reward = 0.0 # update with custom reward shaping 
    return custom_reward

# Example of ToU reward based on energy usage and price of electricity
# ToU reward is based on the ToU (Time of Use) of the agent, which is the amount of the energy time
# the agent spends on the grid times the price of the electricity.
# This example suppose that inside the params there are the following keys:
#   - 'energy_usage': the energy usage of the agent
#   - 'hour': the hour of the day
def tou_reward(params: dict) -> float:
    """
    Calculates a reward value based on the Time of Use (ToU) of energy.

    Args:
        params (dict): Dictionary containing parameters:
            energy_usage (float): The energy usage of the agent.
            hour (int): The current hour of the day (24-hour format).

    Returns:
        float: Reward value.
    """
    
    # ToU dict: {Hour, price}
    tou = {0: 0.25,
           1: 0.25,
           2: 0.25,
           3: 0.25,
           4: 0.25,
           5: 0.25,
           6: 0.41,
           7: 0.41,
           8: 0.41,
           9: 0.41,
           10: 0.41,
           11: 0.30,
           12: 0.30,
           13: 0.30,
           14: 0.30,
           15: 0.30,
           16: 0.27,
           17: 0.27,
           18: 0.27,
           19: 0.27,
           20: 0.27,
           21: 0.27,
           22: 0.25,
           23: 0.25,
           }
    
    # Obtain the price of electricity at the current hour
    current_price = tou[params['hour']]
    # Obtain the energy usage
    energy_usage = params['bat_total_energy_with_battery_KWh']
    
    # The reward is negative as the agent's objective is to minimize energy cost
    tou_reward = -1.0 * energy_usage * current_price

    return tou_reward


def renewable_energy_reward(params: dict) -> float:
    """
    Calculates a reward value based on the usage of renewable energy sources.

    Args:
        params (dict): Dictionary containing parameters:
            renewable_energy_ratio (float): Ratio of energy coming from renewable sources.
            total_energy_consumption (float): Total energy consumption of the data center.

    Returns:
        float: Reward value.
    """
    assert params.get('renewable_energy_ratio') is not None, 'renewable_energy_ratio is not defined. This parameter should be included using some external dataset and added to the reward_info dictionary'
    renewable_energy_ratio = params['renewable_energy_ratio'] # This parameter should be included using some external dataset
    total_energy_consumption = params['bat_total_energy_with_battery_KWh']
    factor = 1.0 # factor to scale the weight of the renewable energy usage

    # Reward = maximize renewable energy usage - minimize total energy consumption
    reward = factor * renewable_energy_ratio  -1.0 * total_energy_consumption
    return reward


def energy_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on energy efficiency.

    Args:
        params (dict): Dictionary containing parameters:
            ITE_load (float): The amount of energy spent on computation (useful work).
            total_energy_consumption (float): Total energy consumption of the data center.

    Returns:
        float: Reward value.
    """
    it_equipment_power = params['dc_ITE_total_power_kW']  
    total_power_consumption = params['dc_total_power_kW']  
    
    reward = it_equipment_power / total_power_consumption
    return reward


def energy_PUE_reward(params: dict) -> float:
    """
    Calculates a reward value based on Power Usage Effectiveness (PUE).

    Args:
        params (dict): Dictionary containing parameters:
            total_energy_consumption (float): Total energy consumption of the data center.
            it_equipment_energy (float): Energy consumed by the IT equipment.

    Returns:
        float: Reward value.
    """
    total_power_consumption = params['dc_total_power_kW']  
    it_equipment_power = params['dc_ITE_total_power_kW']  
    
    # Calculate PUE
    pue = total_power_consumption / it_equipment_power if it_equipment_power != 0 else float('inf')
    
    # We aim to get PUE as close to 1 as possible, hence we take the absolute difference between PUE and 1
    # We use a negative sign since RL seeks to maximize reward, but we want to minimize PUE
    reward = -abs(pue - 1)
    
    return reward


def temperature_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on the efficiency of cooling in the data center.

    Args:
        params (dict): Dictionary containing parameters:
            current_temperature (float): Current temperature in the data center.
            optimal_temperature_range (tuple): Tuple containing the minimum and maximum optimal temperatures for the data center.

    Returns:
        float: Reward value.
    """
    assert params.get('optimal_temperature_range') is not None, 'optimal_temperature_range is not defined. This parameter should be added to the reward_info dictionary'
    current_temperature = params['dc_int_temperature'] 
    optimal_temperature_range = params['optimal_temperature_range']
    min_temp, max_temp = optimal_temperature_range
    
    if min_temp <= current_temperature <= max_temp:
        reward = 1.0
    else:
        if current_temperature < min_temp:
            reward = -abs(current_temperature - min_temp)
        else:
            reward = -abs(current_temperature - max_temp)
    return reward

def water_usage_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on the efficiency of water usage in the data center.
    
    A lower value of water usage results in a higher reward, promoting sustainability
    and efficiency in water consumption.

    Args:
        params (dict): Dictionary containing parameters:
            dc_water_usage (float): The amount of water used by the data center in a given period.

    Returns:
        float: Reward value. The reward is higher for lower values of water usage, 
        promoting reduced water consumption.
    """
    dc_water_usage = params['dc_water_usage']
    
    # Calculate the reward. This is a simple inverse relationship; many other functions could be applied.
    # Adjust the scalar as needed to fit the scale of your rewards or to emphasize the importance of water savings.
    reward = -0.01 * dc_water_usage
    
    return reward

def default_smr_reward(params: dict) -> float:
    """Reward for the SMR agent — carbon-displacing load follower.

    Components
    ----------
    R_carbon    : smr_power_fraction * norm_CI
                  Rewards high SMR output proportional to how dirty the grid
                  is at that moment.

    R_op_cost   : -c_op * smr_power_fraction   (c_op = 0.4)
                  Continuous operational cost that scales with output power.
                  Combined with R_carbon this gives a net signal of
                  smr_power_fraction * (norm_CI - c_op): positive only when
                  the grid is dirtier than ci_threshold.  The agent is
                  indifferent between P_min and P_max at exactly
                  norm_CI == ci_threshold (0.4); it ramps down below the
                  threshold and up above it.

    R_econ      : export_fraction * (norm_CI - ci_threshold) * w_export
                  (ci_threshold = 0.4, w_export = 0.5)
                  Rewards exporting to the grid when it is dirty
                  (norm_CI > 0.4, simulating positive spot price).
                  Penalises exporting when the grid is already clean
                  (norm_CI < 0.4, simulating negative market pricing /
                  renewable curtailment).  Closes the "constant-max-power"
                  loophole: the agent cannot harvest R_econ by running
                  flat-out during clean-grid valleys.

    R_stability : -c_wear * |a_t|   (c_wear = 0.02)
                  Penalises unnecessary control-rod movements (wear).
                  With delta_ramp = 0.05 × P_max the break-even immediate
                  signal requires norm_CI ≈ 0.8; the agent's discounted
                  value function (gamma = 0.995) overcomes this barrier
                  well before CI reaches that level.

    R_boundary  : -5.0 if the action attempts to exceed P_max or drop
                  below P_min.  Strict penalty for physically illegal ramps.

    Coefficient design
    ------------------
    ci_threshold = 0.4   grid "cleanliness" crossover
    c_op         = 0.4   equals ci_threshold so R_carbon + R_op_cost = 0
                          at any power level when norm_CI == ci_threshold
    w_export     = 0.5   export bonus weight (decoupled from bat reward)
    c_wear       = 0.02  ramp-wear penalty (fixed from Phase-6 audit)
    r_boundary   = -5.0  hard stop at physics limits

    Worked examples (no exports, holding action)
    --------------------------------------------
    norm_CI=0.1, p_frac=1.0:  0.10 - 0.40 = -0.30  ← penalised at P_max
    norm_CI=0.4, p_frac=1.0:  0.40 - 0.40 =  0.00  ← indifferent at threshold
    norm_CI=0.8, p_frac=1.0:  0.80 - 0.40 = +0.40  ← rewarded at P_max
    norm_CI=0.1, p_frac=0.2:  0.02 - 0.08 = -0.06  ← preferred over P_max on clean grid

    Args:
        params (dict): Must contain:
            'smr_power_fraction'  – current output / P_max  in [0, 1]
            'smr_grid_export_kW'  – excess power exported to grid (kW)
            'smr_ramp_dir'        – ramp direction taken: {-1, 0, +1}
            'smr_boundary_hit'    – True if action was clamped at a limit
            'norm_CI'             – normalised carbon intensity in [0, 1]
            'max_smr_capacity_mw' – nameplate capacity in MW

    Returns:
        float: Total SMR reward, clipped to [-10, 10].
    """
    norm_ci            = params['norm_CI']
    smr_power_fraction = params['smr_power_fraction']   # [0, 1]
    smr_grid_export_kw = params['smr_grid_export_kW']
    ramp_dir           = params['smr_ramp_dir']         # {-1, 0, 1}
    boundary_hit       = params['smr_boundary_hit']
    max_smr_kw         = params['max_smr_capacity_mw'] * 1000.0

    # Threshold below which exporting is penalised (simulates negative pricing)
    ci_threshold = 0.4

    # R_carbon: reward high SMR output when grid CI is high
    r_carbon = smr_power_fraction * norm_ci

    # R_op_cost: running cost proportional to output power.
    # c_op == ci_threshold ensures R_carbon + R_op_cost =
    # smr_power_fraction * (norm_CI - ci_threshold): zero at the threshold,
    # negative below it, positive above it.
    c_op      = 0.4
    r_op_cost = -c_op * smr_power_fraction

    # R_econ: reward/penalise back-to-grid export based on grid cleanliness.
    # Positive when norm_CI > ci_threshold (dirty grid: displace fossil fuels).
    # Negative when norm_CI < ci_threshold (clean grid: over-generation penalty).
    w_export        = 0.5
    export_fraction = smr_grid_export_kw / max(max_smr_kw, 1e-9)
    r_econ          = export_fraction * (norm_ci - ci_threshold) * w_export

    # R_stability: penalise control-rod ramp actions (wear).
    c_wear      = 0.02
    r_stability = -c_wear * abs(ramp_dir)

    # R_boundary: strict penalty for attempted illegal physics action
    r_boundary = -5.0 if boundary_hit else 0.0

    total_reward = r_carbon + r_op_cost + r_econ + r_stability + r_boundary
    return float(np.clip(total_reward, -10.0, 10.0))


def default_smr_reward_lmp_dispatch(params: dict) -> float:
    """Merit-order economic dispatch reward for the SMR agent.

    Structural motivation
    ---------------------
    Additive linear scalarization (α·R_carbon + β·R_revenue) fails for SMR
    dispatch because carbon displacement and grid revenue are not independent
    objectives — they jointly determine the marginal economic value of each MW
    of SMR output.  The additive form allows the carbon term to unconditionally
    offset the revenue penalty at every power level, creating a reward surface
    whose gradient points toward full power regardless of grid price.

    This function uses *multiplicative merit-order coupling*:

        dispatch_value = α × norm_CI + β × (norm_price − θ)
        R_dispatch     = P_frac × dispatch_value

    When dispatch_value < 0 (cheap + clean grid), the agent receives a
    negative *absolute* reward proportional to its current output.  The
    gradient ∂R/∂P_frac = dispatch_value is directly visible at every
    timestep — the agent does not need to integrate a deferred multi-step
    signal to learn curtailment.  This breaks the local-optimum trap.

    Components
    ----------
    R_dispatch (α, β):
        P_frac × (α·norm_CI + β·(norm_price − θ))
        Positive → run hard; negative → curtail.
        θ (price_threshold, default 0.5) sets the dispatch crossover.

    R_export (w_export = 0.3):
        export_frac × max(0, dispatch_value) × w_export
        Bonus for exporting to the grid *only* when dispatch_value is
        already positive (high price AND/OR dirty grid).  Prevents the
        agent from harvesting export bonuses during cheap hours.

    R_safety (γ):
        -γ × max(0, |ΔT|/40°C − t_safe)²
        Soft quadratic thermal safety constraint.  Zero penalty for normal
        load-following (|ΔT| ≤ t_safe × 40°C ≈ 12°C/step); quadratic
        penalty only for violent transients.  Replaces the always-active
        longevity penalty that unconditionally rewarded Always-On behaviour.

    R_ramp (c_wear = 0.005):
        -c_wear × |ramp_dir| × (1 − 0.8 × urgency)
        Small wear penalty, reduced when |dispatch_value| is large
        (i.e. when economics clearly justify the ramp move).
        c_wear intentionally small (0.005) to avoid blocking economically
        justified ramps — reactor O&M cost is ~$5/MWh vs LMP $40–100/MWh.

    R_boundary:
        -5.0 for attempted physics violation (P < P_min or P > P_max).

    Weight design
    -------------
    Default (α=0.4, β=0.5, γ=0.1, θ=0.5):

        Condition              dispatch_value   Signal
        ──────────────────── ────────────────  ──────────
        Mean  (CI=0.35,p=0.37)   +0.075        Run (✓)
        Cheap (CI=0.25,p=0.20)   −0.050        Curtail (✓)
        Peak  (CI=0.70,p=0.75)   +0.405        Max power (✓)

    Args:
        params (dict): Must contain:
            smr_power_fraction, smr_grid_export_kW, smr_ramp_dir,
            smr_boundary_hit, norm_CI, norm_price, smr_temp_delta,
            max_smr_capacity_mw.
            Optional: alpha_carbon (0.4), beta_revenue (0.5),
            gamma_longevity (0.1), price_threshold (0.5).

    Returns:
        float: Total reward clipped to [−10, 10].
    """
    norm_ci            = params['norm_CI']
    norm_price         = params.get('norm_price', norm_ci)
    smr_power_fraction = params['smr_power_fraction']
    smr_grid_export_kw = params['smr_grid_export_kW']
    ramp_dir           = params['smr_ramp_dir']
    boundary_hit       = params['smr_boundary_hit']
    max_smr_kw         = params['max_smr_capacity_mw'] * 1000.0
    temp_delta         = params.get('smr_temp_delta', 0.0)

    alpha = params.get('alpha_carbon',    0.4)
    beta  = params.get('beta_revenue',    0.5)
    gamma = params.get('gamma_longevity', 0.1)
    theta = params.get('price_threshold', 0.5)   # dispatch crossover point

    # 1. Merit-order dispatch: positive when grid needs the MW, negative when not
    dispatch_value = alpha * norm_ci + beta * (norm_price - theta)
    r_dispatch     = smr_power_fraction * dispatch_value

    # 2. Export quality bonus — only earned when dispatch_value > 0
    w_export        = 0.3
    export_fraction = smr_grid_export_kw / max(max_smr_kw, 1e-9)
    r_export        = export_fraction * max(0.0, dispatch_value) * w_export

    # 3. Soft thermal safety constraint — quadratic only above safe threshold
    #    t_safe = 0.3 ≡ 12 °C per 15-min step (≈ normal slow ramp rate)
    temp_delta_norm = float(np.clip(temp_delta / 40.0, 0.0, 1.0))
    t_safe          = 0.3
    excess          = max(0.0, temp_delta_norm - t_safe)
    r_safety        = -gamma * excess ** 2

    # 4. Context-adaptive ramp wear — discounted when economics justify the move
    c_wear  = 0.005
    urgency = float(np.clip(abs(dispatch_value) / 0.3, 0.0, 1.0))
    r_ramp  = -c_wear * abs(ramp_dir) * (1.0 - 0.8 * urgency)

    # 5. Physics boundary violation
    r_boundary = -5.0 if boundary_hit else 0.0

    total = r_dispatch + r_export + r_safety + r_ramp + r_boundary
    return float(np.clip(total, -10.0, 10.0))


def default_smr_reward_multiobjective(params: dict) -> float:
    """Multi-objective SMR reward: carbon displacement + grid revenue + reactor longevity.

    The SMR is a committed cost (sunk capital), so the goal is to maximise
    value from every MWh it generates rather than deciding whether to run.

    Components
    ----------
    R_carbon    (α):  smr_power_fraction × norm_CI
                      Reward high output when the grid is dirty — each MWh
                      from the SMR displaces a fossil-fuel MWh.

    R_revenue   (β):  export_fraction × (norm_price − 0.5)
                      Centred at the midpoint of the normalised price range.
                      Positive when price is above average (reward export),
                      negative when below (penalise cheap export).  Creates
                      a genuine dispatch tradeoff — the agent must choose
                      *when* to export, not just export as much as possible.

    R_longevity (γ):  −temp_delta_norm
                      Penalise rate of core-temperature change (thermal
                      fatigue).  Large ΔT per timestep accelerates material
                      creep and shortens reactor life.  Normalised by 40 °C
                      (≈ max realistic single-step ΔT at full ramp).

    R_stability :  −c_wear × |a_t|  (c_wear = 0.02)
                   Penalise unnecessary control-rod movements.

    R_boundary  :  −5.0 if action hit P_min / P_max limit.

    Weight defaults (α, β, γ) = (0.5, 0.3, 0.2) sum to 1.0; override via
    env_config keys alpha_carbon / beta_revenue / gamma_longevity.

    Args:
        params (dict): Must contain:
            smr_power_fraction, smr_grid_export_kW, smr_ramp_dir,
            smr_boundary_hit, norm_CI, norm_price, smr_temp_delta,
            max_smr_capacity_mw, alpha_carbon, beta_revenue, gamma_longevity.

    Returns:
        float: Total reward clipped to [−10, 10].
    """
    norm_ci            = params['norm_CI']
    norm_price         = params.get('norm_price', norm_ci)   # fallback: use CI as proxy
    smr_power_fraction = params['smr_power_fraction']
    smr_grid_export_kw = params['smr_grid_export_kW']
    ramp_dir           = params['smr_ramp_dir']
    boundary_hit       = params['smr_boundary_hit']
    max_smr_kw         = params['max_smr_capacity_mw'] * 1000.0
    temp_delta         = params.get('smr_temp_delta', 0.0)

    alpha = params.get('alpha_carbon',    0.5)
    beta  = params.get('beta_revenue',    0.3)
    gamma = params.get('gamma_longevity', 0.2)

    # R_carbon: displace dirty grid generation
    r_carbon = smr_power_fraction * norm_ci

    # R_revenue: earn from high-price grid export, penalise low-price export.
    # Centred at 0.5 so the signal is negative when price is below average
    # and positive when above — creates a genuine dispatch incentive.
    export_fraction = smr_grid_export_kw / max(max_smr_kw, 1e-9)
    r_revenue = export_fraction * (norm_price - 0.5)

    # R_longevity: penalise thermal cycling (fatigue)
    # 40 °C normalisation ≈ max ΔT at full ramp in one 15-min step
    temp_delta_norm = float(np.clip(temp_delta / 40.0, 0.0, 1.0))
    r_longevity = -temp_delta_norm

    # R_stability: penalise unnecessary control-rod ramps (wear)
    c_wear      = 0.02
    r_stability = -c_wear * abs(ramp_dir)

    # R_boundary: hard penalty for attempted physics violation
    r_boundary = -5.0 if boundary_hit else 0.0

    total = (alpha * r_carbon
             + beta  * r_revenue
             + gamma * r_longevity
             + r_stability
             + r_boundary)
    return float(np.clip(total, -10.0, 10.0))


# Other reward methods can be added here.

REWARD_METHOD_MAP = {
    'default_dc_reward' : default_dc_reward,
    'default_bat_reward': default_bat_reward,
    'default_ls_reward' : default_ls_reward,
    # Add custom reward methods here
    'custom_agent_reward' : custom_agent_reward,
    'tou_reward' : tou_reward,
    'renewable_energy_reward' : renewable_energy_reward,
    'energy_efficiency_reward' : energy_efficiency_reward,
    'energy_PUE_reward' : energy_PUE_reward,
    'temperature_efficiency_reward' : temperature_efficiency_reward,
    'water_usage_efficiency_reward' : water_usage_efficiency_reward,
    'default_smr_reward'                   : default_smr_reward,
    'default_smr_reward_multiobjective'    : default_smr_reward_multiobjective,
    'default_smr_reward_lmp_dispatch'      : default_smr_reward_lmp_dispatch,
}

def get_reward_method(reward_method : str = 'default_dc_reward'):
    """
    Maps the string identifier to the reward function

    Args:
        reward_method (string): Identifier for the reward function.

    Returns:
        function: Reward function.
    """
    assert reward_method in REWARD_METHOD_MAP.keys(), f"Specified Reward Method {reward_method} not in REWARD_METHOD_MAP"
    
    return REWARD_METHOD_MAP[reward_method]

