import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SMREnv(gym.Env):
    """Small Modular Reactor (SMR) environment for SustainDC.

    The reactor ramps power up, holds, or ramps down each 15-minute timestep.
    Excess generation beyond current DC demand is logged as grid export.

    Physics
    -------
    P(t) = clip(P(t-1) + a_t * delta_ramp, P_min, P_max)
        a_t in {-1, 0, +1}  mapped from Discrete(3) actions {0, 1, 2}

    Core temperature follows a first-order thermal lag:
        T(t) = T(t-1) + (T_ss(P(t)) - T(t-1)) * (1 - exp(-1 / tau))
    where tau is expressed in 15-minute timesteps (default 4 == 1 hour).

    Notes
    -----
    The obs vector returned by step() is a zeros placeholder; the real
    11-D observation is assembled by SustainDC._create_smr_state() using
    shared CI/workload/temperature data from the environment managers.
    update_dc_demand() must be called by SustainDC before each step().
    """

    # --- Thermal model constants ---
    _T_COOLANT_BASE  = 150.0   # °C  coolant inlet / minimum core temp
    _DELTA_T_MAX     = 150.0   # °C  additional rise from min → max power
    _TAU_STEPS       = 4.0     # thermal time constant (15-min timesteps)
    _T_SAFETY_LIMIT  = 320.0   # °C  soft safety threshold used in reward

    # Discrete action → ramp direction
    ACTION_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, env_config: dict):
        super().__init__()

        self.max_power_mw   = float(env_config['max_smr_capacity_mw'])
        self.min_power_mw   = self.max_power_mw * float(env_config['smr_min_power_fraction'])
        self.delta_ramp     = self.max_power_mw * float(env_config['smr_ramp_rate_fraction'])
        self.dc_load_max_kw = float(env_config.get('dc_load_max_kw', self.max_power_mw * 1000.0))

        self.action_space = spaces.Discrete(3)
        # Shape must match the 11-D vector produced by SustainDC._create_smr_state
        self.observation_space = spaces.Box(
            low=np.float32(-1.0 * np.ones(11)),
            high=np.float32(np.ones(11)),
        )

        init_frac             = float(env_config.get('init_power_fraction', 0.8))
        self.current_power_mw = self.max_power_mw * init_frac
        self.core_temp        = self._steady_state_temp(self.current_power_mw)
        self.dc_demand_kw     = 0.0

    # ------------------------------------------------------------------
    # External update hook — called inside SustainDC._perform_actions()
    # before each call to step(), so export is computed against the
    # DC load from the *current* timestep.
    # ------------------------------------------------------------------
    def update_dc_demand(self, dc_demand_kw: float):
        """Set current DC power demand (kW) before calling step()."""
        self.dc_demand_kw = float(dc_demand_kw)

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    def _steady_state_temp(self, power_mw: float) -> float:
        """Return the steady-state core temperature at a given power level."""
        power_range = max(self.max_power_mw - self.min_power_mw, 1e-9)
        fraction    = (power_mw - self.min_power_mw) / power_range
        return self._T_COOLANT_BASE + fraction * self._DELTA_T_MAX

    def _step_core_temp(self, new_power_mw: float) -> float:
        """Advance core temperature by one timestep toward steady state."""
        t_ss  = self._steady_state_temp(new_power_mw)
        alpha = 1.0 - np.exp(-1.0 / self._TAU_STEPS)
        return self.core_temp + (t_ss - self.core_temp) * alpha

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        init_frac             = float((options or {}).get('init_power_fraction', 0.8))
        self.current_power_mw = self.max_power_mw * init_frac
        self.core_temp        = self._steady_state_temp(self.current_power_mw)
        self.dc_demand_kw     = 0.0
        info = self._build_info(ramp_dir=0, boundary_hit=False)
        # Placeholder obs; real obs assembled by SustainDC._create_smr_state
        return np.zeros(11, dtype=np.float32), info

    def step(self, action: int):
        """Advance reactor physics by one 15-minute timestep.

        Args:
            action (int): 0 = ramp down, 1 = hold, 2 = ramp up.

        Returns:
            obs        : zeros placeholder (11,) — overridden by SustainDC
            reward     : 0.0 — reward computed externally by reward_creator
            terminated : False — episode end managed by Time_Manager
            truncated  : False
            info       : dict with smr_power_output_kW, smr_core_temp,
                         smr_grid_export_kW, smr_ramp_dir, smr_boundary_hit,
                         smr_power_fraction
        """
        ramp_dir = self.ACTION_MAP[int(action)]

        # Detect attempted violations *before* clamping
        at_max       = self.current_power_mw >= self.max_power_mw - 1e-6
        at_min       = self.current_power_mw <= self.min_power_mw + 1e-6
        boundary_hit = (ramp_dir == 1 and at_max) or (ramp_dir == -1 and at_min)

        # P(t) = clip(P(t-1) + a_t * delta_ramp, P_min, P_max)
        self.current_power_mw = float(np.clip(
            self.current_power_mw + ramp_dir * self.delta_ramp,
            self.min_power_mw,
            self.max_power_mw,
        ))

        # First-order thermal update
        self.core_temp = float(self._step_core_temp(self.current_power_mw))

        info = self._build_info(ramp_dir, boundary_hit)
        return np.zeros(11, dtype=np.float32), 0.0, False, False, info

    # ------------------------------------------------------------------
    def _build_info(self, ramp_dir: int, boundary_hit: bool) -> dict:
        smr_output_kw  = self.current_power_mw * 1000.0
        grid_export_kw = max(0.0, smr_output_kw - self.dc_demand_kw)
        return {
            'smr_power_output_kW': smr_output_kw,
            'smr_power_fraction':  self.current_power_mw / self.max_power_mw,
            'smr_core_temp':       self.core_temp,
            'smr_grid_export_kW':  grid_export_kw,
            'smr_ramp_dir':        ramp_dir,
            'smr_boundary_hit':    boundary_hit,
        }
