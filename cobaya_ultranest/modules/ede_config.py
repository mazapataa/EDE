import numpy as np
CLASSY_PATH = "/Users/alfonsozapata/Documents/EDE_update/EDE/class_ede"

OUTPUT_DIR = "ede_output"
RUN_NAME = "ede_real_run"

# ==========================================
# SAMPLING PARAMETERS
# ==========================================
SAMPLING_PARAMS = {
    'logA':      {'label': r'\log(10^{10} A_\mathrm{s})', 'range': (2.8, 3.2)},
    'n_s':       {'label': r'n_\mathrm{s}',               'range': (0.9, 1.0)},
    'H0':        {'label': r'H_0',                        'range': (60.0, 80.0)},
    'omega_b':   {'label': r'\Omega_\mathrm{b} h^2',      'range': (0.022, 0.0226)},
    'omega_cdm': {'label': r'\Omega_\mathrm{c} h^2',      'range': (0.11, 0.13)},
    'm_ncdm':    {'label': r'm_\nu',                      'range': (0.0, 0.1)},
    'w0_fld':    {'label': r'w_{0,\mathrm{DE}}',          'range': (-2.0, -0.5)},
    'Omega_EDE': {'label': r'\Omega_{\mathrm{EDE}}',      'range': (0.0, 0.1)},
    'tau_reio':  {'label': r'\tau_\mathrm{reio}',         'range': (0.01, 0.08)}
}


PARAM_NAMES = list(SAMPLING_PARAMS.keys())
PARAM_LABELS = [v['label'] for v in SAMPLING_PARAMS.values()]
PRIOR_RANGES = [v['range'] for v in SAMPLING_PARAMS.values()]

# ==========================================
# 3. COBAYA MODEL INFO
# ==========================================
COBAYA_INFO = {
    "theory": {
        "classy": {
            "path": CLASSY_PATH,
            "extra_args": {
                "fluid_equation_of_state": "EDE",
                "use_ppf": True,
                "N_ncdm": 1,
                "N_ur": 2.0328,
                "deg_ncdm": 3.0,
                "T_ncdm": 0.71611,
                "tol_initial_Omega_r": 0.1
            }
        }
    },
    "likelihood": {
        "bao.desi_dr2": None,
        # "sn.pantheonplus": None
    },
    "params": {
        # Sampling params (priors here are for Cobaya internal checks, 
        # actual sampling volume is controlled by SAMPLING_PARAMS ranges above)
        "logA":      {"prior": {"min": 1.61, "max": 3.91}, "drop": True, "latex": r"\log(10^{10} A_\mathrm{s})"},
        "A_s":       {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\mathrm{s}"},
        "n_s":       {"prior": {"min": 0.8, "max": 1.2}, "latex": "n_\mathrm{s}"},
        "H0":        {"prior": {"min": 20, "max": 100}, "latex": "H_0"},
        "omega_b":   {"prior": {"min": 0.005, "max": 0.1}, "latex": r"\Omega_\mathrm{b} h^2"},
        "omega_cdm": {"prior": {"min": 0.001, "max": 0.99}, "latex": r"\Omega_\mathrm{c} h^2"},
        "m_ncdm":    {"prior": {"min": 0, "max": 1.667}, "latex": r"m_\nu"},
        "w0_fld":    {"prior": {"min": -3, "max": 1}, "latex": r"w_{0,\mathrm{DE}}"},
        "Omega_EDE": {"prior": {"min": 0.0, "max": 0.5}, "latex": r"\Omega_{\mathrm{EDE}}"},
        "tau_reio":  {"prior": {"min": 0.01, "max": 0.8}, "latex": r"\tau_\mathrm{reio}"},
        
        # Derived/Fixed
        "Omega_m":      {"latex": r"\Omega_\mathrm{m}"},
        "Omega_Lambda": {"value": 0.0}
    }
}