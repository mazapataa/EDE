import numpy as np
import scipy.stats as st
import cobaya
from cobaya.model import get_model
import ultranest
import pickle
import matplotlib.pyplot as plt

# Import GetDist for plotting
try:
    from getdist import plots, MCSamples
    use_getdist = True
    print("GetDist available - will create triangle plot")
except ImportError:
    use_getdist = False
    print("WARNING: GetDist not available - triangle plot will be skipped")

# ---------------------------
# Cobaya configuration (info)
# ---------------------------
info = {
    "theory": {
        "classy": {
            "path": "/Users/alfonsozapata/Documents/EDE_update/EDE/class_ede",
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
        "logA": {
            "prior": {"min": 1.61, "max": 3.91},
            "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
            "proposal": 0.001,
            "latex": r"\log(10^{10} A_\mathrm{s})",
            "drop": True
        },
        "A_s": {
            "value": "lambda logA: 1e-10*np.exp(logA)",
            "latex": "A_\mathrm{s}"
        },
        "n_s": {
            "prior": {"min": 0.8, "max": 1.2},
            "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
            "proposal": 0.002,
            "latex": "n_\mathrm{s}"
        },
        "H0": {
            "prior": {"min": 20, "max": 100},
            "ref": {"dist": "norm", "loc": 67.5, "scale": 0.5},
            "proposal": 0.5,
            "latex": "H_0"
        },  
        "omega_b": {
            "prior": {"min": 0.005, "max": 0.1},
            "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
            "proposal": 0.0001,
            "latex": r"\Omega_\mathrm{b} h^2"
        },
        "omega_cdm": {
            "prior": {"min": 0.001, "max": 0.99},
            "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
            "proposal": 0.0005,
            "latex": r"\Omega_\mathrm{c} h^2"
        },
        "Omega_m": {"latex": r"\Omega_\mathrm{m}"},
        "m_ncdm": {
            "prior": {"min": 0, "max": 1.667},
            "ref": {"dist": "norm", "loc": 0.0067, "scale": 0.033},
            "proposal": 0.01,
            "latex": r"m_\nu"
        },
        "w0_fld": {
            "prior": {"min": -3, "max": 1},
            "ref": {"dist": "norm", "loc": -0.99, "scale": 0.02},
            "proposal": 0.02,
            "latex": r"w_{0,\mathrm{DE}}"
        },
        "Omega_EDE": {
            "prior": {"min": 0.0, "max": 0.5},
            "ref": {"dist": "norm", "loc": 0.004, "scale": 0.0003},
            "latex": r"\Omega_{\mathrm{EDE}}"
        },
        "Omega_Lambda": {"value": 0.0},
        "tau_reio": {
            "prior": {"min": 0.01, "max": 0.8},
            "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006},
            "proposal": 0.003,
            "latex": r"\tau_\mathrm{reio}"
        }
    }
}

# Build Cobaya model
print("Building Cobaya model...")
model = get_model(info)
print("Model built successfully!")

# The list of parameters we will sample (order matters)
param_names = [
    'logA', 'n_s', 'H0', 'omega_b','omega_cdm', 'm_ncdm', 'w0_fld', 'Omega_EDE', 'tau_reio']

# LaTeX labels for triangle plot
param_labels = [
    r'\log(10^{10} A_\mathrm{s})',
    r'n_\mathrm{s}',
    r'H_0',
    r'\Omega_\mathrm{b} h^2',
    r'\Omega_\mathrm{c} h^2',
    r'm_\nu',
    r'w_{0,\mathrm{DE}}',
    r'\Omega_{\mathrm{EDE}}',
    r'\tau_\mathrm{reio}'
]

# ---------------------------
# Uniform prior transform
# ---------------------------
def prior_transform(cube):
    """
    Transform unit cube to parameter values using uniform priors.
    """
    u = np.asarray(cube)
    
    # Define uniform prior ranges for each parameter
    prior_ranges = [
        (2.8, 3.2),        # logA: [3.0, 3.1]
        (0.9, 1.0),      # n_s: [0.95, 0.98]
        (60.00, 80.00),    # H0: [65.0, 75.0]
        (0.022, 0.0226),   # omega_b: [0.022, 0.0226]
        (0.11, 0.13),      # omega_cdm: [0.11, 0.13]
        (0.0, 0.1),        # m_ncdm: [0.0, 0.1]
        (-2.0, -0.5),      # w0_fld: [-1.5, -0.5]
        (0.0, 0.1),       # Omega_EDE: [0.0, 0.01]
        (0.01, 0.08)       # tau_reio: [0.04, 0.07]
    ]

    def map_row(v):
        return np.array([
            v[i] * (hi - lo) + lo for i, (lo, hi) in enumerate(prior_ranges)
        ])

    if u.ndim == 1:
        return map_row(u)
    elif u.ndim == 2:
        return np.array([map_row(v) for v in u])
    else:
        raise ValueError("prior_transform expects 1D or 2D input")
        
# ---------------------------
# Vectorized likelihood
# ---------------------------
def my_likelihood(theta):
    theta = np.asarray(theta)

    def eval_single(x):
        params_dict = {n: float(v) for n, v in zip(param_names, x)}
        try:
            return float(model.logposterior(params_dict).loglike)
        except Exception:
            return -np.inf

    if theta.ndim == 1:
        return eval_single(theta)
    elif theta.ndim == 2:
        return np.array([eval_single(row) for row in theta])
    else:
        raise ValueError("theta must be 1D or 2D")

# ---------------------------
# Function to create triangle plot
# ---------------------------
def create_triangle_plot(samples, weights, param_names, param_labels, output_file='triangle_plot.pdf'):
    """
    Create a triangle (corner) plot using GetDist and save as PDF.
    
    Parameters:
    -----------
    samples : array_like
        Parameter samples from the posterior
    weights : array_like
        Weights for each sample
    param_names : list
        List of parameter names
    param_labels : list
        List of LaTeX labels for parameters
    output_file : str
        Output filename (should end in .pdf)
    """
    if not use_getdist:
        print("WARNING: GetDist not available. Cannot create triangle plot.")
        return
    
    print("\n" + "="*60)
    print("Creating triangle plot with GetDist...")
    print("="*60)
    
    try:
        # Create GetDist MCSamples object
        gd_samples = MCSamples(
            samples=samples,
            weights=weights,
            names=param_names,
            labels=param_labels,
            label='EDE Model'
        )
        
        # Create triangle plot
        g = plots.get_subplot_plotter(width_inch=12)
        g.settings.num_plot_contours = 2
        g.settings.solid_contour_palefactor = 0.6
        g.settings.alpha_filled_add = 0.85
        
        g.triangle_plot(
            gd_samples,
            filled=True,
            title_limit=1
        )
        
        # Save as PDF
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"✓ Triangle plot saved: {output_file}")
        plt.close()
        
        # Print parameter statistics
        print("\n" + "="*60)
        print("Parameter Statistics from Triangle Plot:")
        print("="*60)
        
        for i, (name, label) in enumerate(zip(param_names, param_labels)):
            mean = np.average(samples[:, i], weights=weights)
            std = np.sqrt(np.average((samples[:, i] - mean)**2, weights=weights))
            median = np.percentile(samples[:, i], 50)
            q16 = np.percentile(samples[:, i], 16)
            q84 = np.percentile(samples[:, i], 84)
            
            print(f"{name:12s}: {mean:9.5f} ± {std:8.5f}")
            print(f"             Median: {median:9.5f} (+{q84-median:8.5f}, -{median-q16:8.5f})")
        
        print("="*60)
        
    except Exception as e:
        print(f"✗ ERROR creating triangle plot: {e}")
        import traceback
        traceback.print_exc()

# ---------------------------
# Run UltraNest
# ---------------------------
if __name__ == "__main__":
    # Test at fiducial point
    test_params = {
        'logA': 3.05,
        'n_s': 0.965,
        'H0': 67.5,
        'omega_b': 0.0224,
        'omega_cdm': 0.12,
        'm_ncdm': 0.0,
        'w0_fld': -0.99,
        'Omega_EDE': 0.004,
        'tau_reio': 0.055
    }
    
    print("\n" + "="*60)
    print("Testing likelihood at fiducial point...")
    print("="*60)
    try:
        test_loglike = model.logposterior(test_params).loglike
        print(f"✓ Fiducial log-likelihood: {test_loglike:.2f}")
    except Exception as e:
        print(f"✗ ERROR at fiducial point: {e}")

    print("\n" + "="*60)
    print("Starting UltraNest Nested Sampling")
    print("="*60)
    print(f"Parameters: {len(param_names)}")
    print(f"Fixed: N_ur = 2.0328")
    print("="*60 + "\n")

    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        my_likelihood,
        prior_transform,
        log_dir="ede_real_run",  # Different directory for real run
        #resume=False,            # Start fresh
        num_test_samples=0       # Disable validation
        )

    # REAL RUN WITH PROPER SETTINGS
    result = sampler.run(
        min_num_live_points=400,      # Standard for cosmological applications
        cluster_num_live_points=40,   # Default value
        dlogz=0.5,                    # Good convergence criterion for cosmology
        max_ncalls=1000000,           # Large limit for thorough sampling
        min_ess=1000,                 # Sufficient effective samples
        max_num_improvement_loops=5,  # Good refinement
        show_status=True,
    )

    # Save comprehensive results
    print("\n" + "="*60)
    print("REAL RUN COMPLETED")
    print("="*60)
    sampler.print_results()

    # FIXED: Extract samples and weights correctly from UltraNest
    # UltraNest stores weighted_samples and weighted_samples['weights']
    print("\nExtracting samples and weights...")
    
    # Method 1: Try to get from weighted_samples
    if 'weighted_samples' in result:
        weighted_samples = result['weighted_samples']
        samples = weighted_samples['points']
        weights = weighted_samples['weights']
        print("✓ Extracted from weighted_samples")
    # Method 2: Try samples directly and compute weights
    elif 'samples' in result:
        samples = result['samples']
        # Compute normalized weights from logwt if available
        if 'logwt' in result:
            logwt = result['logwt']
            # Normalize weights
            max_logwt = np.max(logwt)
            weights = np.exp(logwt - max_logwt)
            weights = weights / np.sum(weights)
            print("✓ Extracted samples and computed weights from logwt")
        else:
            # Equal weights as fallback
            weights = np.ones(len(samples)) / len(samples)
            print("⚠ Using equal weights (logwt not available)")
    else:
        print("✗ ERROR: Could not extract samples from result")
        print(f"Available keys in result: {list(result.keys())}")
        # Try to save what we have
        with open('ede_ultranest_results_partial.pkl', 'wb') as f:
            pickle.dump(result, f)
        print("Saved partial results to: ede_ultranest_results_partial.pkl")
        raise KeyError("Cannot extract samples and weights from UltraNest result")
    
    print(f"Number of samples: {len(samples)}")
    print(f"Sum of weights: {np.sum(weights):.6f}")
    
    # Collect available data safely
    data_to_save = {
        'result': result,
        'param_names': param_names,
        'param_labels': param_labels,
        'ncall': sampler.ncall,
        'logz': result['logz'],
        'logzerr': result['logzerr'],
        'samples': samples,
        'weights': weights,
        'N_ur_fixed': 2.0328,
        'note': 'EDE model with N_ur fixed at 2.0328, H0 sampled [65, 75]'
    }

    # Add additional fields if available
    for key in ['logl', 'logwt', 'logvol', 'information']:
        if key in result:
            data_to_save[key] = result[key]

    # Save results
    with open('ede_ultranest_results.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print("\n✓ Results saved: ede_ultranest_results.pkl")

    # Create triangle plot
    create_triangle_plot(
        samples=samples,
        weights=weights,
        param_names=param_names,
        param_labels=param_labels,
        output_file='ede_triangle_plot.pdf'
    )
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Log-evidence: {result['logz']:.2f} ± {result['logzerr']:.2f}")
    print(f"Number of samples: {len(samples)}")
    eff_samples = np.sum(weights)**2 / np.sum(weights**2)
    print(f"Effective samples: {eff_samples:.0f}")
    print("\nOutput files:")
    print("  - ede_ultranest_results.pkl (all data)")
    print("  - ede_triangle_plot.pdf (corner plot)")
    print("  - ede_real_run/ (UltraNest run directory)")
    print("="*60)
