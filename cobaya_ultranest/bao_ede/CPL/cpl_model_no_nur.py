import numpy as np
import scipy.stats as st
import cobaya
from cobaya.model import get_model
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import os
import pickle
import json
import sys
import ctypes
import subprocess

# Import GetDist for plotting
try:
    from getdist import plots, MCSamples
    use_getdist = True
except ImportError:
    print("GetDist not available, falling back to corner plots")
    import corner
    use_getdist = False

# ---------------------------
# MPI Setup
# ---------------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    use_mpi = size > 1
except ImportError:
    rank = 0
    size = 1
    use_mpi = False
    comm = None

# Only print from rank 0
def print_rank0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

# ---------------------------
# Cobaya configuration (info) - CPL parameterization without N_ur sampling
# ---------------------------
info = {
    "theory": {
        "classy": {
            "path": '/Users/alfonsozapata/Documents/EDE_update/EDE/class_ede',
            "extra_args": {
                "fluid_equation_of_state": "CLP",
                "use_ppf": True,
                "N_ncdm": 1,
                "N_ur": 2.0328,
                "deg_ncdm": 3.0,
                "T_ncdm": 0.71611,
            #    "tol_initial_Omega_r": 0.1
            }
        }
    },
    "likelihood": {
        "bao.desi_dr2": None,
    #    "sn.pantheonplus": None
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
            "prior": {"min": 0.0, "max": 2.0},
            "ref": {"dist": "norm", "loc": 0.12, "scale": 0.01},
            "proposal": 0.01,
            "latex": r"m_\nu"
        },
        "w0_fld": {
            "prior": {"min": -2.0, "max": 0.0},
            "ref": {"dist": "norm", "loc": -1.0, "scale": 0.02},
            "proposal": 0.02,
            "latex": r"w_{0,\mathrm{DE}}"
        },
        "wa_fld": {
            "prior": {"min": -2.0, "max": 2.0},
            "ref": {"dist": "norm", "loc": 0.0, "scale": 0.1},
            "proposal": 0.05,
            "latex": r"w_{a,\mathrm{DE}}"
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

# Build Cobaya model with library path fix
print_rank0("Building Cobaya model...")
print_rank0(f"MPI: Using {size} processes (rank {rank})")

# Fix for GLIBCXX compatibility issue
try:
    ctypes.CDLL("libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)
except:
    pass

result = subprocess.run(['gcc', '--print-file-name=libstdc++.so.6'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    gcc_lib_path = os.path.dirname(result.stdout.strip())
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{gcc_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = gcc_lib_path

try:
    model = get_model(info)
    print_rank0("Model built successfully!")
except ImportError as e:
    if "GLIBCXX" in str(e):
        print_rank0("\n" + "="*60)
        print_rank0("GLIBCXX COMPATIBILITY ERROR DETECTED")
        print_rank0("="*60)
        print_rank0("\nThe CLASS module requires a newer C++ standard library.")
        print_rank0("\nTo fix this, run ONE of these solutions:\n")
        print_rank0("Solution 1 - Update conda environment:")
        print_rank0("  conda install -c conda-forge libstdcxx-ng\n")
        print_rank0("Solution 2 - Use system libraries:")
        print_rank0("  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
        print_rank0("  # Then restart Python\n")
        print_rank0("Solution 3 - Recompile CLASS in current environment:")
        print_rank0("  cd /Users/alfonsozapata/Documents/EDE_update/EDE/class_ede")
        print_rank0("  make clean")
        print_rank0("  python setup.py build")
        print_rank0("  python setup.py install\n")
        print_rank0("="*60)
        sys.exit(1)
    else:
        raise

# CPL sampled parameters - 9 parameters (N_ur fixed at 2.0328)
param_names = ['logA', 'n_s', 'H0', 'omega_b', 'omega_cdm', 'm_ncdm', 'w0_fld', 'wa_fld', 'tau_reio']
param_labels = [r'$\log(10^{10}A_s)$', r'$n_s$', r'$H_0$', r'$\Omega_b h^2$', 
                r'$\Omega_c h^2$', r'$m_\nu$', r'$w_0$', r'$w_a$', r'$\tau_{reio}$']
param_labels_latex = [r'\log(10^{10} A_\mathrm{s})', 'n_\mathrm{s}', r'H_0',
                      r'\Omega_\mathrm{b} h^2', r'\Omega_\mathrm{c} h^2', 
                      r'm_\nu', 'w_0', 'w_a', r'\tau_\mathrm{reio}']

# Prior bounds for all parameters
prior_bounds = {
    'logA': (1.61, 3.91),
    'n_s': (0.8, 1.2),
    'H0': (20.0, 100.0),
    'omega_b': (0.005, 0.1),
    'omega_cdm': (0.001, 0.99),
    'm_ncdm': (0.0, 2.0),
    'w0_fld': (-2.0, 0.0),
    'wa_fld': (-2.0, 2.0),
    'tau_reio': (0.01, 0.8)
}

# Fiducial values for initial points
fiducial = {
    'logA': 3.05,
    'n_s': 0.965,
    'H0': 67.5,
    'omega_b': 0.0224,
    'omega_cdm': 0.12,
    'm_ncdm': 0.12,
    'w0_fld': -1.0,
    'wa_fld': 0.0,
    'tau_reio': 0.055
}

# ---------------------------
# Prior transform for dynesty
# ---------------------------
def prior_transform(u):
    """Transform unit cube [0,1]^9 to parameter space with uniform priors"""
    params = np.zeros(9)
    
    # Simple uniform priors within bounds
    for i, pname in enumerate(param_names):
        low, high = prior_bounds[pname]
        params[i] = low + u[i] * (high - low)
    
    return params


# ---------------------------
# Likelihood function with better error handling
# ---------------------------
fail_count = 0
success_count = 0

def log_likelihood(theta):
    """Compute log-likelihood using Cobaya model with CLASS"""
    global fail_count, success_count
    
    params_dict = {
        'logA': float(theta[0]),
        'n_s': float(theta[1]),
        'H0': float(theta[2]),
        'omega_b': float(theta[3]),
        'omega_cdm': float(theta[4]),
        'm_ncdm': float(theta[5]),
        'w0_fld': float(theta[6]),
        'wa_fld': float(theta[7]),
        'tau_reio': float(theta[8])
    }
    
    # Physical validity checks
    try:
        # Check that sum of densities is reasonable
       # omega_total = params_dict['omega_b'] + params_dict['omega_cdm']
       # if omega_total < 0.05 or omega_total > 0.5:
       #     fail_count += 1
       #     return -np.inf
        
        # Check H0 is reasonable
        if params_dict['H0'] < 20 or params_dict['H0'] > 100:
            fail_count += 1
            return -np.inf
        
        # Check m_ncdm is non-negative
        if params_dict['m_ncdm'] < 0:
            fail_count += 1
            return -np.inf
        
        logpost = model.logposterior(params_dict)
        loglike = float(logpost.loglike)
        
        # Check if likelihood is valid
        if not np.isfinite(loglike):
            fail_count += 1
            return -np.inf
        
        success_count += 1
        if success_count % 100 == 0 and rank == 0:
            print(f"Success: {success_count}, Fails: {fail_count}, LogL: {loglike:.2f}")
        
        return loglike
        
    except Exception as e:
        fail_count += 1
        if fail_count % 50 == 0 and rank == 0:
            print(f"Likelihood error (total fails: {fail_count}): {e}")
            print(f"Parameters: H0={params_dict['H0']:.2f}, omega_cdm={params_dict['omega_cdm']:.4f}, w0={params_dict['w0_fld']:.3f}, wa={params_dict['wa_fld']:.3f}")
        return -np.inf


# ---------------------------
# MPI Pool for parallel likelihood evaluation
# ---------------------------
class MPIPool:
    """Simple MPI pool for parallel function evaluation"""
    
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
    
    def map(self, func, tasks):
        """Distribute tasks across MPI processes"""
        if self.size == 1:
            return [func(task) for task in tasks]
        
        if self.rank == 0:
            n_tasks = len(tasks)
            results = [None] * n_tasks
            
            task_id = 0
            for worker in range(1, min(self.size, n_tasks + 1)):
                if task_id < n_tasks:
                    self.comm.send((task_id, tasks[task_id]), dest=worker, tag=1)
                    task_id += 1
            
            n_received = 0
            while n_received < n_tasks:
                status = MPI.Status()
                result = self.comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
                worker = status.Get_source()
                
                result_id, result_value = result
                results[result_id] = result_value
                n_received += 1
                
                if task_id < n_tasks:
                    self.comm.send((task_id, tasks[task_id]), dest=worker, tag=1)
                    task_id += 1
                else:
                    self.comm.send((-1, None), dest=worker, tag=1)
            
            return results
        else:
            while True:
                task_id, task = self.comm.recv(source=0, tag=1)
                if task_id == -1:
                    break
                result = func(task)
                self.comm.send((task_id, result), dest=0, tag=2)
            return None


# ---------------------------
# Parallelized likelihood for dynesty
# ---------------------------
if use_mpi:
    pool = MPIPool(comm)
    
    def parallel_log_likelihood(theta):
        """Wrapper for MPI-parallelized likelihood"""
        if np.ndim(theta) == 1:
            return log_likelihood(theta)
        else:
            if rank == 0:
                results = pool.map(log_likelihood, theta)
                return np.array(results)
            else:
                pool.map(log_likelihood, theta)
                return None
    
    likelihood_func = parallel_log_likelihood
else:
    likelihood_func = log_likelihood


# ---------------------------
# Run dynesty nested sampling
# ---------------------------
def run_nested_sampling(ndim=9, nlive=500, dlogz=0.005, output_dir='./cpl_dynesty_output'):
    """Run nested sampling using dynesty with MPI support"""
    
    if rank != 0 and use_mpi:
        while True:
            try:
                pool.map(log_likelihood, [])
            except:
                break
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    print_rank0("\n" + "="*60)
    print_rank0("Starting Dynesty Nested Sampling - CPL with CLASS (N_ur fixed)")
    print_rank0("="*60)
    print_rank0(f"Number of live points: {nlive}")
    print_rank0(f"Target evidence tolerance: dlogz = {dlogz}")
    print_rank0(f"Output directory: {output_dir}")
    print_rank0(f"Number of parameters: {ndim}")
    print_rank0(f"MPI processes: {size}")
    print_rank0(f"\nParameter ranges:")
    for pname in param_names:
        bounds = prior_bounds[pname]
        fid = fiducial[pname]
        print_rank0(f"  {pname:12s}: [{bounds[0]:.4f}, {bounds[1]:.4f}] (fiducial: {fid:.4f})")
    
    print_rank0(f"\n  Note: N_ur fixed at 2.0328")
    print_rank0(f"        m_ncdm prior range: 0.0 to 2.0 eV")
    
    sampler = NestedSampler(
        likelihood_func,
        prior_transform,
        ndim,
        nlive=nlive,
        bound='multi',
        sample='rwalk',
        walks=25,  # Number of random walk steps
        pool=pool if use_mpi else None,
        queue_size=size if use_mpi else None
    )
    
    print_rank0("\nRunning nested sampling...")
    print_rank0("This may take a while. Progress will be displayed below.")
    
    sampler.run_nested(dlogz=dlogz, print_progress=True)
    
    print_rank0("\nNested sampling completed!")
    print_rank0(f"Total successful likelihood evaluations: {success_count}")
    print_rank0(f"Total failed likelihood evaluations: {fail_count}")
    
    results = sampler.results
    
    # Save results
    with open(os.path.join(output_dir, 'dynesty_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print_rank0(f"Saved: {os.path.join(output_dir, 'dynesty_results.pkl')}")
    
    return results


def save_chains_and_info(results, output_dir='./cpl_dynesty_output'):
    """Save chains, weights, and summary statistics"""
    
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    loglikelihood = results.logl
    
    # Save raw chain
    chain_data = np.column_stack([samples, weights, loglikelihood])
    header = ' '.join(param_names) + ' weight loglikelihood'
    np.savetxt(
        os.path.join(output_dir, 'chain_raw.txt'),
        chain_data,
        header=header,
        fmt='%.8e'
    )
    print_rank0(f"Saved: {os.path.join(output_dir, 'chain_raw.txt')}")
    
    # Save equal-weighted samples
    n_samples = len(samples)
    indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)
    equal_weight_samples = samples[indices]
    
    np.savetxt(
        os.path.join(output_dir, 'chain_equal_weights.txt'),
        equal_weight_samples,
        header=' '.join(param_names),
        fmt='%.8e'
    )
    print_rank0(f"Saved: {os.path.join(output_dir, 'chain_equal_weights.txt')}")
    
    # Save evidence info
    logz = results.logz[-1] if hasattr(results.logz, '__len__') else results.logz
    logzerr = results.logzerr[-1] if hasattr(results.logzerr, '__len__') else results.logzerr
    
    info_dict = {
        'log_evidence': float(logz),
        'log_evidence_error': float(logzerr),
        'n_samples': len(samples),
        'n_live': results.nlive if hasattr(results, 'nlive') else 'N/A',
        'parameter_names': param_names,
        'parameter_labels': param_labels_latex,
        'sampling_priors': {k: list(v) for k, v in prior_bounds.items()},
        'fiducial_values': fiducial,
        'mpi_processes': size,
        'success_count': success_count,
        'fail_count': fail_count,
        'sampling_mode': 'CPL_N_ur_fixed',
        'N_ur_value': 2.0328,
        'note': 'CPL model with N_ur fixed at 2.0328, m_ncdm sampled'
    }
    
    if hasattr(results, 'information'):
        info = results.information
        info_val = float(info[-1]) if hasattr(info, '__len__') else float(info)
        info_dict['information_nats'] = info_val
    
    with open(os.path.join(output_dir, 'sampling_info.json'), 'w') as f:
        json.dump(info_dict, f, indent=4)
    print_rank0(f"Saved: {os.path.join(output_dir, 'sampling_info.json')}")


def analyze_results(results, output_dir='./cpl_dynesty_output'):
    """Analyze and display results"""
    
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    loglikelihood = results.logl
    mean = np.average(samples, weights=weights, axis=0)
    cov = np.cov(samples.T, aweights=weights)
    std = np.sqrt(np.diag(cov))
    
    quantiles = [np.percentile(samples[:, i], [2.5, 16, 50, 84, 97.5]) 
                 for i in range(len(param_names))]
    
    # Calculate chi-squared values
    chi2_values = -2.0 * loglikelihood
    chi2_min = np.min(chi2_values[np.isfinite(chi2_values)])
    best_fit_idx = np.argmin(chi2_values)
    best_fit_params = samples[best_fit_idx]
    
    print_rank0("\n" + "="*60)
    print_rank0("RESULTS SUMMARY - CPL WITH CLASS (N_ur fixed)")
    print_rank0("="*60)
    
    logz = results.logz[-1] if hasattr(results.logz, '__len__') else results.logz
    logzerr = results.logzerr[-1] if hasattr(results.logzerr, '__len__') else results.logzerr
    print_rank0(f"\nLog-evidence: ln(Z) = {logz:.2f} ± {logzerr:.2f}")
    print_rank0(f"Chi-squared (minimum): χ²_min = {chi2_min:.2f}")
    
    if hasattr(results, 'information'):
        info = results.information
        info_val = info[-1] if hasattr(info, '__len__') else info
        print_rank0(f"Information: H = {info_val:.2f} nats")
    
    print_rank0("\nParameter Estimates:")
    print_rank0("-" * 80)
    
    summary_lines = [
        "="*80,
        "RESULTS SUMMARY - CPL MODEL WITH CLASS (N_ur=2.0328 fixed)",
        "="*80,
        "",
        f"Log-evidence: ln(Z) = {logz:.6f} ± {logzerr:.6f}",
        f"Chi-squared (minimum): χ²_min = {chi2_min:.3f}",
        ""
    ]
    
    if hasattr(results, 'information'):
        info = results.information
        info_val = info[-1] if hasattr(info, '__len__') else info
        summary_lines.append(f"Information: H = {info_val:.3f} nats")
        summary_lines.append("")
    
    summary_lines.extend([
        "="*80,
        "PARAMETER ESTIMATES (Posterior Mean ± Std)",
        "="*80,
        ""
    ])
    
    for i, (name, label) in enumerate(zip(param_names, param_labels_latex)):
        q = quantiles[i]
        line1 = f"{name:12s} = {mean[i]:9.5f} ± {std[i]:8.5f}"
        line2 = f"             Median: {q[2]:9.5f} (+{q[3]-q[2]:8.5f}, -{q[2]-q[1]:8.5f})"
        line3 = f"             95% CI: [{q[0]:9.5f}, {q[4]:9.5f}]"
        
        print_rank0(line1)
        print_rank0(line2)
        print_rank0(line3)
        print_rank0("")
        
        summary_lines.extend([line1, line2, line3, ""])
    
    # Add best-fit parameters (at chi2_min)
    summary_lines.extend([
        "="*80,
        f"BEST-FIT PARAMETERS (at χ²_min = {chi2_min:.3f})",
        "="*80,
        ""
    ])
    
    for i, name in enumerate(param_names):
        summary_lines.append(f"{name:12s} = {best_fit_params[i]:9.6f}")
    
    summary_lines.append("")
    
    print_rank0("="*80)
    print_rank0("Comparison with ΛCDM (w0=-1, wa=0)")
    print_rank0("="*80)
    
    w0_idx = param_names.index('w0_fld')
    wa_idx = param_names.index('wa_fld')
    
    w0_tension = abs(mean[w0_idx] + 1.0) / std[w0_idx] if std[w0_idx] > 0 else 0.0
    wa_tension = abs(mean[wa_idx]) / std[wa_idx] if std[wa_idx] > 0 else 0.0
    
    tension_line1 = f"Δw0 = {mean[w0_idx]+1:.5f} ± {std[w0_idx]:.5f}  ({w0_tension:.2f}σ from ΛCDM)"
    tension_line2 = f"Δwa = {mean[wa_idx]:.5f} ± {std[wa_idx]:.5f}  ({wa_tension:.2f}σ from ΛCDM)"
    
    print_rank0(tension_line1)
    print_rank0(tension_line2)
    
    summary_lines.extend([
        "="*80,
        "COMPARISON WITH ΛCDM (w0=-1, wa=0)",
        "="*80,
        "",
        tension_line1,
        tension_line2,
        "",
        "="*80
    ])
    
    with open(os.path.join(output_dir, 'parameter_summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))
    print_rank0(f"\nSaved: {os.path.join(output_dir, 'parameter_summary.txt')}")
    
    return samples, weights


def create_plots(results, samples, weights, output_dir='./cpl_dynesty_output'):
    """Create triangle plot using GetDist"""
    
    print_rank0("\nCreating triangle plot...")
    
    if use_getdist:
        # Create GetDist samples
        gd_samples = MCSamples(
            samples=samples,
            weights=weights,
            names=param_names,
            labels=param_labels_latex,
            label='CPL+CLASS'
        )
        
        # Triangle plot
        g = plots.get_subplot_plotter(width_inch=16)
        g.settings.num_plot_contours = 2
        g.triangle_plot(
            gd_samples,
            filled=True,
            title_limit=1
        )
        
        plt.savefig(os.path.join(output_dir, 'triangle_plot.pdf'), 
                    bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, 'triangle_plot.png'), 
                    bbox_inches='tight', dpi=150)
        plt.close()
        
        print_rank0(f"Saved: {os.path.join(output_dir, 'triangle_plot.pdf')}")
        print_rank0(f"Saved: {os.path.join(output_dir, 'triangle_plot.png')}")
        
    else:
        # Fallback to corner plots
        fig = corner.corner(
            samples,
            weights=weights,
            labels=param_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        plt.savefig(os.path.join(output_dir, 'corner_plot.pdf'), 
                    bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, 'corner_plot.png'), 
                    bbox_inches='tight', dpi=150)
        plt.close()
        
        print_rank0(f"Saved: {os.path.join(output_dir, 'corner_plot.pdf')}")
        print_rank0(f"Saved: {os.path.join(output_dir, 'corner_plot.png')}")


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Test at fiducial point first
    print_rank0("\nTesting likelihood at fiducial point...")
    try:
        loglike_test = model.logposterior(fiducial).loglike
        print_rank0(f"✓ Fiducial log-likelihood: {loglike_test:.2f}")
        if not np.isfinite(loglike_test):
            print_rank0("✗ ERROR: Fiducial point returns invalid likelihood!")
            sys.exit(1)
    except Exception as e:
        print_rank0(f"✗ ERROR at fiducial point: {e}")
        sys.exit(1)
    
    # Test a few random points
    print_rank0("\nTesting random points from prior...")
    from scipy.stats import truncnorm
    test_count = 0
    for i in range(20):
        u = np.random.uniform(0, 1, 9)
        theta_test = prior_transform(u)
        ll = log_likelihood(theta_test)
        if np.isfinite(ll) and ll > -1e40:
            test_count += 1
            print_rank0(f"  Test {i+1}: LogL = {ll:.2f} ✓")
        else:
            print_rank0(f"  Test {i+1}: LogL = {ll:.2f} ✗")
    
    if test_count == 0:
        print_rank0("\n✗ ERROR: No valid points found in prior! Check parameter ranges.")
        sys.exit(1)
    
    print_rank0(f"\n✓ {test_count}/10 test points returned valid likelihoods")
    print_rank0("Proceeding with nested sampling...\n")
    
    output_dir = './cpl_dynesty_output'
    
    # Run nested sampling
    results = run_nested_sampling(ndim=9, nlive=500, dlogz=0.005, output_dir=output_dir)
    
    # Only rank 0 does post-processing
    if rank == 0 and results is not None:
        save_chains_and_info(results, output_dir=output_dir)
        samples, weights = analyze_results(results, output_dir=output_dir)
        create_plots(results, samples, weights, output_dir=output_dir)
        
        print_rank0("\n" + "="*60)
        print_rank0("Analysis complete!")
        print_rank0("="*60)
        print_rank0(f"All results saved in: {output_dir}")