import numpy as np
import pickle
import matplotlib.pyplot as plt
from cobaya.model import get_model

try:
    from getdist import plots, MCSamples
    USE_GETDIST = True
except ImportError:
    USE_GETDIST = False

class cplModelWrapper:
    """
    Wraps the Cobaya model to provide clean interface for UltraNest.
    """
    def __init__(self, info, param_names):
        print("Building Cobaya model...")
        self.model = get_model(info)
        self.param_names = param_names
        print("Model built successfully!")

    def log_likelihood(self, theta):
        """
        Vectorized likelihood function for UltraNest.
        """
        theta = np.asarray(theta)
        
        def eval_single(x):
            params_dict = {n: float(v) for n, v in zip(self.param_names, x)}
            try:
                # Cobaya returns logposterior = loglike + logprior
                # We specifically want loglike for UltraNest
                result = self.model.logposterior(params_dict)
                return float(result.loglike)
            except Exception:
                return -np.inf

        if theta.ndim == 1:
            return eval_single(theta)
        elif theta.ndim == 2:
            return np.array([eval_single(row) for row in theta])
        else:
            raise ValueError("Theta must be 1D or 2D")

class PriorTransformer:
    """
    Handles the transformation from unit cube [0,1] to parameter space.
    """
    def __init__(self, ranges):
        self.ranges = ranges  # List of (min, max) tuples

    def transform(self, cube):
        u = np.asarray(cube)
        
        # Pre-compute arrays for vectorization
        los = np.array([r[0] for r in self.ranges])
        his = np.array([r[1] for r in self.ranges])
        widths = his - los

        def map_row(v):
            return v * widths + los

        if u.ndim == 1:
            return map_row(u)
        elif u.ndim == 2:
            return np.array([map_row(v) for v in u])
        else:
            raise ValueError("Prior transform expects 1D or 2D input")

def save_results(result, sampler, param_names, param_labels, filename='cpl_results.pkl'):
    """
    Extracts samples and saves detailed results to a pickle file.
    """
    print(f"\nProcessing results for: {filename}")
    
    # 1. Extract samples
    if 'weighted_samples' in result:
        samples = result['weighted_samples']['points']
        weights = result['weighted_samples']['weights']
    elif 'samples' in result:
        samples = result['samples']
        logwt = result.get('logwt', np.zeros(len(samples)))
        weights = np.exp(logwt - np.max(logwt))
        weights /= np.sum(weights)
    else:
        print("Warning: Could not extract specific samples. Saving raw result only.")
        samples, weights = None, None

    # 2. Compile data dict
    data = {
        'result': result,
        'param_names': param_names,
        'param_labels': param_labels,
        'ncall': sampler.ncall,
        'logz': result.get('logz'),
        'logzerr': result.get('logzerr'),
        'samples': samples,
        'weights': weights
    }

    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved pickle: {filename}")
    return samples, weights

def create_triangle_plot(samples, weights, names, labels, output_file):
    """
    Generates the corner plot if GetDist is installed.
    """
    if not USE_GETDIST or samples is None:
        print("Skipping plot (GetDist missing or no samples).")
        return

    print(f"Creating triangle plot: {output_file}")
    try:
        gd_samples = MCSamples(samples=samples, weights=weights, names=names, labels=labels, label='CPL Model')
        
        g = plots.get_subplot_plotter(width_inch=10)
        g.settings.num_plot_contours = 2
        g.triangle_plot(gd_samples, filled=True)
        
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print("✓ Plot saved.")
        
        # Simple stats print
        print("\n--- Parameter Constraints ---")
        for i, name in enumerate(names):
            mean = np.average(samples[:, i], weights=weights)
            std = np.sqrt(np.average((samples[:, i] - mean)**2, weights=weights))
            print(f"{name:10s}: {mean:8.4f} ± {std:8.4f}")
            
    except Exception as e:
        print(f"Error creating plot: {e}")