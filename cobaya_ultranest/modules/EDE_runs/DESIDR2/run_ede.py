import os
import ultranest
import numpy as np

# Import our modules
import ede_config as cfg
import ede_utils as utils

def main():
    print("========================================")
    print("   EDE COSMOLOGY ANALYSIS (Modular)     ")
    print("========================================")

    # 1. Initialize logic helpers
    #    The wrapper handles the Cobaya model internally
    ede_wrapper = utils.EdeModelWrapper(cfg.COBAYA_INFO, cfg.PARAM_NAMES)
    
    #    The transformer handles the prior mapping
    prior_helper = utils.PriorTransformer(cfg.PRIOR_RANGES)

    # 2. Sanity Check: Test Fiducial Point
    print("\n--- Testing Fiducial Point ---")
    # Approximate mean of the priors for a test point
    fiducial_point = np.array([np.mean(r) for r in cfg.PRIOR_RANGES])
    
    try:
        logl = ede_wrapper.log_likelihood(fiducial_point)
        print(f"✓ Fiducial Log-Likelihood: {logl:.4f}")
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return

    # 3. Setup UltraNest
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    sampler = ultranest.ReactiveNestedSampler(
        param_names=cfg.PARAM_NAMES,
        loglike=ede_wrapper.log_likelihood,
        transform=prior_helper.transform,
        log_dir=os.path.join(cfg.OUTPUT_DIR, cfg.RUN_NAME),
        resume='subfolder', # Safe resume mode
    )

    # 4. Run Sampler
    print(f"\n--- Starting Sampling ({len(cfg.PARAM_NAMES)} params) ---")
    result = sampler.run(
        min_num_live_points=400,
        cluster_num_live_points=40,
        dlogz=0.5,
        max_ncalls=1000000,
        show_status=True
    )

    # 5. Save and Plot
    print("\n--- Processing Results ---")
    pkl_path = os.path.join(cfg.OUTPUT_DIR, "ede_results.pkl")
    plot_path = os.path.join(cfg.OUTPUT_DIR, "ede_triangle.pdf")

    samples, weights = utils.save_results(
        result, sampler, 
        cfg.PARAM_NAMES, cfg.PARAM_LABELS, 
        filename=pkl_path
    )

    utils.create_triangle_plot(
        samples, weights, 
        cfg.PARAM_NAMES, cfg.PARAM_LABELS, 
        output_file=plot_path
    )

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()  