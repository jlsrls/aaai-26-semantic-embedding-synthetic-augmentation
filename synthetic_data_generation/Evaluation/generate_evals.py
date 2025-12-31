"""Synthetic Data Evaluation Pipeline

This script allows the user to get evaluation results after the data 
generation pipeline has been ran for all models and all waves. The 
purpose of this script is setting up synthetic data, real data, and 
output file paths, and calling the utilities from evaluate.py for each 
experimental run, i.e., each pair (model, wave). See evaluate.py for 
documentation on the evaluation methods used. 

The inputs to this pipeline are the real and synthetic data file paths, 
as well as the evaluation metadata file paths. Evaluation metadata
dictionaries are required for each wave, as specified in the Evaluator
class documentation in evaluate.py.

The outputs to this pipeline are explained in 
Evaluation/Evaluation results/README.md.

Running the pipeline requires:
- sdmetrics: routines for synthetic data evaluation.
- Plotly: manipulating output visualizations from sdmetrics.
- Pillow: manipulating images.
- tqdm: progress bar.
- Pandas: tabular data manipulation.

"""


import pandas as pd
from evaluate import Evaluator, record_eval_metrics
from pathlib import Path


def main():
    # Input directories
    eval_dir = Path().cwd() / "Evaluation"
    eval_meta_dir = eval_dir / "Evaluation metadata"
    real_data_dir = Path().cwd() / "Original data"
    syn_data_dir = Path().cwd() / "Final_gen_data"

    # Output directory
    output_eval_metrics_dir = eval_dir / "Evaluation results"

    models = ["gpt-5", "gpt-4o", "claude-sonnet-4"]
    waves = ["wave1", "wave2", "wave3", "wave4", "wave5", "wave6", "wave7",
             "wave8a", "wave8b", "wave9", "wave10", "wave11", "wave12", "wave13", "wave14"]

    # Dictionaries for compiling cross-wave model results
    c2st_data = {model: [] for model in models}
    density_shape_data = {model: [] for model in models}
    density_trend_data = {model: [] for model in models}
    density_overall_data = {model: [] for model in models}

    # Iterate over each pair (model, wave)
    for model in models:
        for wave_name in waves:
            # Get real data, synthetic data and evaluation metadata paths
            real_wave_path = real_data_dir / f"{wave_name}.csv"
            syn_wave_path = syn_data_dir / model / f"gen_data_{wave_name}.csv"
            eval_meta_wave_path = eval_meta_dir / f"{wave_name}_eval_meta.json"
            # Create output plot path
            output_plot_path = eval_dir / model / f"{model}_{wave_name}_results.png"

            # Initialize evaluator object
            eval_obj_wave = Evaluator(real_wave_path, syn_wave_path, eval_meta_wave_path)

            # Evaluate experimental run with all metrics
            eval_metrics = eval_obj_wave.evaluate()
            eval_results = record_eval_metrics(eval_metrics)

            # # Create image containing the distribution plot of each variable
            # Image functionality is not strictly necessary to run evaluation pipeline
            # img = eval_obj_wave.plot_density()
            # # Save image to file
            # img.save(output_plot_path)

            # Update data dictionaries
            c2st_data[model].append(eval_results["c2st"])
            density_shape_data[model].append(eval_results["density_shape"])
            density_trend_data[model].append(eval_results["density_trend"])
            density_overall_data[model].append(eval_results["density_overall"])

    # Output evaluation results to file
    c2st_df = pd.DataFrame(data=c2st_data)
    density_overall_df = pd.DataFrame(data=density_overall_data)
    density_shape_df = pd.DataFrame(data=density_shape_data)
    density_trend_df = pd.DataFrame(data=density_trend_data)

    c2st_df.to_csv(output_eval_metrics_dir / "c2st.csv")
    density_overall_df.to_csv(output_eval_metrics_dir / "density_overall.csv")
    density_shape_df.to_csv(output_eval_metrics_dir / "density_shape.csv")
    density_trend_df.to_csv(output_eval_metrics_dir / "density_trend.csv")


if __name__ == "__main__":
    main()