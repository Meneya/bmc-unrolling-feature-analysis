#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = "bmc_feature_analysis"
GLOBAL_PLOT_NAME = "global_drift_decay_curve.png"

def main():
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        print(f"Error: Output directory '{OUTPUT_DIR}' does not exist. Run the main pipeline first.")
        return

    print("Gathering incremental drift data across all benchmarks...")
    
    # Locate all incremental_drift.csv files recursively
    drift_files = list(output_path.glob("**/incremental_drift.csv"))
    
    if not drift_files:
        print("No 'incremental_drift.csv' files found. Ensure your pipeline completed successfully.")
        return
        
    print(f"Found data for {len(drift_files)} processed benchmarks.")

    # Aggregate data into a long-form dataframe to handle unequal sequence lengths gracefully
    all_series = []
    
    for fpath in drift_files:
        # Determine a clean benchmark ID based on its relative path
        rel_parts = fpath.relative_to(output_path).parts[:-1]
        bench_id = "/".join(rel_parts)
        
        try:
            df = pd.read_csv(fpath)
            if "frame" in df.columns and "incremental_drift" in df.columns:
                df["benchmark"] = bench_id
                all_series.append(df[["frame", "incremental_drift", "benchmark"]])
        except Exception as e:
            print(f"Skipping corrupt file {fpath}: {e}")

    if not all_series:
        print("No valid data could be parsed.")
        return

    # Combine everything into one giant master DataFrame
    master_df = pd.concat(all_series, ignore_index=True)

    # Compute statistical aggregates for every distinct timeframe index
    stats = master_df.groupby("frame")["incremental_drift"].agg(["mean", "std", "median", "count"]).reset_index()

    # Filter out frames where fewer than 3 benchmarks are active to prevent skewed tail-end statistics
    stats = stats[stats["count"] >= 3]

    print("Generating publication-quality visualization...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(11, 7))

    # 1. Plot individual trajectory lines faintly in the background
    for bench_id, group in master_df.groupby("benchmark"):
        # Truncate plotting to match the frame range cleared by our statistical filter
        group_filtered = group[group["frame"].isin(stats["frame"])]
        plt.plot(
            group_filtered["frame"], 
            group_filtered["incremental_drift"], 
            color="tab:blue", 
            alpha=0.15, 
            linewidth=1.2,
            label="_nolegend_" # Prevents the legend from exploding with 50+ identical items
        )

    # 2. Plot the variance band (Mean +/- 1 Standard Deviation)
    under_line = stats["mean"] - stats["std"]
    over_line = stats["mean"] + stats["std"]
    
    # Clip lower variance bound at 0 since Euclidean distance/norm cannot be negative
    under_line = np.clip(under_line, a_min=0, a_max=None)

    plt.fill_between(
        stats["frame"], 
        under_line, 
        over_line, 
        color="tab:red", 
        alpha=0.2, 
        label="±1 Std. Dev. Band"
    )

    # 3. Plot the bold Global Mean Trend Line
    plt.plot(
        stats["frame"], 
        stats["mean"], 
        color="darkred", 
        linewidth=3.0, 
        linestyle="-", 
        label="Global Mean Drift"
    )

    # 4. Plot the Global Median Line for skewness comparison
    plt.plot(
        stats["frame"], 
        stats["median"], 
        color="black", 
        linewidth=2.0, 
        linestyle="--", 
        label="Global Median Drift"
    )

    # Label adjustments and formatting
    plt.title("Global BMC Feature Drift Decay Across All Benchmarks", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("BMC Frame Index ($k$)", fontsize=12)
    plt.ylabel("Incremental Feature Drift ($\|x(k+1) - x(k)\|$)", fontsize=12)
    
    plt.xlim(stats["frame"].min(), stats["frame"].max())
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=11)
    
    plt.tight_layout()
    
    # Save directly in the execution scope
    plot_output_path = Path(".") / GLOBAL_PLOT_NAME
    plt.savefig(plot_output_path, dpi=300)
    plt.close()

    print(f"Success! Global drift-decay visualization saved to: {plot_output_path.resolve()}")

if __name__ == "__main__":
    main()