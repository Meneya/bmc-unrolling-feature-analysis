#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats

##############################################################################
# PLOTTING AESTHETICS (IEEE Style)
##############################################################################

def setup_plotting():
    """Configures matplotlib for IEEE-style publication quality."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Fonts and Text
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # Lines and Grid
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['savefig.bbox'] = 'tight'

##############################################################################
# DATA LOADING & STATISTICAL PROCESSING
##############################################################################

def load_and_process_data(input_dir, stable_threshold_pct=0.05, consecutive_frames=5):
    """
    Loads benchmark data recursively, normalizes drift using robust percentiles, 
    calculates distance to terminal feature vector, and determines stabilization frames.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory {input_dir} not found.")

    all_frame_data = []
    summary_data = []

    # Recursively search for all incremental_drift.csv files down the tree
    drift_files = list(input_path.rglob("incremental_drift.csv"))

    if not drift_files:
        print(f"Error: No 'incremental_drift.csv' files found under {input_path.resolve()}")
        return pd.DataFrame(), pd.DataFrame()

    for drift_file in drift_files:
        bench_dir = drift_file.parent
        embed_file = bench_dir / "satzilla_embeddings.npy"

        if not embed_file.exists():
            continue

        # Reconstruct benchmark_id matching the nested hierarchy
        benchmark_id = str(bench_dir.relative_to(input_path))
        benchmark_name = bench_dir.name
        
        # Load data
        df = pd.read_csv(drift_file)
        embeddings = np.load(embed_file)
        
        if df.empty or len(embeddings) < 2:
            continue
            
        # 1. Robust Drift Normalization (Using 95th Percentile)
        p95_drift = np.percentile(df['incremental_drift'], 95)
        if p95_drift > 0:
            df['norm_drift'] = df['incremental_drift'] / p95_drift
        else:
            df['norm_drift'] = 0.0

        # 2. Distance to Final Representation
        final_embedding = embeddings[-1]
        dist_to_final = np.linalg.norm(embeddings - final_embedding, axis=1)
        
        p95_dist = np.percentile(dist_to_final, 95)
        if p95_dist > 0:
            norm_dist_to_final = dist_to_final / p95_dist
        else:
            norm_dist_to_final = np.zeros_like(dist_to_final)
            
        # Alignment mapping
        df['norm_dist_to_final'] = norm_dist_to_final[:len(df)]

        # 3. Compute Stabilization Frame using dynamic threshold
        stable_streak = 0
        stabilization_frame = np.nan
        total_frames = len(embeddings)
        
        for idx, row in df.iterrows():
            if row['norm_dist_to_final'] < stable_threshold_pct: 
                stable_streak += 1
                if stable_streak >= consecutive_frames:
                    stabilization_frame = row['frame'] - consecutive_frames + 1
                    break
            else:
                stable_streak = 0
                
        df['benchmark'] = benchmark_id
        all_frame_data.append(df)
        
        summary_data.append({
            'benchmark': benchmark_id,
            'total_frames': total_frames,
            'stabilization_frame': stabilization_frame,
            'stabilized': not np.isnan(stabilization_frame),
            'stabilization_ratio': stabilization_frame / total_frames if not np.isnan(stabilization_frame) else np.nan
        })

    if not all_frame_data:
        return pd.DataFrame(), pd.DataFrame()

    master_df = pd.concat(all_frame_data, ignore_index=True)
    summary_df = pd.DataFrame(summary_data)
    
    return master_df, summary_df

##############################################################################
# VISUALIZATION GENERATORS
##############################################################################

def plot_global_drift(master_df, output_dir, fmt, dpi):
    plt.figure(figsize=(7, 5))
    
    for bench_id, group in master_df.groupby("benchmark"):
        plt.plot(group["frame"], group["incremental_drift"], 
                 color="tab:gray", alpha=0.35, linewidth=1.0)
                 
    agg = master_df.groupby("frame")["incremental_drift"].agg(['mean', 'sem', 'count'])
    agg = agg[agg['count'] >= 3] 
    
    ci95 = agg['sem'] * stats.t.ppf(1 - 0.05/2, agg['count'] - 1)
    
    plt.fill_between(agg.index, np.clip(agg['mean'] - ci95, 0, None), agg['mean'] + ci95, 
                     color="tab:blue", alpha=0.4, label="95% CI")
    plt.plot(agg.index, agg['mean'], color="darkblue", linewidth=2.5, label="Global Mean Drift")
    
    plt.title("Global Feature Drift Decay")
    plt.xlabel("BMC Frame Index ($k$)")
    plt.ylabel("Incremental Feature Drift ($L_2$)")
    plt.xlim(master_df['frame'].min(), agg.index.max())
    plt.legend(loc="upper right")
    plt.savefig(output_dir / f"fig1_global_drift.{fmt}", dpi=dpi)
    plt.close()

def plot_distance_to_final(master_df, threshold, output_dir, fmt, dpi):
    plt.figure(figsize=(7, 5))
    
    for bench_id, group in master_df.groupby("benchmark"):
        plt.plot(group["frame"], group["norm_dist_to_final"], 
                 color="tab:gray", alpha=0.35, linewidth=1.0)
                 
    agg = master_df.groupby("frame")["norm_dist_to_final"].agg(['mean', 'sem', 'count'])
    agg = agg[agg['count'] >= 3] 
    
    ci95 = agg['sem'] * stats.t.ppf(1 - 0.05/2, agg['count'] - 1)
    
    plt.fill_between(agg.index, np.clip(agg['mean'] - ci95, 0, None), agg['mean'] + ci95, 
                     color="tab:orange", alpha=0.4, label="95% CI")
    plt.plot(agg.index, agg['mean'], color="darkorange", linewidth=2.5, label="Mean Normalized Distance")
    
    threshold_label = f"{int(threshold * 100)}% Distance Threshold"
    plt.axhline(threshold, color='black', linestyle=':', alpha=0.8, label=threshold_label)
    plt.title("Distance to Terminal Feature Representation")
    plt.xlabel("BMC Frame Index ($k$)")
    plt.ylabel("Normalized $L_2$ Distance to Final Frame")
    plt.xlim(master_df['frame'].min(), agg.index.max())
    plt.legend(loc="upper right")
    plt.savefig(output_dir / f"fig2_distance_to_final.{fmt}", dpi=dpi)
    plt.close()

def plot_similarity_convergence(master_df, output_dir, fmt, dpi):
    if "similarity_next" not in master_df.columns:
        print("Warning: 'similarity_next' not found. Skipping Similarity Convergence plot.")
        return

    plt.figure(figsize=(7, 5))
    agg = master_df.groupby("frame")["similarity_next"].agg(['mean', 'sem', 'count'])
    agg = agg[agg['count'] >= 3]
    
    ci95 = agg['sem'] * stats.t.ppf(1 - 0.05/2, agg['count'] - 1)
    
    plt.fill_between(agg.index, np.clip(agg['mean'] - ci95, -1, 1), np.clip(agg['mean'] + ci95, -1, 1), 
                     color="tab:green", alpha=0.4, label="95% CI")
    plt.plot(agg.index, agg['mean'], color="darkgreen", linewidth=2.5, label="Mean Cosine Similarity")
    
    plt.axhline(1.0, color='black', linestyle=':', alpha=0.8)
    plt.title("Cosine Similarity Convergence ($S_{k, k+1}$)")
    plt.xlabel("BMC Frame Index ($k$)")
    plt.ylabel("Cosine Similarity")
    plt.xlim(master_df['frame'].min(), agg.index.max())
    plt.ylim(master_df['similarity_next'].min() - 0.1, 1.05)
    plt.legend(loc="lower right")
    plt.savefig(output_dir / f"fig3_similarity_convergence.{fmt}", dpi=dpi)
    plt.close()

def plot_stabilization_histogram(summary_df, threshold, output_dir, fmt, dpi):
    plt.figure(figsize=(7, 5))
    stable_frames = summary_df['stabilization_frame'].dropna()
    
    if len(stable_frames) == 0:
        print("Warning: No benchmarks stabilized under current thresholds.")
        return
        
    plt.hist(stable_frames, bins=range(int(stable_frames.min()), int(stable_frames.max()) + 5, 2), 
             color='tab:purple', alpha=0.7, edgecolor='black', linewidth=1.2)
                 
    plt.axvline(stable_frames.median(), color='black', linestyle='dashed', 
                linewidth=1.5, label=f'Median Frame: {stable_frames.median():.0f}')
    
    plt.title("Distribution of Stabilization Frames")
    plt.xlabel(f"Frame $k^*$ (Onset of 5 consecutive frames with $< {int(threshold * 100)}\%$ distance to final)")
    plt.ylabel("Number of Benchmarks")
    plt.legend()
    plt.savefig(output_dir / f"fig4_stabilization_distribution.{fmt}", dpi=dpi)
    plt.close()

def plot_stabilization_scatter(summary_df, output_dir, fmt, dpi):
    plt.figure(figsize=(7, 5))
    df_clean = summary_df.dropna(subset=['stabilization_frame'])
    
    if df_clean.empty:
        return

    plt.scatter(df_clean['total_frames'], df_clean['stabilization_frame'], 
                color='tab:red', alpha=0.7, edgecolor='black', s=50)
    
    max_val = max(df_clean['total_frames'].max(), df_clean['stabilization_frame'].max())
    plt.plot([0, max_val], [0, max_val], color='black', linestyle=':', alpha=0.5, label='y=x (Never Stabilizes)')
    
    plt.title("Stabilization Frame vs. Total Explored Depth")
    plt.xlabel("Total Frames Explored")
    plt.ylabel("Stabilization Frame ($k^*$)")
    plt.legend()
    plt.savefig(output_dir / f"fig5_stabilization_scatter.{fmt}", dpi=dpi)
    plt.close()

##############################################################################
# MAIN EXECUTIVE ENTRYPOINT
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Aggregate global drift and stability metrics for IWSBP.")
    parser.add_argument("--input", type=str, default="bmc_feature_analysis", help="Directory containing benchmark subfolders.")
    parser.add_argument("--output", type=str, default="bmc_global_plots", help="Directory to save outputs.")
    parser.add_argument("--threshold", type=float, default=0.10, help="Percentile threshold (default: 0.10 for 10%).")
    parser.add_argument("--consecutive", type=int, default=5, help="Consecutive frames for stabilization.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures.")
    parser.add_argument("--format", type=str, default="pdf", choices=['png', 'pdf', 'eps'], help="Format for saved figures.")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    master_df, summary_df = load_and_process_data(input_dir, args.threshold, args.consecutive)
    
    if master_df.empty:
        print("\n[!] Processing Error: No valid data aggregated. Verify your execution paths.")
        return
        
    summary_df.to_csv(output_dir / "stabilization_summary.csv", index=False)
    
    global_stats = master_df.groupby('frame').agg({
        'incremental_drift': ['mean', 'std'],
        'norm_drift': ['mean', 'std'],
        'norm_dist_to_final': ['mean', 'std']
    })
    global_stats.columns = ['_'.join(col) for col in global_stats.columns]
    global_stats.to_csv(output_dir / "global_statistics.csv")
    
    st_df = summary_df.dropna(subset=['stabilization_frame'])
    total_benchmarks = len(summary_df)
    stabilized_count = len(st_df)
    
    print("\n" + "="*50)
    print(" HEADLINE STATISTICS FOR IWSBP PAPER")
    print("="*50)
    if stabilized_count > 0:
        pct_threshold = int(args.threshold * 100)
        print(f"Total Benchmarks Analyzed: {total_benchmarks}")
        print(f"Benchmarks Reaching Stability: {stabilized_count} ({stabilized_count/total_benchmarks*100:.1f}%)")
        print(f"Median Stabilization Frame: {st_df['stabilization_frame'].median():.1f}")
        print(f"Mean Stabilization Frame: {st_df['stabilization_frame'].mean():.1f}")
        print(f"Average Depth Ratio at Stabilization: {st_df['stabilization_ratio'].mean()*100:.1f}%")
        print("-" * 50)
        print(f"CONCLUSION: {stabilized_count/total_benchmarks*100:.0f}% of benchmarks reach within {pct_threshold}% of their ")
        print(f"terminal representation before exploring {st_df['stabilization_ratio'].mean()*100:.0f}% of their total depth.")
    else:
        print(f"CONCLUSION: 0% of benchmarks stabilized under the {int(args.threshold * 100)}% threshold limit.")
    print("="*50 + "\n")
    
    setup_plotting()
    plot_global_drift(master_df, output_dir, args.format, args.dpi)
    plot_distance_to_final(master_df, args.threshold, output_dir, args.format, args.dpi)
    plot_similarity_convergence(master_df, output_dir, args.format, args.dpi)
    plot_stabilization_histogram(summary_df, args.threshold, output_dir, args.format, args.dpi)
    plot_stabilization_scatter(summary_df, output_dir, args.format, args.dpi)
    
    print(f"Success! All outputs saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()