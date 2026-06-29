import os
import re
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

##############################################################################
# FEATURE TAXONOMY — Rigorous topological filtering
##############################################################################
#
# To isolate *purely structural* (topological) drift from confounding
# non-structural signals, we exclude three categories of SATzilla features
# before computing cosine similarity, L2 drift, and PCA trajectories:
#
#   1. TEMPORAL — Wall-clock time spent by each feature-extraction phase.
#      These measure computational cost, not formula topology.
#
#   2. PROBLEM-SIZE — Raw counts of variables, clauses, and reduction
#      statistics.  In BMC these grow *monotonically* with unrolling depth;
#      leaving them in would make drift a trivial artifact of size growth
#      rather than a genuine topological change.
#
#   3. SOLVER-PERFORMANCE — Outcomes from LP relaxation, local-search
#      probes (SAPS / GSAT), and the binary `solved` flag.  These
#      capture solver behavior on the instance, not the instance structure.
#      (Normalized probe statistics — coefficient-of-variation and ratio
#       features — are retained because they are size-independent.)
#
# All remaining features (~347) encode structural properties of the
# clause-variable graph, its degree distributions, clustering, modularity,
# algebraic connectivity, diameter, clause-length distribution, etc.
##############################################################################

# --- Category 1: TEMPORAL (suffix-based detection) ---
# Every SATzilla timing column ends with the suffix "time" (case-insensitive),
# e.g. "Pre-featuretime", "lpTIME", "CG-featuretime".  No non-temporal
# column has this suffix, so a simple endswith() check is both sufficient
# and future-proof against new SATzilla feature phases.
TEMPORAL_SUFFIX = "time"

# --- Category 2: PROBLEM-SIZE (raw counts, monotonic in BMC depth) ---
PROBLEM_SIZE_EXACT = [
    "nvarsOrig",
    "nclausesOrig",
    "nvars",
    "nclauses",
    "reducedVars",
    "reducedClauses",
    "UNARY",
    "BINARY+",
    "TRINARY+",
]

# --- Category 3: SOLVER-PERFORMANCE ---
LP_EXACT = [
    "LP_OBJ",
    "LPSLack-mean", "LPSLack-coeff-variation",
    "LPSLack-min", "LPSLack-max",
    "lpIntRatio",
    "solved",
]

# SAPS / GSAT raw (size-dependent) probe metrics
LS_RAW_EXACT = [
    # SAPS
    "saps_BestSolution_Mean",
    "saps_BestSolution_CoeffVariance",
    "saps_FirstLocalMinStep_Mean",
    "saps_FirstLocalMinStep_CoeffVariance",
    "saps_FirstLocalMinStep_Median",
    "saps_FirstLocalMinStep_Q.10",
    "saps_FirstLocalMinStep_Q.90",
    "saps_BestAvgImprovement_Mean",
    # GSAT
    "gsat_BestSolution_Mean",
    "gsat_BestSolution_CoeffVariance",
    "gsat_FirstLocalMinStep_Mean",
    "gsat_FirstLocalMinStep_CoeffVariance",
    "gsat_FirstLocalMinStep_Median",
    "gsat_FirstLocalMinStep_Q.10",
    "gsat_FirstLocalMinStep_Q.90",
    "gsat_BestAvgImprovement_Mean",
]

# Variable-reduction counts (size-dependent)
VAR_REDUCTION_PATTERNS = ["vars-reduced-depth"]


def select_topological_features(df):
    """
    Given a raw SATzilla feature DataFrame (one row per frame),
    return a new DataFrame containing *only* topological / structural
    features, with temporal, problem-size, and solver-performance
    columns explicitly excluded.

    Prints a diagnostic summary of what was kept vs. dropped.
    """
    cols = df.columns.tolist()

    # --- Build the drop set ---
    drop = set()

    # 1. Temporal: suffix-based — any column whose name ends with "time"
    temporal_dropped = [c for c in cols if c.lower().endswith(TEMPORAL_SUFFIX)]
    drop.update(temporal_dropped)

    # 2. Problem-size exact matches
    drop.update(PROBLEM_SIZE_EXACT)

    # 3. LP / solver-performance exact matches
    drop.update(LP_EXACT)
    drop.update(LS_RAW_EXACT)

    # 4. Variable-reduction pattern matches
    for c in cols:
        if any(pat in c for pat in VAR_REDUCTION_PATTERNS):
            drop.add(c)

    # Only drop columns that actually exist
    drop = {c for c in drop if c in cols}

    keep = [c for c in cols if c not in drop]

    print(f"  Feature filtering: {len(cols)} total -> {len(keep)} topological retained, "
          f"{len(drop)} dropped")
    if len(drop) > 0:
        print(f"  Dropped columns ({len(drop)}):")
        # Group by category for readability
        temporal_dropped = sorted(temporal_dropped)
        size_dropped = [c for c in drop if c in PROBLEM_SIZE_EXACT or any(p in c for p in VAR_REDUCTION_PATTERNS)]
        solver_dropped = [c for c in drop if c not in temporal_dropped and c not in size_dropped]
        if temporal_dropped:
            print(f"    Temporal ({len(temporal_dropped)}): {', '.join(sorted(temporal_dropped))}")
        if size_dropped:
            print(f"    Problem-size ({len(size_dropped)}): {', '.join(sorted(size_dropped))}")
        if solver_dropped:
            print(f"    Solver-perf ({len(solver_dropped)}): {', '.join(sorted(solver_dropped))}")

    return df[keep].copy()


##############################################################################
# CONFIGURATION
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="BMC Feature Evolution Analysis"
    )

    parser.add_argument(
        "--abc",
        required=True,
        help="Path to ABC executable"
    )

    parser.add_argument(
        "--satzilla",
        required=True,
        help="Path to SATzilla feature extractor"
    )

    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Directory containing .aig benchmarks"
    )

    parser.add_argument(
        "--output",
        default="bmc_feature_analysis",
        help="Output directory"
    )

    parser.add_argument(
        "--max-frame",
        type=int,
        default=50,
        help="Maximum BMC frame"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="SATzilla timeout (seconds)"
    )

    return parser.parse_args()

##############################################################################
# UTILITIES
##############################################################################

def run_cmd(cmd):
    print(cmd)
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return result.returncode, result.stdout

##############################################################################
# PARSE CNF STATS
##############################################################################

def parse_cnf_stats(output):
    vars_ = np.nan
    clauses_ = np.nan
    lits_ = np.nan

    m = re.search(
        r'Vars\s*=\s*(\d+).*Clauses\s*=\s*(\d+).*Literals\s*=\s*(\d+)',
        output,
        re.S
    )
    if m:
        vars_ = int(m.group(1))
        clauses_ = int(m.group(2))
        lits_ = int(m.group(3))

    return vars_, clauses_, lits_

def parse_cnf_header_from_file(cnf_file_path):
    """Fallback parser to harvest stats from cached DIMACS CNF files."""
    try:
        with open(cnf_file_path, 'r') as f:
            for line in f:
                if line.startswith('p cnf'):
                    parts = line.split()
                    if len(parts) >= 4:
                        return int(parts[2]), int(parts[3]), np.nan
    except Exception:
        pass
    return np.nan, np.nan, np.nan

##############################################################################
# GENERATE CNF
##############################################################################

def generate_frame_cnf(aig, frame, cnf_file, args):
    abc_cmd = (
        f'read {aig}; '
        f'fold; '
        f'&get; '
        f'&frames -sa -F {frame}; '
        f'&write_cnf {cnf_file}'
    )
    cmd = f'{args.abc} -c "{abc_cmd}"'
    rc, output = run_cmd(cmd)

    if rc != 0:
        raise RuntimeError(output)

    return parse_cnf_stats(output)

##############################################################################
# SATZILLA
##############################################################################

def run_satzilla(cnf_file, csv_file, args):
    cmd = (
        f'{args.satzilla} '
        f'-all '
        f'--timeout {args.timeout} '
        f'{cnf_file} '
        f'{csv_file}'
    )
    rc, output = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(output)

##############################################################################
# LOAD SATZILLA CSV
##############################################################################

def load_feature_vector(csv_file):
    df = pd.read_csv(csv_file)
    if len(df) == 0:
        raise RuntimeError("Empty SATzilla CSV")

    row = df.iloc[0]
    features = {}
    for col in df.columns:
        try:
            features[col] = float(row[col])
        except:
            pass
    return pd.Series(features)

##############################################################################
# PLOTS
##############################################################################

def plot_consecutive_similarity(frames, sims, outfile, benchmark):
    plt.figure(figsize=(8,5))
    plt.plot(frames[:-1], sims, marker='o', color='tab:blue')
    plt.xlabel("Frame")
    plt.ylabel("Cosine Similarity")
    plt.title(f"{benchmark}\nConsecutive Frame Similarity")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_similarity_to_first(frames, sims, outfile, benchmark):
    plt.figure(figsize=(8,5))
    plt.plot(frames, sims, marker='o', color='tab:orange')
    plt.xlabel("Frame")
    plt.ylabel("Similarity to Frame 1")
    plt.title(f"{benchmark}\nSimilarity to Frame 1")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_drift(frames, drift, outfile, benchmark):
    plt.figure(figsize=(8,5))
    plt.plot(frames[:-1], drift, marker='o', color='tab:red')
    plt.xlabel("Frame")
    plt.ylabel("L2 Drift")
    plt.title(f"{benchmark}\nFeature Drift (L2)")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_heatmap(S, outfile, benchmark):
    plt.figure(figsize=(8,6))
    plt.imshow(S, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    plt.title(f"{benchmark}\nSimilarity Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_pca(X, frames, outfile, benchmark):
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.plot(Y[:,0], Y[:,1], marker='o', linestyle='-', alpha=0.5, color='purple')
    for i, frame in enumerate(frames):
        plt.annotate(str(frame), (Y[i,0], Y[i,1]), fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{benchmark}\nPCA Trajectory Across Frames")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

##############################################################################
# PROCESS ONE BENCHMARK
##############################################################################

def process_benchmark(aig_file, args):
    # Retain the folder hierarchy relative to args.benchmarks
    rel_path = Path(aig_file).relative_to(args.benchmarks)
    benchmark_id = str(rel_path.with_suffix('')) # e.g., 'isvlsi_sat_bms/mult2'
    benchmark_name = rel_path.stem

    print("\n" + "="*80)
    print(f"Processing: {benchmark_id}")
    print("="*80)

    # Set up mirrored directory structure
    bench_dir = Path(args.output) / benchmark_id
    cnf_dir = bench_dir / "cnfs"
    feat_dir = bench_dir / "satzilla"

    cnf_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    frame_stats = []
    feature_vectors = []
    valid_frames = []

    for frame in range(1, args.max_frame + 1):
        cnf_file = cnf_dir / f"frame_{frame}.cnf"
        csv_file = feat_dir / f"frame_{frame}.csv"

        # Check CNF cache and fix structural stats missing bug
        if cnf_file.exists():
            vars_, clauses_, lits_ = parse_cnf_header_from_file(cnf_file)
        else:
            try:
                vars_, clauses_, lits_ = generate_frame_cnf(aig_file, frame, cnf_file, args)
            except Exception as e:
                print(f"[{benchmark_name}] Stopping CNF unrolling at frame {frame} due to ABC limit/error.")
                break

        # Check SATzilla Cache
        if not csv_file.exists():
            try:
                run_satzilla(cnf_file, csv_file, args)
            except Exception as e:
                print(f"[{benchmark_name}] SATzilla failed or timed out at frame {frame}")
                break

        # Extract features
        try:
            vec = load_feature_vector(csv_file)
        except Exception:
            print(
                f"Corrupt feature file: {csv_file}"
            )

            csv_file.unlink(missing_ok=True)

            try:
                run_satzilla(cnf_file, csv_file, args)

                vec = load_feature_vector(csv_file)

            except Exception:
                break

        frame_stats.append({
            "frame": frame,
            "vars": vars_,
            "clauses": clauses_,
            "literals": lits_
        })
        feature_vectors.append(vec)
        valid_frames.append(frame)

    if len(feature_vectors) < 2:
        print(f"[{benchmark_name}] Aborted: Too few frames processed (< 2).")
        return None

    # Processing and data normalization
    X = pd.DataFrame(feature_vectors)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    raw_features = X.copy()

    # ------------------------------------------------------------------
    # TOPOLOGICAL FEATURE SELECTION
    # ------------------------------------------------------------------
    X_topo = select_topological_features(X)

    # Standardize features (StandardScaler safely turns constant columns into 0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_topo)

    # Core Similarity Computations
    S = cosine_similarity(X_scaled)
    
    # Vectorized extraction from global matrix S
    consecutive = list(np.diag(S, k=1))
    first_sim = list(S[0, :])

    # --- Vectorized Incremental Drift ---
    delta = X_scaled[1:] - X_scaled[:-1]
    drift = np.linalg.norm(delta, axis=1).tolist()

    # Save metrics locally in the sub-benchmark folder
    pd.DataFrame(frame_stats).to_csv(bench_dir / "abc_stats.csv", index=False)
    raw_features.to_csv(bench_dir / "all_features.csv", index=False)
    np.save(bench_dir / "satzilla_embeddings.npy", X_scaled)
    np.save(bench_dir / "similarity_matrix.npy", S)

    # Also save the list of retained topological feature names for reproducibility
    pd.Series(X_topo.columns.tolist()).to_csv(
        bench_dir / "topological_features.csv", index=False, header=["feature_name"]
    )

    pd.DataFrame({
        "frame": valid_frames[:-1],
        "similarity_next": consecutive,
        "incremental_drift": drift
    }).to_csv(bench_dir / "incremental_drift.csv", index=False)

    # Plot generations
    plot_consecutive_similarity(valid_frames, consecutive, bench_dir / "consecutive_similarity.png", benchmark_name)
    plot_similarity_to_first(valid_frames, first_sim, bench_dir / "similarity_to_frame1.png", benchmark_name)
    plot_drift(valid_frames, drift, bench_dir / "drift.png", benchmark_name)
    plot_heatmap(S, bench_dir / "similarity_heatmap.png", benchmark_name)
    plot_pca(X_scaled, valid_frames, bench_dir / "pca_trajectory.png", benchmark_name)

    # Generate isolated benchmark summary metadata
    summary = {
        "benchmark": benchmark_id,
        "num_frames": len(valid_frames),
        "num_topological_features": X_topo.shape[1],
        "mean_similarity": float(np.mean(consecutive)),
        "min_similarity": float(np.min(consecutive)),
        "max_similarity": float(np.max(consecutive)),
        "std_similarity": float(np.std(consecutive)),
        "mean_drift": float(np.mean(drift)),
        "max_drift": float(np.max(drift))
    }

    pd.DataFrame([summary]).to_csv(bench_dir / "benchmark_summary.csv", index=False)
    return summary

##############################################################################
# MAIN EXECUTIVE ENTRYPOINT
##############################################################################

def main():

    args = parse_args()

    print("=" * 80)
    print("BMC Feature Evolution Analysis")
    print("=" * 80)

    print(f"ABC        : {args.abc}")
    print(f"SATzilla   : {args.satzilla}")
    print(f"Benchmarks : {args.benchmarks}")
    print(f"Output     : {args.output}")
    print(f"Max Frame  : {args.max_frame}")
    print(f"Timeout    : {args.timeout}s")
    print()
    print("Resume mode: ENABLED")
    print()

    Path(args.output).mkdir(exist_ok=True)
    aigs = []

    # Recursively gather targets
    for root, dirs, files in os.walk(args.benchmarks):
        for f in files:
            if f.endswith(".aig") or f.endswith(".aiger"):
                aigs.append(os.path.join(root, f))

    aigs = sorted(aigs)
    print(f"Found {len(aigs)} benchmarks across subdirectories.")

    global_summary = []
    for aig in aigs:
        try:
            s = process_benchmark(aig, args)
            if s is not None:
                global_summary.append(s)
        except Exception as e:
            print(f"\nCRITICAL FAILURE PROCESSING: {aig}")
            print(f"Error Details: {e}\n")

    # Output execution summary exactly into the executed execution scope
    if len(global_summary) > 0:
        df = pd.DataFrame(global_summary)
        summary_path = Path("bmc_feature_analysis") / "global_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nPipeline Execution Complete! Summary generated at: {summary_path.resolve()}")

if __name__ == "__main__":
    main()
