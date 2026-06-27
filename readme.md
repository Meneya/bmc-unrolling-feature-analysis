# BMC Feature Evolution Analysis

This repository provides a rigorous framework to study how the structural and topological characteristics of SAT formulas generated during Bounded Model Checking (BMC) evolve as the unrolling depth increases. 

Unlike traditional performance-based analyses, this pipeline explicitly isolates **purely structural dynamics** from confounding signals such as raw formula growth and solver-specific heuristics.

## The Pipeline

1. **CNF Generation**: Unrolls sequential AIG circuits into combinational CNF formulas across sequential frames using ABC.
2. **Feature Extraction**: Captures multi-dimensional structural and behavioral features for every single frame using SATzilla.
3. **Rigorous Topological Filtering**: Eliminates problem-size, timing, and solver performance variables to isolate the core graph topology.
4. **Data Normalization**: Standardizes feature distributions and handles variable inf/NaN signatures securely.
5. **Geometric Evaluation**:
   * **Cosine Similarity Matrices**: Tracks global structural recurrence and inter-frame distance.
   * **$L_2$ Incremental Drift**: Computes continuous topological displacement vectors.
   * **PCA Trajectories**: Projects high-dimensional feature dynamics into 2D spaces for visual tracking.
6. **Visualization**: Automatically plots multi-scale trends and compiles global summary statistics.

---

## Motivation

As the BMC unrolling depth increases, the underlying CNF formula grows monotonically in size. If basic dimensions (like variable or clause counts) are factored into feature drift calculations, the metrics simply reflect that the formula is getting bigger—a trivial artifact of unrolling.

By filtering down to a pure topological subset (e.g., clause-variable graph degree distributions, clustering coefficients, modularity, algebraic connectivity, and diameter), this project investigates whether the *shape* of these formulas continues to diverge or stabilizes within a localized regime.

Discovering these persistent structural properties motivates:
* **Transfer learning** across model checking depths.
* **Adaptive SAT heuristics** optimized for evolving topologies.
* **Incremental clause deletion policies** informed by structural drift.

---

## Feature Taxonomy & Filtering Framework

To guarantee scientific validity, the pipeline processes the raw feature vectors through a strict exclusion filter before computing any similarity matrix or trajectory:

| Category | Description / Scope | Reason for Exclusion | Examples Dropped |
| :--- | :--- | :--- | :--- |
| **1. Temporal** | Extraction phase wall-clock times | Reflects host CPU performance, not formula structure | `Pre-featuretime`, `CG-featuretime` |
| **2. Problem-Size** | Raw counts of vars, clauses, and reductions | Grows monotonically in BMC; creates trivial drift artifacts | `nvars`, `nclauses`, `reducedVars`, `UNARY` |
| **3. Solver-Perf** | LP relaxation values and raw local search probes | Measures external solver behavior, not formula properties | `solved`, `LP_OBJ`, `saps_BestSolution_Mean` |

---

## Requirements

* Python 3.10+
* ABC System Synthesis and Verification Solver
* SATzilla Feature Extractor

Install required Python dependencies:
```bash
pip install -r requirements.txt
```

## Running
Runs ABC unrolling, executes SATzilla, applies the topological filter, and generates plots.

```bash
python analyze_bmc_features.py \
    --abc /path/to/abc \
    --satzilla /path/to/satzilla_executable \
    --benchmarks benchmarks \
    --output bmc_feature_analysis \
    --max-frame 50 \
    --timeout 300
```

## Output

For every benchmark:

```text
output_dir/benchmark_id/
├── abc_stats.csv              # Tracked raw counts (variables, clauses, literals)
├── all_features.csv           # Untouched raw SATzilla output matrix (143 features)
├── topological_features.csv   # List of the 91 feature labels preserved for the analysis
├── benchmark_summary.csv      # Local metrics (mean/min/max similarity and drift statistics)
├── incremental_drift.csv      # Frame-by-frame calculated L2 distances
├── consecutive_similarity.png # Plot tracking similarity between frame k and k+1
├── similarity_to_frame1.png   # Plot tracking structural decay relative to initial frame
├── drift.png                  # Step-wise metric trajectory visualization
├── similarity_heatmap.png     # Full-scale frame-vs-frame similarity matrix matrix
└── pca_trajectory.png         # 2D spatial trajectory map of the formula's evolution
```

## Global Analysis

A centralized summary file (``global_summary.csv``) along with publication-quality global trend figures are output to the root of the designated directory, detailing global macro-trends and shock profiles across your entire benchmark suite.

## Citation

If you use this code, please cite the accompanying paper.
Satyam Shubham, submitted to 17th International Workshop on Boolean Problems, IWSBP 2026, Bremen, Germany.

