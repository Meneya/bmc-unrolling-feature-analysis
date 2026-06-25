# BMC Feature Evolution Analysis

This repository studies how the structural characteristics of SAT formulas generated during Bounded Model Checking (BMC) evolve as the unrolling depth increases.

The pipeline:

1. Generates CNFs from AIG benchmarks using ABC.
2. Extracts SATzilla features from every unrolling frame.
3. Normalizes the feature vectors.
4. Computes:

   * cosine similarity
   * feature drift
   * PCA trajectories
5. Produces visualizations and summary statistics.

## Motivation

As the BMC depth increases, the underlying CNF grows monotonically. This project investigates whether the structural properties of these formulas continue to diverge or eventually converge to a stable regime.

Such convergence could motivate:

* transfer learning across frames
* adaptive SAT heuristics
* incremental feature prediction
* dynamic symmetry-breaking strategies

## Requirements

* Python 3.10+
* ABC
* SATzilla

Python packages:

```bash
pip install -r requirements.txt
```

## Running

```bash
python analyze_bmc_features.py \
    --abc /path/to/abc \
    --satzilla /path/to/features \
    --benchmarks benchmarks \
    --output bmc_feature_analysis \
    --max-frame 50 \
    --timeout 300
```

## Output

For every benchmark:

```text
benchmark/
├── abc_stats.csv
├── benchmark_summary.csv
├── incremental_drift.csv
├── consecutive_similarity.png
├── similarity_to_frame1.png
├── drift.png
├── similarity_heatmap.png
└── pca_trajectory.png
```

## Global Analysis

```bash
python aggregate_global_drift.py
```

This generates:

```text
global_drift_decay_curve.png
```

showing the aggregate drift behaviour across all benchmarks.

## Citation

If you use this code, please cite the accompanying paper.

