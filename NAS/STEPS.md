# Genetic NAS Experiment Steps (IMX500)

## 1) Minimum experiment matrix

To make results acceptable in research practice, run **multiple seeds** and compare against at least one baseline.

- Algorithms:
  - `baseline_sga` (baseline)
  - `regularized_evolution` (strong NAS evolutionary method)
- Independent runs per algorithm:
  - **Recommended:** 10 runs
  - **Minimum acceptable:** 5 runs
- Same budget per run:
  - same `population-size`, `generations`, `offspring-per-generation`
  - same train/eval datasets and preprocessing
  - same epochs-per-candidate (e.g., 3)

This gives 20 total runs (recommended) or 10 total runs (minimum).

## 2) What to report

For each algorithm, report across seeds:

- **Best quantized accuracy** per run (primary metric)
- **Mean ± std** of best quantized accuracy
- **Median** and **IQR** (robustness)
- **Success rate** (fraction of candidates that compile)
- **Search efficiency**:
  - time-to-best
  - candidates evaluated
  - compilable candidates evaluated
- **Resource profile of final model**:
  - compile success
  - model package output
  - any available memory/size indicators from compile artifacts

## 3) Statistical testing

Use statistical tests over final per-run results:

- If distribution is unclear/non-Gaussian: **Mann–Whitney U** test
- Otherwise (normality plausible): **Welch’s t-test**
- Report **effect size**:
  - Cliff’s delta (non-parametric) or Cohen’s d (parametric)
- Report **95% confidence intervals** for mean or median differences (bootstrap CI is fine)

## 4) Reproducibility checklist

For every run, save:

- full CLI args
- random seed
- generation-by-generation history
- final selected architecture config
- training logs per candidate
- quantization and compile logs
- environment summary (Python, PyTorch, MCT, ONNX Runtime versions)

Keep all runs in unique timestamped directories to avoid collisions.

## 5) Practical recommendation for your setup

Given compilation + quantization are expensive, a practical starting protocol:

- `population-size`: 20
- `offspring-per-generation`: 8
- `generations`: 12
- `epochs-per-candidate`: 3
- `images-per-class-eval`: 1 or 2 for search-time speed

Then do a **final retrain/re-eval** of top-3 models with longer training and a larger eval subset for the final paper-quality number.

## 6) Suggested tables in paper/report

- **Table A:** Algorithm comparison (mean±std best quant acc, success rate, wall-clock)
- **Table B:** Best model configs per algorithm (resolution, widths, depths)
- **Table C:** Ablation (mutation rate, tournament/sample size, offspring count)
- **Figure:** Best fitness vs generation (mean curve across seeds with shaded std)

## 7) Common community expectations

- Multiple independent runs (not single-seed claims)
- Fair equal-budget comparison
- Explicit variance reporting (not only best case)
- Statistical significance + effect size
- Publicly reproducible scripts/configuration

## 8) Automated multi-run pipeline (new)

Use `multi_run_nas_experiment.py` to run **multiple independent seeds** for both `baseline_sga` and `regularized_evolution`, then automatically compute statistical tests and generate PNG visualizations.

### Main goals covered

- fair equal-budget comparison across algorithms
- independent seeds for each run
- run-by-run and generation-by-generation logging
- statistically grounded comparison with effect sizes and multiple-testing correction
- publication-friendly visual diagnostics

### Install analysis dependencies

```bash
pip install -r NAS/requirements.txt
```

### Example command

```bash
python NAS/multi_run_nas_experiment.py \
  --algorithms baseline_sga regularized_evolution \
  --runs-per-algorithm 10 \
  --base-seed 1200 \
  --seed-stride 97 \
  --generations 12 \
  --population-size 20 \
  --offspring-per-generation 8 \
  --epochs-per-candidate 3 \
  --continue-on-failure
```

### Output structure

- `experiment_config.json`: full experiment configuration and environment context
- `experiment.log`: human-readable orchestration log
- `experiment_events.jsonl`: structured event log for post-hoc analysis
- `runs_manifest.jsonl`: per-run status events
- `run_records.json` and `run_records.csv`: aggregated per-run outcomes
- `statistics.json`: statistical comparisons and effect sizes
- `experiment_summary.json`: high-level success/failure counts
- `raw_runs/`: raw output roots used by `genetic_NAS_runner.py`
- `visualizations/`: generated PNG plots (including `visualizations/live/` for current run status snapshots)

Inside each per-run directory produced by `genetic_NAS_runner.py`, there is also:

- `top_3_architectures.json`: top-3 candidates for that run (ranked by fitness among compiled candidates; fallback to all candidates if none compiled), including configs and key metrics.

### Parallel execution layout (new)

`multi_run_nas_experiment.py` now stores artifacts in algorithm-specific subdirectories under the chosen `--output-root`:

- `<output-root>/baseline_sga/...`
- `<output-root>/regularized_evolution/...`

Each algorithm subdirectory contains its own:

- `run_records.json`, `run_records.csv`
- `statistics.json`
- `experiment_summary.json`
- `visualizations/`
- `runs_manifest.jsonl`, `experiment_events.jsonl`

The root `<output-root>/` also stores merged artifacts if multiple algorithms are run in one process.

## 10) Two-machine parallel workflow (SGA + Reg. Evolution)

Use the dedicated PBS launchers (one per machine/GPU):

- `sga_multi_run_nas_experiment.pbs`
- `reg_evo_multi_run_nas_experiment.pbs`

Both are configured to run one algorithm only and keep rich run-level logs/artifacts.

### Merge two independent outputs into one comparison bundle

When both jobs finish (possibly from different machines), run:

```bash
python NAS/merge_parallel_nas_experiments.py \
  --sga-dir NAS/<path_to_sga_output>/baseline_sga \
  --reg-evo-dir NAS/<path_to_reg_evo_output>/regularized_evolution \
  --output-dir NAS/merged_parallel_comparison
```

This produces in `NAS/merged_parallel_comparison`:

- merged `run_records.json` + `run_records.csv`
- merged `statistics.json`
- merged `experiment_summary.json`
- `merge_sources.json` (provenance of source runs)
- comparison `visualizations/*.png`

Then generate publication report from merged results:

```bash
python NAS/generate_publication_report.py \
  --experiment-dir NAS/merged_parallel_comparison \
  --title "Baseline SGA vs Regularized Evolution (Parallel Runs)" \
  --author "Your Name / Team"
```

### Statistical analysis implemented

For each metric (`best_quant_acc1`, `best_fitness`, `compile_success_rate`, `elapsed_seconds`, `total_candidates_evaluated`, `compiled_candidates`):

- normality check: Shapiro-Wilk (when sample size allows)
- parametric branch: Welch's t-test (if both groups are approximately normal)
- non-parametric branch: Mann-Whitney U (default otherwise)
- effect size:
  - Cohen's d for Welch branch
  - Cliff's delta for Mann-Whitney branch
- uncertainty: bootstrap CI for mean difference
- multiplicity control: Holm-Bonferroni adjusted p-values

### Visualizations generated (PNG)

- experiment progress timeline (`overall_progress.png`)
- live run state (`visualizations/live/*.png`) with:
  - best/mean fitness by generation
  - compile success rate as candidates are evaluated
- final distribution comparisons:
  - `distribution_best_quant_acc1.png`
  - `distribution_compile_success_rate.png`
  - `distribution_elapsed_seconds.png`
- convergence summary across seeds (`convergence_best_fitness.png`)
- run-level tradeoff scatter (`run_tradeoff_scatter.png`)
- statistical significance summary (`statistical_pvalues.png`)
- effect-size summary (`effect_sizes.png`)

### Reproducibility notes

- seeds are auto-generated to be unique across algorithm and run index
- every launched command is logged
- each run keeps full stdout logs and original artifacts
- all generated statistics/plots are reproducible from saved run records

## 9) Publication-ready report generation

After a multi-run experiment finishes, generate a paper/report-friendly Markdown document with tables and embedded figures:

```bash
python NAS/generate_publication_report.py \
  --experiment-dir NAS/multi_run_experiments/<experiment_run_dir> \
  --title "Baseline SGA vs Regularized Evolution" \
  --author "Your Name / Team"
```

Optional notes and output directory override:

```bash
python NAS/generate_publication_report.py \
  --experiment-dir NAS/multi_run_experiments/<experiment_run_dir> \
  --output-dir NAS/multi_run_experiments/<experiment_run_dir>/publication_report \
  --notes "Final paper ablation set"
```

### Report artifacts

The script creates:

- `publication_report/publication_report.md`
- `publication_report/publication_report_summary.json`
- `publication_report/tables/algorithm_summary.csv`
- `publication_report/tables/statistical_tests.csv`
- `publication_report/tables/runs_overview.csv`
- `publication_report/tables/best_models.csv`

### Included report content

- experiment setup overview (key CLI/config parameters)
- run-level table across all seeds
- per-algorithm summary table (mean/std/median/IQR/min/max)
- pairwise statistical test table (test type, p-values, Holm correction, effect sizes, CI)
- best discovered model per algorithm with architectural configuration
- embedded figures from `visualizations/`

This gives a strong starting point for paper appendices, internal reports, and reproducibility packages.
