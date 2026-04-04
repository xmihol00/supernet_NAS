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
