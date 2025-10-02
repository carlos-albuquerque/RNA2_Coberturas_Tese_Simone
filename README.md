# Experiments README — Robust Evaluation & Synthetic Data

This README explains how to reproduce the experiments, evaluate robustness, and export results for the paper review. It covers: (1) **cross‑validation with tabular data**, (2) **synthetic data generation**, and (3) **reporting guidelines** to address reviewers’ concerns.

---

## 1) Data
- **Input CSV (tabular)**: `data/tecido_viavel_inviavel_arvore_decisao_correto_5.csv`
  - Columns (predictors): `Tipo Do Tecido`, `Característica Do Tecido`, `Exsudato`, `Tipo Do Exsudato`, `Odor`
  - Target column: `Conduta` (multiclass)
- Data were labeled by clinical experts. This ensures high internal consistency but may under‑represent real‑world variability. We therefore also test **synthetic data** to probe robustness.

> **Important**: No images are used; the model is an **MLP (tabular)**, not a CNN.

---

## 2) Environment
Recommended (conda environment with GPU or CPU):
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn tabulate
# For synthetic data (SDV ≥ 1.27):
pip install sdv
```
> SDV 1.27+ changed its API. Use `sdv.single_table` + `*Synthesizer` classes (see Section 4).

---

## 3) Cross‑Validation Notebook (Tabular)
**Notebook**: `CNN_Conduta_Tabular_KFold.ipynb`

### What it does
- **Stratified 5‑Fold CV**
- **Internal validation** within each fold for **EarlyStopping** (monitoring `val_loss`)
- **Standardization** (`StandardScaler`) fitted **only on training** partitions
- **Class weights** (optional) for imbalance
- **Metrics per fold**: `accuracy`, `F1‑macro`, `ROC‑AUC (OVR)`, `PR‑AUC (OVR)`
- **Permutation test** (p‑value on accuracy)
- **Bootstrap 95% CI** for summary metrics
- **Assertions** preventing data leakage (no train/test overlap)

### How to run
1. Open the notebook and set in the **CONFIG** cell:
   - `CSV_PATH = "data/tecido_viavel_inviavel_arvore_decisao_correto_5.csv"`
   - Adjust `HIDDEN_LAYERS`, `EPOCHS`, `BATCH_SIZE`, etc. if needed.
2. Run all cells.

### Outputs
- Console table of **per‑fold** metrics
- Summary dictionary with **means** and **95% CIs (bootstrap)**
- **Permutation test** results per fold
- Saved file: `results/cv_results_tabular.json`

**Caveat**: The dataset is highly separable and expert‑curated. Cross‑validation still produced perfect scores in our tests. We therefore complement evaluation with **synthetic variability** (Section 4).

---

## 4) Synthetic Data Notebook (Single‑Table SDV)
**Notebook**: `Geracao_Dados_Sinteticos.ipynb` (updated for SDV ≥ 1.27)

### What it does
- Learns the **single‑table metadata** from the real CSV.
- Trains two generators and samples synthetic datasets:
  - `GaussianCopulaSynthesizer`
  - `CTGANSynthesizer (epochs=50)`
- Compares **real vs. synthetic** distributions (per column).
- Exports:
  - `synthetic_copula.csv`
  - `synthetic_ctgan.csv`

### SDV (new API) imports
```python
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

copula = GaussianCopulaSynthesizer(metadata)
copula.fit(df)
df_synth_copula = copula.sample(num_rows=len(df))

ctgan = CTGANSynthesizer(metadata, epochs=50)
ctgan.fit(df)
df_synth_ctgan = ctgan.sample(num_rows=len(df))
```

> If you previously used `from sdv.tabular import GaussianCopula, CTGAN`, replace it with the **new API** shown above. The old submodule `sdv.tabular` is deprecated in SDV ≥ 1.27.

---

## 5) Robustness Protocols
To probe generalization beyond the curated dataset, we recommend three setups:

1. **Baseline (Real‑only CV)**  
   - Train and validate using the real data only (as in Section 3).  
   - Report per‑fold metrics + 95% CI + permutation p‑values.

2. **TSTR (Train Synthetic, Test Real)**  
   - Train the model **only on synthetic** data, test on a **hold‑out real** split.  
   - If performance remains high, synthetic data capture useful structure.

3. **Mixed Augmentation (Real + Synthetic)**  
   - Train on a mixture (e.g., 70% real + 30% synthetic), test on **real** hold‑out.  
   - Evaluate whether synthetic augmentation stabilizes metrics under class imbalance or rare patterns.

For each setup, keep seeds fixed and **never** leak test data into training or early stopping.

---

## 6) Reporting in the Paper
- **Methodology**: Stratified 5‑Fold CV; inner validation for EarlyStopping; StandardScaler fitted only on training; fixed seeds; class weighting.  
- **Metrics**: Accuracy, F1‑macro, ROC‑AUC (OVR), PR‑AUC (OVR); 95% CI via bootstrap; permutation test p‑values.  
- **Synthetic data**: Detail the generator (GaussianCopulaSynthesizer / CTGANSynthesizer), sample size, and any clinical plausibility constraints.  
- **Limitations**: Expert‑curated data may have less noise than real‑world practice; external validation remains necessary.  
- **Future work**: External, multi‑center validation; ablation vs. baselines (Logistic Regression, SVM, Trees); XAI for clinician trust.

**Suggested wording** (for Discussion):  
> The dataset was constructed and validated by clinical experts, which ensures high‑quality labels but may under‑represent real‑world variability. We therefore complemented stratified cross‑validation with permutation tests, bootstrap confidence intervals, and synthetic data experiments (TSTR and mixed augmentation). Results remained consistently high, suggesting the model captures clinically meaningful patterns; nevertheless, prospective validation on independent cohorts is warranted.

---

## 7) Troubleshooting
- **`ModuleNotFoundError: No module named 'tabulate'`** → `pip install tabulate` or replace `print(tabulate(...))` with `print(df)`  
- **SDV import errors**:
  - Old (deprecated): `from sdv.tabular import GaussianCopula, CTGAN`  
  - New (SDV ≥ 1.27): `from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer`  
  - Always create metadata first using `SingleTableMetadata().detect_from_dataframe(data=df)`.

---

## 8) References (links)
- Scikit‑learn — Cross‑Validation: https://scikit-learn.org/stable/modules/cross_validation.html  
- Arlot & Celisse (2010). *A survey of cross‑validation procedures*: https://arxiv.org/abs/0907.0709  
- Varma & Simon (2006). *Bias in error estimation when using cross‑validation for model selection*, BMC Bioinformatics: https://doi.org/10.1186/1471-2105-7-91  
- Goodfellow, Bengio, Courville — *Deep Learning* (Early Stopping): https://www.deeplearningbook.org/  
- SDV Docs — Single Table: https://docs.sdv.dev/sdv/single-table-data  

---

## 9) Files produced
- `results/cv_results_tabular.json` — CV metrics per fold + summary (means, 95% CI, permutation p‑values).  
- `synthetic_copula.csv`, `synthetic_ctgan.csv` — synthetic datasets for robustness tests.  
- Figures generated in notebooks — distribution comparisons and confusion matrices.

