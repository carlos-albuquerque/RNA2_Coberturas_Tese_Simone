# CNN_Conduta — Robust Evaluation with K-Fold

This package organizes your CNN experiment and adds **robust validation** to address reviewers’ concerns: *stratified k-fold*, **bootstrap confidence intervals**, and **permutation tests**. It also includes **automated checks** to prevent data leakage.

### Data
- **CSV**: use `data/tecido_viavel_inviavel_arvore_decisao_correto_5.csv`

## Evaluations
1. **Stratified K-Fold (default k=5)** — provides a more stable estimate of average performance.  
2. **Bootstrap confidence intervals** — report mean and 95% CI for each metric.  
3. **Permutation test** — p-value to check whether performance is significantly better than chance.  
4. **Fixed seeds** — ensures reproducibility (`PYTHONHASHSEED`, `numpy`, `tensorflow`).  
5. **Automated checks** — guarantee *no overlap* between training and test sets and deterministic splits.

## Output
- `results/cv_results.json` with per-fold metrics and summary statistics with confidence intervals.  
- Fold-by-fold table printed to the terminal.

## References
- Scikit-learn: Cross-validation [https://scikit-learn.org/stable/modules/cross_validation.html]  
- Arlot, S., & Celisse, A. (2010). *A survey of cross-validation procedures*. [https://arxiv.org/abs/0907.0709]  
- Varma, S., & Simon, R. (2006). *Bias in error estimation when using cross-validation for model selection*. *BMC Bioinformatics*. [https://doi.org/10.1186/1471-2105-7-91]  
- Goodfellow et al., *Deep Learning*, Early Stopping (Chap. 7). [https://www.deeplearningbook.org/]  
