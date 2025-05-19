# PropCare + DLCE: Uplift Modeling in Recommender Systems

This repository contains the official implementation for the bachelor thesis **"Uplift Modeling in Recommender Systems"** (HSE, 2025). The project addresses causal recommendation with a focus on uplift modeling — estimating the **causal effect** of showing an item to a user.

It extends and enhances two recent models:

- **PropCare** (NeurIPS 2023): Estimation of propensity scores from implicit feedback.
- **DLCE** (RecSys 2020): Debiased Learning for the Causal Effect of recommendation using Inverse Propensity Scoring.

Additional contributions include improvements to PropCare and a **Doubly Robust** extension of DLCE, along with the researching the hybrid approaches of finding the balance between the uplift and relevance.

---

## 📁 Repository Structure

| File                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `main.py`              | Script to run training and evaluation                                       |
| `train.py`             | Dataset loading and training loop for PropCare                              |
| `PropCare_new.py`      | Enhanced version of the PropCare                                            |
| `PropCare_old.py`      | Original implementation of PropCare                                         |
| `prediction_models.py` | Original implementation of DLCE, DR-DLCE and other baselines                |
| `uplift_relevance.py`  | Re-ranking models: Pareto, logistic regression, LightGBM Ranker             |
| `evaluator.py`         | Evaluation metrics: CP@10, CP@100, CDCG, NDCG@10, etc.                      |
| `CJBR.py`, `EM.py`     | Baseline models: combinational joint learning and EM-based approach         |

---

## 📦 Dataset

We use the **Dunnhumby** semi-synthetic dataset, both:
- `original` — uniform treatment assignment
- `personalized` — user-specific assignment probabilities

Follow the setup instructions from the original data source:  
https://arxiv.org/abs/2008.04563

---

## 🚀 How to Run

Before running the project, install dependencies:

```bash
pip install -r requirements.txt
```

Run experiments with:

```bash
python -u main.py --dataset d    # Dunnhumby original
python -u main.py --dataset p    # Dunnhumby personalized
```

You can modify model type and training configs in `main.py`.

| Flag | Description |
|------|-------------|
| `propensity_model` | Propensity estimation model: `mod_propcare` (default), `propcare`, `em`, `cjbr` |
| `prediction_model` | Prediction/uplift model: `drdlmf` (Doubly Robust), `dlmf`, `mf`, `cause`, `causeneigh` |
| `uplift_relevance` | Strategy to combine uplift and relevance: `no`, `pareto`, `lgbm`, `logreg`, `if` |
| `ablation_variant` | For ablation study: disable components like dropout, BCE losses, etc. (e.g., `no_dropout`) | and etc.

Example: Run improved PropCare with DR-DLMF and LightGBM reranker on personalized data

```bash
python -u main.py --dataset p --propensity_model mod_propcare --prediction_model drdlmf --uplift_relevance lgbm
```

<!-- ## 💡 Citation

If you use this code in your research or project, please cite:

```
Tanya Tomayly (2025). Uplift Modeling in Recommender Systems. Bachelor Thesis, HSE University.
``` -->

## 🤝 Acknowledgements

Thanks to Zhongzhou Liu for the [open-source implementation](https://github.com/mediumboat/PropCare) of some baseline models and evaluation code.