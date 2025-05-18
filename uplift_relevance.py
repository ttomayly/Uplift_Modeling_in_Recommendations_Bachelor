import numpy as np
from pathlib import Path
from numpy.random.mtrand import RandomState
import random
import pandas as pd
from evaluator import Evaluator
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import numpy as np
import pandas as pd

def prepare_rerank_features(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6
    df = df.copy()
    df["uplift_x_rel"] = df["uplift"] * df["relevance"]
    df["rel_minus_uplift"] = df["relevance"] - df["uplift"]
    df["uplift_adj"] = df["uplift"] / (df["propensity"] + epsilon)
    df["log_uplift"] = np.where(df["uplift"] > 0, np.log(df["uplift"] + epsilon), 0.0)
    df["log_relevance"] = np.where(df["relevance"] > 0, np.log(df["relevance"] + epsilon), 0.0)
    df["prop_x_rel"] = df["propensity"] * df["relevance"]
    df["uplift_div_rel"] = df["uplift"] / (df["relevance"] + epsilon)

    return df[[
        "uplift", "relevance", 
        # "propensity",
        # "uplift_x_rel", "rel_minus_uplift", "uplift_adj",
        # "log_uplift", "log_relevance", "prop_x_rel", "uplift_div_rel"
    ]]

def fast_pareto(uplift, relevance):
    indices = np.argsort(-uplift)
    best_relevance = -np.inf
    pareto = []

    for i in indices:
        if relevance[i] > best_relevance:
            pareto.append(i)
            best_relevance = relevance[i]

    return np.array(pareto)


def train_reranker_logreg(train_df: pd.DataFrame):
    df = pd.DataFrame({
        "uplift": train_df["pred_upl"],
        "relevance": train_df["pred_rel"],
        "propensity": train_df["propensity"],
        "label": train_df["outcome"]
    })
    X = prepare_rerank_features(df)
    y = df["label"].values

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
    pipeline.fit(X, y)
    return pipeline

def predict_reranker(model, test_df: pd.DataFrame):
    df = pd.DataFrame({
        "uplift": test_df["pred_upl"],
        "relevance": test_df["pred_rel"],
        "propensity": test_df["propensity_estimate"]
    })
    X = prepare_rerank_features(df)
    return model.predict_proba(X)[:, 1]  # вероятность класса 1

def predict_reranker_lgbm(model, test_df: pd.DataFrame):
    df = pd.DataFrame({
        "uplift": test_df["pred_upl"],
        "relevance": test_df["pred_rel"],
        "propensity": test_df["propensity_estimate"]
    })
    X = prepare_rerank_features(df)
    preds = model.predict(X)
    return preds

# import optuna
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split

# def train_reranker_lgbm_with_tuning(train_df: pd.DataFrame):
#     df = pd.DataFrame({
#         "uplift": train_df["pred_upl"],
#         "relevance": train_df["pred_rel"],
#         "propensity": train_df["propensity"],
#         "label": train_df["outcome"],
#         "idx_user": train_df["idx_user"]
#     })

#     X = prepare_rerank_features(df)
#     y = df["label"]
#     groups = df.groupby("idx_user").size().values

#     # Разделим на train/valid
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
#     train_groups = [len(g) for _, g in df.iloc[X_train.index].groupby("idx_user")]
#     valid_groups = [len(g) for _, g in df.iloc[X_valid.index].groupby("idx_user")]

#     lgb_train = lgb.Dataset(X_train, y_train, group=train_groups)
#     lgb_valid = lgb.Dataset(X_valid, y_valid, group=valid_groups, reference=lgb_train)

#     def objective(trial):
#         params = {
#             "objective": "lambdarank",
#             "metric": "ndcg",
#             "ndcg_eval_at": [10],
#             "boosting": "gbdt",
#             "verbosity": -1,
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#             "num_leaves": trial.suggest_int("num_leaves", 15, 63),
#             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
#             "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#             "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
#             "seed": 42,
#             "feature_pre_filter": False
#         }

#         model = lgb.train(params, lgb_train, valid_sets=[lgb_valid], 
#                           num_boost_round=100)

#         return model.best_score["valid_0"]["ndcg@10"]

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=25)  # Можно увеличить n_trials для лучшего результата

#     best_params = study.best_params
#     best_params.update({
#         "objective": "lambdarank",
#         "metric": "ndcg",
#         "ndcg_eval_at": [10],
#         "boosting": "gbdt",
#         "verbosity": -1,
#         "seed": 42
#     })

#     print("Best hyperparameters:", best_params)

#     final_train = lgb.Dataset(X, label=y, group=groups)
#     best_model = lgb.train(best_params, final_train, num_boost_round=100)

#     return best_model

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score  # или precision, recall, f1, etc.

# def train_reranker_logreg(train_df: pd.DataFrame):
#     df = pd.DataFrame({
#         "uplift": train_df["pred_upl"],
#         "relevance": train_df["pred_rel"],
#         "propensity": train_df["propensity"],
#         "label": train_df["outcome"]
#     })
#     X = prepare_rerank_features(df)
#     y = df["label"].values

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     def objective(trial):
#         penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
#         solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
#         C = trial.suggest_float("C", 0.001, 10.0, log=True)

#         # only saga supports elasticnet
#         if penalty == "elasticnet" and solver != "saga":
#             raise optuna.exceptions.TrialPruned()

#         params = {
#             "penalty": penalty,
#             "C": C,
#             "solver": solver,
#             "max_iter": 1000,
#         }

#         if penalty == "elasticnet":
#             params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

#         pipeline = make_pipeline(
#             StandardScaler(),
#             LogisticRegression(**params)
#         )

#         pipeline.fit(X_train, y_train)
#         preds = pipeline.predict_proba(X_val)[:, 1]
#         return roc_auc_score(y_val, preds)  # можно поменять на f1_score, accuracy и т.д.

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=30)

#     best_params = study.best_trial.params
#     print("Best params (LogReg):", best_params)

#     final_params = best_params.copy()
#     if "l1_ratio" not in final_params:
#         final_params["l1_ratio"] = None  # для совместимости
#     final_model = make_pipeline(
#         StandardScaler(),
#         LogisticRegression(
#             penalty=final_params["penalty"],
#             C=final_params["C"],
#             solver=final_params["solver"],
#             l1_ratio=final_params["l1_ratio"],
#             max_iter=1000
#         )
#     )
#     final_model.fit(X, y)
#     return final_model



def train_reranker_lgbm_with_tuning(train_df: pd.DataFrame):
    df = pd.DataFrame({
        "uplift": train_df["pred_upl"],
        "relevance": train_df["pred_rel"],
        "propensity": train_df["propensity"],
        "label": train_df["outcome"],
        "idx_user": train_df["idx_user"]
    })

    X = prepare_rerank_features(df)
    y = df["label"]
    group_sizes = df.groupby("idx_user").size().values

    lgb_train = lgb.Dataset(X, label=y, group=group_sizes)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "boosting": "gbdt",
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42
    }

    model = lgb.train(params, lgb_train, num_boost_round=100)
    return model

def train_reranker_lgbm_classifier(train_df: pd.DataFrame):
    df = pd.DataFrame({
        "uplift": train_df["pred_upl"],
        "relevance": train_df["pred_rel"],
        "propensity": train_df["propensity"],
        "label": train_df["outcome"]
    })

    X = prepare_rerank_features(df)
    y = df["label"]

    lgb_train = lgb.Dataset(X, label=y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "boosting": "gbdt",
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42
    }

    model = lgb.train(params, lgb_train, num_boost_round=100)
    return model
