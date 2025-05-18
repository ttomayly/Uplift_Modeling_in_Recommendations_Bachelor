import argparse
from relevance.train import prepare_data, train_propensity
from relevance.train import plotpath
from prediction.prediction_models import DLMF, PopularBase, MF, CausalNeighborBase, CausEProd, DLMF_DR, DLMF_MLP
import numpy as np
from relevance.CJBPR import CJBPR
from relevance.EM import train_propensity as train_propensity_em
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from uplift_relevance import predict_reranker, train_reranker_lgbm_with_tuning, train_reranker_lgbm_classifier, train_reranker_logreg, prepare_rerank_features, predict_reranker_lgbm, fast_pareto, prepare_rerank_features
import tensorflow as tf
from evaluator import Evaluator
import pickle
import os
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--dimension", default=128, type=int, help="number of features per user/item.")
parser.add_argument("--estimator_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Propensity and Relevance estimators")
parser.add_argument("--embedding_layer_units",
                    default=[256, 128, 64],
                    type=list,
                    help="number of nodes each layer for shared embedding layer.")
parser.add_argument("--click_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Click estimators")
parser.add_argument("--epoch", default=30, type=int,
                    help="Number of epochs in the training")
parser.add_argument("--lambda_1", default=10.0, type=float,
                    help="weight for popularity loss.")
parser.add_argument("--lambda_2", default=0.1, type=float,
                    help="weight for relavance loss.")
parser.add_argument("--lambda_3", default=0.1, type=float,
                    help="weight for propensity loss.")
parser.add_argument("--dataset", default='d', type=str,
                    help="the dataset used")
parser.add_argument("--batch_size", default=5096, type=int,
                    help="the batch size")
parser.add_argument("--repeat", default=1, type=int,
                    help="how many time to run the model")
parser.add_argument("--add", default='default', type=str,
                    help="additional information")
parser.add_argument("--p_weight", default=0.4, type=float,
                    help="weight for p_loss")
parser.add_argument("--saved_DLMF", default='n', type=str,
                    help="use saved weights of DLMF")
parser.add_argument("--to_prob", default=True, type=bool,
                    help="normalize as probability")
parser.add_argument('--ablation_variant', type=str, default='baseline',
                    choices=['baseline', 'no_dropout', 'no_layernorm', 'no_film', 'no_bce_losses', 'no_init_relevance'])
parser.add_argument('--propensity_model', type=str, default='mod_propcare',
                    choices=['mod_propcare','propcare', 'em', 'cjbr'])
parser.add_argument('--prediction_model', type=str, default='drdlmf',
                    choices=['drdlmf', 'dlmf', 'mf', 'cause', 'causeneigh'])
parser.add_argument('--uplift_relevance', type=str, default='no',
                    choices=['no', 'pareto', 'lgbm', 'logreg', 'if'])
flag = parser.parse_args()

def main(flag=flag):
    cp10list_pred = []
    cp100list_pred = []
    cdcglist_pred = []

    cp10list_rel = []
    cp100list_rel = []
    cdcglist_rel = []

    cp10list_pop = []
    cp100list_pop = []
    cdcglist_pop = []

    ndcglist_rel = []
    ndcglist_pred = []
    ndcglist_pop = []

    recalllist_rel = []
    recalllist_pred = []
    recalllist_pop = []

    precisionlist_rel = []
    precisionlist_pred = []
    precisionlist_pop = []

    random_seed = int(233)
    for epoch in range(flag.repeat):
        train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)

        random_seed += 1
        tf.random.set_seed(
            random_seed
        )

        if flag.propensity_model == 'cjbr':
            model = CJBPR(num_users, num_items, flag)
        elif flag.propensity_model == 'propcare':
            model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)
        elif flag.propensity_model == 'mod_propcare':
            model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)
        elif flag.propensity_model == 'em':
            model = train_propensity_em(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)

        train_user = tf.convert_to_tensor(train_df["idx_user"].to_numpy(), dtype=tf.int32)
        train_item = tf.convert_to_tensor(train_df["idx_item"].to_numpy(), dtype=tf.int64)
        train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item))

        test_user = tf.convert_to_tensor(test_df["idx_user"].to_numpy(), dtype=tf.int32)
        test_item = tf.convert_to_tensor(test_df["idx_item"].to_numpy(), dtype=tf.int64)
        test_data = tf.data.Dataset.from_tensor_slices((test_user, test_item))
        p_pred = None

        for u, i in train_data.batch(5000):
            if flag.propensity_model == 'cjbr':
                p_batch, r_batch = model((u, i), training=False)
            else:
                _, p_batch, _, _ = model((u, i), training=False)
            if p_pred is None:
                p_pred = p_batch
            else:
                p_pred = tf.concat((p_pred, p_batch), axis=0)

        p_pred = p_pred.numpy()
        p_pred_t = 0.25 * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
        p_pred_t = np.clip((p_pred_t + 0.5), 0.0, 1.0)

        flag.thres = 0.70

        t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
        p_pred = p_pred * 0.8
  
        train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
        train_df["treated"] = t_pred

        if flag.dataset == "d" or flag.dataset == "1d":
            cap = 0.03
            lr = 0.001
            rf = 0.01
            itr = 100e6
        if flag.dataset == "p" or flag.dataset == "1p":
            lr = 0.001
            cap = 0.5
            rf = 0.001
            itr = 100e6

        if flag.uplift_relevance == 'no':
            if flag.prediction_model == 'drdlmf':
                # with open("dlmf_weights.pkl", "rb") as f:
                #     saved_state = pickle.load(f)
                # recommender = DLMF_DR(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf, use_DR=True)
                # recommender.__dict__.update(saved_state)
                recommender = DLMF_DR(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf, use_DR=True)
                recommender.train(train_df, iter=itr)
            elif flag.prediction_model == 'dlmf':
                recommender = DLMF(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
                recommender.train(train_df, iter=itr)
            elif flag.prediction_model == 'mf':
                recommender_relevance = MF(num_users, num_items)
                recommender_relevance.train(train_df, iter=itr)     
            elif flag.prediction_model == 'cause':
                recommender = CausEProd(num_users, num_items)
                recommender.train(train_df, iter=itr)
            elif flag.prediction_model == 'causeneigh':
                recommender = CausalNeighborBase(num_users, num_items)
                recommender.train(train_df, iter=itr)


        if flag.uplift_relevance != 'no':
            recommender_uplift = DLMF_DR(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf, use_DR=True)
            recommender_uplift.train(train_df, iter=itr)
            
            recommender_relevance = MF(num_users, num_items)
            recommender_relevance.train(train_df, iter=itr)        

        cp10_tmp_list_pred = []
        cp100_tmp_list_pred = []
        cdcg_tmp_list_pred = []

        cp10_tmp_list_rel = []
        cp100_tmp_list_rel = []
        cdcg_tmp_list_rel = []

        cp10_tmp_list_pop = []
        cp100_tmp_list_pop = []
        cdcg_tmp_list_pop = []

        ndcg_tmp_list_rel = []
        ndcg_tmp_list_pred = []
        ndcg_tmp_list_pop = []

        recall_tmp_list_rel = []
        recall_tmp_list_pred = []
        recall_tmp_list_pop = []

        precision_tmp_list_rel = []
        precision_tmp_list_pred = []
        precision_tmp_list_pop = []

        if flag.uplift_relevance != 'no':
            train_df["pred_upl"] = recommender_uplift.predict(train_df)
            train_df["pred_rel"] = recommender_relevance.predict(train_df)


        if flag.uplift_relevance == 'logreg':
            reranker_model = train_reranker_logreg(train_df)
        if flag.uplift_relevance == 'lgbm':
            reranker_model = train_reranker_lgbm_with_tuning(train_df)

        
        for t in range(num_times):
            test_df_t = test_df[test_df["idx_time"] == t]
            user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
            item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
            test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
            r_pred_test = None
            p_pred_test = None

            for u, i in test_t_data.batch(5000):
                if flag.propensity_model == 'cjbr':
                    p_batch, r_batch = model((u, i), training=False)
                else:
                    _, p_batch, r_batch, _ = model((u, i), training=False)
                if r_pred_test is None:
                    r_pred_test = r_batch
                    p_pred_test = p_batch
                else:
                    r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
                    p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

            p_pred_test = p_pred_test.numpy()
            r_pred_test = r_pred_test.numpy()
            p_pred_test_t = 0.25 * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
            p_pred_test_t = np.clip((p_pred_test_t + 0.5), 0.0, 1.0)

            t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)
            p_pred_test = p_pred_test * 0.8
            r_pred_test = r_pred_test * 0.8
            test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
            test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
            test_df_t["treated_estimate"] = t_test_pred
            outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
            outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
            outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
        
            test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
            test_df_t["treated_estimate"] = t_test_pred
            causal_effect_estimate = \
                test_df_t["outcome_estimate"] * \
                (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
            test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

            train_df = train_df[train_df.outcome>0]
            popularity = train_df["idx_item"].value_counts().reset_index()
            popularity.columns = ["idx_item", "popularity"]
            test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")

            if flag.uplift_relevance != 'no':
                test_df_t["pred_upl"] = recommender_uplift.predict(test_df_t)
                test_df_t["pred_rel"] = recommender_relevance.predict(test_df_t)

            if flag.uplift_relevance == 'pareto':
                pareto_indices = fast_pareto(
                    test_df_t["pred_upl"].to_numpy(),
                    test_df_t["pred_rel"].to_numpy()
                )

                pareto_df = test_df_t.iloc[pareto_indices].copy()

                scaler = MinMaxScaler()
                pareto_df[["norm_upl", "norm_rel"]] = scaler.fit_transform(
                    pareto_df[["pred_upl", "pred_rel"]]
                )
                pareto_df["pred"] = pareto_df["pred_upl"] * pareto_df["pred_rel"]
                test_df_t["pred"] = 0
                test_df_t.loc[pareto_df.index, "pred"] = pareto_df["pred"]

                # pareto_df["rank_score"] = pareto_df["pred_upl"].rank(ascending=False) + pareto_df["pred_rel"].rank(ascending=False)
                # pareto_df["pred"] = 1 / pareto_df["rank_score"]
                # test_df_t["pred"] = 0
                # test_df_t.loc[pareto_df.index, "pred"] = pareto_df["pred"]

            if flag.uplift_relevance == 'if':
                rel_threshold = test_df_t["pred_rel"].quantile(0.6)

                filtered = test_df_t[test_df_t["pred_rel"] > rel_threshold].copy()
                filtered["pred"] = filtered["pred_upl"].rank(method="first", ascending=False)
                filtered["pred"] = 1 - (filtered["pred"] / filtered["pred"].max())
                test_df_t["pred"] = 0
                test_df_t.loc[filtered.index, "pred"] = filtered["pred"]

            if flag.uplift_relevance == 'logreg':
                test_df_t["pred"] = predict_reranker(reranker_model, test_df_t)

            if flag.uplift_relevance == 'lgbm':
                test_df_t["pred"] = predict_reranker_lgbm(reranker_model, test_df_t)

            if flag.uplift_relevance == 'no':
                test_df_t["pred"] = recommender.predict(test_df_t)

            evaluator = Evaluator()

            # kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
            # spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
            # pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

            # print(f"Kendall Tau: {kendall_score:.4f}")
            # print(f"Spearman Rho: {spearman_score:.4f}")
            # print(f"Average Rank Position Difference: {pos_diff:.4f}")


            ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
            ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
            ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

            recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
            recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
            recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

            precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
            precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
            precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

            cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
            cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
            cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

            cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
            cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
            cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

            cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
            cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
            cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

            _ = evaluator.get_sorted(test_df_t)

        cp10_pred = np.mean(cp10_tmp_list_pred)
        cp100_pred = np.mean(cp100_tmp_list_pred)
        cdcg_pred = np.mean(cdcg_tmp_list_pred)

        cp10_rel = np.mean(cp10_tmp_list_rel)
        cp100_rel = np.mean(cp100_tmp_list_rel)
        cdcg_rel = np.mean(cdcg_tmp_list_rel)

        cp10_pop = np.mean(cp10_tmp_list_pop)
        cp100_pop = np.mean(cp100_tmp_list_pop)
        cdcg_pop = np.mean(cdcg_tmp_list_pop)

        ndcg_rel = np.mean(ndcg_tmp_list_rel)
        ndcg_pred = np.mean(ndcg_tmp_list_pred)
        ndcg_pop = np.mean(ndcg_tmp_list_pop)

        recall_rel = np.mean(recall_tmp_list_rel)
        recall_pred = np.mean(recall_tmp_list_pred)
        recall_pop = np.mean(recall_tmp_list_pop)

        precision_rel = np.mean(precision_tmp_list_rel)
        precision_pred = np.mean(precision_tmp_list_pred)
        precision_pop = np.mean(precision_tmp_list_pop)

        cp10list_pred.append(cp10_pred)
        cp100list_pred.append(cp100_pred)
        cdcglist_pred.append(cdcg_pred)

        cp10list_rel.append(cp10_rel)
        cp100list_rel.append(cp100_rel)
        cdcglist_rel.append(cdcg_rel)

        cp10list_pop.append(cp10_pop)
        cp100list_pop.append(cp100_pop)
        cdcglist_pop.append(cdcg_pop)

        ndcglist_rel.append(ndcg_rel)
        ndcglist_pred.append(ndcg_pred)
        ndcglist_pop.append(ndcg_pop)

        recalllist_rel.append(recall_rel)
        recalllist_pred.append(recall_pred)
        recalllist_pop.append(recall_pop)

        precisionlist_rel.append(precision_rel)
        precisionlist_pred.append(precision_pred)
        precisionlist_pop.append(precision_pop)
    

    with open(plotpath + "/result_" + flag.dataset +".txt", "a+") as f:
        print("CP10S:", np.mean(cp10list_pred), np.std(cp10list_pred), file=f)
        print("CP10R:", np.mean(cp10list_rel), np.std(cp10list_rel), file=f)
        print("CP10P:", np.mean(cp10list_pop), np.std(cp10list_pop), file=f)

        print("CP100S:", np.mean(cp100list_pred), np.std(cp100list_pred), file=f)
        print("CP100R:", np.mean(cp100list_rel), np.std(cp100list_rel), file=f)
        print("CP100P:", np.mean(cp100list_pop), np.std(cp100list_pop), file=f)
        
        print("CDCGS:", np.mean(cdcglist_pred), np.std(cdcglist_pred), file=f)
        print("CDCGR:", np.mean(cdcglist_rel), np.std(cdcglist_rel), file=f)
        print("CDCGP:", np.mean(cdcglist_pop), np.std(cdcglist_pop), file=f)

        print("NDCG10R:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)
        print("NDCG10S:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)
        print("NDCG10P:", np.mean(ndcglist_pop), np.std(ndcglist_pop), file=f)

        print("Recall10R:", np.mean(recalllist_rel), np.std(recalllist_rel), file=f)
        print("Recall10S:", np.mean(recalllist_pred), np.std(recalllist_pred), file=f)
        print("Recall10P:", np.mean(recalllist_pop), np.std(recalllist_pop), file=f)

        print("Precision10R:", np.mean(precisionlist_rel), np.std(precisionlist_rel), file=f)
        print("Precision10S:", np.mean(precisionlist_pred), np.std(precisionlist_pred), file=f)
        print("Precision10P:", np.mean(precisionlist_pop), np.std(precisionlist_pop), file=f) 
        print("--------------------------------", file=f)    
            
            
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    main(flag)
