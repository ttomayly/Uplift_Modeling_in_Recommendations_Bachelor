# import argparse
# from train import prepare_data, train_propensity
# from train import plotpath, Causal_Model
# from baselines_new import DLMF, PopularBase, MF, CausalNeighborBase, CausEProd, DLMF2, DLMF3,DLMF4, DLMF_MLP
# # from baselines import DLMF, PopularBase, MF, CausalNeighborBase, DLMF2
# # from baselines_ import DLMF
# import numpy as np
# from CJBR_new import CJBPR
# import tensorflow as tf
# from evaluator import Evaluator
# import pickle
# import os
# import pandas as pd
# import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument("--dimension", default=128, type=int, help="number of features per user/item.")
# parser.add_argument("--estimator_layer_units",
#                     default=[64, 32, 16, 8],
#                     type=list,
#                     help="number of nodes each layer for MLP layers in Propensity and Relevance estimators")
# parser.add_argument("--embedding_layer_units",
#                     default=[256, 128, 64],
#                     type=list,
#                     help="number of nodes each layer for shared embedding layer.")
# parser.add_argument("--click_layer_units",
#                     default=[64, 32, 16, 8],
#                     type=list,
#                     help="number of nodes each layer for MLP layers in Click estimators")
# parser.add_argument("--epoch", default=30, type=int,
#                     help="Number of epochs in the training")
# parser.add_argument("--lambda_1", default=10.0, type=float,
#                     help="weight for popularity loss.")
# parser.add_argument("--lambda_2", default=0.1, type=float,
#                     help="weight for relavance loss.")
# parser.add_argument("--lambda_3", default=0.1, type=float,
#                     help="weight for propensity2 loss.")
# parser.add_argument("--dataset", default='d', type=str,
#                     help="the dataset used")
# parser.add_argument("--batch_size", default=5096, type=int,
#                     help="the batch size")
# parser.add_argument("--repeat", default=1, type=int,
#                     help="how many time to run the model")
# parser.add_argument("--add", default='default', type=str,
#                     help="additional information")
# parser.add_argument("--p_weight", default=0.4, type=float,
#                     help="weight for p_loss")
# parser.add_argument("--saved_DLMF", default='n', type=str,
#                     help="use saved weights of DLMF")
# parser.add_argument("--to_prob", default=True, type=bool,
#                     help="normalize as probability")
# flag = parser.parse_args()

# # Функция нормализации: (с параметром to_prob=True - как у авторов гита, с to_prob=False - как в статье)
# def get_norm(vec, to_prob=True, mu=0.5, sigma=0.25):
#     vec_norm = (vec - np.mean(vec))/(np.std(vec))
#     if to_prob:
#         vec_norm = sigma * vec_norm
#         vec_norm = np.clip((vec_norm + mu), 0.0, 1.0)
#     return vec_norm

# def main(flag=flag):
#     ndcglist_rel = []
#     ndcglist_pred = []
#     ndcglist_pop = []

#     recalllist_rel = []
#     recalllist_pred = []
#     recalllist_pop = []

#     precisionlist_rel = []
#     precisionlist_pred = []
#     precisionlist_pop = []
    
#     cp10list_pred = []
#     cp100list_pred = []
#     cdcglist_pred = []

#     cp10list_rel = []
#     cp100list_rel = []
#     cdcglist_rel = []

#     cp10list_pop = []
#     cp100list_pop = []
#     cdcglist_pop = []

#     # prelist_pred = []
#     # ndcglist_pred = []

#     # prelist_rel = []
#     # ndcglist_rel = []

#     # prelist_pop = []
#     # ndcglist_pop = []

#     random_seed = int(233)
#     for epoch in range(flag.repeat):
#         train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)
#         # test_df = pd.read_csv('df_sorted_names.csv')

#         random_seed += 1
#         tf.random.set_seed(
#             random_seed
#         )
#         model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)
#         # model = CJBPR(num_users, num_items, flag)

#         # gpus = tf.config.experimental.list_physical_devices('GPU')
#         # if gpus:
#         #     try:
#         #         for gpu in gpus:
#         #             tf.config.experimental.set_memory_growth(gpu, True)
#         #     except RuntimeError as e:
#         #         print(e)
#         # # Создаем модель
#         # model = CJBPR(train_df, vali_df, test_df)

#         # # Обучаем модель
#         # model.fit()

#         # Train model (uses propensity scores from train_df)
#         # print("Training model...")
#         # model.train(train_df)

#         train_user = tf.convert_to_tensor(train_df["idx_user"].to_numpy(), dtype=tf.int32)
#         train_item = tf.convert_to_tensor(train_df["idx_item"].to_numpy(), dtype=tf.int64)
#         train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item))

#         test_user = tf.convert_to_tensor(test_df["idx_user"].to_numpy(), dtype=tf.int32)
#         test_item = tf.convert_to_tensor(test_df["idx_item"].to_numpy(), dtype=tf.int64)
#         test_data = tf.data.Dataset.from_tensor_slices((test_user, test_item))
#         p_pred = None

#         for u, i in train_data.batch(5000):
#             _, p_batch, _, _ = model((u, i), training=False)
#             if p_pred is None:
#                 p_pred = p_batch
#             else:
#                 p_pred = tf.concat((p_pred, p_batch), axis=0)

#         p_pred = p_pred.numpy()

#         # for u, i in train_data.batch(5000):
#         #     _, p_batch = model((u, i), training=False)  # Now returns tuple of (r_pred, p_pred)
#         #     p_batch = tf.reshape(p_batch, [-1])  # Flatten to 1D if needed
            
#         #     if p_pred is None:
#         #         p_pred = p_batch
#         #     else:
#         #         p_pred = tf.concat((p_pred, p_batch), axis=0)

#         # p_pred = p_pred.numpy()
#         # p_pred_t = get_norm(p_pred, to_prob=flag.to_prob)

#         # if flag.dataset == "d" or "p" and flag.to_prob:
#         #     flag.thres = 0.70
#         # elif flag.dataset == "ml" and flag.to_prob:
#         #     flag.thres = 0.65
#         # elif flag.dataset == "d" or "p" and not flag.to_prob:
#         #     flag.thres = 0.2
#         # elif flag.dataset == "ml" and not flag.to_prob:
#         #     flag.thres = 0.15

#         p_pred_t = 0.25 * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
#         p_pred_t = np.clip((p_pred_t + 0.5), 0.0, 1.0)

#         if flag.dataset == "d" or "p":
#             flag.thres = 0.70
#         elif flag.dataset == "ml":
#             flag.thres = 0.65

#         # # Параметр c
#         # if flag.dataset == "d" or "p":
#         #     flag.c = 0.2
#         # elif flag.dataset == "ml":
#         #     flag.c = 0.2

#         t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
#         if flag.dataset == "d" or "p":
#             p_pred = p_pred * 0.8 ## 0.8
#         if flag.dataset == "ml":
#             p_pred = p_pred * 0.2

#         train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
#         train_df["treated"] = t_pred

#         if flag.dataset == "d":
#             cap = 0.03
#             lr = 0.001
#             rf = 0.01
#             itr = 100e6
#         if flag.dataset == "p":
#             lr = 0.001
#             cap = 0.5
#             rf = 0.001
#             itr = 100e6
#         if flag.dataset == "ml":
#             lr = 0.001
#             cap = 0.3
#             rf = 0.1
#             itr = 100e6

#         with open("dlmf_weights (5).pkl", "rb") as f:
#             saved_state = pickle.load(f)

#         recommender = DLMF(num_users, num_items, capping_T = cap, 
#                            capping_C = cap, learn_rate = lr, reg_factor = rf)

#         recommender.__dict__.update(saved_state)
#         print("DLMF weights loaded successfully!")

#         # recommender = PopularBase(num_users, num_items)
#         # recommender.train(train_df, iter=itr)

#         # recommender = CausEProd(num_users, num_items)
#         # recommender.train(train_df, iter=itr)

#         # recommender = DLMF2(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)

#         # recommender = DLMF(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)

#         # recommender = DLMF3(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf, use_DR=True)
#         # recommender.train(train_df, iter=itr)
        

#         # recommender = DLMF_MLP(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)


        
#         # recommender = DR_Estimator(num_users, num_items, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)
#         # recommender = CausalNeighborBase(num_users, num_items)
#         # recommender.train(train_df, iter=itr)

#         # recommender = MF(num_users, num_items)
#         # recommender.train(train_df, iter=itr)        

#         cp10_tmp_list_pred = []
#         cp100_tmp_list_pred = []
#         cdcg_tmp_list_pred = []

#         cp10_tmp_list_rel = []
#         cp100_tmp_list_rel = []
#         cdcg_tmp_list_rel = []

#         cp10_tmp_list_pop = []
#         cp100_tmp_list_pop = []
#         cdcg_tmp_list_pop = []

#         # ndcg_tmp_list_pred = []
#         # prec_tmp_list_pred = []

#         # ndcg_tmp_list_rel = []
#         # prec_tmp_list_rel = []

#         # ndcg_tmp_list_pop = []
#         # prec_tmp_list_pop = []


#         # посчитать обычные ndcg/recall по outcome при сортировке 
#         # а) по relevance б) по pred в) по popular (популярность просто по датасету)

#         # test_df = pd.read_csv('/Users/tanyatomayly/Downloads/df_sorted.csv')

#         ndcg_tmp_list_rel = []
#         ndcg_tmp_list_pred = []
#         ndcg_tmp_list_pop = []

#         recall_tmp_list_rel = []
#         recall_tmp_list_pred = []
#         recall_tmp_list_pop = [] 

#         precision_tmp_list_rel = []
#         precision_tmp_list_pred = []
#         precision_tmp_list_pop = []

#         if flag.dataset == 'd' or 'p':
#             for t in range(num_times):
#                 test_df_t = test_df[test_df["idx_time"] == t]
#                 user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
#                 item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
#                 test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
#                 r_pred_test = None
#                 p_pred_test = None

#                 for u, i in test_t_data.batch(5000):
#                     _, p_batch, r_batch, _ = model((u, i), training=False)
#                     # p_batch, r_batch = model((u, i), training=False)
#                     if r_pred_test is None:
#                         r_pred_test = r_batch
#                         p_pred_test = p_batch
#                     else:
#                         r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
#                         p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

#                 p_pred_test = p_pred_test.numpy()
#                 r_pred_test = r_pred_test.numpy()
#                 p_pred_test_t = 0.25 * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
#                 p_pred_test_t = np.clip((p_pred_test_t + 0.5), 0.0, 1.0)

#                 # p_pred_test_t = get_norm(p_pred_test, to_prob=flag.to_prob)
#                 t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)
#                 p_pred_test = p_pred_test * 0.8
#                 r_pred_test = r_pred_test * 0.8
#                 test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
#                 test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
#                 test_df_t["treated_estimate"] = t_test_pred
#                 # outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
#                 # outcome_estimate = get_norm(outcome_estimate, to_prob=flag.to_prob)
#                 # test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
#                 outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
#                 outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
#                 outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
#                 test_df_t["outcome_estimate"] = np.where(outcome_estimate >= 0.7, 1.0, 0.0)

#                 # test_df_t = pd.read_csv('/Users/tanyatomayly/Downloads/df_sorted.csv')
#                 train_df = train_df[train_df.outcome>0]
#                 popularity = train_df["idx_item"].value_counts().reset_index()
#                 popularity.columns = ["idx_item", "popularity"]
#                 test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
#                 test_df_t["pred"] = recommender.predict(test_df_t)
#                 evaluator = Evaluator()

#                 causal_effect_estimate = \
#                     test_df_t["outcome_estimate"] * \
#                     (test_df_t["outcome_estimate"] / test_df_t["propensity_estimate"] - \
#                     (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
#                 test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

#                 # kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 # spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 # pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

#                 kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'popularity', 'relevance_estimate')
#                 spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'popularity', 'relevance_estimate')
#                 pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'popularity', 'relevance_estimate')
#                 print(f"Kendall Tau: {kendall_score:.4f}")
#                 print(f"Spearman Rho: {spearman_score:.4f}")
#                 print(f"Average Rank Position Difference: {pos_diff:.4f}")

#                 # user_item_counts = test_df_t.groupby('idx_user')['idx_item'].count().reset_index()
#                 # user_item_counts.columns = ['idx_user', 'num_items']
#                 # print(user_item_counts['num_items'].describe())

#                 # df_ranking_pop = evaluator.get_ranking_popularity(test_df_t, num_rec=10)
#                 # print(df_ranking_pop.groupby('idx_user')['idx_item'].nunique().describe())

#                 # top_items = test_df_t['idx_item'].value_counts().head(10).index
#                 # print(test_df_t[test_df_t['idx_item'].isin(top_items)].groupby('idx_item')['outcome'].mean())

#                 # print(test_df_t.duplicated(subset=['idx_user', 'idx_item']).sum())


#                 ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
#                 ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
#                 ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

#                 recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
#                 recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
#                 recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

#                 precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
#                 precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
#                 precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

#                 cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
#                 cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
#                 cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

#                 cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
#                 cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
#                 cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

#                 cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
#                 cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
#                 cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

#                 # prec_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'Prec', 10))
#                 # ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCG', 100))

#                 # prec_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecR', 10))
#                 # ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 100000))

#                 # prec_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecP', 10))
#                 # ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 100000))

#                 _ = evaluator.get_sorted(test_df_t)
#         else:
#             for t in [0]:
#                 test_df_t = test_df[test_df["idx_time"] == t]
#                 user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
#                 item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
#                 test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
#                 r_pred_test = None
#                 p_pred_test = None

#                 for u, i in test_t_data.batch(5000):
#                     _, p_batch, r_batch, _ = model((u, i), training=False)
#                     if r_pred_test is None:
#                         r_pred_test = r_batch
#                         p_pred_test = p_batch
#                     else:
#                         r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
#                         p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

#                 p_pred_test = p_pred_test.numpy()
#                 r_pred_test = r_pred_test.numpy()
#                 p_pred_test_t = 0.25 * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
#                 p_pred_test_t = np.clip((p_pred_test_t + 0.5), 0.0, 1.0)

#                 t_test_pred = np.where(p_pred_test_t >= 0.65, 1.0, 0.0)
#                 p_pred_test = p_pred_test * 0.2
#                 r_pred_test = r_pred_test * 0.2
#                 test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
#                 test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
#                 test_df_t["treated_estimate"] = t_test_pred
                
#                 outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
#                 outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
#                 outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
#                 test_df_t["outcome_estimate"] = np.where(outcome_estimate >= 0.7, 1.0, 0.0)

#                 causal_effect_estimate = \
#                     test_df_t["outcome_estimate"] * \
#                     (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
#                     (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
#                 test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

#                 train_df = train_df[train_df.outcome>0]
#                 popularity = train_df["idx_item"].value_counts().reset_index()
#                 popularity.columns = ["idx_item", "popularity"]
#                 test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
#                 test_df_t["pred"] = recommender.predict(test_df_t)
#                 evaluator = Evaluator()
#                 cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
#                 cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
#                 cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

#                 ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
#                 ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
#                 ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

#                 recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
#                 recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
#                 recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

#                 precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
#                 precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
#                 precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

#                 cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
#                 cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
#                 cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

#                 cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
#                 cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
#                 cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

#                 kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

#                 print(f"Kendall Tau: {kendall_score:.4f}")
#                 print(f"Spearman Rho: {spearman_score:.4f}")
#                 print(f"Average Rank Position Difference: {pos_diff:.4f}")

#                 _ = evaluator.get_sorted(test_df_t)



#         ndcg_rel = np.mean(ndcg_tmp_list_rel)
#         ndcg_pred = np.mean(ndcg_tmp_list_pred)
#         ndcg_pop = np.mean(ndcg_tmp_list_pop)

#         recall_rel = np.mean(recall_tmp_list_rel)
#         recall_pred = np.mean(recall_tmp_list_pred)
#         recall_pop = np.mean(recall_tmp_list_pop)


#         precision_rel = np.mean(precision_tmp_list_rel)
#         precision_pred = np.mean(precision_tmp_list_pred)
#         precision_pop = np.mean(precision_tmp_list_pop)


#         cp10_pred = np.mean(cp10_tmp_list_pred)
#         cp100_pred = np.mean(cp100_tmp_list_pred)
#         cdcg_pred = np.mean(cdcg_tmp_list_pred)

#         cp10_rel = np.mean(cp10_tmp_list_rel)
#         cp100_rel = np.mean(cp100_tmp_list_rel)
#         cdcg_rel = np.mean(cdcg_tmp_list_rel)

#         cp10_pop = np.mean(cp10_tmp_list_pop)
#         cp100_pop = np.mean(cp100_tmp_list_pop)
#         cdcg_pop = np.mean(cdcg_tmp_list_pop)

#         # prec_pred = np.mean(prec_tmp_list_pred)
#         # ndcg_pred = np.mean(ndcg_tmp_list_pred)

#         # prec_rel = np.mean(prec_tmp_list_rel)
#         # ndcg_rel = np.mean(ndcg_tmp_list_rel)

#         # prec_pop = np.mean(prec_tmp_list_pop)
#         # ndcg_pop = np.mean(ndcg_tmp_list_pop)

#         cp10list_pred.append(cp10_pred)
#         cp100list_pred.append(cp100_pred)
#         cdcglist_pred.append(cdcg_pred)

#         cp10list_rel.append(cp10_rel)
#         cp100list_rel.append(cp100_rel)
#         cdcglist_rel.append(cdcg_rel)

#         cp10list_pop.append(cp10_pop)
#         cp100list_pop.append(cp100_pop)
#         cdcglist_pop.append(cdcg_pop)

#         # prelist_pred.append(prec_pred)
#         # ndcglist_pred.append(ndcg_pred)

#         # prelist_rel.append(prec_rel)
#         # ndcglist_rel.append(ndcg_rel)

#         # prelist_pop.append(prec_pop)
#         # ndcglist_pop.append(ndcg_pop)


#         ndcglist_rel.append(ndcg_rel)
#         ndcglist_pred.append(ndcg_pred)
#         ndcglist_pop.append(ndcg_pop)

#         recalllist_rel.append(recall_rel)
#         recalllist_pred.append(recall_pred)
#         recalllist_pop.append(recall_pop)

#         precisionlist_rel.append(precision_rel)
#         precisionlist_pred.append(precision_pred)
#         precisionlist_pop.append(precision_pop)       

    
#     with open(plotpath+"/result_" + flag.dataset +".txt", "a+") as f:
#         print("NDCG10R:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)
#         print("NDCG10S:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)
#         print("NDCG10P:", np.mean(ndcglist_pop), np.std(ndcglist_pop), file=f)

#         print("Recall10R:", np.mean(recalllist_rel), np.std(recalllist_rel), file=f)
#         print("Recall10S:", np.mean(recalllist_pred), np.std(recalllist_pred), file=f)
#         print("Recall10P:", np.mean(recalllist_pop), np.std(recalllist_pop), file=f)

#         print("Precision10R:", np.mean(precisionlist_rel), np.std(precisionlist_rel), file=f)
#         print("Precision10S:", np.mean(precisionlist_pred), np.std(precisionlist_pred), file=f)
#         print("Precision10P:", np.mean(precisionlist_pop), np.std(precisionlist_pop), file=f)        

#         print("CP10S:", np.mean(cp10list_pred), np.std(cp10list_pred), file=f)
#         print("CP10R:", np.mean(cp10list_rel), np.std(cp10list_rel), file=f)
#         print("CP10P:", np.mean(cp10list_pop), np.std(cp10list_pop), file=f)

#         print("CP100S:", np.mean(cp100list_pred), np.std(cp100list_pred), file=f)
#         print("CP100R:", np.mean(cp100list_rel), np.std(cp100list_rel), file=f)
#         print("CP100P:", np.mean(cp100list_pop), np.std(cp100list_pop), file=f)
        
#         print("CDCGS:", np.mean(cdcglist_pred), np.std(cdcglist_pred), file=f)
#         print("CDCGR:", np.mean(cdcglist_rel), np.std(cdcglist_rel), file=f)
#         print("CDCGP:", np.mean(cdcglist_pop), np.std(cdcglist_pop), file=f)

#         # print("RelP:", np.mean(prelist_pred), np.std(prelist_pred), file=f)
#         # print("NDCGP:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)

#         # print("RelR:", np.mean(prelist_rel), np.std(prelist_rel), file=f)
#         # print("NDCGR:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)

#         # print("RelPop:", np.mean(prelist_rel), np.std(prelist_rel), file=f)
#         # print("NDCGPop:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)

            
# if __name__ == "__main__":
#     physical_devices = tf.config.list_physical_devices('GPU')
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     except:
#         pass
#     main(flag)

import argparse
from train import prepare_data, train_propensity
from train import plotpath, Causal_Model
from baselines_new import DLMF, PopularBase, MF, CausalNeighborBase, CausEProd, DLMF2, DLMF3,DLMF4, DLMF_MLP
# from baselines import DLMF, PopularBase, MF, CausalNeighborBase, DLMF2
# from baselines_ import DLMF
import numpy as np
# from CJBR_new import CJBPR
# from EM import PropensityModel, train_propensity

# mlp_reranker.py
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def prepare_rerank_features(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6
    df = df.copy()
    df["uplift_x_rel"] = df["uplift"] * df["relevance"]
    df["rel_minus_uplift"] = df["relevance"] - df["uplift"]
    df["uplift_adj"] = df["uplift"] / (df["propensity"] + epsilon)
    df["log_uplift"] = np.log(df["uplift"] + epsilon)
    df["log_relevance"] = np.log(df["relevance"] + epsilon)
    df["prop_x_rel"] = df["propensity"] * df["relevance"]
    df["uplift_div_rel"] = df["uplift"] / (df["relevance"] + epsilon)

    return df[[
        "uplift", "relevance", "propensity",
        "uplift_x_rel", "rel_minus_uplift", "uplift_adj",
        "log_uplift", "log_relevance", "prop_x_rel", "uplift_div_rel"
    ]]

def train_rerank_mlp(X_train, y_train):
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    model.fit(X_train, y_train, batch_size=256, epochs=10, validation_split=0.1, verbose=0)
    return model

def fast_pareto(uplift, relevance):
    indices = np.argsort(-uplift)  # сортировка по убыванию uplift
    best_relevance = -np.inf
    pareto = []

    for i in indices:
        if relevance[i] > best_relevance:
            pareto.append(i)
            best_relevance = relevance[i]

    return np.array(pareto)


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
flag = parser.parse_args()
import lightgbm as lgb

# Функция нормализации: (с параметром to_prob=True - как у авторов гита, с to_prob=False - как в статье)
def get_norm(vec, to_prob=True, mu=0.5, sigma=0.25):
    vec_norm = (vec - np.mean(vec))/(np.std(vec))
    if to_prob:
        vec_norm = sigma * vec_norm
        vec_norm = np.clip((vec_norm + mu), 0.0, 1.0)
    return vec_norm

import numpy as np
import pandas as pd

def prepare_rerank_features(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6
    df = df.copy()
    df["uplift_x_rel"] = df["uplift"] * df["relevance"]
    df["rel_minus_uplift"] = df["relevance"] - df["uplift"]
    df["uplift_adj"] = df["uplift"] / (df["propensity"] + epsilon)
    df["log_uplift"] = np.log(np.clip(df["uplift"], a_min=epsilon, a_max=None))
    df["log_relevance"] = np.log(np.clip(df["relevance"], a_min=epsilon, a_max=None))
    df["prop_x_rel"] = df["propensity"] * df["relevance"]
    df["uplift_div_rel"] = df["uplift"] / (df["relevance"] + epsilon)

    return df[[
        "uplift", "relevance", "propensity",
        "uplift_x_rel", "rel_minus_uplift", "uplift_adj",
        "log_uplift", "log_relevance", "prop_x_rel", "uplift_div_rel"
    ]]

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_reranker_logreg(train_df: pd.DataFrame):
    df = pd.DataFrame({
        "uplift": train_df["uplift"],
        "relevance": train_df["relevance"],
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

    cp10list_pers_pop = []
    cp100list_pers_pop = []
    cdcglist_pers_pop = []

    ndcglist_rel = []
    ndcglist_pred = []
    ndcglist_pop = []
    ndcglist_pers_pop = []

    recalllist_rel = []
    recalllist_pred = []
    recalllist_pop = []
    recalllist_pers_pop = []

    precisionlist_rel = []
    precisionlist_pred = []
    precisionlist_pop = []
    precisionlist_pers_pop = []

    # prelist_pred = []
    # ndcglist_pred = []

    # prelist_rel = []
    # ndcglist_rel = []

    # prelist_pop = []
    # ndcglist_pop = []

    random_seed = int(233)
    for epoch in range(flag.repeat):
        train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)
        # test_df = pd.read_csv('df_sorted_names.csv')

        random_seed += 1
        tf.random.set_seed(
            random_seed
        )
        model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)


        # model = CJBPR(num_users, num_items, flag)

        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        #     except RuntimeError as e:
        #         print(e)
        # Создаем модель
        # model = CJBPR(train_df, vali_df, test_df)

        # # Обучаем модель
        # model.fit()

        # Train model (uses propensity scores from train_df)
        # print("Training model...")
        # model.train(train_df)

        train_user = tf.convert_to_tensor(train_df["idx_user"].to_numpy(), dtype=tf.int32)
        train_item = tf.convert_to_tensor(train_df["idx_item"].to_numpy(), dtype=tf.int64)
        train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item))

        test_user = tf.convert_to_tensor(test_df["idx_user"].to_numpy(), dtype=tf.int32)
        test_item = tf.convert_to_tensor(test_df["idx_item"].to_numpy(), dtype=tf.int64)
        test_data = tf.data.Dataset.from_tensor_slices((test_user, test_item))
        p_pred = None

        for u, i in train_data.batch(5000):
            # p_batch, r_batch = model((u, i), training=False)
            _, p_batch, _, _ = model((u, i), training=False)
            if p_pred is None:
                p_pred = p_batch
            else:
                p_pred = tf.concat((p_pred, p_batch), axis=0)

        p_pred = p_pred.numpy()

        # for u, i in train_data.batch(5000):
        #     _, p_batch = model((u, i), training=False)  # Now returns tuple of (r_pred, p_pred)
        #     p_batch = tf.reshape(p_batch, [-1])  # Flatten to 1D if needed
            
        #     if p_pred is None:
        #         p_pred = p_batch
        #     else:
        #         p_pred = tf.concat((p_pred, p_batch), axis=0)

        # p_pred = p_pred.numpy()
        # p_pred_t = get_norm(p_pred, to_prob=flag.to_prob)

        # if flag.dataset == "d" or "p" and flag.to_prob:
        #     flag.thres = 0.70
        # elif flag.dataset == "ml" and flag.to_prob:
        #     flag.thres = 0.65
        # elif flag.dataset == "d" or "p" and not flag.to_prob:
        #     flag.thres = 0.2
        # elif flag.dataset == "ml" and not flag.to_prob:
        #     flag.thres = 0.15

        p_pred_t = 0.25 * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
        p_pred_t = np.clip((p_pred_t + 0.5), 0.0, 1.0)

        if flag.dataset == "d" or "p":
            flag.thres = 0.70
        elif flag.dataset == "ml":
            flag.thres = 0.65

        # # Параметр c
        # if flag.dataset == "d" or "p":
        #     flag.c = 0.2
        # elif flag.dataset == "ml":
        #     flag.c = 0.2

        t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
        if flag.dataset == "d" or "p":
            p_pred = p_pred * 0.8 ## 0.8
        if flag.dataset == "ml":
            p_pred = p_pred * 0.2

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
        if flag.dataset == "ml":
            lr = 0.001
            cap = 0.3
            rf = 0.1
            itr = 100e6

        with open("dlmf_weights.pkl", "rb") as f:
            saved_state = pickle.load(f)

        # recommender = DLMF(num_users, num_items, capping_T = cap, 
        #                    capping_C = cap, learn_rate = lr, reg_factor = rf)

        recommender_uplift = DLMF3(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf, use_DR=True)


        recommender_uplift.__dict__.update(saved_state)
        print("DLMF weights loaded successfully!")

        # recommender = PopularBase(num_users, num_items)
        # recommender.train(train_df, iter=itr)

        # recommender = CausEProd(num_users, num_items)
        # recommender.train(train_df, iter=itr)

        # recommender = DLMF2(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
        # recommender.train(train_df, iter=itr)

        # recommender = DLMF(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
        # recommender.train(train_df, iter=itr)

        # recommender_uplift = DLMF3(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf, use_DR=True)
        # recommender_uplift.train(train_df, iter=itr)
        

        # recommender = DLMF_MLP(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
        # recommender.train(train_df, iter=itr)


        
        # recommender = DR_Estimator(num_users, num_items, learn_rate=lr, reg_factor=rf)
        # recommender.train(train_df, iter=itr)
        # recommender = CausalNeighborBase(num_users, num_items)
        # recommender.train(train_df, iter=itr)

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

        cp10_tmp_list_pers_pop = []
        cp100_tmp_list_pers_pop = []
        cdcg_tmp_list_pers_pop = []
        
        ndcg_tmp_list_rel = []
        ndcg_tmp_list_pred = []
        ndcg_tmp_list_pop = []
        ndcg_tmp_list_pers_pop = []

        recall_tmp_list_rel = []
        recall_tmp_list_pred = []
        recall_tmp_list_pop = []
        recall_tmp_list_pers_pop = []

        precision_tmp_list_rel = []
        precision_tmp_list_pred = []
        precision_tmp_list_pop = []
        precision_tmp_list_pers_pop = []

        # # Перед этим — предсказать uplift/relevance на train_df:
        train_df["uplift"] = recommender_uplift.predict(train_df)
        train_df["relevance"] = recommender_relevance.predict(train_df)

        # # epsilon = 1e-6
        # # train_df["uplift_adj"] = train_df["uplift"] / (train_df["propensity"] + epsilon)
        # train_df["rel2"] = train_df["relevance"] * train_df["relevance"]
        # train_df["uplift_x_rel"] = train_df["uplift"] * train_df["relevance"]
        # train_df["uplift_minus_rel"] = np.abs(train_df["uplift"] - train_df["relevance"])

        # features = ["uplift", "relevance", "propensity", 
        #             # "uplift_adj", 
        #             "rel2",
        #             "uplift_x_rel", "uplift_minus_rel"
        #             ]
        # X_train = train_df[features]
        # y_train = train_df["outcome"]

        # # LGBM Group info — например, по пользователям:
        # group_sizes = train_df.groupby("idx_user").size().values
        # lgb_train = lgb.Dataset(X_train, label=y_train, group=group_sizes)

        # params = {
        #     "objective": "lambdarank",
        #     "metric": "ndcg",
        #     "ndcg_eval_at": [10],
        #     "learning_rate": 0.05,
        #     "num_leaves": 31,
        #     "min_data_in_leaf": 20,
        #     "verbose": -1,
        # }

        # ranker = lgb.train(params, lgb_train, num_boost_round=100)

        reranker_model = train_reranker_logreg(train_df)  # train_df содержит uplift, relevance, propensity, outcome


        if flag.dataset == 'd' or 'p':
            for t in range(num_times):
                test_df_t = test_df[test_df["idx_time"] == t]
                user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
                item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
                test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
                r_pred_test = None
                p_pred_test = None

                for u, i in test_t_data.batch(5000):
                    _, p_batch, r_batch, _ = model((u, i), training=False)
                    # p_batch, r_batch = model((u, i), training=False)
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

                # p_pred_test_t = get_norm(p_pred_test, to_prob=flag.to_prob)
                t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)
                p_pred_test = p_pred_test * 0.8
                r_pred_test = r_pred_test * 0.8
                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
                test_df_t["treated_estimate"] = t_test_pred
                # outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                # outcome_estimate = get_norm(outcome_estimate, to_prob=flag.to_prob)
                # test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
                outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
                outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
                test_df_t["outcome_estimate"] = np.where(outcome_estimate >= 0.7, 1.0, 0.0)
                
                # causal_effect_estimate = \
                #     test_df_t["outcome_estimate"] * \
                #     (test_df_t["outcome_estimate"] / test_df_t["propensity_estimate"] - \
                #     (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                # test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

                # train_df = train_df[train_df.outcome>0]
                # train_df['personal_popular'] = train_df.groupby(['idx_user', 'idx_item'])['outcome'].transform('sum')
                # pers_popular = train_df[['idx_user', 'idx_item', 'personal_popular']]
                # test_df_t = test_df_t.merge(pers_popular, on=['idx_user', 'idx_item'], how="left")

                # train_df = train_df[train_df.outcome>0]
                # popularity = train_df["idx_item"].value_counts().reset_index()
                # popularity.columns = ["idx_item", "popularity"]
                # test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                # test_df_t['popularity'] = (test_df_t['popularity'] - np.min(test_df_t['popularity'])) \
                #                             / (np.max(test_df_t['popularity']) - np.min(test_df_t['popularity']))
                # test_df_t['frequency'] = test_df_t['personal_popular']
                # test_df_t['personal_popular'] = test_df_t['personal_popular'] + test_df_t['popularity']

                # # test_df_t = pd.read_csv('/Users/tanyatomayly/Downloads/df_sorted.csv')
                # train_df = train_df[train_df.outcome>0]
                # popularity = train_df["idx_item"].value_counts().reset_index()
                # popularity.columns = ["idx_item", "popularity"]
                # test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                # test_df_t["pred"] = recommender.predict(test_df_t)

                # test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                # test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)

                # outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                # outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
                # outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
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
                test_df_t['popularity'] = (test_df_t['popularity'] - np.min(test_df_t['popularity'])) \
                                            / (np.max(test_df_t['popularity']) - np.min(test_df_t['popularity']))
                test_df_t['frequency'] = test_df_t['personal_popular']
                test_df_t['personal_popular'] = test_df_t['personal_popular'] + test_df_t['popularity']

                # test_df_t = test_df_t.sample(n=10000, random_state=42)

                test_df_t["pred_upl"] = recommender_uplift.predict(test_df_t)
                test_df_t["pred_rel"] = recommender_relevance.predict(test_df_t)


                test_df_t["pred"] =  test_df_t["pred_upl"] hi can I be

                # pareto_indices = fast_pareto(
                #     test_df_t["pred_upl"].to_numpy(),
                #     test_df_t["pred_rel"].to_numpy()
                # )

                # test_df_t["pred"] = 0
                # test_df_t.loc[pareto_indices, "pred"] = 1

                # Предположим, test_df_t — это DataFrame с предсказаниями
                # pred_upl — предсказание uplift
                # pred_rel — предсказание relevance

                # Шаг 1: Подготовим матрицу признаков
                # F = test_df_t[["pred_upl", "pred_rel"]].to_numpy() * -1  # отрицание — так как pymoo минимизирует

                # Шаг 2: Найдём индексы Pareto-фронта
                # Получение индексов фронта
                # nds = NonDominatedSorting().do(F, only_non_dominated_front=True)

                # # Создание новой колонки
                # test_df_t["pred"] = 0  # по умолчанию всем 0
                # test_df_t.loc[nds, "pred"] = 1  # фронту присваиваем 1

                # test_df_t["pred"] = predict_reranker(reranker_model, test_df_t)

                # from sklearn.preprocessing import MinMaxScaler

                # test_df_t["uplift"] = recommender_uplift.predict(test_df_t)
                # test_df_t["relevance"] = recommender_relevance.predict(test_df_t)

                # # # Те же фичи:
                # # test_df_t["uplift_adj"] = test_df_t["uplift"] / (test_df_t["propensity_estimate"] + epsilon)
                # test_df_t["rel2"] = test_df_t["relevance"] * test_df_t["relevance"]
                # test_df_t["uplift_x_rel"] = test_df_t["uplift"] * test_df_t["relevance"]
                # test_df_t["uplift_minus_rel"] = np.abs(test_df_t["uplift"] - test_df_t["relevance"])

                # X_test = test_df_t[features]
                # test_df_t["pred"] = ranker.predict(X_test)


                # from sklearn.linear_model import LogisticRegression
                # from sklearn.model_selection import train_test_split

                # import lightgbm as lgb

                # # === Фичи и цель ===
                # df = pd.DataFrame({
                #     "idx_user": test_df_t["idx_user"],
                #     "idx_item": test_df_t["idx_item"],
                #     "uplift": test_df_t["pred_upl"],
                #     "relevance": test_df_t["pred_rel"],
                #     "propensity": test_df_t["propensity_estimate"],
                #     "label": test_df_t["outcome"],  # binary: click / purchase
                # })

                # # === Интерактивные признаки ===
                # epsilon = 1e-6
                # df["uplift_adj"] = df["uplift"] / (df["propensity"] + epsilon)
                # df["uplift_x_rel"] = df["uplift"] * df["relevance"]
                # df["uplift_minus_rel"] = np.abs(df["uplift"] - df["relevance"])

                # # === Порядок фичей ===
                # features = ["uplift", "relevance", "propensity", "uplift_adj", "uplift_x_rel", "uplift_minus_rel"]
                # X = df[features]
                # y = df["label"]

                # # === Группировка: число айтемов на пользователя ===
                # group_sizes = df.groupby("idx_user").size().values  # например, по 100 айтемов на пользователя

                # # === Dataset for LGBM Ranker ===
                # lgb_train = lgb.Dataset(X, label=y, group=group_sizes)

                # params = {
                #     "objective": "lambdarank",
                #     "metric": "ndcg",
                #     "ndcg_eval_at": [10],
                #     "learning_rate": 0.05,
                #     "num_leaves": 31,
                #     "min_data_in_leaf": 20,
                #     "verbose": -1,
                # }

                # ranker = lgb.train(
                #     params,
                #     lgb_train,
                #     num_boost_round=100,
                # )

                # df["pred_score"] = ranker.predict(X)

                # # Запись обратно
                # test_df_t["pred"] = df["pred_score"]




                # df = pd.DataFrame({
                #     "uplift": test_df_t["pred_upl"],
                #     "relevance": test_df_t["pred_rel"],
                #     "propensity_estimate": test_df_t["propensity_estimate"],
                #     "label": test_df_t['outcome'],  # бинарная: click / purchase / treatment outcome
                # })

                # df["interaction"] = df["uplift"] * df["relevance"]

                # X = df[["uplift", "relevance", "propensity_estimate"]]
                # X["uplift"] = X["uplift"] * 2  # усиливаем вклад uplift
                # X["uplift_corr"] = df["uplift"] / (df["propensity_estimate"] + 1e-6)

                # model = LogisticRegression()
                # model.fit(X, df["label"])
                # test_df_t["pred"] = model.predict_proba(X)[:, 1]

                # y = df["label"]

                # model = LogisticRegression()
                # model.fit(X, y)

                # # Предсказания для теста
                # test_df_t["pred"] = model.predict_proba(X)[:, 1]


                # scaler = MinMaxScaler()

                # test_df_t["uplift_norm"] = scaler.fit_transform(test_df_t[["pred_upl"]])
                # test_df_t["relevance_norm"] = scaler.fit_transform(test_df_t[["pred_rel"]])

                # # Стабилизация логарифмом (добавляем epsilon для избежания log(0))
                # epsilon = 1e-6
                # alpha = 0.3
                # test_df_t["pred"] = (
                #     alpha * np.log(test_df_t["uplift_norm"] + epsilon) +
                #     (1 - alpha) * np.log(test_df_t["relevance_norm"] + epsilon)
                # )


                # # # Складываем нормализованные значения
                # # test_df_t["pred"] = new["pred_norm"] + new["pred_mf_norm"]

                # # print(test_df_t)
                evaluator = Evaluator()



                # train_df['personal_popular'] = train_df.groupby(['idx_user', 'idx_item'])['outcome'].transform('sum')
                # pers_popular = train_df[['idx_user', 'idx_item', 'personal_popular']]
                # test_df_t = test_df_t.merge(pers_popular, on=['idx_user', 'idx_item'], how="left")
                # train_df = train_df[train_df.outcome>0]
                # popularity = train_df["idx_item"].value_counts().reset_index()
                # popularity.columns = ["idx_item", "popularity"]
                # test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                
                # test_df_t['popularity'] = (test_df_t['popularity'] - np.min(test_df_t['popularity'])) \
                #                             / (np.max(test_df_t['popularity']) - np.min(test_df_t['popularity']))
                # test_df_t['frequency'] = test_df_t['personal_popular']
                # test_df_t['personal_popular'] = test_df_t['personal_popular'] + test_df_t['popularity']

                # test_df_t = pd.read_csv('/Users/tanyatomayly/Downloads/df_sorted.csv')
                # train_df = train_df[train_df.outcome>0]
                # popularity = train_df["idx_item"].value_counts().reset_index()
                # popularity.columns = ["idx_item", "popularity"]
                # test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                # test_df_t["pred"] = recommender.predict(test_df_t)
                # evaluator = Evaluator()




                kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

                # kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'popularity', 'relevance_estimate')
                # spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'popularity', 'relevance_estimate')
                # pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'popularity', 'relevance_estimate')
                print(f"Kendall Tau: {kendall_score:.4f}")
                print(f"Spearman Rho: {spearman_score:.4f}")
                print(f"Average Rank Position Difference: {pos_diff:.4f}")

                # user_item_counts = test_df_t.groupby('idx_user')['idx_item'].count().reset_index()
                # user_item_counts.columns = ['idx_user', 'num_items']
                # print(user_item_counts['num_items'].describe())

                # df_ranking_pop = evaluator.get_ranking_popularity(test_df_t, num_rec=10)
                # print(df_ranking_pop.groupby('idx_user')['idx_item'].nunique().describe())

                # top_items = test_df_t['idx_item'].value_counts().head(10).index
                # print(test_df_t[test_df_t['idx_item'].isin(top_items)].groupby('idx_item')['outcome'].mean())

                # print(test_df_t.duplicated(subset=['idx_user', 'idx_item']).sum())


                ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
                ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
                ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))
                ndcg_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'NDCGPP', 10))

                recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
                recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
                recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))
                recall_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'NDCGPP', 10))

                precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
                precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
                precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))
                precision_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'NDCGPP', 10))

                cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
                cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
                cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

                cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
                cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
                cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

                cp10_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'CPrecPP', 10))
                cp100_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'CPrecPP', 100))
                cdcg_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'CDCGPP', 100000))


                cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
                cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
                cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

                # prec_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'Prec', 10))
                # ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCG', 100))

                # prec_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecR', 10))
                # ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 100000))

                # prec_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecP', 10))
                # ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 100000))

                _ = evaluator.get_sorted(test_df_t)
        else:
            for t in [0]:
                test_df_t = test_df[test_df["idx_time"] == t]
                user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
                item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
                test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
                r_pred_test = None
                p_pred_test = None

                for u, i in test_t_data.batch(5000):
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

                t_test_pred = np.where(p_pred_test_t >= 0.65, 1.0, 0.0)
                p_pred_test = p_pred_test * 0.2
                r_pred_test = r_pred_test * 0.2
                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
                test_df_t["treated_estimate"] = t_test_pred
                
                outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
                outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
                test_df_t["outcome_estimate"] = np.where(outcome_estimate >= 0.7, 1.0, 0.0)

                causal_effect_estimate = \
                    test_df_t["outcome_estimate"] * \
                    (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                    (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

                train_df = train_df[train_df.outcome>0]
                popularity = train_df["idx_item"].value_counts().reset_index()
                popularity.columns = ["idx_item", "popularity"]
                test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                test_df_t["pred"] = recommender.predict(test_df_t)
                evaluator = Evaluator()
                cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
                cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
                cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

                ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
                ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
                ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

                recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
                recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
                recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

                precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
                precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
                precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

                cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
                cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
                cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

                cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
                cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
                cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

                kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

                print(f"Kendall Tau: {kendall_score:.4f}")
                print(f"Spearman Rho: {spearman_score:.4f}")
                print(f"Average Rank Position Difference: {pos_diff:.4f}")

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

        cp10_pers_pop = np.mean(cp10_tmp_list_pers_pop)
        cp100_pers_pop = np.mean(cp100_tmp_list_pers_pop)
        cdcg_pers_pop = np.mean(cdcg_tmp_list_pers_pop)

        ndcg_rel = np.mean(ndcg_tmp_list_rel)
        ndcg_pred = np.mean(ndcg_tmp_list_pred)
        ndcg_pop = np.mean(ndcg_tmp_list_pop)
        ndcg_pers_pop = np.mean(ndcg_tmp_list_pers_pop)

        recall_rel = np.mean(recall_tmp_list_rel)
        recall_pred = np.mean(recall_tmp_list_pred)
        recall_pop = np.mean(recall_tmp_list_pop)
        recall_pers_pop = np.mean(recall_tmp_list_pers_pop)

        precision_rel = np.mean(precision_tmp_list_rel)
        precision_pred = np.mean(precision_tmp_list_pred)
        precision_pop = np.mean(precision_tmp_list_pop)
        precision_pers_pop = np.mean(precision_tmp_list_pers_pop)

        cp10list_pred.append(cp10_pred)
        cp100list_pred.append(cp100_pred)
        cdcglist_pred.append(cdcg_pred)

        cp10list_rel.append(cp10_rel)
        cp100list_rel.append(cp100_rel)
        cdcglist_rel.append(cdcg_rel)

        cp10list_pop.append(cp10_pop)
        cp100list_pop.append(cp100_pop)
        cdcglist_pop.append(cdcg_pop)

        cp10list_pers_pop.append(cp10_pers_pop)
        cp100list_pers_pop.append(cp100_pers_pop)
        cdcglist_pers_pop.append(cdcg_pers_pop)

        ndcglist_rel.append(ndcg_rel)
        ndcglist_pred.append(ndcg_pred)
        ndcglist_pop.append(ndcg_pop)
        ndcglist_pers_pop.append(ndcg_pers_pop)

        recalllist_rel.append(recall_rel)
        recalllist_pred.append(recall_pred)
        recalllist_pop.append(recall_pop)
        recalllist_pers_pop.append(recall_pers_pop)

        precisionlist_rel.append(precision_rel)
        precisionlist_pred.append(precision_pred)
        precisionlist_pop.append(precision_pop)
        precisionlist_pers_pop.append(precision_pers_pop)       


    

    with open(plotpath + "/result_" + flag.dataset +".txt", "a+") as f:
        print("CP10S:", np.mean(cp10list_pred), np.std(cp10list_pred), file=f)
        print("CP10R:", np.mean(cp10list_rel), np.std(cp10list_rel), file=f)
        print("CP10P:", np.mean(cp10list_pop), np.std(cp10list_pop), file=f)
        print("CP10PP:", np.mean(cp10list_pers_pop), np.std(cp10list_pers_pop), file=f)

        print("CP100S:", np.mean(cp100list_pred), np.std(cp100list_pred), file=f)
        print("CP100R:", np.mean(cp100list_rel), np.std(cp100list_rel), file=f)
        print("CP100P:", np.mean(cp100list_pop), np.std(cp100list_pop), file=f)
        print("CP100PP:", np.mean(cp100list_pers_pop), np.std(cp100list_pers_pop), file=f)
        
        print("CDCGS:", np.mean(cdcglist_pred), np.std(cdcglist_pred), file=f)
        print("CDCGR:", np.mean(cdcglist_rel), np.std(cdcglist_rel), file=f)
        print("CDCGP:", np.mean(cdcglist_pop), np.std(cdcglist_pop), file=f)
        print("CDCGPP:", np.mean(cdcglist_pers_pop), np.std(cdcglist_pers_pop), file=f)

        print("NDCG10R:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)
        print("NDCG10S:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)
        print("NDCG10P:", np.mean(ndcglist_pop), np.std(ndcglist_pop), file=f)
        print("NDCG10PP:", np.mean(ndcglist_pers_pop), np.std(ndcglist_pers_pop), file=f)

        print("Recall10R:", np.mean(recalllist_rel), np.std(recalllist_rel), file=f)
        print("Recall10S:", np.mean(recalllist_pred), np.std(recalllist_pred), file=f)
        print("Recall10P:", np.mean(recalllist_pop), np.std(recalllist_pop), file=f)
        print("Recall10PP:", np.mean(recalllist_pers_pop), np.std(recalllist_pers_pop), file=f)

        print("Precision10R:", np.mean(precisionlist_rel), np.std(precisionlist_rel), file=f)
        print("Precision10S:", np.mean(precisionlist_pred), np.std(precisionlist_pred), file=f)
        print("Precision10P:", np.mean(precisionlist_pop), np.std(precisionlist_pop), file=f) 
        print("Precision10PP:", np.mean(precisionlist_pers_pop), np.std(precisionlist_pers_pop), file=f) 
        print("--------------------------------", file=f)    
            
            
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    main(flag)

# import argparse
# from train import prepare_data, train_propensity
# from train import plotpath, Causal_Model
# from baselines_new import DLMF, PopularBase, MF, CausalNeighborBase, CausEProd, DLMF2, DLMF3,DLMF4, DLMF_MLP
# # from baselines import DLMF, PopularBase, MF, CausalNeighborBase, DLMF2
# # from baselines_ import DLMF
# import numpy as np
# from CJBR_new import CJBPR
# import tensorflow as tf
# from evaluator import Evaluator
# import pickle
# import os
# import pandas as pd
# import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument("--dimension", default=128, type=int, help="number of features per user/item.")
# parser.add_argument("--estimator_layer_units",
#                     default=[64, 32, 16, 8],
#                     type=list,
#                     help="number of nodes each layer for MLP layers in Propensity and Relevance estimators")
# parser.add_argument("--embedding_layer_units",
#                     default=[256, 128, 64],
#                     type=list,
#                     help="number of nodes each layer for shared embedding layer.")
# parser.add_argument("--click_layer_units",
#                     default=[64, 32, 16, 8],
#                     type=list,
#                     help="number of nodes each layer for MLP layers in Click estimators")
# parser.add_argument("--epoch", default=30, type=int,
#                     help="Number of epochs in the training")
# parser.add_argument("--lambda_1", default=10.0, type=float,
#                     help="weight for popularity loss.")
# parser.add_argument("--lambda_2", default=0.1, type=float,
#                     help="weight for relavance loss.")
# parser.add_argument("--lambda_3", default=0.1, type=float,
#                     help="weight for propensity2 loss.")
# parser.add_argument("--dataset", default='d', type=str,
#                     help="the dataset used")
# parser.add_argument("--batch_size", default=5096, type=int,
#                     help="the batch size")
# parser.add_argument("--repeat", default=1, type=int,
#                     help="how many time to run the model")
# parser.add_argument("--add", default='default', type=str,
#                     help="additional information")
# parser.add_argument("--p_weight", default=0.4, type=float,
#                     help="weight for p_loss")
# parser.add_argument("--saved_DLMF", default='n', type=str,
#                     help="use saved weights of DLMF")
# parser.add_argument("--to_prob", default=True, type=bool,
#                     help="normalize as probability")
# flag = parser.parse_args()

# # Функция нормализации: (с параметром to_prob=True - как у авторов гита, с to_prob=False - как в статье)
# def get_norm(vec, to_prob=True, mu=0.5, sigma=0.25):
#     vec_norm = (vec - np.mean(vec))/(np.std(vec))
#     if to_prob:
#         vec_norm = sigma * vec_norm
#         vec_norm = np.clip((vec_norm + mu), 0.0, 1.0)
#     return vec_norm

# def main(flag=flag):
#     ndcglist_rel = []
#     ndcglist_pred = []
#     ndcglist_pop = []

#     recalllist_rel = []
#     recalllist_pred = []
#     recalllist_pop = []

#     precisionlist_rel = []
#     precisionlist_pred = []
#     precisionlist_pop = []
    
#     cp10list_pred = []
#     cp100list_pred = []
#     cdcglist_pred = []

#     cp10list_rel = []
#     cp100list_rel = []
#     cdcglist_rel = []

#     cp10list_pop = []
#     cp100list_pop = []
#     cdcglist_pop = []

#     # prelist_pred = []
#     # ndcglist_pred = []

#     # prelist_rel = []
#     # ndcglist_rel = []

#     # prelist_pop = []
#     # ndcglist_pop = []

#     random_seed = int(233)
#     for epoch in range(flag.repeat):
#         train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)
#         # test_df = pd.read_csv('df_sorted_names.csv')

#         random_seed += 1
#         tf.random.set_seed(
#             random_seed
#         )
#         model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)

#         # model = CJBPR(num_users, num_items, flag)

#         # gpus = tf.config.experimental.list_physical_devices('GPU')
#         # if gpus:
#         #     try:
#         #         for gpu in gpus:
#         #             tf.config.experimental.set_memory_growth(gpu, True)
#         #     except RuntimeError as e:
#         #         print(e)
#         # # Создаем модель
#         # model = CJBPR(train_df, vali_df, test_df)

#         # # Обучаем модель
#         # model.fit()

#         # Train model (uses propensity scores from train_df)
#         # print("Training model...")
#         # model.train(train_df)

#         train_user = tf.convert_to_tensor(train_df["idx_user"].to_numpy(), dtype=tf.int32)
#         train_item = tf.convert_to_tensor(train_df["idx_item"].to_numpy(), dtype=tf.int64)
#         train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item))

#         test_user = tf.convert_to_tensor(test_df["idx_user"].to_numpy(), dtype=tf.int32)
#         test_item = tf.convert_to_tensor(test_df["idx_item"].to_numpy(), dtype=tf.int64)
#         test_data = tf.data.Dataset.from_tensor_slices((test_user, test_item))
#         p_pred = None

#         for u, i in train_data.batch(5000):
#             _, p_batch, _, _ = model((u, i), training=False)
#             if p_pred is None:
#                 p_pred = p_batch
#             else:
#                 p_pred = tf.concat((p_pred, p_batch), axis=0)

#         p_pred = p_pred.numpy()

#         # for u, i in train_data.batch(5000):
#         #     _, p_batch = model((u, i), training=False)  # Now returns tuple of (r_pred, p_pred)
#         #     p_batch = tf.reshape(p_batch, [-1])  # Flatten to 1D if needed
            
#         #     if p_pred is None:
#         #         p_pred = p_batch
#         #     else:
#         #         p_pred = tf.concat((p_pred, p_batch), axis=0)

#         # p_pred = p_pred.numpy()
#         # p_pred_t = get_norm(p_pred, to_prob=flag.to_prob)

#         # if flag.dataset == "d" or "p" and flag.to_prob:
#         #     flag.thres = 0.70
#         # elif flag.dataset == "ml" and flag.to_prob:
#         #     flag.thres = 0.65
#         # elif flag.dataset == "d" or "p" and not flag.to_prob:
#         #     flag.thres = 0.2
#         # elif flag.dataset == "ml" and not flag.to_prob:
#         #     flag.thres = 0.15

#         p_pred_t = 0.25 * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
#         p_pred_t = np.clip((p_pred_t + 0.5), 0.0, 1.0)

#         if flag.dataset == "d" or "p":
#             flag.thres = 0.70
#         elif flag.dataset == "ml":
#             flag.thres = 0.65

#         # # Параметр c
#         # if flag.dataset == "d" or "p":
#         #     flag.c = 0.2
#         # elif flag.dataset == "ml":
#         #     flag.c = 0.2

#         t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
#         if flag.dataset == "d" or "p":
#             p_pred = p_pred * 0.8 ## 0.8
#         if flag.dataset == "ml":
#             p_pred = p_pred * 0.2

#         train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
#         train_df["treated"] = t_pred

#         if flag.dataset == "d":
#             cap = 0.03
#             lr = 0.001
#             rf = 0.01
#             itr = 100e6
#         if flag.dataset == "p":
#             lr = 0.001
#             cap = 0.5
#             rf = 0.001
#             itr = 100e6
#         if flag.dataset == "ml":
#             lr = 0.001
#             cap = 0.3
#             rf = 0.1
#             itr = 100e6

#         # with open("dlmf_weights.pkl", "rb") as f:
#         #     saved_state = pickle.load(f)

#         # recommender = DLMF(num_users, num_items, capping_T = cap, 
#         #                    capping_C = cap, learn_rate = lr, reg_factor = rf)

#         # recommender.__dict__.update(saved_state)
#         # print("DLMF weights loaded successfully!")

#         # recommender = PopularBase(num_users, num_items)
#         # recommender.train(train_df, iter=itr)

#         # recommender = CausEProd(num_users, num_items)
#         # recommender.train(train_df, iter=itr)

#         # recommender = DLMF2(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)

#         # recommender = DLMF(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)

#         recommender = DLMF3(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf, use_DR=True)
#         recommender.train(train_df, iter=itr)
        

#         # recommender = DLMF_MLP(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)


        
#         # recommender = DR_Estimator(num_users, num_items, learn_rate=lr, reg_factor=rf)
#         # recommender.train(train_df, iter=itr)
#         # recommender = CausalNeighborBase(num_users, num_items)
#         # recommender.train(train_df, iter=itr)

#         # recommender = MF(num_users, num_items)
#         # recommender.train(train_df, iter=itr)        

#         cp10_tmp_list_pred = []
#         cp100_tmp_list_pred = []
#         cdcg_tmp_list_pred = []

#         cp10_tmp_list_rel = []
#         cp100_tmp_list_rel = []
#         cdcg_tmp_list_rel = []

#         cp10_tmp_list_pop = []
#         cp100_tmp_list_pop = []
#         cdcg_tmp_list_pop = []

#         # ndcg_tmp_list_pred = []
#         # prec_tmp_list_pred = []

#         # ndcg_tmp_list_rel = []
#         # prec_tmp_list_rel = []

#         # ndcg_tmp_list_pop = []
#         # prec_tmp_list_pop = []


#         # посчитать обычные ndcg/recall по outcome при сортировке 
#         # а) по relevance б) по pred в) по popular (популярность просто по датасету)

#         # test_df = pd.read_csv('/Users/tanyatomayly/Downloads/df_sorted.csv')

#         ndcg_tmp_list_rel = []
#         ndcg_tmp_list_pred = []
#         ndcg_tmp_list_pop = []

#         recall_tmp_list_rel = []
#         recall_tmp_list_pred = []
#         recall_tmp_list_pop = [] 

#         precision_tmp_list_rel = []
#         precision_tmp_list_pred = []
#         precision_tmp_list_pop = []

#         if flag.dataset == 'd' or 'p':
#             for t in range(num_times):
#                 test_df_t = test_df[test_df["idx_time"] == t]
#                 user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
#                 item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
#                 test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
#                 r_pred_test = None
#                 p_pred_test = None

#                 for u, i in test_t_data.batch(5000):
#                     _, p_batch, r_batch, _ = model((u, i), training=False)
#                     # p_batch, r_batch = model((u, i), training=False)
#                     if r_pred_test is None:
#                         r_pred_test = r_batch
#                         p_pred_test = p_batch
#                     else:
#                         r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
#                         p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

#                 p_pred_test = p_pred_test.numpy()
#                 r_pred_test = r_pred_test.numpy()
#                 p_pred_test_t = 0.25 * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
#                 p_pred_test_t = np.clip((p_pred_test_t + 0.5), 0.0, 1.0)

#                 # p_pred_test_t = get_norm(p_pred_test, to_prob=flag.to_prob)
#                 t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)
#                 p_pred_test = p_pred_test * 0.8
#                 r_pred_test = r_pred_test * 0.8
#                 test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
#                 test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
#                 test_df_t["treated_estimate"] = t_test_pred
#                 # outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
#                 # outcome_estimate = get_norm(outcome_estimate, to_prob=flag.to_prob)
#                 # test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
#                 outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
#                 outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
#                 outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
#                 test_df_t["outcome_estimate"] = np.where(outcome_estimate >= 0.7, 1.0, 0.0)

#                 # test_df_t = pd.read_csv('/Users/tanyatomayly/Downloads/df_sorted.csv')
#                 train_df = train_df[train_df.outcome>0]
#                 popularity = train_df["idx_item"].value_counts().reset_index()
#                 popularity.columns = ["idx_item", "popularity"]
#                 test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
#                 test_df_t["pred"] = recommender.predict(test_df_t)
#                 evaluator = Evaluator()

#                 causal_effect_estimate = \
#                     test_df_t["outcome_estimate"] * \
#                     (test_df_t["outcome_estimate"] / test_df_t["propensity_estimate"] - \
#                     (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
#                 test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

#                 # kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 # spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 # pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

#                 kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'popularity', 'relevance_estimate')
#                 spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'popularity', 'relevance_estimate')
#                 pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'popularity', 'relevance_estimate')
#                 print(f"Kendall Tau: {kendall_score:.4f}")
#                 print(f"Spearman Rho: {spearman_score:.4f}")
#                 print(f"Average Rank Position Difference: {pos_diff:.4f}")

#                 # user_item_counts = test_df_t.groupby('idx_user')['idx_item'].count().reset_index()
#                 # user_item_counts.columns = ['idx_user', 'num_items']
#                 # print(user_item_counts['num_items'].describe())

#                 # df_ranking_pop = evaluator.get_ranking_popularity(test_df_t, num_rec=10)
#                 # print(df_ranking_pop.groupby('idx_user')['idx_item'].nunique().describe())

#                 # top_items = test_df_t['idx_item'].value_counts().head(10).index
#                 # print(test_df_t[test_df_t['idx_item'].isin(top_items)].groupby('idx_item')['outcome'].mean())

#                 # print(test_df_t.duplicated(subset=['idx_user', 'idx_item']).sum())


#                 ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
#                 ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
#                 ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

#                 recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
#                 recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
#                 recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

#                 precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
#                 precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
#                 precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

#                 cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
#                 cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
#                 cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

#                 cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
#                 cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
#                 cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

#                 cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
#                 cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
#                 cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

#                 # prec_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'Prec', 10))
#                 # ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCG', 100))

#                 # prec_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecR', 10))
#                 # ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 100000))

#                 # prec_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecP', 10))
#                 # ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 100000))

#                 _ = evaluator.get_sorted(test_df_t)
#         else:
#             for t in [0]:
#                 test_df_t = test_df[test_df["idx_time"] == t]
#                 user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
#                 item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
#                 test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
#                 r_pred_test = None
#                 p_pred_test = None

#                 for u, i in test_t_data.batch(5000):
#                     _, p_batch, r_batch, _ = model((u, i), training=False)
#                     if r_pred_test is None:
#                         r_pred_test = r_batch
#                         p_pred_test = p_batch
#                     else:
#                         r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
#                         p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

#                 p_pred_test = p_pred_test.numpy()
#                 r_pred_test = r_pred_test.numpy()
#                 p_pred_test_t = 0.25 * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
#                 p_pred_test_t = np.clip((p_pred_test_t + 0.5), 0.0, 1.0)

#                 t_test_pred = np.where(p_pred_test_t >= 0.65, 1.0, 0.0)
#                 p_pred_test = p_pred_test * 0.2
#                 r_pred_test = r_pred_test * 0.2
#                 test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
#                 test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
#                 test_df_t["treated_estimate"] = t_test_pred
                
#                 outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
#                 outcome_estimate = 0.25 * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
#                 outcome_estimate = np.clip((outcome_estimate + 0.5), 0.0, 1.0)
#                 test_df_t["outcome_estimate"] = np.where(outcome_estimate >= 0.7, 1.0, 0.0)

#                 causal_effect_estimate = \
#                     test_df_t["outcome_estimate"] * \
#                     (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
#                     (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
#                 test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

#                 train_df = train_df[train_df.outcome>0]
#                 popularity = train_df["idx_item"].value_counts().reset_index()
#                 popularity.columns = ["idx_item", "popularity"]
#                 test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
#                 test_df_t["pred"] = recommender.predict(test_df_t)
#                 evaluator = Evaluator()
#                 cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
#                 cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
#                 cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

#                 ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
#                 ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
#                 ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

#                 recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
#                 recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
#                 recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

#                 precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
#                 precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
#                 precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

#                 cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
#                 cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
#                 cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

#                 cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
#                 cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
#                 cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

#                 kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
#                 pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

#                 print(f"Kendall Tau: {kendall_score:.4f}")
#                 print(f"Spearman Rho: {spearman_score:.4f}")
#                 print(f"Average Rank Position Difference: {pos_diff:.4f}")

#                 _ = evaluator.get_sorted(test_df_t)



#         ndcg_rel = np.mean(ndcg_tmp_list_rel)
#         ndcg_pred = np.mean(ndcg_tmp_list_pred)
#         ndcg_pop = np.mean(ndcg_tmp_list_pop)

#         recall_rel = np.mean(recall_tmp_list_rel)
#         recall_pred = np.mean(recall_tmp_list_pred)
#         recall_pop = np.mean(recall_tmp_list_pop)


#         precision_rel = np.mean(precision_tmp_list_rel)
#         precision_pred = np.mean(precision_tmp_list_pred)
#         precision_pop = np.mean(precision_tmp_list_pop)


#         cp10_pred = np.mean(cp10_tmp_list_pred)
#         cp100_pred = np.mean(cp100_tmp_list_pred)
#         cdcg_pred = np.mean(cdcg_tmp_list_pred)

#         cp10_rel = np.mean(cp10_tmp_list_rel)
#         cp100_rel = np.mean(cp100_tmp_list_rel)
#         cdcg_rel = np.mean(cdcg_tmp_list_rel)

#         cp10_pop = np.mean(cp10_tmp_list_pop)
#         cp100_pop = np.mean(cp100_tmp_list_pop)
#         cdcg_pop = np.mean(cdcg_tmp_list_pop)

#         # prec_pred = np.mean(prec_tmp_list_pred)
#         # ndcg_pred = np.mean(ndcg_tmp_list_pred)

#         # prec_rel = np.mean(prec_tmp_list_rel)
#         # ndcg_rel = np.mean(ndcg_tmp_list_rel)

#         # prec_pop = np.mean(prec_tmp_list_pop)
#         # ndcg_pop = np.mean(ndcg_tmp_list_pop)

#         cp10list_pred.append(cp10_pred)
#         cp100list_pred.append(cp100_pred)
#         cdcglist_pred.append(cdcg_pred)

#         cp10list_rel.append(cp10_rel)
#         cp100list_rel.append(cp100_rel)
#         cdcglist_rel.append(cdcg_rel)

#         cp10list_pop.append(cp10_pop)
#         cp100list_pop.append(cp100_pop)
#         cdcglist_pop.append(cdcg_pop)

#         # prelist_pred.append(prec_pred)
#         # ndcglist_pred.append(ndcg_pred)

#         # prelist_rel.append(prec_rel)
#         # ndcglist_rel.append(ndcg_rel)

#         # prelist_pop.append(prec_pop)
#         # ndcglist_pop.append(ndcg_pop)


#         ndcglist_rel.append(ndcg_rel)
#         ndcglist_pred.append(ndcg_pred)
#         ndcglist_pop.append(ndcg_pop)

#         recalllist_rel.append(recall_rel)
#         recalllist_pred.append(recall_pred)
#         recalllist_pop.append(recall_pop)

#         precisionlist_rel.append(precision_rel)
#         precisionlist_pred.append(precision_pred)
#         precisionlist_pop.append(precision_pop)       

    
#     with open(plotpath+"/result_" + flag.dataset +".txt", "a+") as f:
#         print("NDCG10R:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)
#         print("NDCG10S:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)
#         print("NDCG10P:", np.mean(ndcglist_pop), np.std(ndcglist_pop), file=f)

#         print("Recall10R:", np.mean(recalllist_rel), np.std(recalllist_rel), file=f)
#         print("Recall10S:", np.mean(recalllist_pred), np.std(recalllist_pred), file=f)
#         print("Recall10P:", np.mean(recalllist_pop), np.std(recalllist_pop), file=f)

#         print("Precision10R:", np.mean(precisionlist_rel), np.std(precisionlist_rel), file=f)
#         print("Precision10S:", np.mean(precisionlist_pred), np.std(precisionlist_pred), file=f)
#         print("Precision10P:", np.mean(precisionlist_pop), np.std(precisionlist_pop), file=f)        

#         print("CP10S:", np.mean(cp10list_pred), np.std(cp10list_pred), file=f)
#         print("CP10R:", np.mean(cp10list_rel), np.std(cp10list_rel), file=f)
#         print("CP10P:", np.mean(cp10list_pop), np.std(cp10list_pop), file=f)

#         print("CP100S:", np.mean(cp100list_pred), np.std(cp100list_pred), file=f)
#         print("CP100R:", np.mean(cp100list_rel), np.std(cp100list_rel), file=f)
#         print("CP100P:", np.mean(cp100list_pop), np.std(cp100list_pop), file=f)
        
#         print("CDCGS:", np.mean(cdcglist_pred), np.std(cdcglist_pred), file=f)
#         print("CDCGR:", np.mean(cdcglist_rel), np.std(cdcglist_rel), file=f)
#         print("CDCGP:", np.mean(cdcglist_pop), np.std(cdcglist_pop), file=f)

#         # print("RelP:", np.mean(prelist_pred), np.std(prelist_pred), file=f)
#         # print("NDCGP:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)

#         # print("RelR:", np.mean(prelist_rel), np.std(prelist_rel), file=f)
#         # print("NDCGR:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)

#         # print("RelPop:", np.mean(prelist_rel), np.std(prelist_rel), file=f)
#         # print("NDCGPop:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)

            
# if __name__ == "__main__":
#     physical_devices = tf.config.list_physical_devices('GPU')
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     except:
#         pass
#     main(flag)
