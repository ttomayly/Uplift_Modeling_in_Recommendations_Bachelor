# # import os
# # import datetime
# # import pickle
# # from pathlib import Path

# # import tensorflow as tf
# # import numpy as np
# # from tqdm import tqdm
# # from scipy.stats import kendalltau, pearsonr
# # from sklearn.metrics import mean_squared_error
# # from models import Causal_Model

# # import pandas as pd
# # from pathlib import Path
# # import numpy as np
# # import tensorflow as tf
# # from tqdm import tqdm
# # from models import Causal_Model
# # from scipy.stats import kendalltau
# # from evaluator import Evaluator
# # from sklearn.metrics import mean_absolute_error
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # import random
# # import datetime
# # import os
# # import itertools
# # # from dataset import dh_original, dh_personalized, ml_data

# # plotpath = "./results/"
# # if not os.path.isdir(plotpath):
# #     os.makedirs(plotpath)

# # def save_flag(flag, path):
# #     with open(os.path.join(path, "config.pkl"), "wb") as f:
# #         pickle.dump(flag, f)

# # def load_flag(path):
# #     with open(os.path.join(path, "config.pkl"), "rb") as f:
# #         return pickle.load(f)


# # plotpath = "./results/"
# # if not os.path.isdir(plotpath):
# #     os.makedirs(plotpath)
# # def diff(list1, list2):
# #     return list(set(list2).difference(set(list1)))


# # def sparse_gather(indices, values, selected_indices, axis=0):
# #     """
# #     indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
# #     values:  [ value1,                                 , ..., valuen]
# #     """
# #     mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
# #     to_select = tf.where(mask)[:, 1]
# #     user_item = tf.gather(indices, to_select, axis=0)
# #     user = tf.gather(user_item, 0, axis=1)
# #     item = tf.gather(user_item, 1, axis=1)
# #     values = tf.gather(values, to_select, axis=0)
# #     return user, item, values


# # def count_freq(x):
# #     unique, counts = np.unique(x, return_counts=True)
# #     return np.asarray((unique, counts)).T


# # def prepare_data(flag):
# #     dataset = flag.dataset
# #     data_path = None
# #     if dataset == "d":
# #         print("dunn_cate (original) is used.")
# #         # data = pd.read_csv('dh_original.csv')
# #         # data = dh_original.copy()
# #         data_path = Path("Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/original_rp0.40/result")
# #     elif dataset == "p":
# #         print("dunn_cate (personalized) is used.")
# #         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210")
# #         # data = dh_personalized.copy()
# #         # data = pd.read_csv('dh_personalized.csv')
# #     elif dataset == "ml":
# #         data_path = Path("Uplift_Data/MovieLens/ML_100k_logrank100_offset5.0_scaling1.0")
# #         # data = ml_data.copy()
# #         # data = pd.read_csv('ml.csv')
# #         print("ML-100k is used")
# #     train_data = data_path / "data_train.csv"
# #     vali_data = data_path / "data_vali.csv"
# #     test_data = data_path / "data_test.csv"
# #     train_df = pd.read_csv(train_data)
# #     vali_df = pd.read_csv(vali_data)
# #     test_df = pd.read_csv(test_data)

# #     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# #     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
# #     user_ids = np.sort(
# #         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
# #     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# #     item_ids = np.sort(
# #         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
# #     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
# #     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
# #     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
# #     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
# #     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
# #     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
# #     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
# #     num_users = len(user_ids)
# #     num_items = len(item_ids)
# #     print(num_items)
# #     if dataset == "d" or dataset == "p":
# #         num_times = len(train_df["idx_time"].unique().tolist())
# #     else: 
# #         num_times = 1
# #         train_df["idx_time"] = 0
# #         vali_df["idx_time"] = 0
# #         test_df["idx_time"] = 0
# #     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
# #     train_df_positive = train_df[train_df["outcome"] > 0]
# #     counts = count_freq(train_df_positive['idx_item'].to_numpy())
# #     np_counts = np.zeros(num_items)
# #     print(np_counts.shape)
# #     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

# #     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

# # def load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular):
# #     """
# #     Загружает модель и конфиг из checkpoint_path
# #     """
# #     flag = load_flag(checkpoint_path)
# #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# #     model.load_weights(os.path.join(checkpoint_path, "config.pkl"))
# #     print(f"Model and flag loaded from {checkpoint_path}")
# #     return model, flag


# # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):

# #     checkpoint_path = "results/default/"  # путь к нужной папке с моделью
# #     model, flag = load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular)
# #     # model = Causal_Model(num_users, num_items, flag, None, None, popular)
# #     # optim_val_car = -float('inf')

# #     # experiment_name = flag.add if hasattr(flag, 'add') else f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# #     # save_path = os.path.join(plotpath, experiment_name)

# #     # if not os.path.exists(save_path):
# #     #     os.makedirs(save_path)

# #     # print(f"Training and saving to: {save_path}")

# #     # for epoch in range(flag.epoch):
# #     # for epoch in range(2):
# #     #     print(f"\nEpoch {epoch+1}/{flag.epoch}: Sampling negative items...")
# #     #     j_list = [
# #     #         np.random.choice([j for j in range(num_items) if j != i])
# #     #         for i in train_df["idx_item"].to_numpy()
# #     #     ]

# #     #     train_data = tf.data.Dataset.from_tensor_slices((
# #     #         train_df["idx_user"].to_numpy(),
# #     #         train_df["idx_item"].to_numpy(),
# #     #         np.array(j_list),
# #     #         train_df["outcome"].to_numpy()
# #     #     ))

# #     #     with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# #     #         t.set_description(f'Training Epoch {epoch}')
# #     #         for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# #     #             _ = model.propensity_train((user, item, item_j, value))
# #     #             t.update()

# #         # # Валидация
# #         # vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# #         # p_pred = []

# #         # for u, i in vali_data.batch(5000):
# #         #     _, p_batch, _, _ = model((u, i), training=False)
# #         #     p_pred.append(p_batch)

# #         # p_pred = tf.concat(p_pred, axis=0).numpy().squeeze()
# #         # p_true = vali_df["propensity"].to_numpy().squeeze()
# #         # p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred) + 1e-8)

# #     #     tau_res, _ = kendalltau(p_pred, p_true)
# #     #     mse = mean_squared_error(p_true, p_pred)

# #     #     print(f"Kendall's tau: {tau_res:.4f}, MSE: {mse:.4f}")

# #     #     if abs(tau_res) > optim_val_car:
# #     #         optim_val_car = abs(tau_res)
# #     #         model.save_weights(os.path.join(save_path, ".weights.h5"))
# #     #         save_flag(flag, save_path)
# #     #         print(f"\nModel weights and config saved in {save_path}")

# #     # # Загрузка лучших весов после обучения
# #     # model.load_weights(os.path.join(save_path, ".weights.h5"))
# #     # print("\nBest model weights loaded for evaluation or further training.")

# #     return model


# import pandas as pd
# from pathlib import Path
# import numpy as np
# import tensorflow as tf
# from tqdm import tqdm
# from models_new import Causal_Model
# # from models_new2 import Causal_Model
# # from CJBR import CJBPR
# from scipy.stats import kendalltau
# from evaluator import Evaluator
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import random
# import datetime
# import os
# import itertools
# # from dataset import dh_original, dh_personalized, ml_data

# plotpath = "./results/"
# if not os.path.isdir(plotpath):
#     os.makedirs(plotpath)
# def diff(list1, list2):
#     return list(set(list2).difference(set(list1)))


# def sparse_gather(indices, values, selected_indices, axis=0):
#     """
#     indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
#     values:  [ value1,                                 , ..., valuen]
#     """
#     mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
#     to_select = tf.where(mask)[:, 1]
#     user_item = tf.gather(indices, to_select, axis=0)
#     user = tf.gather(user_item, 0, axis=1)
#     item = tf.gather(user_item, 1, axis=1)
#     values = tf.gather(values, to_select, axis=0)
#     return user, item, values


# def count_freq(x):
#     unique, counts = np.unique(x, return_counts=True)
#     return np.asarray((unique, counts)).T



# def prepare_data(flag):
#     dataset = flag.dataset
#     data_path = None
#     if dataset == "d":
#         print("dunn_cate (original) is used.")
#         # data = pd.read_csv('dh_original.csv')
#         # data = dh_original.copy()
#         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40")
#     elif dataset == "p":
#         print("dunn_cate (personalized) is used.")
#         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210")
#         # data = dh_personalized.copy()
#         # data = pd.read_csv('dh_personalized.csv')
#     elif dataset == "ml":
#         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/synthetic/ML_100k_logrank100_offset5.0_scaling1.0")
#         # data = ml_data.copy()
#         # data = pd.read_csv('ml.csv')
#         print("ML-100k is used")
#     train_data = data_path / "data_train.csv"
#     vali_data = data_path / "data_vali.csv"
#     test_data = data_path / "data_test.csv"
#     train_df = pd.read_csv(train_data)
#     vali_df = pd.read_csv(vali_data)
#     test_df = pd.read_csv(test_data)

#     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
#     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
#     user_ids = np.sort(
#         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
#     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
#     item_ids = np.sort(
#         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
#     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
#     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
#     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
#     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
#     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
#     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
#     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
#     num_users = len(user_ids)
#     num_items = len(item_ids)
#     print(num_items)
#     if dataset == "d" or dataset == "p":
#         num_times = len(train_df["idx_time"].unique().tolist())
#     else: 
#         num_times = 1
#         train_df["idx_time"] = 0
#         vali_df["idx_time"] = 0
#         test_df["idx_time"] = 0
#     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
#     train_df_positive = train_df[train_df["outcome"] > 0]
#     counts = count_freq(train_df_positive['idx_item'].to_numpy())
#     np_counts = np.zeros(num_items)
#     print(np_counts.shape)
#     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

#     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

# def snips_estimate(y, z_hat, p_hat):
#         """
#         SNIPS оценка uplift
#         """
#         p_hat = tf.clip_by_value(p_hat, 0.0001, 0.9999)
#         treated = tf.cast(z_hat == 1, tf.float32)
#         control = tf.cast(z_hat == 0, tf.float32)

#         y_treated = treated * y / p_hat
#         denom_treated = treated / p_hat

#         y_control = control * y / (1.0 - p_hat)
#         denom_control = control / (1.0 - p_hat)

#         tau_snips = (
#             tf.reduce_sum(y_treated) / tf.reduce_sum(denom_treated)
#             - tf.reduce_sum(y_control) / tf.reduce_sum(denom_control)
#         )
#         return tau_snips

# def evaluate_snips_propcare(model, dataset, threshold=0.2):
#         """
#         Оценка модели PropCare через SNIPS uplift

#         dataset: tf.data.Dataset или генератор, выдающий (user, item, y_true)
#         threshold: порог на нормализованное propensity
#         """
#         all_y, all_z_hat, all_p_hat = [], [], []

#         for batch in dataset:
#             user, item, y_true = batch
#             _, p_hat, _, _ = model((user, item), training=False)

#             # Z-скор нормализация
#             norm_p = (p_hat - tf.reduce_mean(p_hat)) / (tf.math.reduce_std(p_hat) + 1e-6)
#             z_hat = tf.cast(norm_p >= threshold, tf.float32)

#             all_y.append(tf.reshape(y_true, [-1]))
#             all_z_hat.append(tf.reshape(z_hat, [-1]))
#             all_p_hat.append(tf.reshape(p_hat, [-1]))

#         y_all = tf.concat(all_y, axis=0)
#         z_hat_all = tf.concat(all_z_hat, axis=0)
#         p_hat_all = tf.concat(all_p_hat, axis=0)

#         uplift_snips = snips_estimate(y_all, z_hat_all, p_hat_all)
#         print(f"[SNIPS Evaluation] Estimated Uplift: {uplift_snips.numpy():.4f}")
#         return uplift_snips.numpy()


# def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
#     from scipy.stats import kendalltau, pearsonr
#     from sklearn.metrics import mean_absolute_error, mean_squared_error

#     model = Causal_Model(num_users, num_items, flag, None, None, popular)
#     sample_user = tf.constant([0]) 
#     sample_item = tf.constant([0]) 
#     _ = model((sample_user, sample_item))  # This builds all layers 
#     model.load_weights("/Users/tanyatomayly/Desktop/PropCare-main/results/.weights.h5")
#     # # Сохранение весов после обучения
    
#     # # model.save_weights("./results/default/.weights.h5")  # Замени путь на нужный

#     # optim_val_car = 0
#     # train_df = train_df[train_df["outcome"] > 0]
#     # for epoch in range(flag.epoch):
#     # # for epoch in range(1):
#     #     print("Sampling negative items...", end=" ")
#     #     j_list = []
#     #     for i in train_df["idx_item"].to_numpy():
#     #         j = np.random.randint(0, num_items)
#     #         while j == i:
#     #             j = np.random.randint(0, num_items)
#     #         j_list.append(j)
#     #     print("Done")
#     #     j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
#     #     train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
#     #     with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
#     #         t.set_description('Training Epoch %i' % epoch)
#     #         for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
#     #             step = model.propensity_train((user, item, item_j, value))
#     #             t.update()
#     #     vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
#     #     gamma_pred = None
#     #     p_pred = None
#     #     r_pred = None
#     #     for u, i in vali_data.batch(5000):
#     #         gamma_batch, p_batch, r_batch, _ = model((u, i), training=False)
#     #         if gamma_pred is None:
#     #             gamma_pred = gamma_batch
#     #         else:
#     #             gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
#     #         if p_pred is None:
#     #             p_pred = p_batch
#     #         else:
#     #             p_pred = tf.concat((p_pred, p_batch), axis=0)
#     #         if r_pred is None:
#     #             r_pred = r_batch
#     #         else:
#     #             r_pred = tf.concat((r_pred, r_batch), axis=0)

                
#     #     # p_pred = tf.reshape(p_pred, [-1])
#     #     # y_true = tf.convert_to_tensor(vali_df["outcome"].to_numpy(), dtype=tf.float32)

#     #     # norm_p = (p_pred - tf.reduce_mean(p_pred)) / (tf.math.reduce_std(p_pred) + 1e-6)
#     #     # z_hat = tf.cast(norm_p >= 0.2, tf.float32)

#     #     # uplift_snips = snips_estimate(y_true, z_hat, p_pred).numpy()
#     #     # print(f"[SNIPS uplift estimate] τ = {uplift_snips:.4f}")

#     #     p_true = np.squeeze(vali_df["propensity"].to_numpy())
#     #     p_pred = np.squeeze(p_pred.numpy())
#     #     p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
#     #     # p_pred = np.squeeze(p_pred)
#     #     # p_range = np.max(p_pred) - np.min(p_pred)
#     #     # if p_range == 0:
#     #     #     p_pred = np.zeros_like(p_pred)
#     #     # else:
#     #     #     p_pred = (p_pred - np.min(p_pred)) / p_range

#     #     # if np.any(np.isnan(p_pred)):
#     #     #     print("Warning: NaNs in predicted propensity! Replacing with 0.")
#     #     #     p_pred = np.nan_to_num(p_pred, nan=0.0)

#     #     tau_res, _ = kendalltau(p_pred, p_true)
#     #     pearsonres, _ = pearsonr(p_pred, p_true)
#     #     mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
#     #     val_obj = tau_res
#     #     if val_obj > optim_val_car:
#     #         optim_val_car = val_obj
#     #         if not os.path.isdir(plotpath+ '/' + flag.add):
#     #             os.makedirs(plotpath+ '/' + flag.add)
#     #         model.save_weights(plotpath + flag.add + "/.weights.h5")
#     #         print("Model saved!")
#     #         print(plotpath + flag.add + "/.weights.h5")
#     # model.load_weights(plotpath + flag.add + "/.weights.h5")
#     return model

# if __name__ == "__main__":
#     pass

# # HERE


# # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
# #     from scipy.stats import kendalltau, pearsonr
# #     from sklearn.metrics import mean_absolute_error, mean_squared_error

# #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# #     optim_val_car = -float('inf')  # Убедитесь, что начальное значение минимальное
# #     # train_df = train_df[train_df["outcome"] > 0]
# #     for epoch in range(flag.epoch):
# #         print("Sampling negative items...", end=" ")
# #         j_list = []
# #         for i in train_df["idx_item"].to_numpy():
# #             j = np.random.randint(0, num_items)
# #             while j == i:
# #                 j = np.random.randint(0, num_items)
# #             j_list.append(j)
# #         print("Done")
# #         j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
# #         train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
# #         with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# #             t.set_description('Training Epoch %i' % epoch)
# #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# #                 step = model.propensity_train((user, item, item_j, value))
# #                 t.update()
# #         vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# #         gamma_pred = None
# #         p_pred = None
# #         for u, i in vali_data.batch(5000):
# #             gamma_batch, p_batch, _, _ = model((u, i), training=False)
# #             if gamma_pred is None:
# #                 gamma_pred = gamma_batch
# #             else:
# #                 gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
# #             if p_pred is None:
# #                 p_pred = p_batch
# #             else:
# #                 p_pred = tf.concat((p_pred, p_batch), axis=0)
# # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # #         p_pred = np.squeeze(p_pred.numpy())
# # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # #         tau_res, _ = kendalltau(p_pred, p_true)
# # #         # pearsonres, _ = pearsonr(p_pred, p_true)
# # #         # mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
# # #         val_obj = tau_res

# # #         save_path = os.path.join(plotpath, flag.add)

# # #         if not os.path.exists(save_path):
# # #             os.makedirs(save_path)

# # #         if abs(val_obj) > optim_val_car:
# # #             optim_val_car = val_obj
# # #     #         if not os.path.isdir(plotpath+ '/' + flag.add):
# # #     #             os.makedirs(plotpath+ '/' + flag.add)
# # #     #         # model.save_weights(plotpath+ '/' + flag.add + "/saved_model")
# # #     #         model.save_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # #     #         print("Model saved!")
# # #     # # model.load_weights(plotpath+ '/' + flag.add + "/saved_model")
# # #     # model.load_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # #             model.save_weights(os.path.join(save_path, "saved_model.keras"))
# # #             print("Model weights saved!")

# # #             if os.path.isfile(os.path.join(save_path, "saved_model.keras")):
# # #                 print("Weights file verified.")
# # #             else:
# # #                 raise FileNotFoundError("Saved weights file not found.")

# # #         model.load_weights(os.path.join(save_path, "saved_model.keras"))
# # #         print("Weights successfully loaded.")


# # #     return model

# # # if __name__ == "__main__":
# # #     pass

# # # import os
# # # from scipy.stats import kendalltau


# #         # Нормализация и вычисление метрик
# #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# #         p_pred = np.squeeze(p_pred.numpy())
# #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# #         tau_res, _ = kendalltau(p_pred, p_true)
# #         val_obj = tau_res

# #         # Формируем уникальное имя поддиректории для сохранения весов
# #         experiment_name = f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# #         save_path = os.path.join(plotpath, experiment_name)
# #         print(f"Сохранение модели в путь: {save_path}")

# #         # Проверяем и создаем директорию, если она отсутствует
# #         if not os.path.exists(save_path):
# #             os.makedirs(save_path)
# #             print(f"Директория {save_path} создана.")

# #         # Определяем путь для сохранения весов
# #         weights_file = os.path.join(save_path, ".weights.h5")

# #         # Сохраняем лучшие веса модели, если текущая метрика лучше
# #         # if abs(val_obj) > optim_val_car:
# #         optim_val_car = val_obj
            
# #             # Сохранение весов модели
# #         model.save_weights(weights_file)
# #         print(f"Model weights saved at: {weights_file}")

# #         # Проверка существования файла перед загрузкой
# #         if os.path.isfile(weights_file):
# #             # Загружаем веса модели
# #             model.load_weights(weights_file)
# #             print("Weights successfully loaded.")
# #         else:
# #             raise FileNotFoundError(f"Файл весов {weights_file} не найден!")

# import os
# import datetime
# import pickle
# from pathlib import Path

# import tensorflow as tf
# import numpy as np
# from tqdm import tqdm
# from scipy.stats import kendalltau, pearsonr
# from sklearn.metrics import mean_squared_error
# from models import Causal_Model

# import pandas as pd
# from pathlib import Path
# import numpy as np
# import tensorflow as tf
# from tqdm import tqdm
# from models import Causal_Model
# from scipy.stats import kendalltau
# from evaluator import Evaluator
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import random
# import datetime
# import os
# import itertools
# # from dataset import dh_original, dh_personalized, ml_data

# plotpath = "./results/"
# if not os.path.isdir(plotpath):
#     os.makedirs(plotpath)

# def save_flag(flag, path):
#     with open(os.path.join(path, "config.pkl"), "wb") as f:
#         pickle.dump(flag, f)

# def load_flag(path):
#     with open(os.path.join(path, "config.pkl"), "rb") as f:
#         return pickle.load(f)


# plotpath = "./results/"
# if not os.path.isdir(plotpath):
#     os.makedirs(plotpath)
# def diff(list1, list2):
#     return list(set(list2).difference(set(list1)))


# def sparse_gather(indices, values, selected_indices, axis=0):
#     """
#     indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
#     values:  [ value1,                                 , ..., valuen]
#     """
#     mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
#     to_select = tf.where(mask)[:, 1]
#     user_item = tf.gather(indices, to_select, axis=0)
#     user = tf.gather(user_item, 0, axis=1)
#     item = tf.gather(user_item, 1, axis=1)
#     values = tf.gather(values, to_select, axis=0)
#     return user, item, values


# def count_freq(x):
#     unique, counts = np.unique(x, return_counts=True)
#     return np.asarray((unique, counts)).T


# def prepare_data(flag):
#     dataset = flag.dataset
#     data_path = None
#     if dataset == "d":
#         print("dunn_cate (original) is used.")
#         # data = pd.read_csv('dh_original.csv')
#         # data = dh_original.copy()
#         data_path = Path("Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/original_rp0.40/result")
#     elif dataset == "p":
#         print("dunn_cate (personalized) is used.")
#         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210")
#         # data = dh_personalized.copy()
#         # data = pd.read_csv('dh_personalized.csv')
#     elif dataset == "ml":
#         data_path = Path("Uplift_Data/MovieLens/ML_100k_logrank100_offset5.0_scaling1.0")
#         # data = ml_data.copy()
#         # data = pd.read_csv('ml.csv')
#         print("ML-100k is used")
#     train_data = data_path / "data_train.csv"
#     vali_data = data_path / "data_vali.csv"
#     test_data = data_path / "data_test.csv"
#     train_df = pd.read_csv(train_data)
#     vali_df = pd.read_csv(vali_data)
#     test_df = pd.read_csv(test_data)

#     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
#     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
#     user_ids = np.sort(
#         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
#     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
#     item_ids = np.sort(
#         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
#     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
#     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
#     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
#     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
#     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
#     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
#     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
#     num_users = len(user_ids)
#     num_items = len(item_ids)
#     print(num_items)
#     if dataset == "d" or dataset == "p":
#         num_times = len(train_df["idx_time"].unique().tolist())
#     else: 
#         num_times = 1
#         train_df["idx_time"] = 0
#         vali_df["idx_time"] = 0
#         test_df["idx_time"] = 0
#     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
#     train_df_positive = train_df[train_df["outcome"] > 0]
#     counts = count_freq(train_df_positive['idx_item'].to_numpy())
#     np_counts = np.zeros(num_items)
#     print(np_counts.shape)
#     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

#     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

# def load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular):
#     """
#     Загружает модель и конфиг из checkpoint_path
#     """
#     flag = load_flag(checkpoint_path)
#     model = Causal_Model(num_users, num_items, flag, None, None, popular)
#     model.load_weights(os.path.join(checkpoint_path, "config.pkl"))
#     print(f"Model and flag loaded from {checkpoint_path}")
#     return model, flag


# def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):

#     checkpoint_path = "results/default/"  # путь к нужной папке с моделью
#     model, flag = load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular)
#     # model = Causal_Model(num_users, num_items, flag, None, None, popular)
#     # optim_val_car = -float('inf')

#     # experiment_name = flag.add if hasattr(flag, 'add') else f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     # save_path = os.path.join(plotpath, experiment_name)

#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)

#     # print(f"Training and saving to: {save_path}")

#     # for epoch in range(flag.epoch):
#     # for epoch in range(2):
#     #     print(f"\nEpoch {epoch+1}/{flag.epoch}: Sampling negative items...")
#     #     j_list = [
#     #         np.random.choice([j for j in range(num_items) if j != i])
#     #         for i in train_df["idx_item"].to_numpy()
#     #     ]

#     #     train_data = tf.data.Dataset.from_tensor_slices((
#     #         train_df["idx_user"].to_numpy(),
#     #         train_df["idx_item"].to_numpy(),
#     #         np.array(j_list),
#     #         train_df["outcome"].to_numpy()
#     #     ))

#     #     with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
#     #         t.set_description(f'Training Epoch {epoch}')
#     #         for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
#     #             _ = model.propensity_train((user, item, item_j, value))
#     #             t.update()

#         # # Валидация
#         # vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
#         # p_pred = []

#         # for u, i in vali_data.batch(5000):
#         #     _, p_batch, _, _ = model((u, i), training=False)
#         #     p_pred.append(p_batch)

#         # p_pred = tf.concat(p_pred, axis=0).numpy().squeeze()
#         # p_true = vali_df["propensity"].to_numpy().squeeze()
#         # p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred) + 1e-8)

#     #     tau_res, _ = kendalltau(p_pred, p_true)
#     #     mse = mean_squared_error(p_true, p_pred)

#     #     print(f"Kendall's tau: {tau_res:.4f}, MSE: {mse:.4f}")

#     #     if abs(tau_res) > optim_val_car:
#     #         optim_val_car = abs(tau_res)
#     #         model.save_weights(os.path.join(save_path, ".weights.h5"))
#     #         save_flag(flag, save_path)
#     #         print(f"\nModel weights and config saved in {save_path}")

#     # # Загрузка лучших весов после обучения
#     # model.load_weights(os.path.join(save_path, ".weights.h5"))
#     # print("\nBest model weights loaded for evaluation or further training.")

#     return model


import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# from models_new import Causal_Model
# from models_new2 import Causal_Model
# from CJBR import CJBPR
# from scipy.stats import kendalltau
from evaluator import Evaluator
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import datetime
import os
import itertools
# from dataset import dh_original, dh_personalized, ml_data

plotpath = "./results/"
if not os.path.isdir(plotpath):
    os.makedirs(plotpath)
def diff(list1, list2):
    return list(set(list2).difference(set(list1)))


def sparse_gather(indices, values, selected_indices, axis=0):
    """
    indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
    values:  [ value1,                                 , ..., valuen]
    """
    mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
    to_select = tf.where(mask)[:, 1]
    user_item = tf.gather(indices, to_select, axis=0)
    user = tf.gather(user_item, 0, axis=1)
    item = tf.gather(user_item, 1, axis=1)
    values = tf.gather(values, to_select, axis=0)
    return user, item, values


def count_freq(x):
    unique, counts = np.unique(x, return_counts=True)
    return np.asarray((unique, counts)).T

def preprocess_data(datapath, dataset):
    train_data = datapath / "data_train.csv"
    vali_data = datapath / "data_vali.csv"
    test_data = datapath / "data_test.csv"
    train_df = pd.read_csv(train_data)
    vali_df = pd.read_csv(vali_data)
    test_df = pd.read_csv(test_data)

    # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
    user_ids = np.sort(
        pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    item_ids = np.sort(
        pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
    item2item_encoded = {x: i for i, x in enumerate(item_ids)}
    train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
    train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
    vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
    vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
    test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
    test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
    num_users = len(user_ids)
    num_items = len(item_ids)
    if dataset == "d" or dataset == "p":
        num_times = len(train_df["idx_time"].unique().tolist())
    else: 
        num_times = 1
        train_df["idx_time"] = 0
        vali_df["idx_time"] = 0
        test_df["idx_time"] = 0
    train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
    return train_df, vali_df, test_df, num_users, num_items, num_times

def prepare_data(flag):
    dataset = flag.dataset[-1]
    data_path = None
    if flag.dataset == "1d":
        print("dunn_cate (original) 1 week is used.")
        data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/original_rp0.40")
        train_df, vali_df, test_df, _, _, _ = preprocess_data(data_path, dataset)
        data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/1week/original_rp0.40")
        _, _, _, num_users, num_items, num_times = preprocess_data(data_path, dataset)
        train_df['personal_popular'] = train_df.groupby(['idx_user', 'idx_item'])['outcome'].transform('sum')
        train_df = train_df[train_df['idx_time'] == 0]
        df_merge = train_df[['idx_user', 'idx_item', 'personal_popular']]
        vali_df = vali_df[vali_df['idx_time'] == 0]
        test_df = test_df[test_df['idx_time'] == 0]
        vali_df = pd.merge(vali_df, df_merge, on=['idx_user', 'idx_item'], how='left')
        test_df = pd.merge(test_df, df_merge, on=['idx_user', 'idx_item'], how='left')
    elif flag.dataset == "10d":
        print("dunn_cate (original) 10 weeks is used.")
        data_path = Path("c/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/original_rp0.40")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    elif flag.dataset == "1p":
        print("dunn_cate (personalized) is 1 week used.")
        data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/rank_rp0.40_sf1.00_nr210")
        train_df, vali_df, test_df, _, _, _ = preprocess_data(data_path, dataset)
        data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/1week/rank_rp0.40_sf1.00_nr210")
        _, _, _, num_users, num_items, num_times = preprocess_data(data_path, dataset)
        train_df['personal_popular'] = train_df.groupby(['idx_user', 'idx_item'])['outcome'].transform('sum')
        train_df = train_df[train_df['idx_time'] == 0]
        df_merge = train_df[['idx_user', 'idx_item', 'personal_popular']]
        vali_df = vali_df[vali_df['idx_time'] == 0]
        test_df = test_df[test_df['idx_time'] == 0]
        vali_df = pd.merge(vali_df, df_merge, on=['idx_user', 'idx_item'], how='left')
        test_df = pd.merge(test_df, df_merge, on=['idx_user', 'idx_item'], how='left')
    elif flag.dataset == "10p":
        print("dunn_cate (personalized) 10 weeks is used.")
        data_path = Path("./CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/rank_rp0.40_sf2.00_nr210")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    elif flag.dataset == "ml":
        print("ML-100k is used")
        data_path = Path("./CausalNBR/data/synthetic/ML_100k_logrank100_offset5.0_scaling1.0")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    
    train_df_positive = train_df[train_df["outcome"] > 0]
    counts = count_freq(train_df_positive['idx_item'].to_numpy())
    np_counts = np.zeros(num_items)
    np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

    return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts
    

# def prepare_data(flag):
#     dataset = flag.dataset
#     data_path = None
#     if dataset == "d":
#         print("dunn_cate (original) is used.")
#         # data = pd.read_csv('dh_original.csv')
#         # data = dh_original.copy()
#         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/1week/original_rp0.40")
#     elif dataset == "p":
#         print("dunn_cate (personalized) is used.")
#         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/1week/rank_rp0.40_sf1.00_nr210")
#         # data = dh_personalized.copy()
#         # data = pd.read_csv('dh_personalized.csv')
#     elif dataset == "ml":
#         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/synthetic/ML_100k_logrank100_offset5.0_scaling1.0")
#         # data = ml_data.copy()
#         # data = pd.read_csv('ml.csv')
#         print("ML-100k is used")
#     train_data = data_path / "data_train.csv"
#     vali_data = data_path / "data_vali.csv"
#     test_data = data_path / "data_test.csv"
#     train_df = pd.read_csv(train_data)
#     vali_df = pd.read_csv(vali_data)
#     test_df = pd.read_csv(test_data)

#     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
#     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
#     user_ids = np.sort(
#         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
#     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
#     item_ids = np.sort(
#         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
#     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
#     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
#     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
#     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
#     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
#     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
#     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
#     num_users = len(user_ids)
#     num_items = len(item_ids)
#     print(num_items)
#     if dataset == "d" or dataset == "p":
#         num_times = len(train_df["idx_time"].unique().tolist())
#     else: 
#         num_times = 1
#         train_df["idx_time"] = 0
#         vali_df["idx_time"] = 0
#         test_df["idx_time"] = 0
#     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
#     train_df_positive = train_df[train_df["outcome"] > 0]
#     counts = count_freq(train_df_positive['idx_item'].to_numpy())
#     np_counts = np.zeros(num_items)
#     print(np_counts.shape)
#     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

#     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

def snips_estimate(y, z_hat, p_hat):
        """
        SNIPS оценка uplift
        """
        p_hat = tf.clip_by_value(p_hat, 0.0001, 0.9999)
        treated = tf.cast(z_hat == 1, tf.float32)
        control = tf.cast(z_hat == 0, tf.float32)

        y_treated = treated * y / p_hat
        denom_treated = treated / p_hat

        y_control = control * y / (1.0 - p_hat)
        denom_control = control / (1.0 - p_hat)

        tau_snips = (
            tf.reduce_sum(y_treated) / tf.reduce_sum(denom_treated)
            - tf.reduce_sum(y_control) / tf.reduce_sum(denom_control)
        )
        return tau_snips

def evaluate_snips_propcare(model, dataset, threshold=0.2):
        """
        Оценка модели PropCare через SNIPS uplift

        dataset: tf.data.Dataset или генератор, выдающий (user, item, y_true)
        threshold: порог на нормализованное propensity
        """
        all_y, all_z_hat, all_p_hat = [], [], []

        for batch in dataset:
            user, item, y_true = batch
            _, p_hat, _, _ = model((user, item), training=False)

            # Z-скор нормализация
            norm_p = (p_hat - tf.reduce_mean(p_hat)) / (tf.math.reduce_std(p_hat) + 1e-6)
            z_hat = tf.cast(norm_p >= threshold, tf.float32)

            all_y.append(tf.reshape(y_true, [-1]))
            all_z_hat.append(tf.reshape(z_hat, [-1]))
            all_p_hat.append(tf.reshape(p_hat, [-1]))

        y_all = tf.concat(all_y, axis=0)
        z_hat_all = tf.concat(all_z_hat, axis=0)
        p_hat_all = tf.concat(all_p_hat, axis=0)

        uplift_snips = snips_estimate(y_all, z_hat_all, p_hat_all)
        print(f"[SNIPS Evaluation] Estimated Uplift: {uplift_snips.numpy():.4f}")
        return uplift_snips.numpy()

from itertools import product
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# from scipy.stats import kendalltau
from itertools import product
from models_new import Causal_Model  # путь уточни при необходимости
# from models_old import Causal_Model  # путь уточни при необходимости

# def train_propensity_with_search(train_df, vali_df, flag, num_users, num_items, popular, plotpath="./results"):
#     """
#     Подбирает гиперпараметры PropCare по валидации на каждом шаге (валидация внутри эпох),
#     сохраняет веса только один раз — для лучшего результата.
#     """
#     param_grid = {
#         'lambda_1': [1.0, 10.0],
#         'lambda_2': [0.1, 1.0],
#         'lambda_3': [0.1, 1.0],
#         'dimension': [64, 128],
#     }

#     best_score = -float("inf")
#     best_flag_values = None
#     best_model_path = os.path.join(plotpath, "best_prop_weights.h5")

#     keys = list(param_grid.keys())
#     combinations = list(product(*param_grid.values()))

#     for values in combinations:
#         param_combo = dict(zip(keys, values))
#         flag.lambda_1 = param_combo['lambda_1']
#         flag.lambda_2 = param_combo['lambda_2']
#         flag.lambda_3 = param_combo['lambda_3']
#         flag.dimension = param_combo['dimension']

#         model = Causal_Model(num_users, num_items, flag, None, None, popular)
#         _ = model((tf.constant([0]), tf.constant([0])))

#         for epoch in range(3):
#             j_list = []
#             for i in train_df["idx_item"].to_numpy():
#                 j = np.random.randint(0, num_items)
#                 while j == i:
#                     j = np.random.randint(0, num_items)
#                 j_list.append(j)

#             train_data = tf.data.Dataset.from_tensor_slices((
#                 train_df["idx_user"].to_numpy(),
#                 train_df["idx_item"].to_numpy(),
#                 j_list,
#                 train_df["outcome"].to_numpy()
#             ))

#             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
#                 model.propensity_train((user, item, item_j, value))

#             # Валидация после каждой эпохи
#             vali_data = tf.data.Dataset.from_tensor_slices(
#                 (vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy())
#             )
#             p_pred = None
#             for u, i in vali_data.batch(5000):
#                 _, p_batch, _, _ = model((u, i), training=False)
#                 p_pred = tf.concat([p_pred, p_batch], axis=0) if p_pred is not None else p_batch

#             p_true = np.squeeze(vali_df["propensity"].to_numpy())
#             p_pred = np.squeeze(p_pred.numpy())
#             p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred) + 1e-8)

#             tau_res, _ = kendalltau(p_pred, p_true)
#             print(f"[τ = {tau_res:.4f}] Epoch {epoch} for params: {param_combo}")

#             if tau_res > best_score:
#                 best_score = tau_res
#                 best_flag_values = param_combo.copy()
#                 model.save_weights(best_model_path)
#                 print(f"✅ New best model saved with τ = {tau_res:.4f}")

#     return best_flag_values, best_score

# def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular, plotpath="./results"):
#     print("🚀 Подбор гиперпараметров PropCare...")
#     best_flag_values, best_score = train_propensity_with_search(train_df, vali_df, flag, num_users, num_items, popular, plotpath)

#     # Установим лучшие значения
#     flag.lambda_1 = best_flag_values['lambda_1']
#     flag.lambda_2 = best_flag_values['lambda_2']
#     flag.lambda_3 = best_flag_values['lambda_3']
#     flag.dimension = best_flag_values['dimension']

#     # Создаём и загружаем модель с лучшими весами
#     model = Causal_Model(num_users, num_items, flag, None, None, popular)
#     _ = model((tf.constant([0]), tf.constant([0])))
#     model.load_weights(os.path.join(plotpath, "best_prop_weights.h5"))

#     return model




def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
    from scipy.stats import kendalltau, pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model = Causal_Model(num_users, num_items, flag, None, None, popular)
    sample_user = tf.constant([0]) 
    sample_item = tf.constant([0]) 
    _ = model((sample_user, sample_item))  # This builds all layers 
    # model.load_weights("/Users/tanyatomayly/Desktop/PropCare-main/results/default/.weights.h5")
    model.load_weights("vesa_orig_dr/.weights.h5")
    
    # model.save_weights("source myenv/bin/activate./results/default/.weights.h5")  # Замени путь на нужный

    # optim_val_car = 0
    # # train_df = train_df[train_df["outcome"] > 0]
    # for epoch in range(flag.epoch):
    # # for epoch in range(1):
    #     print("Sampling negative items...", end=" ")
    #     j_list = []
    #     for i in train_df["idx_item"].to_numpy():
    #         j = np.random.randint(0, num_items)
    #         while j == i:
    #             j = np.random.randint(0, num_items)
    #         j_list.append(j)
    #     print("Done")
    #     j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
    #     train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
    #     with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
    #         t.set_description('Training Epoch %i' % epoch)
    #         for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
    #             step = model.propensity_train((user, item, item_j, value))
    #             t.update()
    #     vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
    #     gamma_pred = None
    #     p_pred = None
    #     r_pred = None
    #     for u, i in vali_data.batch(5000):
    #         gamma_batch, p_batch, r_batch, _ = model((u, i), training=False)
    #         if gamma_pred is None:
    #             gamma_pred = gamma_batch
    #         else:
    #             gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
    #         if p_pred is None:
    #             p_pred = p_batch
    #         else:
    #             p_pred = tf.concat((p_pred, p_batch), axis=0)
    #         if r_pred is None:
    #             r_pred = r_batch
    #         else:
    #             r_pred = tf.concat((r_pred, r_batch), axis=0)

                
    #     # p_pred = tf.reshape(p_pred, [-1])
    #     # y_true = tf.convert_to_tensor(vali_df["outcome"].to_numpy(), dtype=tf.float32)

    #     # norm_p = (p_pred - tf.reduce_mean(p_pred)) / (tf.math.reduce_std(p_pred) + 1e-6)
    #     # z_hat = tf.cast(norm_p >= 0.2, tf.float32)

    #     # uplift_snips = snips_estimate(y_true, z_hat, p_pred).numpy()
    #     # print(f"[SNIPS uplift estimate] τ = {uplift_snips:.4f}")

    #     p_true = np.squeeze(vali_df["propensity"].to_numpy())
    #     p_pred = np.squeeze(p_pred.numpy())
    #     p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
    #     # p_pred = np.squeeze(p_pred)
    #     # p_range = np.max(p_pred) - np.min(p_pred)
    #     # if p_range == 0:
    #     #     p_pred = np.zeros_like(p_pred)
    #     # else:
    #     #     p_pred = (p_pred - np.min(p_pred)) / p_range

    #     # if np.any(np.isnan(p_pred)):
    #     #     print("Warning: NaNs in predicted propensity! Replacing with 0.")
    #     #     p_pred = np.nan_to_num(p_pred, nan=0.0)

    #     tau_res, _ = kendalltau(p_pred, p_true)
    #     pearsonres, _ = pearsonr(p_pred, p_true)
    #     mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
    #     val_obj = tau_res
    #     if val_obj > optim_val_car:
    #         print(tau_res)
    #         optim_val_car = val_obj
    #         if not os.path.isdir(plotpath+ '/' + flag.add):
    #             os.makedirs(plotpath+ '/' + flag.add)
    #         model.save_weights(plotpath + flag.add + "/.weights.h5")
    #         print("Model saved!")
    #         print(plotpath + flag.add + "/.weights.h5")
    # model.load_weights(plotpath + flag.add + "/.weights.h5")

    return model

if __name__ == "__main__":
    pass

# HERE


# # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
# # #     from scipy.stats import kendalltau, pearsonr
# # #     from sklearn.metrics import mean_absolute_error, mean_squared_error

# # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # #     optim_val_car = -float('inf')  # Убедитесь, что начальное значение минимальное
# # #     # train_df = train_df[train_df["outcome"] > 0]
# # #     for epoch in range(flag.epoch):
# # #         print("Sampling negative items...", end=" ")
# # #         j_list = []
# # #         for i in train_df["idx_item"].to_numpy():
# # #             j = np.random.randint(0, num_items)
# # #             while j == i:
# # #                 j = np.random.randint(0, num_items)
# # #             j_list.append(j)
# # #         print("Done")
# # #         j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
# # #         train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
# # #         with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# # #             t.set_description('Training Epoch %i' % epoch)
# # #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # #                 step = model.propensity_train((user, item, item_j, value))
# # #                 t.update()
# # #         vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# # #         gamma_pred = None
# # #         p_pred = None
# # #         for u, i in vali_data.batch(5000):
# # #             gamma_batch, p_batch, _, _ = model((u, i), training=False)
# # #             if gamma_pred is None:
# # #                 gamma_pred = gamma_batch
# # #             else:
# # #                 gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
# # #             if p_pred is None:
# # #                 p_pred = p_batch
# # #             else:
# # #                 p_pred = tf.concat((p_pred, p_batch), axis=0)
# # # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # # #         p_pred = np.squeeze(p_pred.numpy())
# # # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # # #         tau_res, _ = kendalltau(p_pred, p_true)
# # # #         # pearsonres, _ = pearsonr(p_pred, p_true)
# # # #         # mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
# # # #         val_obj = tau_res

# # # #         save_path = os.path.join(plotpath, flag.add)

# # # #         if not os.path.exists(save_path):
# # # #             os.makedirs(save_path)

# # # #         if abs(val_obj) > optim_val_car:
# # # #             optim_val_car = val_obj
# # # #     #         if not os.path.isdir(plotpath+ '/' + flag.add):
# # # #     #             os.makedirs(plotpath+ '/' + flag.add)
# # # #     #         # model.save_weights(plotpath+ '/' + flag.add + "/saved_model")
# # # #     #         model.save_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # # #     #         print("Model saved!")
# # # #     # # model.load_weights(plotpath+ '/' + flag.add + "/saved_model")
# # # #     # model.load_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # # #             model.save_weights(os.path.join(save_path, "saved_model.keras"))
# # # #             print("Model weights saved!")

# # # #             if os.path.isfile(os.path.join(save_path, "saved_model.keras")):
# # # #                 print("Weights file verified.")
# # # #             else:
# # # #                 raise FileNotFoundError("Saved weights file not found.")

# # # #         model.load_weights(os.path.join(save_path, "saved_model.keras"))
# # # #         print("Weights successfully loaded.")


# # # #     return model

# # # # if __name__ == "__main__":
# # # #     pass

# # # # import os
# # # # from scipy.stats import kendalltau


# # #         # Нормализация и вычисление метрик
# # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # #         p_pred = np.squeeze(p_pred.numpy())
# # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # #         tau_res, _ = kendalltau(p_pred, p_true)
# # #         val_obj = tau_res

# # #         # Формируем уникальное имя поддиректории для сохранения весов
# # #         experiment_name = f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # #         save_path = os.path.join(plotpath, experiment_name)
# # #         print(f"Сохранение модели в путь: {save_path}")

# # #         # Проверяем и создаем директорию, если она отсутствует
# # #         if not os.path.exists(save_path):
# # #             os.makedirs(save_path)
# # #             print(f"Директория {save_path} создана.")

# # #         # Определяем путь для сохранения весов
# # #         weights_file = os.path.join(save_path, ".weights.h5")

# # #         # Сохраняем лучшие веса модели, если текущая метрика лучше
# # #         # if abs(val_obj) > optim_val_car:
# # #         optim_val_car = val_obj
            
# # #             # Сохранение весов модели
# # #         model.save_weights(weights_file)
# # #         print(f"Model weights saved at: {weights_file}")

# # #         # Проверка существования файла перед загрузкой
# # #         if os.path.isfile(weights_file):
# # #             # Загружаем веса модели
# # #             model.load_weights(weights_file)
# # #             print("Weights successfully loaded.")
# # #         else:
# # #             raise FileNotFoundError(f"Файл весов {weights_file} не найден!")
# # # import os
# # # import datetime
# # # import pickle
# # # from pathlib import Path

# # # import tensorflow as tf
# # # import numpy as np
# # # from tqdm import tqdm
# # # from scipy.stats import kendalltau, pearsonr
# # # from sklearn.metrics import mean_squared_error
# # # from models import Causal_Model

# # # import pandas as pd
# # # from pathlib import Path
# # # import numpy as np
# # # import tensorflow as tf
# # # from tqdm import tqdm
# # # from models import Causal_Model
# # # from scipy.stats import kendalltau
# # # from evaluator import Evaluator
# # # from sklearn.metrics import mean_absolute_error
# # # import matplotlib.pyplot as plt
# # # from sklearn.model_selection import train_test_split
# # # import random
# # # import datetime
# # # import os
# # # import itertools
# # # # from dataset import dh_original, dh_personalized, ml_data

# # # plotpath = "./results/"
# # # if not os.path.isdir(plotpath):
# # #     os.makedirs(plotpath)

# # # def save_flag(flag, path):
# # #     with open(os.path.join(path, "config.pkl"), "wb") as f:
# # #         pickle.dump(flag, f)

# # # def load_flag(path):
# # #     with open(os.path.join(path, "config.pkl"), "rb") as f:
# # #         return pickle.load(f)


# # # plotpath = "./results/"
# # # if not os.path.isdir(plotpath):
# # #     os.makedirs(plotpath)
# # # def diff(list1, list2):
# # #     return list(set(list2).difference(set(list1)))


# # # def sparse_gather(indices, values, selected_indices, axis=0):
# # #     """
# # #     indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
# # #     values:  [ value1,                                 , ..., valuen]
# # #     """
# # #     mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
# # #     to_select = tf.where(mask)[:, 1]
# # #     user_item = tf.gather(indices, to_select, axis=0)
# # #     user = tf.gather(user_item, 0, axis=1)
# # #     item = tf.gather(user_item, 1, axis=1)
# # #     values = tf.gather(values, to_select, axis=0)
# # #     return user, item, values


# # # def count_freq(x):
# # #     unique, counts = np.unique(x, return_counts=True)
# # #     return np.asarray((unique, counts)).T


# # # def prepare_data(flag):
# # #     dataset = flag.dataset
# # #     data_path = None
# # #     if dataset == "d":
# # #         print("dunn_cate (original) is used.")
# # #         # data = pd.read_csv('dh_original.csv')
# # #         # data = dh_original.copy()
# # #         data_path = Path("Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/original_rp0.40/result")
# # #     elif dataset == "p":
# # #         print("dunn_cate (personalized) is used.")
# # #         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210")
# # #         # data = dh_personalized.copy()
# # #         # data = pd.read_csv('dh_personalized.csv')
# # #     elif dataset == "ml":
# # #         data_path = Path("Uplift_Data/MovieLens/ML_100k_logrank100_offset5.0_scaling1.0")
# # #         # data = ml_data.copy()
# # #         # data = pd.read_csv('ml.csv')
# # #         print("ML-100k is used")
# # #     train_data = data_path / "data_train.csv"
# # #     vali_data = data_path / "data_vali.csv"
# # #     test_data = data_path / "data_test.csv"
# # #     train_df = pd.read_csv(train_data)
# # #     vali_df = pd.read_csv(vali_data)
# # #     test_df = pd.read_csv(test_data)

# # #     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# # #     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
# # #     user_ids = np.sort(
# # #         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
# # #     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# # #     item_ids = np.sort(
# # #         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
# # #     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
# # #     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
# # #     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
# # #     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
# # #     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
# # #     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
# # #     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
# # #     num_users = len(user_ids)
# # #     num_items = len(item_ids)
# # #     print(num_items)
# # #     if dataset == "d" or dataset == "p":
# # #         num_times = len(train_df["idx_time"].unique().tolist())
# # #     else: 
# # #         num_times = 1
# # #         train_df["idx_time"] = 0
# # #         vali_df["idx_time"] = 0
# # #         test_df["idx_time"] = 0
# # #     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
# # #     train_df_positive = train_df[train_df["outcome"] > 0]
# # #     counts = count_freq(train_df_positive['idx_item'].to_numpy())
# # #     np_counts = np.zeros(num_items)
# # #     print(np_counts.shape)
# # #     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

# # #     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

# # # def load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular):
# # #     """
# # #     Загружает модель и конфиг из checkpoint_path
# # #     """
# # #     flag = load_flag(checkpoint_path)
# # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # #     model.load_weights(os.path.join(checkpoint_path, "config.pkl"))
# # #     print(f"Model and flag loaded from {checkpoint_path}")
# # #     return model, flag


# # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):

# # #     checkpoint_path = "results/default/"  # путь к нужной папке с моделью
# # #     model, flag = load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular)
# # #     # model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # #     # optim_val_car = -float('inf')

# # #     # experiment_name = flag.add if hasattr(flag, 'add') else f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # #     # save_path = os.path.join(plotpath, experiment_name)

# # #     # if not os.path.exists(save_path):
# # #     #     os.makedirs(save_path)

# # #     # print(f"Training and saving to: {save_path}")

# # #     # for epoch in range(flag.epoch):
# # #     # for epoch in range(2):
# # #     #     print(f"\nEpoch {epoch+1}/{flag.epoch}: Sampling negative items...")
# # #     #     j_list = [
# # #     #         np.random.choice([j for j in range(num_items) if j != i])
# # #     #         for i in train_df["idx_item"].to_numpy()
# # #     #     ]

# # #     #     train_data = tf.data.Dataset.from_tensor_slices((
# # #     #         train_df["idx_user"].to_numpy(),
# # #     #         train_df["idx_item"].to_numpy(),
# # #     #         np.array(j_list),
# # #     #         train_df["outcome"].to_numpy()
# # #     #     ))

# # #     #     with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# # #     #         t.set_description(f'Training Epoch {epoch}')
# # #     #         for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # #     #             _ = model.propensity_train((user, item, item_j, value))
# # #     #             t.update()

# # #         # # Валидация
# # #         # vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# # #         # p_pred = []

# # #         # for u, i in vali_data.batch(5000):
# # #         #     _, p_batch, _, _ = model((u, i), training=False)
# # #         #     p_pred.append(p_batch)

# # #         # p_pred = tf.concat(p_pred, axis=0).numpy().squeeze()
# # #         # p_true = vali_df["propensity"].to_numpy().squeeze()
# # #         # p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred) + 1e-8)

# # #     #     tau_res, _ = kendalltau(p_pred, p_true)
# # #     #     mse = mean_squared_error(p_true, p_pred)

# # #     #     print(f"Kendall's tau: {tau_res:.4f}, MSE: {mse:.4f}")

# # #     #     if abs(tau_res) > optim_val_car:
# # #     #         optim_val_car = abs(tau_res)
# # #     #         model.save_weights(os.path.join(save_path, ".weights.h5"))
# # #     #         save_flag(flag, save_path)
# # #     #         print(f"\nModel weights and config saved in {save_path}")

# # #     # # Загрузка лучших весов после обучения
# # #     # model.load_weights(os.path.join(save_path, ".weights.h5"))
# # #     # print("\nBest model weights loaded for evaluation or further training.")

# # #     return model


# # import pandas as pd
# # from pathlib import Path
# # import numpy as np
# # import tensorflow as tf
# # from tqdm import tqdm
# # from models_new import Causal_Model
# # # from models_new2 import Causal_Model
# # # from models import Causal_Model
# # # from CJBR import CJBPR
# # from scipy.stats import kendalltau
# # from evaluator import Evaluator
# # from sklearn.metrics import mean_absolute_error
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # import random
# # import datetime
# # import os
# # import itertools
# # # from dataset import dh_original, dh_personalized, ml_data

# # plotpath = "./results/"
# # if not os.path.isdir(plotpath):
# #     os.makedirs(plotpath)
# # def diff(list1, list2):
# #     return list(set(list2).difference(set(list1)))


# # def sparse_gather(indices, values, selected_indices, axis=0):
# #     """
# #     indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
# #     values:  [ value1,                                 , ..., valuen]
# #     """
# #     mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
# #     to_select = tf.where(mask)[:, 1]
# #     user_item = tf.gather(indices, to_select, axis=0)
# #     user = tf.gather(user_item, 0, axis=1)
# #     item = tf.gather(user_item, 1, axis=1)
# #     values = tf.gather(values, to_select, axis=0)
# #     return user, item, values


# # def count_freq(x):
# #     unique, counts = np.unique(x, return_counts=True)
# #     return np.asarray((unique, counts)).T



# # def prepare_data(flag):
# #     dataset = flag.dataset
# #     data_path = None
# #     if dataset == "d":
# #         print("dunn_cate (original) is used.")
# #         # data = pd.read_csv('dh_original.csv')
# #         # data = dh_original.copy()
# #         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40")
# #     elif dataset == "p":
# #         print("dunn_cate (personalized) is used.")
# #         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210")
# #         # data = dh_personalized.copy()
# #         # data = pd.read_csv('dh_personalized.csv')
# #     elif dataset == "ml":
# #         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/synthetic/ML_100k_logrank100_offset5.0_scaling1.0")
# #         # data = ml_data.copy()
# #         # data = pd.read_csv('ml.csv')
# #         print("ML-100k is used")
# #     train_data = data_path / "data_train.csv"
# #     vali_data = data_path / "data_vali.csv"
# #     test_data = data_path / "data_test.csv"
# #     train_df = pd.read_csv(train_data)
# #     vali_df = pd.read_csv(vali_data)
# #     test_df = pd.read_csv(test_data)

# #     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# #     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
# #     user_ids = np.sort(
# #         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
# #     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# #     item_ids = np.sort(
# #         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
# #     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
# #     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
# #     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
# #     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
# #     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
# #     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
# #     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
# #     num_users = len(user_ids)
# #     num_items = len(item_ids)
# #     print(num_items)
# #     if dataset == "d" or dataset == "p":
# #         num_times = len(train_df["idx_time"].unique().tolist())
# #     else: 
# #         num_times = 1
# #         train_df["idx_time"] = 0
# #         vali_df["idx_time"] = 0
# #         test_df["idx_time"] = 0
# #     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
# #     train_df_positive = train_df[train_df["outcome"] > 0]
# #     counts = count_freq(train_df_positive['idx_item'].to_numpy())
# #     np_counts = np.zeros(num_items)
# #     print(np_counts.shape)
# #     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

# #     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

# # def snips_estimate(y, z_hat, p_hat):
# #         """
# #         SNIPS оценка uplift
# #         """
# #         p_hat = tf.clip_by_value(p_hat, 0.0001, 0.9999)
# #         treated = tf.cast(z_hat == 1, tf.float32)
# #         control = tf.cast(z_hat == 0, tf.float32)

# #         y_treated = treated * y / p_hat
# #         denom_treated = treated / p_hat

# #         y_control = control * y / (1.0 - p_hat)
# #         denom_control = control / (1.0 - p_hat)

# #         tau_snips = (
# #             tf.reduce_sum(y_treated) / tf.reduce_sum(denom_treated)
# #             - tf.reduce_sum(y_control) / tf.reduce_sum(denom_control)
# #         )
# #         return tau_snips

# # def evaluate_snips_propcare(model, dataset, threshold=0.2):
# #         """
# #         Оценка модели PropCare через SNIPS uplift

# #         dataset: tf.data.Dataset или генератор, выдающий (user, item, y_true)
# #         threshold: порог на нормализованное propensity
# #         """
# #         all_y, all_z_hat, all_p_hat = [], [], []

# #         for batch in dataset:
# #             user, item, y_true = batch
# #             _, p_hat, _, _ = model((user, item), training=False)

# #             # Z-скор нормализация
# #             norm_p = (p_hat - tf.reduce_mean(p_hat)) / (tf.math.reduce_std(p_hat) + 1e-6)
# #             z_hat = tf.cast(norm_p >= threshold, tf.float32)

# #             all_y.append(tf.reshape(y_true, [-1]))
# #             all_z_hat.append(tf.reshape(z_hat, [-1]))
# #             all_p_hat.append(tf.reshape(p_hat, [-1]))

# #         y_all = tf.concat(all_y, axis=0)
# #         z_hat_all = tf.concat(all_z_hat, axis=0)
# #         p_hat_all = tf.concat(all_p_hat, axis=0)

# #         uplift_snips = snips_estimate(y_all, z_hat_all, p_hat_all)
# #         print(f"[SNIPS Evaluation] Estimated Uplift: {uplift_snips.numpy():.4f}")
# #         return uplift_snips.numpy()

# # from itertools import product
# # import os
# # import numpy as np
# # import tensorflow as tf
# # from tqdm import tqdm
# # from scipy.stats import kendalltau
# # from itertools import product
# # from models_new import Causal_Model  # путь уточни при необходимости

# # def train_propensity_with_search(train_df, vali_df, flag, num_users, num_items, popular, plotpath="./results"):
# #     """
# #     Подбирает гиперпараметры PropCare по валидации на каждом шаге (валидация внутри эпох),
# #     сохраняет веса только один раз — для лучшего результата.
# #     """
# #     param_grid = {
# #         'lambda_1': [1.0, 10.0],
# #         'lambda_2': [0.1, 1.0],
# #         'lambda_3': [0.1, 1.0],
# #         'dimension': [64, 128],
# #     }

# #     best_score = -float("inf")
# #     best_flag_values = None
# #     best_model_path = os.path.join(plotpath, "best_prop_weights.h5")

# #     keys = list(param_grid.keys())
# #     combinations = list(product(*param_grid.values()))

# #     for values in combinations:
# #         param_combo = dict(zip(keys, values))
# #         flag.lambda_1 = param_combo['lambda_1']
# #         flag.lambda_2 = param_combo['lambda_2']
# #         flag.lambda_3 = param_combo['lambda_3']
# #         flag.dimension = param_combo['dimension']

# #         model = Causal_Model(num_users, num_items, flag, None, None, popular)
# #         _ = model((tf.constant([0]), tf.constant([0])))

# #         for epoch in range(3):
# #             j_list = []
# #             for i in train_df["idx_item"].to_numpy():
# #                 j = np.random.randint(0, num_items)
# #                 while j == i:
# #                     j = np.random.randint(0, num_items)
# #                 j_list.append(j)

# #             train_data = tf.data.Dataset.from_tensor_slices((
# #                 train_df["idx_user"].to_numpy(),
# #                 train_df["idx_item"].to_numpy(),
# #                 j_list,
# #                 train_df["outcome"].to_numpy()
# #             ))

# #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# #                 model.propensity_train((user, item, item_j, value))

# #             # Валидация после каждой эпохи
# #             vali_data = tf.data.Dataset.from_tensor_slices(
# #                 (vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy())
# #             )
# #             p_pred = None
# #             for u, i in vali_data.batch(5000):
# #                 _, p_batch, _, _ = model((u, i), training=False)
# #                 p_pred = tf.concat([p_pred, p_batch], axis=0) if p_pred is not None else p_batch

# #             p_true = np.squeeze(vali_df["propensity"].to_numpy())
# #             p_pred = np.squeeze(p_pred.numpy())
# #             p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred) + 1e-8)

# #             tau_res, _ = kendalltau(p_pred, p_true)
# #             print(f"[τ = {tau_res:.4f}] Epoch {epoch} for params: {param_combo}")

# #             if tau_res > best_score:
# #                 best_score = tau_res
# #                 best_flag_values = param_combo.copy()
# #                 model.save_weights(best_model_path)
# #                 print(f"✅ New best model saved with τ = {tau_res:.4f}")

# #     return best_flag_values, best_score

# # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular, plotpath="./results"):
# #     print("🚀 Подбор гиперпараметров PropCare...")
# #     best_flag_values, best_score = train_propensity_with_search(train_df, vali_df, flag, num_users, num_items, popular, plotpath)

# #     # Установим лучшие значения
# #     flag.lambda_1 = best_flag_values['lambda_1']
# #     flag.lambda_2 = best_flag_values['lambda_2']
# #     flag.lambda_3 = best_flag_values['lambda_3']
# #     flag.dimension = best_flag_values['dimension']

# #     # Создаём и загружаем модель с лучшими весами
# #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# #     _ = model((tf.constant([0]), tf.constant([0])))
# #     model.load_weights(os.path.join(plotpath, "best_prop_weights.h5"))

# #     return model




# # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
# # #     from scipy.stats import kendalltau, pearsonr
# # #     from sklearn.metrics import mean_absolute_error, mean_squared_error

# # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # #     # sample_user = tf.constant([0]) 
# # #     # sample_item = tf.constant([0]) 
# # #     # _ = model((sample_user, sample_item))  # This builds all layers 
# # #     # model.load_weights("/Users/tanyatomayly/Desktop/PropCare-main/results/default/.weights.h5")
    
# # #     # # model.save_weights("./results/default/.weights.h5")  # Замени путь на нужный

# # #     optim_val_car = 0
# # #     # train_df = train_df[train_df["outcome"] > 0]
# # #     for epoch in range(flag.epoch):
# # #     # for epoch in range(1):
# # #         print("Sampling negative items...", end=" ")
# # #         j_list = []
# # #         for i in train_df["idx_item"].to_numpy():
# # #             j = np.random.randint(0, num_items)
# # #             while j == i:
# # #                 j = np.random.randint(0, num_items)
# # #             j_list.append(j)
# # #         print("Done")
# # #         j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
# # #         train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
# # #         with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# # #             t.set_description('Training Epoch %i' % epoch)
# # #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # #                 step = model.propensity_train((user, item, item_j, value))
# # #                 t.update()
# # #         vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# # #         gamma_pred = None
# # #         p_pred = None
# # #         r_pred = None
# # #         for u, i in vali_data.batch(5000):
# # #             gamma_batch, p_batch, r_batch, _ = model((u, i), training=False)
# # #             if gamma_pred is None:
# # #                 gamma_pred = gamma_batch
# # #             else:
# # #                 gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
# # #             if p_pred is None:
# # #                 p_pred = p_batch
# # #             else:
# # #                 p_pred = tf.concat((p_pred, p_batch), axis=0)
# # #             if r_pred is None:
# # #                 r_pred = r_batch
# # #             else:
# # #                 r_pred = tf.concat((r_pred, r_batch), axis=0)

                
# # #         # p_pred = tf.reshape(p_pred, [-1])
# # #         # y_true = tf.convert_to_tensor(vali_df["outcome"].to_numpy(), dtype=tf.float32)

# # #         # norm_p = (p_pred - tf.reduce_mean(p_pred)) / (tf.math.reduce_std(p_pred) + 1e-6)
# # #         # z_hat = tf.cast(norm_p >= 0.2, tf.float32)

# # #         # uplift_snips = snips_estimate(y_true, z_hat, p_pred).numpy()
# # #         # print(f"[SNIPS uplift estimate] τ = {uplift_snips:.4f}")

# # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # #         p_pred = np.squeeze(p_pred.numpy())
# # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # #         # p_pred = np.squeeze(p_pred)
# # #         # p_range = np.max(p_pred) - np.min(p_pred)
# # #         # if p_range == 0:
# # #         #     p_pred = np.zeros_like(p_pred)
# # #         # else:
# # #         #     p_pred = (p_pred - np.min(p_pred)) / p_range

# # #         # if np.any(np.isnan(p_pred)):
# # #         #     print("Warning: NaNs in predicted propensity! Replacing with 0.")
# # #         #     p_pred = np.nan_to_num(p_pred, nan=0.0)

# # #         tau_res, _ = kendalltau(p_pred, p_true)
# # #         pearsonres, _ = pearsonr(p_pred, p_true)
# # #         mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
# # #         val_obj = tau_res
# # #         if val_obj > optim_val_car:
# # #             optim_val_car = val_obj
# # #             if not os.path.isdir(plotpath+ '/' + flag.add):
# # #                 os.makedirs(plotpath+ '/' + flag.add)
# # #             model.save_weights(plotpath + flag.add + "/.weights.h5")
# # #             print("Model saved!")
# # #             print(plotpath + flag.add + "/.weights.h5")
# # #     model.load_weights(plotpath + flag.add + "/.weights.h5")

# #     # return model

# # if __name__ == "__main__":
# #     pass

# # # HERE


# # # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
# # # #     from scipy.stats import kendalltau, pearsonr
# # # #     from sklearn.metrics import mean_absolute_error, mean_squared_error

# # # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # # #     optim_val_car = -float('inf')  # Убедитесь, что начальное значение минимальное
# # # #     # train_df = train_df[train_df["outcome"] > 0]
# # # #     for epoch in range(flag.epoch):
# # # #         print("Sampling negative items...", end=" ")
# # # #         j_list = []
# # # #         for i in train_df["idx_item"].to_numpy():
# # # #             j = np.random.randint(0, num_items)
# # # #             while j == i:
# # # #                 j = np.random.randint(0, num_items)
# # # #             j_list.append(j)
# # # #         print("Done")
# # # #         j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
# # # #         train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
# # # #         with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# # # #             t.set_description('Training Epoch %i' % epoch)
# # # #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # # #                 step = model.propensity_train((user, item, item_j, value))
# # # #                 t.update()
# # # #         vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# # # #         gamma_pred = None
# # # #         p_pred = None
# # # #         for u, i in vali_data.batch(5000):
# # # #             gamma_batch, p_batch, _, _ = model((u, i), training=False)
# # # #             if gamma_pred is None:
# # # #                 gamma_pred = gamma_batch
# # # #             else:
# # # #                 gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
# # # #             if p_pred is None:
# # # #                 p_pred = p_batch
# # # #             else:
# # # #                 p_pred = tf.concat((p_pred, p_batch), axis=0)
# # # # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # # # #         p_pred = np.squeeze(p_pred.numpy())
# # # # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # # # #         tau_res, _ = kendalltau(p_pred, p_true)
# # # # #         # pearsonres, _ = pearsonr(p_pred, p_true)
# # # # #         # mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
# # # # #         val_obj = tau_res

# # # # #         save_path = os.path.join(plotpath, flag.add)

# # # # #         if not os.path.exists(save_path):
# # # # #             os.makedirs(save_path)

# # # # #         if abs(val_obj) > optim_val_car:
# # # # #             optim_val_car = val_obj
# # # # #     #         if not os.path.isdir(plotpath+ '/' + flag.add):
# # # # #     #             os.makedirs(plotpath+ '/' + flag.add)
# # # # #     #         # model.save_weights(plotpath+ '/' + flag.add + "/saved_model")
# # # # #     #         model.save_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # # # #     #         print("Model saved!")
# # # # #     # # model.load_weights(plotpath+ '/' + flag.add + "/saved_model")
# # # # #     # model.load_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # # # #             model.save_weights(os.path.join(save_path, "saved_model.keras"))
# # # # #             print("Model weights saved!")

# # # # #             if os.path.isfile(os.path.join(save_path, "saved_model.keras")):
# # # # #                 print("Weights file verified.")
# # # # #             else:
# # # # #                 raise FileNotFoundError("Saved weights file not found.")

# # # # #         model.load_weights(os.path.join(save_path, "saved_model.keras"))
# # # # #         print("Weights successfully loaded.")


# # # # #     return model

# # # # # if __name__ == "__main__":
# # # # #     pass

# # # # # import os
# # # # # from scipy.stats import kendalltau


# # # #         # Нормализация и вычисление метрик
# # # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # # #         p_pred = np.squeeze(p_pred.numpy())
# # # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # # #         tau_res, _ = kendalltau(p_pred, p_true)
# # # #         val_obj = tau_res

# # # #         # Формируем уникальное имя поддиректории для сохранения весов
# # # #         experiment_name = f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # # #         save_path = os.path.join(plotpath, experiment_name)
# # # #         print(f"Сохранение модели в путь: {save_path}")

# # # #         # Проверяем и создаем директорию, если она отсутствует
# # # #         if not os.path.exists(save_path):
# # # #             os.makedirs(save_path)
# # # #             print(f"Директория {save_path} создана.")

# # # #         # Определяем путь для сохранения весов
# # # #         weights_file = os.path.join(save_path, ".weights.h5")

# # # #         # Сохраняем лучшие веса модели, если текущая метрика лучше
# # # #         # if abs(val_obj) > optim_val_car:
# # # #         optim_val_car = val_obj
            
# # # #             # Сохранение весов модели
# # # #         model.save_weights(weights_file)
# # # #         print(f"Model weights saved at: {weights_file}")

# # # #         # Проверка существования файла перед загрузкой
# # # #         if os.path.isfile(weights_file):
# # # #             # Загружаем веса модели
# # # #             model.load_weights(weights_file)
# # # #             print("Weights successfully loaded.")
# # # #         else:
# # # #             raise FileNotFoundError(f"Файл весов {weights_file} не найден!")

# # # # import os
# # # # import datetime
# # # # import pickle
# # # # from pathlib import Path

# # # # import tensorflow as tf
# # # # import numpy as np
# # # # from tqdm import tqdm
# # # # from scipy.stats import kendalltau, pearsonr
# # # # from sklearn.metrics import mean_squared_error
# # # # from models import Causal_Model

# # # # import pandas as pd
# # # # from pathlib import Path
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from tqdm import tqdm
# # # # from models import Causal_Model
# # # # from scipy.stats import kendalltau
# # # # from evaluator import Evaluator
# # # # from sklearn.metrics import mean_absolute_error
# # # # import matplotlib.pyplot as plt
# # # # from sklearn.model_selection import train_test_split
# # # # import random
# # # # import datetime
# # # # import os
# # # # import itertools
# # # # # from dataset import dh_original, dh_personalized, ml_data

# # # # plotpath = "./results/"
# # # # if not os.path.isdir(plotpath):
# # # #     os.makedirs(plotpath)

# # # # def save_flag(flag, path):
# # # #     with open(os.path.join(path, "config.pkl"), "wb") as f:
# # # #         pickle.dump(flag, f)

# # # # def load_flag(path):
# # # #     with open(os.path.join(path, "config.pkl"), "rb") as f:
# # # #         return pickle.load(f)


# # # # plotpath = "./results/"
# # # # if not os.path.isdir(plotpath):
# # # #     os.makedirs(plotpath)
# # # # def diff(list1, list2):
# # # #     return list(set(list2).difference(set(list1)))


# # # # def sparse_gather(indices, values, selected_indices, axis=0):
# # # #     """
# # # #     indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
# # # #     values:  [ value1,                                 , ..., valuen]
# # # #     """
# # # #     mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
# # # #     to_select = tf.where(mask)[:, 1]
# # # #     user_item = tf.gather(indices, to_select, axis=0)
# # # #     user = tf.gather(user_item, 0, axis=1)
# # # #     item = tf.gather(user_item, 1, axis=1)
# # # #     values = tf.gather(values, to_select, axis=0)
# # # #     return user, item, values


# # # # def count_freq(x):
# # # #     unique, counts = np.unique(x, return_counts=True)
# # # #     return np.asarray((unique, counts)).T


# # # # def prepare_data(flag):
# # # #     dataset = flag.dataset
# # # #     data_path = None
# # # #     if dataset == "d":
# # # #         print("dunn_cate (original) is used.")
# # # #         # data = pd.read_csv('dh_original.csv')
# # # #         # data = dh_original.copy()
# # # #         data_path = Path("Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/original_rp0.40/result")
# # # #     elif dataset == "p":
# # # #         print("dunn_cate (personalized) is used.")
# # # #         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210")
# # # #         # data = dh_personalized.copy()
# # # #         # data = pd.read_csv('dh_personalized.csv')
# # # #     elif dataset == "ml":
# # # #         data_path = Path("Uplift_Data/MovieLens/ML_100k_logrank100_offset5.0_scaling1.0")
# # # #         # data = ml_data.copy()
# # # #         # data = pd.read_csv('ml.csv')
# # # #         print("ML-100k is used")
# # # #     train_data = data_path / "data_train.csv"
# # # #     vali_data = data_path / "data_vali.csv"
# # # #     test_data = data_path / "data_test.csv"
# # # #     train_df = pd.read_csv(train_data)
# # # #     vali_df = pd.read_csv(vali_data)
# # # #     test_df = pd.read_csv(test_data)

# # # #     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# # # #     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
# # # #     user_ids = np.sort(
# # # #         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
# # # #     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# # # #     item_ids = np.sort(
# # # #         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
# # # #     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
# # # #     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
# # # #     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
# # # #     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
# # # #     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
# # # #     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
# # # #     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
# # # #     num_users = len(user_ids)
# # # #     num_items = len(item_ids)
# # # #     print(num_items)
# # # #     if dataset == "d" or dataset == "p":
# # # #         num_times = len(train_df["idx_time"].unique().tolist())
# # # #     else: 
# # # #         num_times = 1
# # # #         train_df["idx_time"] = 0
# # # #         vali_df["idx_time"] = 0
# # # #         test_df["idx_time"] = 0
# # # #     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
# # # #     train_df_positive = train_df[train_df["outcome"] > 0]
# # # #     counts = count_freq(train_df_positive['idx_item'].to_numpy())
# # # #     np_counts = np.zeros(num_items)
# # # #     print(np_counts.shape)
# # # #     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

# # # #     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

# # # # def load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular):
# # # #     """
# # # #     Загружает модель и конфиг из checkpoint_path
# # # #     """
# # # #     flag = load_flag(checkpoint_path)
# # # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # # #     model.load_weights(os.path.join(checkpoint_path, "config.pkl"))
# # # #     print(f"Model and flag loaded from {checkpoint_path}")
# # # #     return model, flag


# # # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):

# # # #     checkpoint_path = "results/default/"  # путь к нужной папке с моделью
# # # #     model, flag = load_model_from_checkpoint(checkpoint_path, num_users, num_items, popular)
# # # #     # model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # # #     # optim_val_car = -float('inf')

# # # #     # experiment_name = flag.add if hasattr(flag, 'add') else f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # # #     # save_path = os.path.join(plotpath, experiment_name)

# # # #     # if not os.path.exists(save_path):
# # # #     #     os.makedirs(save_path)

# # # #     # print(f"Training and saving to: {save_path}")

# # # #     # for epoch in range(flag.epoch):
# # # #     # for epoch in range(2):
# # # #     #     print(f"\nEpoch {epoch+1}/{flag.epoch}: Sampling negative items...")
# # # #     #     j_list = [
# # # #     #         np.random.choice([j for j in range(num_items) if j != i])
# # # #     #         for i in train_df["idx_item"].to_numpy()
# # # #     #     ]

# # # #     #     train_data = tf.data.Dataset.from_tensor_slices((
# # # #     #         train_df["idx_user"].to_numpy(),
# # # #     #         train_df["idx_item"].to_numpy(),
# # # #     #         np.array(j_list),
# # # #     #         train_df["outcome"].to_numpy()
# # # #     #     ))

# # # #     #     with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# # # #     #         t.set_description(f'Training Epoch {epoch}')
# # # #     #         for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # # #     #             _ = model.propensity_train((user, item, item_j, value))
# # # #     #             t.update()

# # # #         # # Валидация
# # # #         # vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# # # #         # p_pred = []

# # # #         # for u, i in vali_data.batch(5000):
# # # #         #     _, p_batch, _, _ = model((u, i), training=False)
# # # #         #     p_pred.append(p_batch)

# # # #         # p_pred = tf.concat(p_pred, axis=0).numpy().squeeze()
# # # #         # p_true = vali_df["propensity"].to_numpy().squeeze()
# # # #         # p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred) + 1e-8)

# # # #     #     tau_res, _ = kendalltau(p_pred, p_true)
# # # #     #     mse = mean_squared_error(p_true, p_pred)

# # # #     #     print(f"Kendall's tau: {tau_res:.4f}, MSE: {mse:.4f}")

# # # #     #     if abs(tau_res) > optim_val_car:
# # # #     #         optim_val_car = abs(tau_res)
# # # #     #         model.save_weights(os.path.join(save_path, ".weights.h5"))
# # # #     #         save_flag(flag, save_path)
# # # #     #         print(f"\nModel weights and config saved in {save_path}")

# # # #     # # Загрузка лучших весов после обучения
# # # #     # model.load_weights(os.path.join(save_path, ".weights.h5"))
# # # #     # print("\nBest model weights loaded for evaluation or further training.")

# # # #     return model


# # # import pandas as pd
# # # from pathlib import Path
# # # import numpy as np
# # # import tensorflow as tf
# # # from tqdm import tqdm
# # # from models_new import Causal_Model
# # # # from models_new2 import Causal_Model
# # # # from models import Causal_Model
# # # # from CJBR import CJBPR
# # # from scipy.stats import kendalltau
# # # from evaluator import Evaluator
# # # from sklearn.metrics import mean_absolute_error
# # # import matplotlib.pyplot as plt
# # # from sklearn.model_selection import train_test_split
# # # import random
# # # import datetime
# # # import os
# # # import itertools
# # # # from dataset import dh_original, dh_personalized, ml_data

# # # plotpath = "./results/"
# # # if not os.path.isdir(plotpath):
# # #     os.makedirs(plotpath)
# # # def diff(list1, list2):
# # #     return list(set(list2).difference(set(list1)))


# # # def sparse_gather(indices, values, selected_indices, axis=0):
# # #     """
# # #     indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
# # #     values:  [ value1,                                 , ..., valuen]
# # #     """
# # #     mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
# # #     to_select = tf.where(mask)[:, 1]
# # #     user_item = tf.gather(indices, to_select, axis=0)
# # #     user = tf.gather(user_item, 0, axis=1)
# # #     item = tf.gather(user_item, 1, axis=1)
# # #     values = tf.gather(values, to_select, axis=0)
# # #     return user, item, values


# # # def count_freq(x):
# # #     unique, counts = np.unique(x, return_counts=True)
# # #     return np.asarray((unique, counts)).T



# # # def prepare_data(flag):
# # #     dataset = flag.dataset
# # #     data_path = None
# # #     if dataset == "d":
# # #         print("dunn_cate (original) is used.")
# # #         # data = pd.read_csv('dh_original.csv')
# # #         # data = dh_original.copy()
# # #         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40")
# # #     elif dataset == "p":
# # #         print("dunn_cate (personalized) is used.")
# # #         data_path = Path("/Users/tanyatomayly/Downloads/Uplift_Data/DunnHumby/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210")
# # #         # data = dh_personalized.copy()
# # #         # data = pd.read_csv('dh_personalized.csv')
# # #     elif dataset == "ml":
# # #         data_path = Path("/Users/tanyatomayly/Downloads/CausalNBR/data/synthetic/ML_100k_logrank100_offset5.0_scaling1.0")
# # #         # data = ml_data.copy()
# # #         # data = pd.read_csv('ml.csv')
# # #         print("ML-100k is used")
# # #     train_data = data_path / "data_train.csv"
# # #     vali_data = data_path / "data_vali.csv"
# # #     test_data = data_path / "data_test.csv"
# # #     train_df = pd.read_csv(train_data)
# # #     vali_df = pd.read_csv(vali_data)
# # #     test_df = pd.read_csv(test_data)

# # #     # train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# # #     # vali_df, train_df = train_test_split(train_df, test_size=0.5, random_state=42)
# # #     user_ids = np.sort(
# # #         pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
# # #     user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# # #     item_ids = np.sort(
# # #         pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
# # #     item2item_encoded = {x: i for i, x in enumerate(item_ids)}
# # #     train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
# # #     train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
# # #     vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
# # #     vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
# # #     test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
# # #     test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
# # #     num_users = len(user_ids)
# # #     num_items = len(item_ids)
# # #     print(num_items)
# # #     if dataset == "d" or dataset == "p":
# # #         num_times = len(train_df["idx_time"].unique().tolist())
# # #     else: 
# # #         num_times = 1
# # #         train_df["idx_time"] = 0
# # #         vali_df["idx_time"] = 0
# # #         test_df["idx_time"] = 0
# # #     train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
# # #     train_df_positive = train_df[train_df["outcome"] > 0]
# # #     counts = count_freq(train_df_positive['idx_item'].to_numpy())
# # #     np_counts = np.zeros(num_items)
# # #     print(np_counts.shape)
# # #     np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

# # #     return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts

# # # def snips_estimate(y, z_hat, p_hat):
# # #         """
# # #         SNIPS оценка uplift
# # #         """
# # #         p_hat = tf.clip_by_value(p_hat, 0.0001, 0.9999)
# # #         treated = tf.cast(z_hat == 1, tf.float32)
# # #         control = tf.cast(z_hat == 0, tf.float32)

# # #         y_treated = treated * y / p_hat
# # #         denom_treated = treated / p_hat

# # #         y_control = control * y / (1.0 - p_hat)
# # #         denom_control = control / (1.0 - p_hat)

# # #         tau_snips = (
# # #             tf.reduce_sum(y_treated) / tf.reduce_sum(denom_treated)
# # #             - tf.reduce_sum(y_control) / tf.reduce_sum(denom_control)
# # #         )
# # #         return tau_snips

# # # def evaluate_snips_propcare(model, dataset, threshold=0.2):
# # #         """
# # #         Оценка модели PropCare через SNIPS uplift

# # #         dataset: tf.data.Dataset или генератор, выдающий (user, item, y_true)
# # #         threshold: порог на нормализованное propensity
# # #         """
# # #         all_y, all_z_hat, all_p_hat = [], [], []

# # #         for batch in dataset:
# # #             user, item, y_true = batch
# # #             _, p_hat, _, _ = model((user, item), training=False)

# # #             # Z-скор нормализация
# # #             norm_p = (p_hat - tf.reduce_mean(p_hat)) / (tf.math.reduce_std(p_hat) + 1e-6)
# # #             z_hat = tf.cast(norm_p >= threshold, tf.float32)

# # #             all_y.append(tf.reshape(y_true, [-1]))
# # #             all_z_hat.append(tf.reshape(z_hat, [-1]))
# # #             all_p_hat.append(tf.reshape(p_hat, [-1]))

# # #         y_all = tf.concat(all_y, axis=0)
# # #         z_hat_all = tf.concat(all_z_hat, axis=0)
# # #         p_hat_all = tf.concat(all_p_hat, axis=0)

# # #         uplift_snips = snips_estimate(y_all, z_hat_all, p_hat_all)
# # #         print(f"[SNIPS Evaluation] Estimated Uplift: {uplift_snips.numpy():.4f}")
# # #         return uplift_snips.numpy()

# # # # from itertools import product
# # # # import os
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from tqdm import tqdm
# # # # from scipy.stats import kendalltau
# # # # from itertools import product
# # # # from models_new import Causal_Model  # путь уточни при необходимости

# # # # def train_propensity_with_search(train_df, vali_df, flag, num_users, num_items, popular, plotpath="./results"):
# # # #     """
# # # #     Подбирает гиперпараметры PropCare по валидации на каждом шаге (валидация внутри эпох),
# # # #     сохраняет веса только один раз — для лучшего результата.
# # # #     """
# # # #     param_grid = {
# # # #         'lambda_1': [10.0],
# # # #         'lambda_2': [0.5],
# # # #         'lambda_3': [0.1],
# # # #         'dimension': [128],
# # # #     }

# # # #     # import torch
# # # #     # print(torch.cuda.is_available())  # для PyTorch

# # # #     import tensorflow as tf
# # # #     print(tf.config.list_physical_devices('GPU'))  # для TensorFlow


# # # #     best_score = -float("inf")
# # # #     best_flag_values = None
# # # #     best_model_path = os.path.join(plotpath, ".weights.h5")

# # # #     keys = list(param_grid.keys())
# # # #     combinations = list(product(*param_grid.values()))

# # # #     for values in combinations:
# # # #         param_combo = dict(zip(keys, values))
# # # #         flag.lambda_1 = param_combo['lambda_1']
# # # #         flag.lambda_2 = param_combo['lambda_2']
# # # #         flag.lambda_3 = param_combo['lambda_3']
# # # #         flag.dimension = param_combo['dimension']

# # # #         model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # # #         _ = model((tf.constant([0]), tf.constant([0])))

# # # #         for epoch in range(flag.epoch):
# # # #             j_list = []
# # # #             for i in train_df["idx_item"].to_numpy():
# # # #                 j = np.random.randint(0, num_items)
# # # #                 while j == i:
# # # #                     j = np.random.randint(0, num_items)
# # # #                 j_list.append(j)

# # # #             train_data = tf.data.Dataset.from_tensor_slices((
# # # #                 train_df["idx_user"].to_numpy(),
# # # #                 train_df["idx_item"].to_numpy(),
# # # #                 j_list,
# # # #                 train_df["outcome"].to_numpy()
# # # #             ))

# # # #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # # #                 model.propensity_train((user, item, item_j, value))

# # # #             # Валидация после каждой эпохи
# # # #             vali_data = tf.data.Dataset.from_tensor_slices(
# # # #                 (vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy())
# # # #             )
# # # #             p_pred = None
# # # #             for u, i in vali_data.batch(5000):
# # # #                 _, p_batch, _, _ = model((u, i), training=False)
# # # #                 p_pred = tf.concat([p_pred, p_batch], axis=0) if p_pred is not None else p_batch

# # # #             p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # # #             p_pred = np.squeeze(p_pred.numpy())
# # # #             p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred) + 1e-8)

# # # #             tau_res, _ = kendalltau(p_pred, p_true)
# # # #             print(f"[τ = {tau_res:.4f}] Epoch {epoch} for params: {param_combo}")

# # # #             if tau_res > best_score:
# # # #                 best_score = tau_res
# # # #                 best_flag_values = param_combo.copy()
# # # #                 model.save_weights(best_model_path)
# # # #                 print(f"✅ New best model saved with τ = {tau_res:.4f}")

# # # #     return best_flag_values, best_score

# # # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular, plotpath="./results"):
# # # #     print("🚀 Подбор гиперпараметров PropCare...")
# # # #     best_flag_values, best_score = train_propensity_with_search(train_df, vali_df, flag, num_users, num_items, popular, plotpath)

# # # #     # Установим лучшие значения
# # # #     flag.lambda_1 = best_flag_values['lambda_1']
# # # #     flag.lambda_2 = best_flag_values['lambda_2']
# # # #     flag.lambda_3 = best_flag_values['lambda_3']
# # # #     flag.dimension = best_flag_values['dimension']

# # # #     # Создаём и загружаем модель с лучшими весами
# # # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # # #     _ = model((tf.constant([0]), tf.constant([0])))
# # # #     model.load_weights(os.path.join(plotpath, ".weights.h5"))

# # # #     return model




# # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
# # #     from scipy.stats import kendalltau, pearsonr
# # #     from sklearn.metrics import mean_absolute_error, mean_squared_error

# # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # #     # sample_user = tf.constant([0]) 
# # #     # sample_item = tf.constant([0]) 
# # #     # _ = model((sample_user, sample_item))  # This builds all layers 
# # #     # # model.load_weights("/Users/tanyatomayly/Desktop/PropCare-main/results/default/.weights.h5")
# # #     # model.load_weights("/Users/tanyatomayly/Desktop/PropCare-main/results/.weights.h5")
    
# # #     # # # model.save_weights("./results/default/.weights.h5")  # Замени путь на нужный

# # #     optim_val_car = 0
# # #     # train_df = train_df[train_df["outcome"] > 0]
# # #     for epoch in range(flag.epoch):
# # #     # for epoch in range(1):
# # #         print("Sampling negative items...", end=" ")
# # #         j_list = []
# # #         for i in train_df["idx_item"].to_numpy():
# # #             j = np.random.randint(0, num_items)
# # #             while j == i:
# # #                 j = np.random.randint(0, num_items)
# # #             j_list.append(j)
# # #         print("Done")
# # #         j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
# # #         train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
# # #         with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# # #             t.set_description('Training Epoch %i' % epoch)
# # #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # #                 step = model.propensity_train((user, item, item_j, value))
# # #                 t.update()
# # #         vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# # #         gamma_pred = None
# # #         p_pred = None
# # #         r_pred = None
# # #         for u, i in vali_data.batch(5000):
# # #             gamma_batch, p_batch, r_batch, _ = model((u, i), training=False)
# # #             if gamma_pred is None:
# # #                 gamma_pred = gamma_batch
# # #             else:
# # #                 gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
# # #             if p_pred is None:
# # #                 p_pred = p_batch
# # #             else:
# # #            µ≠и        /ю≥≥≥≥≥≥≥≥≥≥≥≥≥≥≥≥≥ж     p_pred = tf.concat((p_pred, p_batch), axis=0)
# # #             if r_pred is None:
# # #                 r_pred = r_batch
# # #             else:
# # #                 r_pred = tf.concat((r_pred, r_batch), axis=0)

                
# # #         # p_pred = tf.reshape(p_pred, [-1])
# # #         # y_true = tf.convert_to_tensor(vali_df["outcome"].to_numpy(), dtype=tf.float32)

# # #         # norm_p = (p_pred - tf.reduce_mean(p_pred)) / (tf.math.reduce_std(p_pred) + 1e-6)
# # #         # z_hat = tf.cast(norm_p >= 0.2, tf.float32)

# # #         # uplift_snips = snips_estimate(y_true, z_hat, p_pred).numpy()
# # #         # print(f"[SNIPS uplift estimate] τ = {uplift_snips:.4f}")

# # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # #         p_pred = np.squeeze(p_pred.numpy())
# # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # #         # p_pred = np.squeeze(p_pred)
# # #         # p_range = np.max(p_pred) - np.min(p_pred)
# # #         # if p_range == 0:
# # #         #     p_pred = np.zeros_like(p_pred)
# # #         # else:
# # #         #     p_pred = (p_pred - np.min(p_pred)) / p_range

# # #         # if np.any(np.isnan(p_pred)):
# # #         #     print("Warning: NaNs in predicted propensity! Replacing with 0.")
# # #         #     p_pred = np.nan_to_num(p_pred, nan=0.0)

# # #         tau_res, _ = kendalltau(p_pred, p_true)
# # #         pearsonres, _ = pearsonr(p_pred, p_true)
# # #         mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
# # #         val_obj = tau_res
# # #         if val_obj > optim_val_car:
# # #             optim_val_car = val_obj
# # #             if not os.path.isdir(plotpath+ '/' + flag.add):
# # #                 os.makedirs(plotpath+ '/' + flag.add)
# # #             model.save_weights(plotpath + flag.add + "/.weights.h5")
# # #             print("Model saved!")
# # #             print(plotpath + flag.add + "/.weights.h5")
# # #     model.load_weights(plotpath + flag.add + "/.weights.h5")

# # #     return model

# # # if __name__ == "__main__":
# # #     pass

# # # # HERE


# # # # # def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
# # # # #     from scipy.stats import kendalltau, pearsonr
# # # # #     from sklearn.metrics import mean_absolute_error, mean_squared_error

# # # # #     model = Causal_Model(num_users, num_items, flag, None, None, popular)
# # # # #     optim_val_car = -float('inf')  # Убедитесь, что начальное значение минимальное
# # # # #     # train_df = train_df[train_df["outcome"] > 0]
# # # # #     for epoch in range(flag.epoch):
# # # # #         print("Sampling negative items...", end=" ")
# # # # #         j_list = []
# # # # #         for i in train_df["idx_item"].to_numpy():
# # # # #             j = np.random.randint(0, num_items)
# # # # #             while j == i:
# # # # #                 j = np.random.randint(0, num_items)
# # # # #             j_list.append(j)
# # # # #         print("Done")
# # # # #         j_list = np.reshape(np.array(j_list, dtype=train_df["idx_item"].to_numpy().dtype), train_df["idx_item"].to_numpy().shape)
# # # # #         train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_df["idx_item"].to_numpy(), j_list, train_df["outcome"].to_numpy()))
# # # # #         with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
# # # # #             t.set_description('Training Epoch %i' % epoch)
# # # # #             for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
# # # # #                 step = model.propensity_train((user, item, item_j, value))
# # # # #                 t.update()
# # # # #         vali_data = tf.data.Dataset.from_tensor_slices((vali_df["idx_user"].to_numpy(), vali_df["idx_item"].to_numpy()))
# # # # #         gamma_pred = None
# # # # #         p_pred = None
# # # # #         for u, i in vali_data.batch(5000):
# # # # #             gamma_batch, p_batch, _, _ = model((u, i), training=False)
# # # # #             if gamma_pred is None:
# # # # #                 gamma_pred = gamma_batch
# # # # #             else:
# # # # #                 gamma_pred = tf.concat((gamma_pred, gamma_batch), axis=0)
# # # # #             if p_pred is None:
# # # # #                 p_pred = p_batch
# # # # #             else:
# # # # #                 p_pred = tf.concat((p_pred, p_batch), axis=0)
# # # # # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # # # # #         p_pred = np.squeeze(p_pred.numpy())
# # # # # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # # # # #         tau_res, _ = kendalltau(p_pred, p_true)
# # # # # #         # pearsonres, _ = pearsonr(p_pred, p_true)
# # # # # #         # mse = mean_squared_error(y_pred=p_pred, y_true=p_true)
# # # # # #         val_obj = tau_res

# # # # # #         save_path = os.path.join(plotpath, flag.add)

# # # # # #         if not os.path.exists(save_path):
# # # # # #             os.makedirs(save_path)

# # # # # #         if abs(val_obj) > optim_val_car:
# # # # # #             optim_val_car = val_obj
# # # # # #     #         if not os.path.isdir(plotpath+ '/' + flag.add):
# # # # # #     #             os.makedirs(plotpath+ '/' + flag.add)
# # # # # #     #         # model.save_weights(plotpath+ '/' + flag.add + "/saved_model")
# # # # # #     #         model.save_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # # # # #     #         print("Model saved!")
# # # # # #     # # model.load_weights(plotpath+ '/' + flag.add + "/saved_model")
# # # # # #     # model.load_weights(plotpath + '/' + flag.add + "/saved_model.keras")

# # # # # #             model.save_weights(os.path.join(save_path, "saved_model.keras"))
# # # # # #             print("Model weights saved!")

# # # # # #             if os.path.isfile(os.path.join(save_path, "saved_model.keras")):
# # # # # #                 print("Weights file verified.")
# # # # # #             else:
# # # # # #                 raise FileNotFoundError("Saved weights file not found.")

# # # # # #         model.load_weights(os.path.join(save_path, "saved_model.keras"))
# # # # # #         print("Weights successfully loaded.")


# # # # # #     return model

# # # # # # if __name__ == "__main__":
# # # # # #     pass

# # # # # # import os
# # # # # # from scipy.stats import kendalltau


# # # # #         # Нормализация и вычисление метрик
# # # # #         p_true = np.squeeze(vali_df["propensity"].to_numpy())
# # # # #         p_pred = np.squeeze(p_pred.numpy())
# # # # #         p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
# # # # #         tau_res, _ = kendalltau(p_pred, p_true)
# # # # #         val_obj = tau_res

# # # # #         # Формируем уникальное имя поддиректории для сохранения весов
# # # # #         experiment_name = f"default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # # # #         save_path = os.path.join(plotpath, experiment_name)
# # # # #         print(f"Сохранение модели в путь: {save_path}")

# # # # #         # Проверяем и создаем директорию, если она отсутствует
# # # # #         if not os.path.exists(save_path):
# # # # #             os.makedirs(save_path)
# # # # #             print(f"Директория {save_path} создана.")

# # # # #         # Определяем путь для сохранения весов
# # # # #         weights_file = os.path.join(save_path, ".weights.h5")

# # # # #         # Сохраняем лучшие веса модели, если текущая метрика лучше
# # # # #         # if abs(val_obj) > optim_val_car:
# # # # #         optim_val_car = val_obj
            
# # # # #             # Сохранение весов модели
# # # # #         model.save_weights(weights_file)
# # # # #         print(f"Model weights saved at: {weights_file}")

# # # # #         # Проверка существования файла перед загрузкой
# # # # #         if os.path.isfile(weights_file):
# # # # #             # Загружаем веса модели
# # # # #             model.load_weights(weights_file)
# # # # #             print("Weights successfully loaded.")
# # # # #         else:
# # # # #             raise FileNotFoundError(f"Файл весов {weights_file} не найден!")
