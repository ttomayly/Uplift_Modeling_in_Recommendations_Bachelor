import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import kendalltau
from evaluator import Evaluator
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tqdm import tqdm
from itertools import product
from PropCare_new import Causal_Model as Causal_Model_New # путь уточни при необходимости
from PropCare_old import Causal_Model as Causal_Model_Old  # путь уточни при необходимости

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



def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
    from scipy.stats import kendalltau, pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if flag.propensity_model == "mod_propcare":
        model = Causal_Model_New(num_users, num_items, flag, None, None, popular)
    elif flag.propensity_model == "propcare":
        model = Causal_Model_Old(num_users, num_items, flag, None, None, popular)

    sample_user = tf.constant([0]) 
    sample_item = tf.constant([0]) 
    _ = model((sample_user, sample_item))  # This builds all layers 
    # model.load_weights("/Users/tanyatomayly/Desktop/PropCare-main/results/default/.weights.h5")
    model.load_weights("no/vesa_pers_dr/.weights.h5")
    
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