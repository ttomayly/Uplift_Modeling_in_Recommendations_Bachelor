# import numpy as np
# from pathlib import Path
# from numpy.random.mtrand import RandomState
# import random
# import pandas as pd
# from evaluator import Evaluator
# import pickle


# class Recommender(object):

#     def __init__(self, num_users, num_items,
#                  colname_user = 'idx_user', colname_item = 'idx_item',
#                  colname_outcome = 'outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):
#         super().__init__()

#         self.num_users = num_users
#         self.num_items = num_items
#         self.colname_user = colname_user
#         self.colname_item = colname_item
#         self.colname_outcome = colname_outcome
#         self.colname_prediction = colname_prediction
#         self.colname_treatment = colname_treatment
#         self.colname_propensity = colname_propensity

#     def train(self, df, iter=100):
#         pass

#     def predict(self, df):
#         pass

#     def recommend(self, df, num_rec=10):
#         pass

#     def func_sigmoid(self, x):
#         if x >= 0:
#             return 1.0 / (1.0 + np.exp(-x))
#         else:
#             return np.exp(x) / (1.0 + np.exp(x))

#     def sample_time(self):
#         return random.randrange(self.num_times)

#     def sample_user(self, idx_time, TP=True, TN=True, CP=True, CN=True):
#         while True:
#             flag_condition = 1
#             u = random.randrange(self.num_users)
#             if TP:
#                 if u not in self.dict_treatment_positive_sets[idx_time]:
#                     flag_condition = 0
#             if TN:
#                 if u not in self.dict_treatment_negative_sets[idx_time]:
#                     flag_condition = 0
#             if CP:
#                 if u not in self.dict_control_positive_sets[idx_time]:
#                     flag_condition = 0
#             if CN:
#                 if u not in self.dict_control_negative_sets[idx_time]:
#                     flag_condition = 0
#             if flag_condition > 0:
#                 return u

#     def sample_treatment(self, idx_time, idx_user):
#         return random.choice(self.dict_treatment_sets[idx_time][idx_user])

#     def sample_control(self, idx_time, idx_user):
#         while True:
#             flag_condition = 1
#             i = random.randrange(self.num_items)
#             if idx_user in self.dict_treatment_positive_sets[idx_time]:
#                 if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
#                     flag_condition = 0
#             if idx_user in self.dict_treatment_negative_sets[idx_time]:
#                 if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
#                     flag_condition = 0
#             if flag_condition > 0:
#                 return i

#     # in case control is rare
#     def sample_control2(self, idx_time, idx_user):
#         cand_control = np.arange(self.num_items)
#         cand_control = cand_control[np.isin(cand_control, self.dict_treatment_sets[idx_time][idx_user])]
#         return random.choice(cand_control)

#     def sample_treatment_positive(self, idx_time, idx_user):
#         return random.choice(self.dict_treatment_positive_sets[idx_time][idx_user])

#     def sample_treatment_negative(self, idx_time, idx_user):
#         return random.choice(self.dict_treatment_negative_sets[idx_time][idx_user])

#     def sample_control_positive(self, idx_time, idx_user):
#         return random.choice(self.dict_control_positive_sets[idx_time][idx_user])

#     def sample_control_negative(self, idx_time, idx_user):
#         while True:
#             flag_condition = 1
#             i = random.randrange(self.num_items)
#             if idx_user in self.dict_treatment_positive_sets[idx_time]:
#                 if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
#                     flag_condition = 0
#             if idx_user in self.dict_treatment_negative_sets[idx_time]:
#                 if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
#                     flag_condition = 0
#             if idx_user in self.dict_control_positive_sets[idx_time]:
#                 if i in self.dict_control_positive_sets[idx_time][idx_user]:
#                     flag_condition = 0
#             if flag_condition > 0:
#                 return i

#     # TP: treatment-positive
#     # CP: control-positive
#     # TN: treatment-negative
#     # TN: control-negative
#     def sample_triplet(self):
#         t = self.sample_time()
#         if random.random() <= self.alpha:  # CN as positive
#             if random.random() <= 0.5:  # TP as positive
#                 if random.random() <= 0.5:  # TP vs. TN
#                     u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
#                     i = self.sample_treatment_positive(t, u)
#                     j = self.sample_treatment_negative(t, u)
#                 else:  # TP vs. CP
#                     u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
#                     i = self.sample_treatment_positive(t, u)
#                     j = self.sample_control_positive(t, u)
#             else:  # CN as positive
#                 if random.random() <= 0.5:  # CN vs. TN
#                     u = self.sample_user(t, TP=False, TN=True, CP=False, CN=True)
#                     i = self.sample_control_negative(t, u)
#                     j = self.sample_treatment_negative(t, u)
#                 else:  # CN vs. CP
#                     u = self.sample_user(t, TP=False, TN=False, CP=True, CN=True)
#                     i = self.sample_control_negative(t, u)
#                     j = self.sample_control_positive(t, u)
#         else:  # CN as negative
#             if random.random() <= 0.333:  # TP vs. CN
#                 u = self.sample_user(t, TP=True, TN=False, CP=False, CN=True)
#                 i = self.sample_treatment_positive(t, u)
#                 j = self.sample_control_negative(t, u)
#             elif random.random() <= 0.5:  # TP vs. TN
#                 u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
#                 i = self.sample_treatment_positive(t, u)
#                 j = self.sample_treatment_negative(t, u)
#             else:  # TP vs. CP
#                 u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
#                 i = self.sample_treatment_positive(t, u)
#                 j = self.sample_control_positive(t, u)

#         return u, i, j

#     def sample_pair(self):
#         t = self.sample_time()
#         if random.random() < 0.5: # pick treatment
#             if random.random() > self.ratio_nega: # TP
#                 u = self.sample_user(t, TP=True, TN=False, CP=False, CN=False)
#                 i = self.sample_treatment_positive(t, u)
#                 flag_positive = 1
#             else: # TN
#                 u = self.sample_user(t, TP=False, TN=True, CP=False, CN=False)
#                 i = self.sample_treatment_negative(t, u)
#                 flag_positive = 0
#         else: # pick control
#             if random.random() > self.ratio_nega:  # CP
#                 u = self.sample_user(t, TP=False, TN=False, CP=True, CN=False)
#                 i = self.sample_control_positive(t, u)
#                 flag_positive = 0
#             else:  # CN
#                 u = self.sample_user(t, TP=False, TN=False, CP=False, CN=True)
#                 i = self.sample_control_negative(t, u)
#                 if random.random() <= self.alpha:  # CN as positive
#                     flag_positive = 1
#                 else:
#                     flag_positive = 0

#         return u, i, flag_positive

#     # getter
#     def get_propensity(self, idx_user, idx_item):
#         return self.dict_propensity[idx_user][idx_item]


# class LMF(Recommender):
#     def __init__(self, num_users, num_items,
#                  metric='AUC', ratio_nega=0.8,
#                  dim_factor=200, with_bias=False,
#                  learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
#                  reg_factor_j=0.01, reg_bias_j=0.01,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)
#         self.metric = metric
#         self.ratio_nega = ratio_nega
#         self.dim_factor = dim_factor
#         self.rng = RandomState(seed=None)
#         self.with_bias = with_bias

#         self.learn_rate = learn_rate
#         self.reg_bias = reg_bias
#         self.reg_factor = reg_factor
#         self.sd_init = sd_init
#         self.reg_bias_j = reg_bias_j
#         self.reg_factor_j = reg_factor_j
#         self.flag_prepared = False

#         self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
#         self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         if self.with_bias:
#             self.user_biases = np.zeros(self.num_users)
#             self.item_biases = np.zeros(self.num_items)
#             self.global_bias = 0.0

#     def prepare_dictionary(self, df, colname_time='idx_time'):
#         print("start prepare dictionary")
#         self.colname_time = colname_time
#         self.num_times = np.max(df.loc[:, self.colname_time]) + 1
#         self.dict_positive_sets = dict()

#         df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]

#         for t in np.arange(self.num_times):
#             df_t = df_posi.loc[df_posi.loc[:, self.colname_time] == t]
#             self.dict_positive_sets[t] = dict()
#             for u in np.unique(df_t.loc[:, self.colname_user]):
#                 self.dict_positive_sets[t][u] = \
#                     np.unique(df_t.loc[df_t.loc[:, self.colname_user] == u, self.colname_item].values)

#         self.flag_prepared = True
#         print("prepared dictionary!")


#     def train(self, df, iter = 10):

#         df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
#         if not self.flag_prepared: # prepare dictionary
#             self.prepare_dictionary(df)
        
#         err = 0
#         current_iter = 0
#         while True:
#             df_train = df_train.sample(frac=1)
#             users = df_train.loc[:, self.colname_user].values
#             items = df_train.loc[:, self.colname_item].values
#             times = df_train.loc[:, self.colname_time].values

#             if self.metric == 'AUC': # BPR
#                 for n in np.arange(len(df_train)):
#                     u = users[n]
#                     i = items[n]
#                     t = times[n]

#                     while True:
#                         j = random.randrange(self.num_items)
#                         if not j in self.dict_positive_sets[t][u]:
#                             break

#                     u_factor = self.user_factors[u, :]
#                     i_factor = self.item_factors[i, :]
#                     j_factor = self.item_factors[j, :]

#                     diff_rating = np.sum(u_factor * (i_factor - j_factor))

#                     if self.with_bias:
#                         diff_rating += (self.item_biases[i] - self.item_biases[j])

#                     coeff = self.func_sigmoid(-diff_rating)

#                     err += coeff

#                     self.user_factors[u, :] += \
#                         self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
#                     self.item_factors[i, :] += \
#                         self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
#                     self.item_factors[j, :] += \
#                         self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

#                     if self.with_bias:
#                         self.item_biases[i] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                         self.item_biases[j] += \
#                             self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

#             current_iter += 1
#             if current_iter >= iter:
#                 return err/iter

#             elif self.metric == 'logloss': # essentially WRMF with downsampling
#                 for n in np.arange(len(df_train)):
#                     u = users[n]
#                     i = items[n]
#                     t = times[n]
#                     flag_positive = 1

#                     if np.random.rand() < self.ratio_nega:
#                         flag_positive = 0
#                         i = np.random.randint(self.num_items)
#                         while True:
#                             if not i in self.dict_positive_sets[t][u]:
#                                 break
#                             else:
#                                 i = np.random.randint(self.num_items)

#                     u_factor = self.user_factors[u, :]
#                     i_factor = self.item_factors[i, :]

#                     rating = np.sum(u_factor * i_factor)

#                     if self.with_bias:
#                         rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

#                     if flag_positive > 0:
#                         coeff = 1 / (1 + np.exp(rating))
#                     else:
#                         coeff = -1 / (1 + np.exp(-rating))

#                     err += np.abs(coeff)

#                     self.user_factors[u, :] += \
#                         self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
#                     self.item_factors[i, :] += \
#                         self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

#                     if self.with_bias:
#                         self.item_biases[i] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                         self.user_biases[u] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
#                         self.global_bias += \
#                             self.learn_rate * (coeff)

#                     current_iter += 1
#                     if current_iter >= iter:
#                         return err / iter

#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred = np.zeros(len(df))
#         for n in np.arange(len(df)):
#             pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
#             if self.with_bias:
#                 pred[n] += self.item_biases[items[n]]
#                 pred[n] += self.user_biases[users[n]]
#                 pred[n] += self.global_bias

#         # pred = 1 / (1 + np.exp(-pred))
#         return pred


# class PopularBase(Recommender):
#     def __init__(self, num_users, num_items,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction)
#         self.df_cnt = df = pd.DataFrame([])

#     def train(self, df, iter = 1):
#         df_cnt = df.groupby(self.colname_item, as_index=False)[self.colname_outcome].sum()
#         df_cnt['prob'] = df_cnt[self.colname_outcome] /self.num_users
#         self.df_cnt = df_cnt

#     def predict(self, df):
#         df = pd.merge(df, self.df_cnt, on=self.colname_item, how='left')
#         return df.loc[:, 'prob'].values

# class DLMF2(Recommender): # This version consider a scale factor alpha
#     def __init__(self, num_users, num_items,
#                  metric='AR_logi', capping_T=0.01, capping_C=0.01,
#                  dim_factor=200, with_bias=False, with_IPS=True,
#                  only_treated=False,
#                  learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
#                  sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
#                  coeff_T = 1.0, coeff_C = 1.0,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)
#         self.metric = metric
#         self.capping_T = capping_T
#         self.capping_C = capping_C
#         self.with_IPS = with_IPS
#         self.dim_factor = dim_factor
#         self.rng = RandomState(seed=None)
#         self.with_bias = with_bias
#         self.coeff_T = coeff_T
#         self.coeff_C = coeff_C
#         self.learn_rate = learn_rate
#         self.reg_bias = reg_factor
#         self.reg_factor = reg_factor
#         self.reg_bias_j = reg_factor
#         self.reg_factor_j = reg_factor
#         self.sd_init = sd_init
#         self.only_treated = only_treated

#         self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
#         self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         if self.with_bias:
#             self.user_biases = np.zeros(self.num_users)
#             self.item_biases = np.zeros(self.num_items)
#             self.global_bias = 0.0
#         self.alpha = 0.5
        
            
#     def train(self, df, iter=100):
#         df_train = df[df[self.colname_outcome] > 0].copy()

#         if self.only_treated:
#             df_train = df_train[df_train[self.colname_treatment] > 0]

#         # Apply capping to avoid extreme values
#         if self.capping_T is not None:
#             treated_mask = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
#             df_train.loc[treated_mask, self.colname_propensity] = self.capping_T

#         if self.capping_C is not None:
#             control_mask = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
#             df_train.loc[control_mask, self.colname_propensity] = 1 - self.capping_C

#         self.propensity = df_train[self.colname_propensity].values
#         current_iter = 0
#         err = 0
#         epsilon = 1e-6

#         while True:
#             df_train = df_train.sample(frac=1)
#             users = df_train[self.colname_user].values
#             items = df_train[self.colname_item].values
#             treat = df_train[self.colname_treatment].values
#             outcome = df_train[self.colname_outcome].values
#             prop = np.clip(self.propensity * self.alpha, 1e-3, 1 - 1e-3)

#             for n in range(len(df_train)):
#                 u = users[n]
#                 i = items[n]
#                 t = treat[n]
#                 y = outcome[n]
#                 p = prop[n]

#                 ite = t * y / p - (1 - t) * y / (1 - p)
#                 z_y_1 = t * y
#                 z_y_0 = (1 - t) * y

#                 # Skip if ITE is nan or inf
#                 if np.isnan(ite) or np.isinf(ite):
#                     continue

#                 # Sample a negative item
#                 while True:
#                     j = random.randrange(self.num_items)
#                     if j != i:
#                         break

#                 u_factor = self.user_factors[u]
#                 i_factor = self.item_factors[i]
#                 j_factor = self.item_factors[j]

#                 diff_rating = np.dot(u_factor, i_factor - j_factor)
#                 if self.with_bias:
#                     diff_rating += self.item_biases[i] - self.item_biases[j]

#                 if self.metric == 'AR_logi':
#                     if ite >= 0:
#                         coeff = ite * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
#                         const_value = z_y_1 * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
#                     else:
#                         coeff = ite * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
#                         const_value = z_y_0 * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
#                 else:
#                     # Fallback if metric is not supported
#                     continue

#                 # Skip if coeff is nan or inf
#                 if np.isnan(coeff) or np.isinf(coeff):
#                     continue

#                 err += abs(coeff)

#                 # SGD updates
#                 self.user_factors[u] += self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
#                 self.item_factors[i] += self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
#                 self.item_factors[j] += self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

#                 alpha_grad = const_value / (np.power(self.alpha, 2) + epsilon)
#                 self.alpha += self.learn_rate * alpha_grad
#                 self.alpha = np.clip(self.alpha, 1e-3, 10)

#                 if self.with_bias:
#                     self.item_biases[i] += self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                     self.item_biases[j] += self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

#                 current_iter += 1
#                 if current_iter % 10000 == 0:
#                     print(f"Iter: {current_iter} | Error: {err / current_iter:.6f} | Alpha: {self.alpha:.6f}")

#                 if current_iter >= iter:
#                     return self.alpha


#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred = np.zeros(len(df))
#         for n in np.arange(len(df)):
#             pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
#             if self.with_bias:
#                 pred[n] += self.item_biases[items[n]]
#                 pred[n] += self.user_biases[users[n]]
#                 pred[n] += self.global_bias

#         # pred = 1 / (1 + np.exp(-pred))
#         return pred



# class DLMF(Recommender):
#     def __init__(self, num_users, num_items,
#                  metric='AR_logi', capping_T=0.01, capping_C=0.01,
#                  dim_factor=200, with_bias=False, with_IPS=True,
#                  only_treated=False,
#                  learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
#                  sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
#                  coeff_T = 1.0, coeff_C = 1.0,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)
#         self.metric = metric
#         self.capping_T = capping_T
#         self.capping_C = capping_C
#         self.with_IPS = with_IPS
#         self.dim_factor = dim_factor
#         self.rng = RandomState(seed=None)
#         self.with_bias = with_bias
#         self.coeff_T = coeff_T
#         self.coeff_C = coeff_C
#         self.learn_rate = learn_rate
#         self.reg_bias = reg_factor
#         self.reg_factor = reg_factor
#         self.reg_bias_j = reg_factor
#         self.reg_factor_j = reg_factor
#         self.sd_init = sd_init
#         self.only_treated = only_treated

#         self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
#         self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         if self.with_bias:
#             self.user_biases = np.zeros(self.num_users)
#             self.item_biases = np.zeros(self.num_items)
#             self.global_bias = 0.0

#     def train(self, df, iter = 100):
#         df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :] # need only positive outcomes
#         if self.only_treated: # train only with treated positive (DLTO)
#             df_train = df_train.loc[df_train.loc[:, self.colname_treatment] > 0, :]

#         if self.capping_T is not None:
#             bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] < self.capping_T, df_train.loc[:, self.colname_treatment] == 1)
#             if np.sum(bool_cap) > 0:
#                 df_train.loc[bool_cap, self.colname_propensity] = self.capping_T
#         if self.capping_C is not None:      
#             bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] > 1 - self.capping_C, df_train.loc[:, self.colname_treatment] == 0)
#             if np.sum(bool_cap) > 0:
#                 df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

#         if self.with_IPS: # point estimate of individual treatment effect (ITE) <- for binary outcome abs(ITE) = IPS
#             df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]/df_train.loc[:, self.colname_propensity] - \
#                                       (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]/(1 - df_train.loc[:, self.colname_propensity])
#             z_y_1 = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]
#             z_y_1 = z_y_1.values
#             z_y_0 = (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]
#             z_y_0 = z_y_0.values

#         else:
#             df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]  - \
#                                       (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]

#         err = 0
#         current_iter = 0
#         while True:
#             df_train = df_train.sample(frac=1)
#             users = df_train.loc[:, self.colname_user].values
#             items = df_train.loc[:, self.colname_item].values
#             ITE = df_train.loc[:, 'ITE'].values

#             if self.metric in ['AR_logi', 'AR_sig', 'AR_hinge']:
#                 for n in np.arange(len(df_train)):

#                     u = users[n]
#                     i = items[n]

#                     while True:
#                         j = random.randrange(self.num_items)
#                         if i != j:
#                             break

#                     u_factor = self.user_factors[u, :]
#                     i_factor = self.item_factors[i, :]
#                     j_factor = self.item_factors[j, :]

#                     diff_rating = np.sum(u_factor * (i_factor - j_factor))
#                     if self.with_bias:
#                         diff_rating += (self.item_biases[i] - self.item_biases[j])

#                     if self.metric == 'AR_logi':
#                         if ITE[n] >= 0:
#                             coeff = ITE[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating) # Z=1, Y=1
#                             const_value = z_y_1[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
#                         else:
#                             coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) # Z=0, Y=1
#                             const_value = z_y_0[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)

#                     elif self.metric == 'AR_sig':
#                         if ITE[n] >= 0:
#                             coeff = ITE[n] * self.coeff_T * self.func_sigmoid(self.coeff_T * diff_rating) * self.func_sigmoid(-self.coeff_T * diff_rating)
#                         else:
#                             coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) * self.func_sigmoid(-self.coeff_C * diff_rating)

#                     elif self.metric == 'AR_hinge':
#                         if ITE[n] >= 0:
#                             if self.coeff_T > 0 and diff_rating < 1.0/self.coeff_T:
#                                 coeff = ITE[n] * self.coeff_T 
#                             else:
#                                 coeff = 0.0
#                         else:
#                             if self.coeff_C > 0 and diff_rating > -1.0/self.coeff_C:
#                                 coeff = ITE[n] * self.coeff_C
#                             else:
#                                 coeff = 0.0

#                     err += np.abs(coeff)

#                     self.user_factors[u, :] += \
#                         self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
#                     self.item_factors[i, :] += \
#                         self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
#                     self.item_factors[j, :] += \
#                         self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

#                     if self.with_bias:
#                         self.item_biases[i] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                         self.item_biases[j] += \
#                             self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

#                     current_iter += 1
#                     if current_iter % 100000 == 0:
#                         print(str(current_iter)+"/"+str(iter))
#                         print()
#                         assert not np.isnan(coeff)
#                         assert not np.isinf(coeff)
#                         print("z_y_1 mean:", np.mean(z_y_1), "z_y_0 mean:", np.mean(z_y_0))
#                         print("Error:", err / iter)

#                     if current_iter >= iter:
#                         with open("dlmf_weights.pkl", "wb") as f:
#                             pickle.dump(self.__dict__, f)
#                             print("DLMF weights saved.")
#                         return err / iter

#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred = np.zeros(len(df))
#         for n in np.arange(len(df)):
#             pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
#             if self.with_bias:
#                 pred[n] += self.item_biases[items[n]]
#                 pred[n] += self.user_biases[users[n]]
#                 pred[n] += self.global_bias

#         # pred = 1 / (1 + np.exp(-pred))
#         return pred
    
# class DLMF4(Recommender):
#     def __init__(self, num_users, num_items,
#                  metric='AR_logi', capping_T=0.01, capping_C=0.01,
#                  dim_factor=200, with_bias=False, with_IPS=True,
#                  only_treated=False, with_DR=False,
#                  learn_rate=0.01, reg_factor=0.01, reg_bias=0.01,
#                  sd_init=0.1, reg_factor_j=0.01, reg_bias_j=0.01,
#                  coeff_T=1.0, coeff_C=1.0,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)
#         self.metric = metric
#         self.capping_T = capping_T
#         self.capping_C = capping_C
#         self.with_IPS = with_IPS
#         self.with_DR = with_DR
#         self.dim_factor = dim_factor
#         self.rng = RandomState(seed=None)
#         self.with_bias = with_bias
#         self.coeff_T = coeff_T
#         self.coeff_C = coeff_C
#         self.learn_rate = learn_rate
#         self.reg_bias = reg_factor
#         self.reg_factor = reg_factor
#         self.reg_bias_j = reg_factor
#         self.reg_factor_j = reg_factor
#         self.sd_init = sd_init
#         self.only_treated = only_treated

#         self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
#         self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         if self.with_bias:
#             self.user_biases = np.zeros(self.num_users)
#             self.item_biases = np.zeros(self.num_items)
#             self.global_bias = 0.0

#     @staticmethod
#     def func_sigmoid(x):
#         return 1 / (1 + np.exp(-x))

#     def train(self, df, iter=100, omega=1.0):
#         df_train = df[df[self.colname_outcome] > 0]
#         if self.only_treated:
#             df_train = df_train[df_train[self.colname_treatment] > 0]

#         if self.capping_T is not None:
#             mask = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
#             df_train.loc[mask, self.colname_propensity] = self.capping_T
#         if self.capping_C is not None:
#             mask = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
#             df_train.loc[mask, self.colname_propensity] = 1 - self.capping_C

#         if self.with_DR:
#             from sklearn.linear_model import LogisticRegression
#             X = df_train[[self.colname_user, self.colname_item]].values
#             y = df_train[self.colname_outcome].values
#             z = df_train[self.colname_treatment].values
#             X1 = X[z == 1]; y1 = y[z == 1]
#             X0 = X[z == 0]; y0 = y[z == 0]
#             self.model_treated = LogisticRegression().fit(X1, y1)
#             self.model_control = LogisticRegression().fit(X0, y0)

#         df_train = df_train.sample(frac=1)
#         users = df_train[self.colname_user].values
#         items = df_train[self.colname_item].values
#         outcomes = df_train[self.colname_outcome].values
#         props = df_train[self.colname_propensity].values
#         treats = df_train[self.colname_treatment].values

#         err = 0
#         current_iter = 0

#         for n in range(len(df_train)):
#             u = users[n]
#             i = items[n]
#             y = outcomes[n]
#             p = props[n]
#             z = treats[n]

#             while True:
#                 j = random.randrange(self.num_items)
#                 if i != j:
#                     break

#             u_factor = self.user_factors[u, :]
#             i_factor = self.item_factors[i, :]
#             j_factor = self.item_factors[j, :]

#             diff_rating = np.sum(u_factor * (i_factor - j_factor))
#             if self.with_bias:
#                 diff_rating += (self.item_biases[i] - self.item_biases[j])

#             if self.with_DR:
#                 x_ij = np.array([[u, i]])
#                 y1_hat = self.model_treated.predict_proba(x_ij)[0, 1]
#                 y0_hat = self.model_control.predict_proba(x_ij)[0, 1]
#                 ite = (z / p - (1 - z) / (1 - p)) * (y - (z * y1_hat + (1 - z) * y0_hat)) + (y1_hat - y0_hat)
#             elif self.with_IPS:
#                 ite = z * y / p - (1 - z) * y / (1 - p)
#             else:
#                 ite = z * y - (1 - z) * y

#             if ite >= 0:
#                 loss = np.log(1 + np.exp(-omega * diff_rating)) * self.coeff_T * ite
#             else:
#                 loss = np.log(1 + np.exp(omega * diff_rating)) * self.coeff_C * ite

#             grad = -omega * self.func_sigmoid(-omega * diff_rating) * ite if ite >= 0 else omega * self.func_sigmoid(omega * diff_rating) * ite

#             self.user_factors[u, :] += self.learn_rate * (grad * (i_factor - j_factor) - self.reg_factor * u_factor)
#             self.item_factors[i, :] += self.learn_rate * (grad * u_factor - self.reg_factor * i_factor)
#             self.item_factors[j, :] += self.learn_rate * (-grad * u_factor - self.reg_factor_j * j_factor)

#             if self.with_bias:
#                 self.item_biases[i] += self.learn_rate * (grad - self.reg_bias * self.item_biases[i])
#                 self.item_biases[j] += self.learn_rate * (-grad - self.reg_bias_j * self.item_biases[j])

#             err += np.abs(grad)
#             current_iter += 1
#             if current_iter % 100000 == 0:
#                 print(f"{current_iter}/{iter}")
#                 print("Error:", err / iter)

#             if current_iter >= iter:
#                 with open("dlmf_weights.pkl", "wb") as f:
#                     pickle.dump(self.__dict__, f)
#                     print("DLMF weights saved.")
#                 return err / iter


# class DLMF3(Recommender):
#     def __init__(self, num_users, num_items,
#                  metric='AR_logi', capping_T=0.01, capping_C=0.01,
#                  dim_factor=200, with_bias=False, with_IPS=True,
#                  only_treated=False, use_DR=False,
#                  learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
#                  sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
#                  coeff_T = 1.0, coeff_C = 1.0,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):

#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)
#         self.metric = metric
#         self.capping_T = capping_T
#         self.capping_C = capping_C
#         self.with_IPS = with_IPS
#         self.use_DR = use_DR
#         self.dim_factor = dim_factor
#         self.rng = RandomState(seed=None)
#         self.with_bias = with_bias
#         self.coeff_T = coeff_T
#         self.coeff_C = coeff_C
#         self.learn_rate = learn_rate
#         self.reg_bias = reg_factor
#         self.reg_factor = reg_factor
#         self.reg_bias_j = reg_factor
#         self.reg_factor_j = reg_factor
#         self.sd_init = sd_init
#         self.only_treated = only_treated

#         self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
#         self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         if self.with_bias:
#             self.user_biases = np.zeros(self.num_users)
#             self.item_biases = np.zeros(self.num_items)
#             self.global_bias = 0.0

#     def func_sigmoid(self, x):
#         return 1.0 / (1.0 + np.exp(-x))

#     def train(self, df, iter = 100):
#         df_train = df[df[self.colname_outcome] > 0].copy()
#         if self.only_treated:
#             df_train = df_train[df_train[self.colname_treatment] == 1]

#         if self.capping_T is not None:
#             bool_cap = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
#             df_train.loc[bool_cap, self.colname_propensity] = self.capping_T

#         if self.capping_C is not None:
#             bool_cap = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
#             df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

#         if self.use_DR:
#             dr_estimator = DR_Estimator(self.num_users, self.num_items, dim_factor=self.dim_factor,
#                                         learn_rate=self.learn_rate, reg_factor=self.reg_factor,
#                                         with_bias=self.with_bias, colname_user=self.colname_user,
#                                         colname_item=self.colname_item, colname_outcome=self.colname_outcome,
#                                         colname_treatment=self.colname_treatment,
#                                         colname_propensity=self.colname_propensity)
#             dr_estimator.train(df_train, iter=iter)
#             df_train['ITE'] = dr_estimator.compute_DR(df_train)
#         else:
#             if self.with_IPS:
#                 df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] / df_train[self.colname_propensity] - \
#                                    (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome] / (1 - df_train[self.colname_propensity])
#             else:
#                 df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] - \
#                                    (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome]

#         z_y_1 = df_train[self.colname_treatment] * df_train[self.colname_outcome]
#         z_y_1 = z_y_1.values
#         z_y_0 = (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome]
#         z_y_0 = z_y_0.values

#         err = 0
#         current_iter = 0
#         while True:
#             df_train = df_train.sample(frac=1)
#             users = df_train[self.colname_user].values
#             items = df_train[self.colname_item].values
#             ITE = df_train['ITE'].values

#             for n in np.arange(len(df_train)):
#                 u = users[n]
#                 i = items[n]
#                 while True:
#                     j = random.randrange(self.num_items)
#                     if i != j:
#                         break

#                 u_factor = self.user_factors[u, :]
#                 i_factor = self.item_factors[i, :]
#                 j_factor = self.item_factors[j, :]

#                 diff_rating = np.sum(u_factor * (i_factor - j_factor))
#                 if self.with_bias:
#                     diff_rating += self.item_biases[i] - self.item_biases[j]

#                 ite = ITE[n]
#                 if self.metric == 'AR_logi':
#                     if ite >= 0:
#                         coeff = ite * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
#                     else:
#                         coeff = ite * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
#                 elif self.metric == 'AR_sig':
#                     if ite >= 0:
#                         sig = self.func_sigmoid(self.coeff_T * diff_rating)
#                         coeff = ite * self.coeff_T * sig * (1 - sig)
#                     else:
#                         sig = self.func_sigmoid(self.coeff_C * diff_rating)
#                         coeff = ite * self.coeff_C * sig * (1 - sig)
#                 elif self.metric == 'AR_hinge':
#                     if ite >= 0 and diff_rating < 1.0 / self.coeff_T:
#                         coeff = ite * self.coeff_T
#                     elif ite < 0 and diff_rating > -1.0 / self.coeff_C:
#                         coeff = ite * self.coeff_C
#                     else:
#                         coeff = 0.0

#                 err += np.abs(coeff)
#                 self.user_factors[u, :] += self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
#                 self.item_factors[i, :] += self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
#                 self.item_factors[j, :] += self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

#                 if self.with_bias:
#                     self.item_biases[i] += self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                     self.item_biases[j] += self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

#                 current_iter += 1
#                 if current_iter >= iter:
#                     with open("dlmf_weights.pkl", "wb") as f:
#                         pickle.dump(self.__dict__, f)
#                         print("DLMF weights saved.")
#                     return err / iter
                            
#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred = np.zeros(len(df))
#         for n in np.arange(len(df)):
#             pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
#             if self.with_bias:
#                 pred[n] += self.item_biases[items[n]]
#                 pred[n] += self.user_biases[users[n]]
#                 pred[n] += self.global_bias
#         return pred



# class RandomBase(Recommender):

#     def __init__(self, num_users, num_items,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction)

#     def train(self, df, iter = 1):
#         pass

#     def predict(self, df):
#         return np.random.rand(df.shape[0])



# class MF(Recommender):
#     def __init__(self, num_users, num_items,
#                  metric='RMSE',
#                  dim_factor=200, with_bias=False,
#                  learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)
#         self.metric = metric
#         self.dim_factor = dim_factor
#         self.rng = RandomState(seed=None)
#         self.with_bias = with_bias

#         self.learn_rate = learn_rate
#         self.reg_bias = reg_bias
#         self.reg_factor = reg_factor
#         self.sd_init = sd_init

#         self.flag_prepared = False

#         self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
#         self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         if self.with_bias:
#             self.user_biases = np.zeros(self.num_users)
#             self.item_biases = np.zeros(self.num_items)
#             self.global_bias = 0.0

#     def prepare_dictionary(self, df, colname_time='idx_time'):
#         print("start prepare dictionary")
#         self.colname_time = colname_time
#         self.num_times = np.max(df.loc[:, self.colname_time]) + 1
#         self.dict_positive_sets = dict()
    
#         df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
    
#         for t in np.arange(self.num_times):
#             df_t = df_posi.loc[df_posi.loc[:, self.colname_time] == t]
#             self.dict_positive_sets[t] = dict()
#             for u in np.unique(df_t.loc[:, self.colname_user]):
#                 self.dict_positive_sets[t][u] = \
#                     np.unique(df_t.loc[df_t.loc[:, self.colname_user] == u, self.colname_item].values)
    
#         self.flag_prepared = True
#         print("prepared dictionary!")


#     def train(self, df, iter = 100):

#         # by default, rating prediction
#         # outcome = rating
#         df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

#         # # in case of binary implicit feedback
#         if self.metric == 'logloss':
#             df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
#             if not self.flag_prepared: # prepare dictionary
#                 self.prepare_dictionary(df)
#         else:
#             df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

#         err = 0
#         current_iter = 0
#         while True:
#             if self.metric == 'RMSE':
#                 df_train = df_train.sample(frac=1)
#                 users = df_train.loc[:, self.colname_user].values
#                 items = df_train.loc[:, self.colname_item].values
#                 outcomes = df_train.loc[:, self.colname_outcome].values

#                 for n in np.arange(len(df_train)):
#                     u = users[n]
#                     i = items[n]
#                     r = outcomes[n]

#                     u_factor = self.user_factors[u, :]
#                     i_factor = self.item_factors[i, :]

#                     rating = np.sum(u_factor * i_factor)
#                     if self.with_bias:
#                         rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

#                     coeff = r - rating
#                     err += np.abs(coeff)

#                     self.user_factors[u, :] += \
#                         self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
#                     self.item_factors[i, :] += \
#                         self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

#                     if self.with_bias:
#                         self.item_biases[i] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                         self.user_biases[u] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
#                         self.global_bias += \
#                             self.learn_rate * (coeff)

#                     current_iter += 1
#                     if current_iter >= iter:
#                         if current_iter % 100000 == 0:
#                             print(str(current_iter)+"/"+str(iter))
#                         return err / iter

#             elif self.metric == 'logloss': # logistic matrix factorization
#                 df_train = df_train.sample(frac=1)
#                 users = df_train.loc[:, self.colname_user].values
#                 items = df_train.loc[:, self.colname_item].values
#                 outcomes = df_train.loc[:, self.colname_outcome].values

#                 for n in np.arange(len(df_train)):
#                     u = users[n]
#                     i = items[n]
#                     r = outcomes[n]

#                     u_factor = self.user_factors[u, :]
#                     i_factor = self.item_factors[i, :]

#                     rating = np.sum(u_factor * i_factor)
#                     if self.with_bias:
#                         rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

#                     if r > 0:
#                         coeff = self.func_sigmoid(-rating)
#                     else:
#                         coeff = - self.func_sigmoid(rating)

#                     self.user_factors[u, :] += \
#                         self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
#                     self.item_factors[i, :] += \
#                         self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

#                     if self.with_bias:
#                         self.item_biases[i] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                         self.user_biases[u] += \
#                             self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
#                         self.global_bias += \
#                             self.learn_rate * (coeff)

#                     current_iter += 1
#                     if current_iter >= iter:
                
#                         return err / iter

#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred = np.zeros(len(df))
#         for n in np.arange(len(df)):
#             pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
#             if self.with_bias:
#                 pred[n] += self.item_biases[items[n]]
#                 pred[n] += self.user_biases[users[n]]
#                 pred[n] += self.global_bias

#         if self.metric == 'logloss':
#             pred = 1 / (1 + np.exp(-pred))
#         return pred


# class DR_Estimator:
#     def __init__(self, num_users, num_items,
#                  dim_factor=64, learn_rate=0.01, reg_factor=0.01, with_bias=True,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_treatment='treated',
#                  colname_propensity='propensity'):

#         self.num_users = num_users
#         self.num_items = num_items
#         self.dim_factor = dim_factor
#         self.learn_rate = learn_rate
#         self.reg_factor = reg_factor
#         self.with_bias = with_bias

#         self.colname_user = colname_user
#         self.colname_item = colname_item
#         self.colname_outcome = colname_outcome
#         self.colname_treatment = colname_treatment
#         self.colname_propensity = colname_propensity

#         self.model_T = MF(num_users, num_items, dim_factor=dim_factor,
#                           learn_rate=learn_rate, reg_factor=reg_factor, with_bias=with_bias,
#                           colname_user=colname_user, colname_item=colname_item,
#                           colname_outcome=colname_outcome, colname_treatment=colname_treatment,
#                           colname_propensity=colname_propensity)

#         self.model_C = MF(num_users, num_items, dim_factor=dim_factor,
#                           learn_rate=learn_rate, reg_factor=reg_factor, with_bias=with_bias,
#                           colname_user=colname_user, colname_item=colname_item,
#                           colname_outcome=colname_outcome, colname_treatment=colname_treatment,
#                           colname_propensity=colname_propensity)

#     def train(self, df, iter=10):
#         df_T = df[df[self.colname_treatment] == 1].copy()
#         df_C = df[df[self.colname_treatment] == 0].copy()

#         self.model_T.train(df_T, iter=iter)
#         self.model_C.train(df_C, iter=iter)

#     def compute_DR(self, df):
#         Z = df[self.colname_treatment].values
#         Y = df[self.colname_outcome].values
#         P = df[self.colname_propensity].values

#         y_T_hat = self.model_T.predict(df)
#         y_C_hat = self.model_C.predict(df)

#         term_T = Z * (Y - y_T_hat) / np.clip(P, 1e-3, 1) + y_T_hat
#         term_C = (1 - Z) * (Y - y_C_hat) / np.clip(1 - P, 1e-3, 1) + y_C_hat

#         return term_T - term_C


# class CausalNeighborBase(Recommender):
#     def __init__(self, num_users, num_items,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  measure_simil='cosine', way_simil='treatment',
#                  way_neighbor='user', num_neighbor=3000,
#                  way_self='exclude',
#                  weight_treated_outcome=0.5,
#                  shrinkage_T=10.0, shrinkage_C=10.0,
#                  scale_similarity=0.33, normalize_similarity=False):

#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction)
#         self.measure_simil = measure_simil
#         self.way_simil = way_simil
#         self.way_neighbor = way_neighbor
#         self.num_neighbor = num_neighbor
#         self.scale_similarity = scale_similarity
#         self.normalize_similarity = normalize_similarity
#         self.weight_treated_outcome = weight_treated_outcome
#         self.shrinkage_T = shrinkage_T
#         self.shrinkage_C = shrinkage_C
#         self.way_self = way_self # exclude/include/only


#     def simil(self, set1, set2, measure_simil):
#         if measure_simil == "jaccard":
#             return self.simil_jaccard(set1, set2)
#         elif measure_simil == "cosine":
#             return self.simil_cosine(set1, set2)

#     def train(self, df, iter=1):
#         df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
#         print("len(df_posi): {}".format(len(df_posi)))

#         dict_items2users = dict() # map an item to users who consumed the item
#         for i in np.arange(self.num_items):
#             dict_items2users[i] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_item] == i, self.colname_user].values)
#         self.dict_items2users = dict_items2users
#         print("prepared dict_items2users")

#         dict_users2items = dict()  # map an user to items which are consumed by the user
#         for u in np.arange(self.num_users):
#             dict_users2items[u] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_user] == u, self.colname_item].values)
#         self.dict_users2items = dict_users2items
#         print("prepared dict_users2items")

#         df_treated = df.loc[df.loc[:, self.colname_treatment] > 0]  # calc similarity by treatment assignment
#         print("len(df_treated): {}".format(len(df_treated)))

#         dict_items2users_treated = dict() # map an item to users who get treatment of the item
#         for i in np.arange(self.num_items):
#             dict_items2users_treated[i] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_item] == i, self.colname_user].values)
#         self.dict_items2users_treated = dict_items2users_treated
#         print("prepared dict_items2users_treated")

#         dict_users2items_treated = dict()  # map an user to items which are treated to the user
#         for u in np.arange(self.num_users):
#             dict_users2items_treated[u] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_user] == u, self.colname_item].values)
#         self.dict_users2items_treated = dict_users2items_treated
#         print("prepared dict_users2items_treated")

#         if self.way_simil == 'treatment':
#             if self.way_neighbor == 'user':
#                 dict_simil_users = {}
#                 sum_simil = np.zeros(self.num_users)
#                 for u1 in np.arange(self.num_users):
#                     if u1 % round(self.num_users/10) == 0:
#                         print("progress of similarity computation: {:.1f} %".format(100 * u1/self.num_users))

#                     items_u1 = self.dict_users2items_treated[u1]
#                     dict_neighbor = {}
#                     if len(items_u1) > 0:
#                         cand_u2 = np.unique(df_treated.loc[np.isin(df_treated.loc[:, self.colname_item], items_u1), self.colname_user].values)
#                         for u2 in cand_u2:
#                             if u2 != u1:
#                                 items_u2 = self.dict_users2items_treated[u2]
#                                 dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

#                         # print("len(dict_neighbor): {}".format(len(dict_neighbor)))
#                         if len(dict_neighbor) > self.num_neighbor:
#                             dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
#                         if self.scale_similarity != 1.0:
#                             dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
#                         if self.normalize_similarity:
#                             dict_neighbor = self.normalize_neighbor(dict_neighbor)
#                         dict_simil_users[u1] = dict_neighbor
#                         sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
#                     else:
#                         dict_simil_users[u1] = dict_neighbor
#                 self.dict_simil_users = dict_simil_users
#                 self.sum_simil = sum_simil

#             elif self.way_neighbor == 'item':
#                 dict_simil_items = {}
#                 sum_simil = np.zeros(self.num_items)
#                 for i1 in np.arange(self.num_items):
#                     if i1 % round(self.num_items/10) == 0:
#                         print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

#                     users_i1 = self.dict_items2users_treated[i1]
#                     dict_neighbor = {}
#                     if len(users_i1) > 0:
#                         cand_i2 = np.unique(
#                             df_treated.loc[np.isin(df_treated.loc[:, self.colname_user], users_i1), self.colname_item].values)
#                         for i2 in cand_i2:
#                             if i2 != i1:
#                                 users_i2 = self.dict_items2users_treated[i2]
#                                 dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

#                         if len(dict_neighbor) > self.num_neighbor:
#                             dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
#                         if self.scale_similarity != 1.0:
#                             dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
#                         if self.normalize_similarity:
#                             dict_neighbor = self.normalize_neighbor(dict_neighbor)
#                         dict_simil_items[i1] = dict_neighbor
#                         sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
#                     else:
#                         dict_simil_items[i1] = dict_neighbor
#                 self.dict_simil_items = dict_simil_items
#                 self.sum_simil = sum_simil
#         else:
#             if self.way_neighbor == 'user':
#                 dict_simil_users = {}
#                 sum_simil = np.zeros(self.num_users)
#                 for u1 in np.arange(self.num_users):
#                     if u1 % round(self.num_users/10) == 0:
#                         print("progress of similarity computation: {:.1f} %".format(100 * u1 / self.num_users))

#                     items_u1 = self.dict_users2items[u1]
#                     dict_neighbor = {}
#                     if len(items_u1) > 0:
#                         cand_u2 = np.unique(
#                             df_posi.loc[np.isin(df_posi.loc[:, self.colname_item], items_u1), self.colname_user].values)
#                         for u2 in cand_u2:
#                             if u2 != u1:
#                                 items_u2 = self.dict_users2items[u2]
#                                 dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

#                         if len(dict_neighbor) > self.num_neighbor:
#                             dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
#                         if self.scale_similarity != 1.0:
#                             dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
#                         if self.normalize_similarity:
#                             dict_neighbor = self.normalize_neighbor(dict_neighbor)
#                         dict_simil_users[u1] = dict_neighbor
#                         sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
#                     else:
#                         dict_simil_users[u1] = dict_neighbor
#                 self.dict_simil_users = dict_simil_users
#                 self.sum_simil = sum_simil

#             elif self.way_neighbor == 'item':
#                 dict_simil_items = {}
#                 sum_simil = np.zeros(self.num_items)
#                 for i1 in np.arange(self.num_items):
#                     if i1 % round(self.num_items/10) == 0:
#                         print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

#                     users_i1 = self.dict_items2users[i1]
#                     dict_neighbor = {}
#                     if len(users_i1) > 0:
#                         cand_i2 = np.unique(
#                             df_posi.loc[np.isin(df_posi.loc[:, self.colname_user], users_i1), self.colname_item].values)
#                         for i2 in cand_i2:
#                             if i2 != i1:
#                                 users_i2 = self.dict_items2users[i2]
#                                 dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

#                         if len(dict_neighbor) > self.num_neighbor:
#                             dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
#                         if self.scale_similarity != 1.0:
#                             dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
#                         if self.normalize_similarity:
#                             dict_neighbor = self.normalize_neighbor(dict_neighbor)
#                         dict_simil_items[i1] = dict_neighbor
#                         sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
#                     else:
#                         dict_simil_items[i1] = dict_neighbor
#                 self.dict_simil_items = dict_simil_items
#                 self.sum_simil = sum_simil


#     def trim_neighbor(self, dict_neighbor, num_neighbor):
#         return dict(sorted(dict_neighbor.items(), key=lambda x:x[1], reverse = True)[:num_neighbor])

#     def normalize_neighbor(self, dict_neighbor):
#         sum_simil = 0.0
#         for v in dict_neighbor.values():
#             sum_simil += v
#         for k, v in dict_neighbor.items():
#             dict_neighbor[k] = v/sum_simil
#         return dict_neighbor

#     def rescale_neighbor(self, dict_neighbor, scaling_similarity=1.0):
#         for k, v in dict_neighbor.items():
#             dict_neighbor[k] = np.power(v, scaling_similarity)
#         return dict_neighbor


#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred = np.zeros(len(df))
#         if self.way_neighbor == 'user':
#             for n in np.arange(len(df)):
#                 u1 = users[n]
#                 simil_users = np.fromiter(self.dict_simil_users[u1].keys(), dtype=int)
#                 i_users_posi = self.dict_items2users[items[n]]  # users who consumed i=items[n]
#                 i_users_treated = self.dict_items2users_treated[items[n]]  # users who are treated i=items[n]
#                 if n % round(len(df)/10) == 0:
#                     print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))
#                     # print("simil_users")
#                     # print(simil_users)
#                     # print(type(simil_users))
#                     # print(np.any(np.isin(simil_users, i_users_posi)))

#                 # initialize for this u1-i pair
#                 value_T = 0.0
#                 denom_T = 0.0
#                 value_C = 0.0
#                 denom_C = 0.0

#                 if np.any(np.isin(simil_users, i_users_posi)):
#                     simil_users = simil_users[np.isin(simil_users, np.unique(np.append(i_users_treated,i_users_posi)))]
#                     for u2 in simil_users:
#                         if u2 in i_users_treated:
#                             denom_T += self.dict_simil_users[u1][u2]
#                             if u2 in i_users_posi:
#                                 value_T += self.dict_simil_users[u1][u2]
#                         else:
#                             value_C += self.dict_simil_users[u1][u2]
#                             # denom_C += self.dict_simil_users[u1][u2]
#                             # if u2 in i_users_posi:
#                             #     value_C += self.dict_simil_users[u1][u2]
#                     denom_C = self.sum_simil[u1] - denom_T # denom_T + denom_C = sum_simil

#                 if self.way_self == 'include': # add data of self u-i
#                     if u1 in i_users_treated:
#                         denom_T += 1.0
#                         if u1 in i_users_posi:
#                             value_T += 1.0
#                     else:
#                         denom_C += 1.0
#                         if u1 in i_users_posi:
#                             value_C += 1.0

#                 if self.way_self == 'only': # force data to self u-i
#                     if u1 in i_users_treated:
#                         denom_T = 1.0
#                         if u1 in i_users_posi:
#                             value_T = 1.0
#                         else:
#                             value_T = 0.0
#                     else:
#                         denom_C = 1.0
#                         if u1 in i_users_posi:
#                             value_C = 1.0
#                         else:
#                             value_C = 0.0

#                 if value_T > 0:
#                     pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
#                 if value_C > 0:
#                     pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)
#             print(pred[:5])
#             print(np.mean(pred))
#             print(np.max(pred))
#             print(np.min(pred))

#         elif self.way_neighbor == 'item':
#             for n in np.arange(len(df)):
#                 i1 = items[n]
#                 simil_items = np.fromiter(self.dict_simil_items[i1].keys(), dtype=int)
#                 u_items_posi = self.dict_users2items[users[n]]  # items that is consumed by u=users[n]
#                 u_items_treated = self.dict_users2items_treated[users[n]] # items that is treated for u=users[n]
#                 if n % round(len(df)/10) == 0:
#                     print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))

#                 # initialize for this u-i1 pair
#                 value_T = 0.0
#                 denom_T = 0.0
#                 value_C = 0.0
#                 denom_C = 0.0

#                 if np.any(np.isin(simil_items, u_items_posi)):
#                     simil_items = simil_items[np.isin(simil_items, np.unique(np.append(u_items_posi, u_items_treated)))]
#                     for i2 in simil_items:
#                         if i2 in u_items_treated: # we assume that treated items are less than untreated items
#                             denom_T += self.dict_simil_items[i1][i2]
#                             if i2 in u_items_posi:
#                                 value_T += self.dict_simil_items[i1][i2]
#                         else:
#                             value_C += self.dict_simil_items[i1][i2]
#                             # denom_C += self.dict_simil_items[i1][i2]
#                             # if i2 in u_items_posi:
#                             #     value_C += self.dict_simil_items[i1][i2]
#                     denom_C = self.sum_simil[i1] - denom_T  # denom_T + denom_C = sum_simil

#                 if self.way_self == 'include': # add data of self u-i
#                     if i1 in u_items_treated:
#                         denom_T += 1.0
#                         if i1 in u_items_posi:
#                             value_T += 1.0
#                     else:
#                         denom_C += 1.0
#                         if i1 in u_items_posi:
#                             value_C += 1.0

#                 if self.way_self == 'only': # force data to self u-i
#                     if i1 in u_items_treated:
#                         denom_T = 1.0
#                         if i1 in u_items_posi:
#                             value_T = 1.0
#                         else:
#                             value_T = 0.0
#                     else:
#                         denom_C = 1.0
#                         if i1 in u_items_posi:
#                             value_C = 1.0
#                         else:
#                             value_C = 0.0

#                 if value_T > 0:
#                     pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
#                 if value_C > 0:
#                     pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)

#         return pred


#     def simil_jaccard(self, x, y):
#         return len(np.intersect1d(x, y))/len(np.union1d(x, y))

#     def simil_cosine(self, x, y):
#         return len(np.intersect1d(x, y))/np.sqrt(len(x)*len(y))

# class CausEProd(Recommender):
#     def __init__(self, num_users, num_items,
#                  metric='logloss',
#                  dim_factor=10, with_bias=False,
#                  learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
#                  reg_causal=0.01,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):
#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)
#         self.metric = metric

#         self.dim_factor = dim_factor
#         self.rng = RandomState(seed=None)
#         self.with_bias = with_bias

#         self.learn_rate = learn_rate
#         self.reg_bias = reg_bias
#         self.reg_factor = reg_factor
#         self.reg_causal = reg_causal
#         self.sd_init = sd_init
#         self.flag_prepared = False

#         # user_factors=user_factors_T=user_factors_C for CausE-Prod
#         self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
#         # item_factors_T=item_factors
#         self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         self.item_factors_C = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
#         if self.with_bias:
#             self.user_biases = np.zeros(self.num_users)
#             self.item_biases = np.zeros(self.num_items)
#             self.global_bias = 0.0


#     def prepare_dictionary(self, df, colname_time='idx_time'):
#         print("start prepare dictionary")
#         self.colname_time = colname_time
#         self.num_times = np.max(df.loc[:, self.colname_time]) + 1
#         self.dict_treatment_positive_sets = dict()
#         self.dict_treatment_negative_sets = dict()
#         self.dict_treatment_sets = dict()
#         self.dict_control_positive_sets = dict()
#         self.dict_control_negative_sets = dict()
#         # skip control_negative for its volume
#         df_train = df.loc[df.loc[:, self.colname_outcome] + df.loc[:, self.colname_treatment] > 0]

#         for t in np.arange(self.num_times):
#             df_t = df_train.loc[df_train.loc[:, self.colname_time] == t]
#             self.dict_treatment_positive_sets[t] = dict()
#             self.dict_treatment_negative_sets[t] = dict()
#             self.dict_treatment_sets[t] = dict()
#             self.dict_control_positive_sets[t] = dict()
#             self.dict_control_negative_sets[t] = dict()

#             for u in np.unique(df_t.loc[:, self.colname_user]):

#                 df_tu = df_t.loc[df_t.loc[:, self.colname_user] == u]
#                 if len(df_tu) < self.num_items:  # check existence of control negatives
#                     self.dict_control_negative_sets[t][u] = []

#                 bool_control = df_tu.loc[:, self.colname_treatment] == 0
#                 if np.any(bool_control):
#                     self.dict_control_positive_sets[t][u] = df_tu.loc[bool_control, self.colname_item].values
#                 # only treatment
#                 bool_treatment = np.logical_not(bool_control)
#                 if np.any(bool_treatment):
#                     df_tu = df_tu.loc[bool_treatment]
#                     bool_positive = df_tu.loc[:, self.colname_outcome] > 0
#                     self.dict_treatment_sets[t][u] = df_tu.loc[:, self.colname_item].values
#                     if np.any(bool_positive):
#                         self.dict_treatment_positive_sets[t][u] = df_tu.loc[bool_positive, self.colname_item].values
#                     bool_negative = np.logical_not(bool_positive)
#                     if np.any(bool_negative):
#                         self.dict_treatment_negative_sets[t][u] = df_tu.loc[bool_negative, self.colname_item].values
#                 # else:
#                 #     self.dict_treatment_sets[t][u] = []

#         self.flag_prepared = True
#         print("prepared dictionary!")


#     # override
#     def sample_pair(self):
#         t = self.sample_time()
#         if random.random() < 0.5: # pick treatment
#             flag_treatment = 1
#             while True: # pick a user with treatment
#                 u = random.randrange(self.num_users)
#                 if u in self.dict_treatment_sets[t]:
#                     break

#             i = self.sample_treatment(t, u)
#             if u in self.dict_treatment_positive_sets[t] and i in self.dict_treatment_positive_sets[t][u]:
#                 flag_positive = 1
#             else:
#                 flag_positive = 0

#         else: # pick control
#             flag_treatment = 0
#             while True: # pick a user with control
#                 u = random.randrange(self.num_users)
#                 if u in self.dict_treatment_sets[t]:
#                     len_T = len(self.dict_treatment_sets[t][u])
#                 else:
#                     len_T = 0
#                 if len_T < self.num_items:
#                     break

#             if len_T > self.num_items * 0.99:
#                 # print(len_T)
#                 i = self.sample_control2(t, u)
#             else:
#                 i = self.sample_control(t, u)

#             if u in self.dict_control_positive_sets[t] and i in self.dict_control_positive_sets[t][u]:
#                 flag_positive = 1
#             else:
#                 flag_positive = 0

#         return u, i, flag_positive, flag_treatment


#     def train(self, df, iter = 100):

#         if not self.flag_prepared: # prepare dictionary
#             self.prepare_dictionary(df)

#         err = 0
#         current_iter = 0
#         if self.metric in ['logloss']:
#             while True:
#                 u, i, flag_positive, flag_treatment = self.sample_pair()

#                 u_factor = self.user_factors[u, :]
#                 i_factor_T = self.item_factors[i, :]
#                 i_factor_C = self.item_factors_C[i, :]

#                 if flag_treatment > 0:
#                     rating = np.sum(u_factor * i_factor_T)
#                 else:
#                     rating = np.sum(u_factor * i_factor_C)

#                 if self.with_bias:
#                     rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

#                 if flag_positive > 0:
#                     coeff = self.func_sigmoid(-rating)
#                 else:
#                     coeff = -self.func_sigmoid(rating)

#                 err += np.abs(coeff)


#                 i_diff_TC = i_factor_T - i_factor_C


#                 if flag_treatment > 0:
#                     self.user_factors[u, :] += \
#                         self.learn_rate * (coeff * i_factor_T - self.reg_factor * u_factor)
#                     self.item_factors[i, :] += \
#                         self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor_T - self.reg_causal * i_diff_TC)
#                     self.item_factors_C[i, :] += \
#                         self.learn_rate * (self.reg_causal * i_diff_TC)
#                 else:
#                     self.user_factors[u, :] += \
#                         self.learn_rate * (coeff * i_factor_C - self.reg_factor * u_factor)
#                     self.item_factors_C[i, :] += \
#                         self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor_C + self.reg_causal * i_diff_TC)
#                     self.item_factors[i, :] += \
#                         self.learn_rate * (-self.reg_causal * i_diff_TC)

#                 if self.with_bias:
#                     self.item_biases[i] += \
#                         self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
#                     self.user_biases[u] += \
#                         self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
#                     self.global_bias += \
#                         self.learn_rate * (coeff)

#                 current_iter += 1
#                 if current_iter >= iter:
#                     return err / iter

#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred_T = np.zeros(len(df))
#         pred_C = np.zeros(len(df))

#         for n in np.arange(len(df)):
#             pred_T[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
#             if self.with_bias:
#                 pred_T[n] += self.item_biases[items[n]]
#                 pred_T[n] += self.user_biases[users[n]]
#                 pred_T[n] += self.global_bias
#             pred_C[n] = np.inner(self.user_factors[users[n], :], self.item_factors_C[items[n], :])
#             if self.with_bias:
#                 pred_C[n] += self.item_biases[items[n]]
#                 pred_C[n] += self.user_biases[users[n]]
#                 pred_C[n] += self.global_bias

#         pred = 1 / (1 + np.exp(-pred_T)) - 1 / (1 + np.exp(-pred_C))

#         return pred

# class DLMF_MLP(Recommender):
#     def __init__(self, num_users, num_items,
#                  metric='AR_logi', capping_T=0.01, capping_C=0.01,
#                  dim_factor=64, hidden_dim=128, with_IPS=True, only_treated=False,
#                  learn_rate=0.01, reg_factor=0.01,
#                  colname_user='idx_user', colname_item='idx_item',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity'):

#         super().__init__(num_users=num_users, num_items=num_items,
#                          colname_user=colname_user, colname_item=colname_item,
#                          colname_outcome=colname_outcome, colname_prediction=colname_prediction,
#                          colname_treatment=colname_treatment, colname_propensity=colname_propensity)

#         self.metric = metric
#         self.capping_T = capping_T
#         self.capping_C = capping_C
#         self.with_IPS = with_IPS
#         self.only_treated = only_treated
#         self.learn_rate = learn_rate
#         self.reg_factor = reg_factor
#         self.dim_factor = dim_factor
#         self.hidden_dim = hidden_dim

#         self.rng = RandomState(seed=42)
#         self.user_factors = self.rng.normal(0, 0.1, size=(self.num_users, dim_factor))
#         self.item_factors = self.rng.normal(0, 0.1, size=(self.num_items, dim_factor))

#         # MLP параметры
#         self.W1 = self.rng.normal(0, 0.1, size=(2 * dim_factor, hidden_dim))
#         self.b1 = np.zeros(hidden_dim)
#         self.W2 = self.rng.normal(0, 0.1, size=(hidden_dim, 1))
#         self.b2 = 0.0

#     def func_sigmoid(self, x):
#         return 1.0 / (1.0 + np.exp(-x))

#     def mlp_forward(self, x):
#         h = np.dot(x, self.W1) + self.b1
#         h = np.tanh(h)
#         return np.dot(h, self.W2)[0] + self.b2

#     def mlp_backward(self, x, grad_out):
#         h = np.dot(x, self.W1) + self.b1                # (hidden_dim,)
#         h_act = np.tanh(h)                              # (hidden_dim,)
#         grad_h = (1 - h_act ** 2) * (self.W2.flatten() * grad_out)  # (hidden_dim,)

#         grad_W2 = np.outer(h_act, np.array([grad_out]))  # (hidden_dim, 1)
#         grad_b2 = grad_out
#         grad_W1 = np.outer(grad_h, x)                    # (hidden_dim, input_dim)
#         grad_b1 = grad_h

#         return grad_W1, grad_b1, grad_W2, grad_b2



#     def train(self, df, iter=10000):
#         df_train = df[df[self.colname_outcome] > 0].copy()
#         if self.only_treated:
#             df_train = df_train[df_train[self.colname_treatment] == 1]

#         if self.capping_T:
#             mask_T = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
#             df_train.loc[mask_T, self.colname_propensity] = self.capping_T

#         if self.capping_C:
#             mask_C = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
#             df_train.loc[mask_C, self.colname_propensity] = 1 - self.capping_C

#         if self.with_IPS:
#             df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] / df_train[self.colname_propensity] - \
#                               (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome] / (1 - df_train[self.colname_propensity])
#         else:
#             df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] - \
#                               (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome]

#         users = df_train[self.colname_user].values
#         items = df_train[self.colname_item].values
#         ITE = df_train['ITE'].values

#         current_iter = 0
#         while current_iter < iter:
#             df_train = df_train.sample(frac=1)
#             for n in range(len(df_train)):
#                 u = users[n]
#                 i = items[n]
#                 while True:
#                     j = random.randint(0, self.num_items - 1)
#                     if j != i:
#                         break

#                 u_vec = self.user_factors[u]
#                 i_vec = self.item_factors[i]
#                 j_vec = self.item_factors[j]

#                 x_pos = np.concatenate([u_vec, i_vec])
#                 x_neg = np.concatenate([u_vec, j_vec])

#                 score_pos = self.mlp_forward(x_pos)
#                 score_neg = self.mlp_forward(x_neg)
#                 diff = score_pos - score_neg

#                 ite = ITE[n]
#                 coeff = 0.0
#                 if self.metric == 'AR_logi':
#                     if ite >= 0:
#                         coeff = ite * self.func_sigmoid(-diff)
#                     else:
#                         coeff = ite * self.func_sigmoid(diff)
#                 elif self.metric == 'AR_hinge':
#                     if ite >= 0 and diff < 1:
#                         coeff = ite
#                     elif ite < 0 and diff > -1:
#                         coeff = ite

#                 grad_pos = coeff
#                 grad_neg = -coeff

#                 gW1_p, gb1_p, gW2_p, gb2_p = self.mlp_backward(x_pos, grad_pos)
#                 gW1_n, gb1_n, gW2_n, gb2_n = self.mlp_backward(x_neg, grad_neg)

#                 self.W1 += self.learn_rate * (gW1_p + gW1_n - self.reg_factor * self.W1)
#                 self.b1 += self.learn_rate * (gb1_p + gb1_n)
#                 self.W2 += self.learn_rate * (gW2_p + gW2_n - self.reg_factor * self.W2)
#                 self.b2 += self.learn_rate * (gb2_p + gb2_n)

#                 # Обновление эмбеддингов
#                 self.user_factors[u] += self.learn_rate * (coeff * (i_vec - j_vec) - self.reg_factor * u_vec)
#                 self.item_factors[i] += self.learn_rate * (coeff * u_vec - self.reg_factor * i_vec)
#                 self.item_factors[j] += self.learn_rate * (-coeff * u_vec - self.reg_factor * j_vec)

#                 current_iter += 1
#                 if current_iter >= iter:
#                     return

#     def predict(self, df):
#         users = df[self.colname_user].values
#         items = df[self.colname_item].values
#         pred = np.zeros(len(df))
#         for n in range(len(df)):
#             u_vec = self.user_factors[users[n]]
#             i_vec = self.item_factors[items[n]]
#             x = np.concatenate([u_vec, i_vec])
#             pred[n] = self.mlp_forward(x)
#         return pred

    
# if __name__ == "__main__":
#     pass

import numpy as np
from pathlib import Path
from numpy.random.mtrand import RandomState
import random
import pandas as pd
from evaluator import Evaluator
import pickle


class Recommender(object):

    def __init__(self, num_users, num_items,
                 colname_user = 'idx_user', colname_item = 'idx_item',
                 colname_outcome = 'outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'
                 , colname_frequency = 'frequency'):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_frequency = colname_frequency

    def train(self, df, iter=100):
        pass

    def predict(self, df):
        pass

    def recommend(self, df, num_rec=10):
        pass

    def func_sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return np.exp(x) / (1.0 + np.exp(x))

    def sample_time(self):
        return random.randrange(self.num_times)

    def sample_user(self, idx_time, TP=True, TN=True, CP=True, CN=True):
        while True:
            flag_condition = 1
            u = random.randrange(self.num_users)
            if TP:
                if u not in self.dict_treatment_positive_sets[idx_time]:
                    flag_condition = 0
            if TN:
                if u not in self.dict_treatment_negative_sets[idx_time]:
                    flag_condition = 0
            if CP:
                if u not in self.dict_control_positive_sets[idx_time]:
                    flag_condition = 0
            if CN:
                if u not in self.dict_control_negative_sets[idx_time]:
                    flag_condition = 0
            if flag_condition > 0:
                return u

    def sample_treatment(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_sets[idx_time][idx_user])

    def sample_control(self, idx_time, idx_user):
        while True:
            flag_condition = 1
            i = random.randrange(self.num_items)
            if idx_user in self.dict_treatment_positive_sets[idx_time]:
                if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_treatment_negative_sets[idx_time]:
                if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
                    flag_condition = 0
            if flag_condition > 0:
                return i

    # in case control is rare
    def sample_control2(self, idx_time, idx_user):
        cand_control = np.arange(self.num_items)
        cand_control = cand_control[np.isin(cand_control, self.dict_treatment_sets[idx_time][idx_user])]
        return random.choice(cand_control)

    def sample_treatment_positive(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_positive_sets[idx_time][idx_user])

    def sample_treatment_negative(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_negative_sets[idx_time][idx_user])

    def sample_control_positive(self, idx_time, idx_user):
        return random.choice(self.dict_control_positive_sets[idx_time][idx_user])

    def sample_control_negative(self, idx_time, idx_user):
        while True:
            flag_condition = 1
            i = random.randrange(self.num_items)
            if idx_user in self.dict_treatment_positive_sets[idx_time]:
                if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_treatment_negative_sets[idx_time]:
                if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_control_positive_sets[idx_time]:
                if i in self.dict_control_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if flag_condition > 0:
                return i

    # TP: treatment-positive
    # CP: control-positive
    # TN: treatment-negative
    # TN: control-negative
    def sample_triplet(self):
        t = self.sample_time()
        if random.random() <= self.alpha:  # CN as positive
            if random.random() <= 0.5:  # TP as positive
                if random.random() <= 0.5:  # TP vs. TN
                    u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
                    i = self.sample_treatment_positive(t, u)
                    j = self.sample_treatment_negative(t, u)
                else:  # TP vs. CP
                    u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
                    i = self.sample_treatment_positive(t, u)
                    j = self.sample_control_positive(t, u)
            else:  # CN as positive
                if random.random() <= 0.5:  # CN vs. TN
                    u = self.sample_user(t, TP=False, TN=True, CP=False, CN=True)
                    i = self.sample_control_negative(t, u)
                    j = self.sample_treatment_negative(t, u)
                else:  # CN vs. CP
                    u = self.sample_user(t, TP=False, TN=False, CP=True, CN=True)
                    i = self.sample_control_negative(t, u)
                    j = self.sample_control_positive(t, u)
        else:  # CN as negative
            if random.random() <= 0.333:  # TP vs. CN
                u = self.sample_user(t, TP=True, TN=False, CP=False, CN=True)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_control_negative(t, u)
            elif random.random() <= 0.5:  # TP vs. TN
                u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_treatment_negative(t, u)
            else:  # TP vs. CP
                u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_control_positive(t, u)

        return u, i, j

    def sample_pair(self):
        t = self.sample_time()
        if random.random() < 0.5: # pick treatment
            if random.random() > self.ratio_nega: # TP
                u = self.sample_user(t, TP=True, TN=False, CP=False, CN=False)
                i = self.sample_treatment_positive(t, u)
                flag_positive = 1
            else: # TN
                u = self.sample_user(t, TP=False, TN=True, CP=False, CN=False)
                i = self.sample_treatment_negative(t, u)
                flag_positive = 0
        else: # pick control
            if random.random() > self.ratio_nega:  # CP
                u = self.sample_user(t, TP=False, TN=False, CP=True, CN=False)
                i = self.sample_control_positive(t, u)
                flag_positive = 0
            else:  # CN
                u = self.sample_user(t, TP=False, TN=False, CP=False, CN=True)
                i = self.sample_control_negative(t, u)
                if random.random() <= self.alpha:  # CN as positive
                    flag_positive = 1
                else:
                    flag_positive = 0

        return u, i, flag_positive

    # getter
    def get_propensity(self, idx_user, idx_item):
        return self.dict_propensity[idx_user][idx_item]


class LMF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AUC', ratio_nega=0.8,
                 dim_factor=200, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 reg_factor_j=0.01, reg_bias_j=0.01,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.ratio_nega = ratio_nega
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.sd_init = sd_init
        self.reg_bias_j = reg_bias_j
        self.reg_factor_j = reg_factor_j
        self.flag_prepared = False

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def prepare_dictionary(self, df, colname_time='idx_time'):
        print("start prepare dictionary")
        self.colname_time = colname_time
        self.num_times = np.max(df.loc[:, self.colname_time]) + 1
        self.dict_positive_sets = dict()

        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]

        for t in np.arange(self.num_times):
            df_t = df_posi.loc[df_posi.loc[:, self.colname_time] == t]
            self.dict_positive_sets[t] = dict()
            for u in np.unique(df_t.loc[:, self.colname_user]):
                self.dict_positive_sets[t][u] = \
                    np.unique(df_t.loc[df_t.loc[:, self.colname_user] == u, self.colname_item].values)

        self.flag_prepared = True
        print("prepared dictionary!")


    def train(self, df, iter = 10):

        df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
        if not self.flag_prepared: # prepare dictionary
            self.prepare_dictionary(df)
        
        err = 0
        current_iter = 0
        while True:
            df_train = df_train.sample(frac=1)
            users = df_train.loc[:, self.colname_user].values
            items = df_train.loc[:, self.colname_item].values
            times = df_train.loc[:, self.colname_time].values

            if self.metric == 'AUC': # BPR
                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    t = times[n]

                    while True:
                        j = random.randrange(self.num_items)
                        if not j in self.dict_positive_sets[t][u]:
                            break

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]
                    j_factor = self.item_factors[j, :]

                    diff_rating = np.sum(u_factor * (i_factor - j_factor))

                    if self.with_bias:
                        diff_rating += (self.item_biases[i] - self.item_biases[j])

                    coeff = self.func_sigmoid(-diff_rating)

                    err += coeff

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                    self.item_factors[j, :] += \
                        self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.item_biases[j] += \
                            self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

            current_iter += 1
            if current_iter >= iter:
                return err/iter

            elif self.metric == 'logloss': # essentially WRMF with downsampling
                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    t = times[n]
                    flag_positive = 1

                    if np.random.rand() < self.ratio_nega:
                        flag_positive = 0
                        i = np.random.randint(self.num_items)
                        while True:
                            if not i in self.dict_positive_sets[t][u]:
                                break
                            else:
                                i = np.random.randint(self.num_items)

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)

                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    if flag_positive > 0:
                        coeff = 1 / (1 + np.exp(rating))
                    else:
                        coeff = -1 / (1 + np.exp(-rating))

                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                        return err / iter

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        # pred = 1 / (1 + np.exp(-pred))
        return pred


class PopularBase(Recommender):
    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)
        self.df_cnt = df = pd.DataFrame([])

    def train(self, df, iter = 1):
        df_cnt = df.groupby(self.colname_item, as_index=False)[self.colname_outcome].sum()
        df_cnt['prob'] = df_cnt[self.colname_outcome] /self.num_users
        self.df_cnt = df_cnt

    def predict(self, df):
        df = pd.merge(df, self.df_cnt, on=self.colname_item, how='left')
        return df.loc[:, 'prob'].values

class DLMF2(Recommender): # This version consider a scale factor alpha
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=200, with_bias=False, with_IPS=True,
                 only_treated=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
                 sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
                 coeff_T = 1.0, coeff_C = 1.0,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.coeff_T = coeff_T
        self.coeff_C = coeff_C
        self.learn_rate = learn_rate
        self.reg_bias = reg_factor
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_factor
        self.reg_factor_j = reg_factor
        self.sd_init = sd_init
        self.only_treated = only_treated

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0
        self.alpha = 0.5
        
            
    def train(self, df, iter=100):
        df_train = df[df[self.colname_outcome] > 0].copy()

        if self.only_treated:
            df_train = df_train[df_train[self.colname_treatment] > 0]

        # Apply capping to avoid extreme values
        if self.capping_T is not None:
            treated_mask = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
            df_train.loc[treated_mask, self.colname_propensity] = self.capping_T

        if self.capping_C is not None:
            control_mask = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
            df_train.loc[control_mask, self.colname_propensity] = 1 - self.capping_C

        self.propensity = df_train[self.colname_propensity].values
        current_iter = 0
        err = 0
        epsilon = 1e-6

        while True:
            df_train = df_train.sample(frac=1)
            users = df_train[self.colname_user].values
            items = df_train[self.colname_item].values
            treat = df_train[self.colname_treatment].values
            outcome = df_train[self.colname_outcome].values
            prop = np.clip(self.propensity * self.alpha, 1e-3, 1 - 1e-3)

            for n in range(len(df_train)):
                u = users[n]
                i = items[n]
                t = treat[n]
                y = outcome[n]
                p = prop[n]

                ite = t * y / p - (1 - t) * y / (1 - p)
                z_y_1 = t * y
                z_y_0 = (1 - t) * y

                # Skip if ITE is nan or inf
                if np.isnan(ite) or np.isinf(ite):
                    continue

                # Sample a negative item
                while True:
                    j = random.randrange(self.num_items)
                    if j != i:
                        break

                u_factor = self.user_factors[u]
                i_factor = self.item_factors[i]
                j_factor = self.item_factors[j]

                diff_rating = np.dot(u_factor, i_factor - j_factor)
                if self.with_bias:
                    diff_rating += self.item_biases[i] - self.item_biases[j]

                if self.metric == 'AR_logi':
                    if ite >= 0:
                        coeff = ite * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
                        const_value = z_y_1 * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
                    else:
                        coeff = ite * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
                        const_value = z_y_0 * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
                else:
                    # Fallback if metric is not supported
                    continue

                # Skip if coeff is nan or inf
                if np.isnan(coeff) or np.isinf(coeff):
                    continue

                err += abs(coeff)

                # SGD updates
                self.user_factors[u] += self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                self.item_factors[i] += self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                self.item_factors[j] += self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                alpha_grad = const_value / (np.power(self.alpha, 2) + epsilon)
                self.alpha += self.learn_rate * alpha_grad
                self.alpha = np.clip(self.alpha, 1e-3, 10)

                if self.with_bias:
                    self.item_biases[i] += self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                    self.item_biases[j] += self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

                current_iter += 1
                if current_iter % 10000 == 0:
                    print(f"Iter: {current_iter} | Error: {err / current_iter:.6f} | Alpha: {self.alpha:.6f}")

                if current_iter >= iter:
                    return self.alpha


    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        # pred = 1 / (1 + np.exp(-pred))
        return pred



class DLMF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=200, with_bias=False, with_IPS=True,
                 only_treated=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
                 sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
                 coeff_T = 1.0, coeff_C = 1.0,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.coeff_T = coeff_T
        self.coeff_C = coeff_C
        self.learn_rate = learn_rate
        self.reg_bias = reg_factor
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_factor
        self.reg_factor_j = reg_factor
        self.sd_init = sd_init
        self.only_treated = only_treated

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def train(self, df, iter = 100):
        df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :] # need only positive outcomes
        if self.only_treated: # train only with treated positive (DLTO)
            df_train = df_train.loc[df_train.loc[:, self.colname_treatment] > 0, :]

        if self.capping_T is not None:
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] < self.capping_T, df_train.loc[:, self.colname_treatment] == 1)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = self.capping_T
        if self.capping_C is not None:      
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] > 1 - self.capping_C, df_train.loc[:, self.colname_treatment] == 0)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

        if self.with_IPS: # point estimate of individual treatment effect (ITE) <- for binary outcome abs(ITE) = IPS
            df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]/df_train.loc[:, self.colname_propensity] - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]/(1 - df_train.loc[:, self.colname_propensity])
            z_y_1 = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]
            z_y_1 = z_y_1.values
            z_y_0 = (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]
            z_y_0 = z_y_0.values

        else:
            df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]  - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]

        err = 0
        current_iter = 0
        while True:
            df_train = df_train.sample(frac=1)
            users = df_train.loc[:, self.colname_user].values
            items = df_train.loc[:, self.colname_item].values
            ITE = df_train.loc[:, 'ITE'].values

            if self.metric in ['AR_logi', 'AR_sig', 'AR_hinge']:
                for n in np.arange(len(df_train)):

                    u = users[n]
                    i = items[n]

                    while True:
                        j = random.randrange(self.num_items)
                        if i != j:
                            break

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]
                    j_factor = self.item_factors[j, :]

                    diff_rating = np.sum(u_factor * (i_factor - j_factor))
                    if self.with_bias:
                        diff_rating += (self.item_biases[i] - self.item_biases[j])

                    if self.metric == 'AR_logi':
                        if ITE[n] >= 0:
                            coeff = ITE[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating) # Z=1, Y=1
                            const_value = z_y_1[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
                        else:
                            coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) # Z=0, Y=1
                            const_value = z_y_0[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)

                    elif self.metric == 'AR_sig':
                        if ITE[n] >= 0:
                            coeff = ITE[n] * self.coeff_T * self.func_sigmoid(self.coeff_T * diff_rating) * self.func_sigmoid(-self.coeff_T * diff_rating)
                        else:
                            coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) * self.func_sigmoid(-self.coeff_C * diff_rating)

                    elif self.metric == 'AR_hinge':
                        if ITE[n] >= 0:
                            if self.coeff_T > 0 and diff_rating < 1.0/self.coeff_T:
                                coeff = ITE[n] * self.coeff_T 
                            else:
                                coeff = 0.0
                        else:
                            if self.coeff_C > 0 and diff_rating > -1.0/self.coeff_C:
                                coeff = ITE[n] * self.coeff_C
                            else:
                                coeff = 0.0

                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                    self.item_factors[j, :] += \
                        self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.item_biases[j] += \
                            self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

                    current_iter += 1
                    if current_iter % 100000 == 0:
                        print(str(current_iter)+"/"+str(iter))
                        print()
                        assert not np.isnan(coeff)
                        assert not np.isinf(coeff)
                        print("z_y_1 mean:", np.mean(z_y_1), "z_y_0 mean:", np.mean(z_y_0))
                        print("Error:", err / iter)

                    if current_iter >= iter:
                        with open("dlmf_weights.pkl", "wb") as f:
                            pickle.dump(self.__dict__, f)
                            print("DLMF weights saved.")
                        return err / iter

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        # pred = 1 / (1 + np.exp(-pred))
        return pred
    
class DLMF4(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=200, with_bias=False, with_IPS=True,
                 only_treated=False, with_DR=False,
                 learn_rate=0.01, reg_factor=0.01, reg_bias=0.01,
                 sd_init=0.1, reg_factor_j=0.01, reg_bias_j=0.01,
                 coeff_T=1.0, coeff_C=1.0,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.with_DR = with_DR
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.coeff_T = coeff_T
        self.coeff_C = coeff_C
        self.learn_rate = learn_rate
        self.reg_bias = reg_factor
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_factor
        self.reg_factor_j = reg_factor
        self.sd_init = sd_init
        self.only_treated = only_treated

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    @staticmethod
    def func_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def train(self, df, iter=100, omega=1.0):
        df_train = df[df[self.colname_outcome] > 0]
        if self.only_treated:
            df_train = df_train[df_train[self.colname_treatment] > 0]

        if self.capping_T is not None:
            mask = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
            df_train.loc[mask, self.colname_propensity] = self.capping_T
        if self.capping_C is not None:
            mask = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
            df_train.loc[mask, self.colname_propensity] = 1 - self.capping_C

        if self.with_DR:
            from sklearn.linear_model import LogisticRegression
            X = df_train[[self.colname_user, self.colname_item]].values
            y = df_train[self.colname_outcome].values
            z = df_train[self.colname_treatment].values
            X1 = X[z == 1]; y1 = y[z == 1]
            X0 = X[z == 0]; y0 = y[z == 0]
            self.model_treated = LogisticRegression().fit(X1, y1)
            self.model_control = LogisticRegression().fit(X0, y0)

        df_train = df_train.sample(frac=1)
        users = df_train[self.colname_user].values
        items = df_train[self.colname_item].values
        outcomes = df_train[self.colname_outcome].values
        props = df_train[self.colname_propensity].values
        treats = df_train[self.colname_treatment].values

        err = 0
        current_iter = 0

        for n in range(len(df_train)):
            u = users[n]
            i = items[n]
            y = outcomes[n]
            p = props[n]
            z = treats[n]

            while True:
                j = random.randrange(self.num_items)
                if i != j:
                    break

            u_factor = self.user_factors[u, :]
            i_factor = self.item_factors[i, :]
            j_factor = self.item_factors[j, :]

            diff_rating = np.sum(u_factor * (i_factor - j_factor))
            if self.with_bias:
                diff_rating += (self.item_biases[i] - self.item_biases[j])

            if self.with_DR:
                x_ij = np.array([[u, i]])
                y1_hat = self.model_treated.predict_proba(x_ij)[0, 1]
                y0_hat = self.model_control.predict_proba(x_ij)[0, 1]
                ite = (z / p - (1 - z) / (1 - p)) * (y - (z * y1_hat + (1 - z) * y0_hat)) + (y1_hat - y0_hat)
            elif self.with_IPS:
                ite = z * y / p - (1 - z) * y / (1 - p)
            else:
                ite = z * y - (1 - z) * y

            if ite >= 0:
                loss = np.log(1 + np.exp(-omega * diff_rating)) * self.coeff_T * ite
            else:
                loss = np.log(1 + np.exp(omega * diff_rating)) * self.coeff_C * ite

            grad = -omega * self.func_sigmoid(-omega * diff_rating) * ite if ite >= 0 else omega * self.func_sigmoid(omega * diff_rating) * ite

            self.user_factors[u, :] += self.learn_rate * (grad * (i_factor - j_factor) - self.reg_factor * u_factor)
            self.item_factors[i, :] += self.learn_rate * (grad * u_factor - self.reg_factor * i_factor)
            self.item_factors[j, :] += self.learn_rate * (-grad * u_factor - self.reg_factor_j * j_factor)

            if self.with_bias:
                self.item_biases[i] += self.learn_rate * (grad - self.reg_bias * self.item_biases[i])
                self.item_biases[j] += self.learn_rate * (-grad - self.reg_bias_j * self.item_biases[j])

            err += np.abs(grad)
            current_iter += 1
            if current_iter % 1000000 == 0:
                print(f"{current_iter}/{iter}")
                print("Error:", err / iter)

            if current_iter >= iter:
                with open("dlmf_weights.pkl", "wb") as f:
                    pickle.dump(self.__dict__, f)
                    print("DLMF weights saved.")
                return err / iter


class DLMF3(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=200, with_bias=False, with_IPS=True,
                 only_treated=False, use_DR=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
                 sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
                 coeff_T = 1.0, coeff_C = 1.0,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity', colname_frequency = 'frequency'):

        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity
                         ,colname_frequency = colname_frequency)
        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.use_DR = use_DR
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.coeff_T = coeff_T
        self.coeff_C = coeff_C
        self.learn_rate = learn_rate
        self.reg_bias = reg_factor
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_factor
        self.reg_factor_j = reg_factor
        self.sd_init = sd_init
        self.only_treated = only_treated
        self.colname_frequency = colname_frequency
        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def func_sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    # def train(self, df, iter=100):
    #     df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
    #     if self.only_treated:  # train only with treated positive (DLTO)
    #         df_train = df_train.loc[df_train.loc[:, self.colname_treatment] > 0, :]

    #     # ✅ Bias инициализация через frequency
    #     if self.with_bias and 'frequency' in df_train.columns:
    #         avg_freq = df_train.groupby(self.colname_item)['frequency'].mean()
    #         item_bias_init = np.log1p(avg_freq).reindex(np.arange(self.num_items)).fillna(0).values
    #         self.item_biases = item_bias_init
    #         print("Item biases initialized from frequency using log(1 + freq).")

    #     # ✅ Propensity capping
    #     if self.capping_T is not None:
    #         bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] < self.capping_T,
    #                                 df_train.loc[:, self.colname_treatment] == 1)
    #         if np.sum(bool_cap) > 0:
    #             df_train.loc[bool_cap, self.colname_propensity] = self.capping_T

    #     if self.capping_C is not None:
    #         bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] > 1 - self.capping_C,
    #                                 df_train.loc[:, self.colname_treatment] == 0)
    #         if np.sum(bool_cap) > 0:
    #             df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

    #     # ✅ ITE расчёт
    #     if self.with_IPS:
    #         df_train.loc[:, 'ITE'] = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome] / df_train.loc[:, self.colname_propensity] - \
    #                                 (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome] / (1 - df_train.loc[:, self.colname_propensity])
    #         z_y_1 = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]
    #         z_y_1 = z_y_1.values
    #         z_y_0 = (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]
    #         z_y_0 = z_y_0.values
    #     else:
    #         df_train.loc[:, 'ITE'] = df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome] - \
    #                                 (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]

    #     err = 0
    #     current_iter = 0
    #     while True:
    #         df_train = df_train.sample(frac=1)
    #         users = df_train.loc[:, self.colname_user].values
    #         items = df_train.loc[:, self.colname_item].values
    #         ITE = df_train.loc[:, 'ITE'].values

    #         if self.metric in ['AR_logi', 'AR_sig', 'AR_hinge']:
    #             for n in np.arange(len(df_train)):
    #                 u = users[n]
    #                 i = items[n]

    #                 while True:
    #                     j = random.randrange(self.num_items)
    #                     if i != j:
    #                         break

    #                 u_factor = self.user_factors[u, :]
    #                 i_factor = self.item_factors[i, :]
    #                 j_factor = self.item_factors[j, :]

    #                 diff_rating = np.sum(u_factor * (i_factor - j_factor))
    #                 if self.with_bias:
    #                     diff_rating += (self.item_biases[i] - self.item_biases[j])

    #                 if self.metric == 'AR_logi':
    #                     if ITE[n] >= 0:
    #                         coeff = ITE[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
    #                         const_value = z_y_1[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
    #                     else:
    #                         coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
    #                         const_value = z_y_0[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)

    #                 elif self.metric == 'AR_sig':
    #                     if ITE[n] >= 0:
    #                         coeff = ITE[n] * self.coeff_T * self.func_sigmoid(self.coeff_T * diff_rating) * self.func_sigmoid(-self.coeff_T * diff_rating)
    #                     else:
    #                         coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) * self.func_sigmoid(-self.coeff_C * diff_rating)

    #                 elif self.metric == 'AR_hinge':
    #                     if ITE[n] >= 0:
    #                         if self.coeff_T > 0 and diff_rating < 1.0 / self.coeff_T:
    #                             coeff = ITE[n] * self.coeff_T
    #                         else:
    #                             coeff = 0.0
    #                     else:
    #                         if self.coeff_C > 0 and diff_rating > -1.0 / self.coeff_C:
    #                             coeff = ITE[n] * self.coeff_C
    #                         else:
    #                             coeff = 0.0

    #                 err += np.abs(coeff)

                    # self.user_factors[u, :] += \
                    #     self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                    # self.item_factors[i, :] += \
                    #     self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                    # self.item_factors[j, :] += \
                    #     self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                    # if self.with_bias:
                    #     self.item_biases[i] += \
                    #         self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                    #     self.item_biases[j] += \
                    #         self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

    #                 current_iter += 1
    #                 if current_iter % 100000 == 0:
    #                     print(str(current_iter) + "/" + str(iter))
    #                 if current_iter >= iter:
    #                     with open("dlmf_weights.pkl", "wb") as f:
    #                         pickle.dump(self.__dict__, f)
    #                         print("DLMF weights saved.")
    #                     return err / iter

    def train(self, df, iter = 100):
        df_train = df[df[self.colname_outcome] > 0].copy()
        if self.only_treated:
            df_train = df_train[df_train[self.colname_treatment] == 1]

        if self.capping_T is not None:
            bool_cap = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
            df_train.loc[bool_cap, self.colname_propensity] = self.capping_T

        if self.capping_C is not None:
            bool_cap = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
            df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

        if self.use_DR:
            dr_estimator = DR_Estimator(self.num_users, self.num_items, dim_factor=self.dim_factor,
                                        learn_rate=self.learn_rate, reg_factor=self.reg_factor,
                                        with_bias=self.with_bias, colname_user=self.colname_user,
                                        colname_item=self.colname_item, colname_outcome=self.colname_outcome,
                                        colname_treatment=self.colname_treatment,
                                        colname_propensity=self.colname_propensity)
            dr_estimator.train(df_train, iter=iter)
            df_train['ITE'] = dr_estimator.compute_DR(df_train)
        else:
            if self.with_IPS:
                df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] / df_train[self.colname_propensity] - \
                                   (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome] / (1 - df_train[self.colname_propensity])
            else:
                df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] - \
                                   (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome]

        z_y_1 = df_train[self.colname_treatment] * df_train[self.colname_outcome]
        z_y_1 = z_y_1.values
        z_y_0 = (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome]
        z_y_0 = z_y_0.values

        err = 0
        current_iter = 0
        while True:
            df_train = df_train.sample(frac=1)
            users = df_train[self.colname_user].values
            items = df_train[self.colname_item].values
            ITE = df_train['ITE'].values

            for n in np.arange(len(df_train)):
                u = users[n]
                i = items[n]
                while True:
                    j = random.randrange(self.num_items)
                    if i != j:
                        break

                u_factor = self.user_factors[u, :]
                i_factor = self.item_factors[i, :]
                j_factor = self.item_factors[j, :]

                diff_rating = np.sum(u_factor * (i_factor - j_factor))
                if self.with_bias:
                    diff_rating += self.item_biases[i] - self.item_biases[j]

                ite = ITE[n]
                if self.metric == 'AR_logi':
                    if ite >= 0:
                        coeff = ite * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
                    else:
                        coeff = ite * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
                elif self.metric == 'AR_sig':
                    if ite >= 0:
                        sig = self.func_sigmoid(self.coeff_T * diff_rating)
                        coeff = ite * self.coeff_T * sig * (1 - sig)
                    else:
                        sig = self.func_sigmoid(self.coeff_C * diff_rating)
                        coeff = ite * self.coeff_C * sig * (1 - sig)
                elif self.metric == 'AR_hinge':
                    if ite >= 0 and diff_rating < 1.0 / self.coeff_T:
                        coeff = ite * self.coeff_T
                    elif ite < 0 and diff_rating > -1.0 / self.coeff_C:
                        coeff = ite * self.coeff_C
                    else:
                        coeff = 0.0

                err += np.abs(coeff)
                self.user_factors[u, :] += self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                self.item_factors[i, :] += self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                self.item_factors[j, :] += self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                if self.with_bias:
                    self.item_biases[i] += self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                    self.item_biases[j] += self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

                current_iter += 1
                if current_iter >= iter:
                    with open("dlmf_weights.pkl", "wb") as f:
                        pickle.dump(self.__dict__, f)
                        print("DLMF weights saved.")
                    return err / iter


    # def train(self, df, iter=100):
    #     df_train = df[df[self.colname_outcome] > 0].copy()
    #     if self.only_treated:
    #         df_train = df_train[df_train[self.colname_treatment] == 1]

    #     if self.capping_T is not None:
    #         bool_cap = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
    #         df_train.loc[bool_cap, self.colname_propensity] = self.capping_T

    #     if self.capping_C is not None:
    #         bool_cap = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
    #         df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

    #     if self.use_DR:
    #         dr_estimator = DR_Estimator(self.num_users, self.num_items, dim_factor=self.dim_factor,
    #                                     learn_rate=self.learn_rate, reg_factor=self.reg_factor,
    #                                     with_bias=self.with_bias, colname_user=self.colname_user,
    #                                     colname_item=self.colname_item, colname_outcome=self.colname_outcome,
    #                                     colname_treatment=self.colname_treatment,
    #                                     colname_propensity=self.colname_propensity)
    #         dr_estimator.train(df_train, iter=10)
    #         df_train['ITE'] = dr_estimator.compute_DR(df_train)
    #     else:
    #         if self.with_IPS:
    #             df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] / df_train[self.colname_propensity] - \
    #                             (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome] / (1 - df_train[self.colname_propensity])
    #         else:
    #             df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] - \
    #                             (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome]

    #     err = 0
    #     current_iter = 0
    #     while True:
    #         df_train = df_train.sample(frac=1)
    #         users = df_train[self.colname_user].values
    #         items = df_train[self.colname_item].values
    #         ITE = df_train['ITE'].values

    #         for n in np.arange(len(df_train)):
    #             u = users[n]
    #             i = items[n]
    #             while True:
    #                 j = random.randrange(self.num_items)
    #                 if i != j:
    #                     break

    #             u_factor = self.user_factors[u, :]
    #             i_factor = self.item_factors[i, :]
    #             j_factor = self.item_factors[j, :]

    #             # new: add user-item interaction factor (ui_factor)
    #             ui_factor = self.colname_frequency[u, i, :]
    #             uj_factor = self.colname_frequency[u, j, :]

    #             diff_rating = np.sum(u_factor * (i_factor - j_factor)) + np.sum(ui_factor - uj_factor)

    #             if self.with_bias:
    #                 diff_rating += self.item_biases[i] - self.item_biases[j]

    #             ite = ITE[n]
    #             if self.metric == 'AR_logi':
    #                 if ite >= 0:
    #                     coeff = ite * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating)
    #                 else:
    #                     coeff = ite * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating)
    #             elif self.metric == 'AR_sig':
    #                 if ite >= 0:
    #                     sig = self.func_sigmoid(self.coeff_T * diff_rating)
    #                     coeff = ite * self.coeff_T * sig * (1 - sig)
    #                 else:
    #                     sig = self.func_sigmoid(self.coeff_C * diff_rating)
    #                     coeff = ite * self.coeff_C * sig * (1 - sig)
    #             elif self.metric == 'AR_hinge':
    #                 if ite >= 0 and diff_rating < 1.0 / self.coeff_T:
    #                     coeff = ite * self.coeff_T
    #                 elif ite < 0 and diff_rating > -1.0 / self.coeff_C:
    #                     coeff = ite * self.coeff_C
    #                 else:
    #                     coeff = 0.0

    #             err += np.abs(coeff)

    #             self.user_factors[u, :] += self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
    #             self.item_factors[i, :] += self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
    #             self.item_factors[j, :] += self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

    #             # new: update Frequency (ui interaction)
    #             self.colname_frequency[u, i, :] += self.learn_rate * (coeff - self.reg_factor * ui_factor)
    #             self.colname_frequency[u, j, :] += self.learn_rate * (-coeff - self.reg_factor * uj_factor)

    #             if self.with_bias:
    #                 self.item_biases[i] += self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
    #                 self.item_biases[j] += self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

    #             current_iter += 1
    #             if current_iter >= iter:
    #                 with open("dlmf_weights.pkl", "wb") as f:
    #                     pickle.dump(self.__dict__, f)
    #                     print("DLMF weights saved.")
    #                 return err / iter

    # def predict(self, df):
    #     users = df[self.colname_user].values
    #     items = df[self.colname_item].values
    #     pred = np.zeros(len(df))
    #     for n in np.arange(len(df)):
    #         pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
    #         if self.with_bias:
    #             pred[n] += self.item_biases[items[n]]
    #             pred[n] += self.user_biases[users[n]]
    #             pred[n] += self.global_bias
    #     return pred
    
    def predict(self, df): 
        users = df[self.colname_user].values 
        items = df[self.colname_item].values 
        frequencies = df[self.colname_frequency].values 
        pred = np.zeros(len(df)) 
        for n in np.arange(len(df)): 
            pred[n] = np.inner(self.user_factors[users[n], :], (self.item_factors[items[n], :])) + frequencies[n] 
            if self.with_bias: 
                pred[n] += self.item_biases[items[n]] 
                pred[n] += self.user_biases[users[n]] 
                pred[n] += self.global_bias 
 
        # pred = 1 / (1 + np.exp(-pred)) 
        return pred



class RandomBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)

    def train(self, df, iter = 1):
        pass

    def predict(self, df):
        return np.random.rand(df.shape[0])



class MF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='RMSE',
                 dim_factor=200, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.sd_init = sd_init

        self.flag_prepared = False

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def prepare_dictionary(self, df, colname_time='idx_time'):
        print("start prepare dictionary")
        self.colname_time = colname_time
        self.num_times = np.max(df.loc[:, self.colname_time]) + 1
        self.dict_positive_sets = dict()
    
        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
    
        for t in np.arange(self.num_times):
            df_t = df_posi.loc[df_posi.loc[:, self.colname_time] == t]
            self.dict_positive_sets[t] = dict()
            for u in np.unique(df_t.loc[:, self.colname_user]):
                self.dict_positive_sets[t][u] = \
                    np.unique(df_t.loc[df_t.loc[:, self.colname_user] == u, self.colname_item].values)
    
        self.flag_prepared = True
        print("prepared dictionary!")


    def train(self, df, iter = 100):

        # by default, rating prediction
        # outcome = rating
        df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

        # # in case of binary implicit feedback
        if self.metric == 'logloss':
            df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :]  # need only positive outcomes
            if not self.flag_prepared: # prepare dictionary
                self.prepare_dictionary(df)
        else:
            df_train = df.loc[~np.isnan(df.loc[:, self.colname_outcome]), :]

        err = 0
        current_iter = 0
        while True:
            if self.metric == 'RMSE':
                df_train = df_train.sample(frac=1)
                users = df_train.loc[:, self.colname_user].values
                items = df_train.loc[:, self.colname_item].values
                outcomes = df_train.loc[:, self.colname_outcome].values

                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    r = outcomes[n]

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)
                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    coeff = r - rating
                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                        if current_iter % 100000 == 0:
                            print(str(current_iter)+"/"+str(iter))
                        return err / iter

            elif self.metric == 'logloss': # logistic matrix factorization
                df_train = df_train.sample(frac=1)
                users = df_train.loc[:, self.colname_user].values
                items = df_train.loc[:, self.colname_item].values
                outcomes = df_train.loc[:, self.colname_outcome].values

                for n in np.arange(len(df_train)):
                    u = users[n]
                    i = items[n]
                    r = outcomes[n]

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)
                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    if r > 0:
                        coeff = self.func_sigmoid(-rating)
                    else:
                        coeff = - self.func_sigmoid(rating)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.user_biases[u] += \
                            self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                        self.global_bias += \
                            self.learn_rate * (coeff)

                    current_iter += 1
                    if current_iter >= iter:
                
                        return err / iter

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        if self.metric == 'logloss':
            pred = 1 / (1 + np.exp(-pred))
        return pred


class DR_Estimator:
    def __init__(self, num_users, num_items,
                 dim_factor=64, learn_rate=0.01, reg_factor=0.01, with_bias=True,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_treatment='treated',
                 colname_propensity='propensity'):

        self.num_users = num_users
        self.num_items = num_items
        self.dim_factor = dim_factor
        self.learn_rate = learn_rate
        self.reg_factor = reg_factor
        self.with_bias = with_bias

        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity

        self.model_T = MF(num_users, num_items, dim_factor=dim_factor,
                          learn_rate=learn_rate, reg_factor=reg_factor, with_bias=with_bias,
                          colname_user=colname_user, colname_item=colname_item,
                          colname_outcome=colname_outcome, colname_treatment=colname_treatment,
                          colname_propensity=colname_propensity)

        self.model_C = MF(num_users, num_items, dim_factor=dim_factor,
                          learn_rate=learn_rate, reg_factor=reg_factor, with_bias=with_bias,
                          colname_user=colname_user, colname_item=colname_item,
                          colname_outcome=colname_outcome, colname_treatment=colname_treatment,
                          colname_propensity=colname_propensity)

    def train(self, df, iter=10):
        df_T = df[df[self.colname_treatment] == 1].copy()
        df_C = df[df[self.colname_treatment] == 0].copy()

        self.model_T.train(df_T, iter=iter)
        self.model_C.train(df_C, iter=iter)

    def compute_DR(self, df):
        Z = df[self.colname_treatment].values
        Y = df[self.colname_outcome].values
        P = df[self.colname_propensity].values

        y_T_hat = self.model_T.predict(df)
        y_C_hat = self.model_C.predict(df)

        term_T = Z * (Y - y_T_hat) / np.clip(P, 1e-3, 1) + y_T_hat
        term_C = (1 - Z) * (Y - y_C_hat) / np.clip(1 - P, 1e-3, 1) + y_C_hat

        return term_T - term_C


class CausalNeighborBase(Recommender):
    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 measure_simil='cosine', way_simil='treatment',
                 way_neighbor='user', num_neighbor=3000,
                 way_self='exclude',
                 weight_treated_outcome=0.5,
                 shrinkage_T=10.0, shrinkage_C=10.0,
                 scale_similarity=0.33, normalize_similarity=False):

        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)
        self.measure_simil = measure_simil
        self.way_simil = way_simil
        self.way_neighbor = way_neighbor
        self.num_neighbor = num_neighbor
        self.scale_similarity = scale_similarity
        self.normalize_similarity = normalize_similarity
        self.weight_treated_outcome = weight_treated_outcome
        self.shrinkage_T = shrinkage_T
        self.shrinkage_C = shrinkage_C
        self.way_self = way_self # exclude/include/only


    def simil(self, set1, set2, measure_simil):
        if measure_simil == "jaccard":
            return self.simil_jaccard(set1, set2)
        elif measure_simil == "cosine":
            return self.simil_cosine(set1, set2)

    def train(self, df, iter=1):
        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]
        print("len(df_posi): {}".format(len(df_posi)))

        dict_items2users = dict() # map an item to users who consumed the item
        for i in np.arange(self.num_items):
            dict_items2users[i] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_item] == i, self.colname_user].values)
        self.dict_items2users = dict_items2users
        print("prepared dict_items2users")

        dict_users2items = dict()  # map an user to items which are consumed by the user
        for u in np.arange(self.num_users):
            dict_users2items[u] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_user] == u, self.colname_item].values)
        self.dict_users2items = dict_users2items
        print("prepared dict_users2items")

        df_treated = df.loc[df.loc[:, self.colname_treatment] > 0]  # calc similarity by treatment assignment
        print("len(df_treated): {}".format(len(df_treated)))

        dict_items2users_treated = dict() # map an item to users who get treatment of the item
        for i in np.arange(self.num_items):
            dict_items2users_treated[i] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_item] == i, self.colname_user].values)
        self.dict_items2users_treated = dict_items2users_treated
        print("prepared dict_items2users_treated")

        dict_users2items_treated = dict()  # map an user to items which are treated to the user
        for u in np.arange(self.num_users):
            dict_users2items_treated[u] = np.unique(df_treated.loc[df_treated.loc[:, self.colname_user] == u, self.colname_item].values)
        self.dict_users2items_treated = dict_users2items_treated
        print("prepared dict_users2items_treated")

        if self.way_simil == 'treatment':
            if self.way_neighbor == 'user':
                dict_simil_users = {}
                sum_simil = np.zeros(self.num_users)
                for u1 in np.arange(self.num_users):
                    if u1 % round(self.num_users/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * u1/self.num_users))

                    items_u1 = self.dict_users2items_treated[u1]
                    dict_neighbor = {}
                    if len(items_u1) > 0:
                        cand_u2 = np.unique(df_treated.loc[np.isin(df_treated.loc[:, self.colname_item], items_u1), self.colname_user].values)
                        for u2 in cand_u2:
                            if u2 != u1:
                                items_u2 = self.dict_users2items_treated[u2]
                                dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

                        # print("len(dict_neighbor): {}".format(len(dict_neighbor)))
                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_users[u1] = dict_neighbor
                        sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_users[u1] = dict_neighbor
                self.dict_simil_users = dict_simil_users
                self.sum_simil = sum_simil

            elif self.way_neighbor == 'item':
                dict_simil_items = {}
                sum_simil = np.zeros(self.num_items)
                for i1 in np.arange(self.num_items):
                    if i1 % round(self.num_items/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

                    users_i1 = self.dict_items2users_treated[i1]
                    dict_neighbor = {}
                    if len(users_i1) > 0:
                        cand_i2 = np.unique(
                            df_treated.loc[np.isin(df_treated.loc[:, self.colname_user], users_i1), self.colname_item].values)
                        for i2 in cand_i2:
                            if i2 != i1:
                                users_i2 = self.dict_items2users_treated[i2]
                                dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_items[i1] = dict_neighbor
                        sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_items[i1] = dict_neighbor
                self.dict_simil_items = dict_simil_items
                self.sum_simil = sum_simil
        else:
            if self.way_neighbor == 'user':
                dict_simil_users = {}
                sum_simil = np.zeros(self.num_users)
                for u1 in np.arange(self.num_users):
                    if u1 % round(self.num_users/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * u1 / self.num_users))

                    items_u1 = self.dict_users2items[u1]
                    dict_neighbor = {}
                    if len(items_u1) > 0:
                        cand_u2 = np.unique(
                            df_posi.loc[np.isin(df_posi.loc[:, self.colname_item], items_u1), self.colname_user].values)
                        for u2 in cand_u2:
                            if u2 != u1:
                                items_u2 = self.dict_users2items[u2]
                                dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_users[u1] = dict_neighbor
                        sum_simil[u1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_users[u1] = dict_neighbor
                self.dict_simil_users = dict_simil_users
                self.sum_simil = sum_simil

            elif self.way_neighbor == 'item':
                dict_simil_items = {}
                sum_simil = np.zeros(self.num_items)
                for i1 in np.arange(self.num_items):
                    if i1 % round(self.num_items/10) == 0:
                        print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

                    users_i1 = self.dict_items2users[i1]
                    dict_neighbor = {}
                    if len(users_i1) > 0:
                        cand_i2 = np.unique(
                            df_posi.loc[np.isin(df_posi.loc[:, self.colname_user], users_i1), self.colname_item].values)
                        for i2 in cand_i2:
                            if i2 != i1:
                                users_i2 = self.dict_items2users[i2]
                                dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

                        if len(dict_neighbor) > self.num_neighbor:
                            dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                        if self.scale_similarity != 1.0:
                            dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                        if self.normalize_similarity:
                            dict_neighbor = self.normalize_neighbor(dict_neighbor)
                        dict_simil_items[i1] = dict_neighbor
                        sum_simil[i1] = np.sum(np.array(list(dict_neighbor.values())))
                    else:
                        dict_simil_items[i1] = dict_neighbor
                self.dict_simil_items = dict_simil_items
                self.sum_simil = sum_simil


    def trim_neighbor(self, dict_neighbor, num_neighbor):
        return dict(sorted(dict_neighbor.items(), key=lambda x:x[1], reverse = True)[:num_neighbor])

    def normalize_neighbor(self, dict_neighbor):
        sum_simil = 0.0
        for v in dict_neighbor.values():
            sum_simil += v
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = v/sum_simil
        return dict_neighbor

    def rescale_neighbor(self, dict_neighbor, scaling_similarity=1.0):
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = np.power(v, scaling_similarity)
        return dict_neighbor


    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        if self.way_neighbor == 'user':
            for n in np.arange(len(df)):
                u1 = users[n]
                simil_users = np.fromiter(self.dict_simil_users[u1].keys(), dtype=int)
                i_users_posi = self.dict_items2users[items[n]]  # users who consumed i=items[n]
                i_users_treated = self.dict_items2users_treated[items[n]]  # users who are treated i=items[n]
                if n % round(len(df)/10) == 0:
                    print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))
                    # print("simil_users")
                    # print(simil_users)
                    # print(type(simil_users))
                    # print(np.any(np.isin(simil_users, i_users_posi)))

                # initialize for this u1-i pair
                value_T = 0.0
                denom_T = 0.0
                value_C = 0.0
                denom_C = 0.0

                if np.any(np.isin(simil_users, i_users_posi)):
                    simil_users = simil_users[np.isin(simil_users, np.unique(np.append(i_users_treated,i_users_posi)))]
                    for u2 in simil_users:
                        if u2 in i_users_treated:
                            denom_T += self.dict_simil_users[u1][u2]
                            if u2 in i_users_posi:
                                value_T += self.dict_simil_users[u1][u2]
                        else:
                            value_C += self.dict_simil_users[u1][u2]
                            # denom_C += self.dict_simil_users[u1][u2]
                            # if u2 in i_users_posi:
                            #     value_C += self.dict_simil_users[u1][u2]
                    denom_C = self.sum_simil[u1] - denom_T # denom_T + denom_C = sum_simil

                if self.way_self == 'include': # add data of self u-i
                    if u1 in i_users_treated:
                        denom_T += 1.0
                        if u1 in i_users_posi:
                            value_T += 1.0
                    else:
                        denom_C += 1.0
                        if u1 in i_users_posi:
                            value_C += 1.0

                if self.way_self == 'only': # force data to self u-i
                    if u1 in i_users_treated:
                        denom_T = 1.0
                        if u1 in i_users_posi:
                            value_T = 1.0
                        else:
                            value_T = 0.0
                    else:
                        denom_C = 1.0
                        if u1 in i_users_posi:
                            value_C = 1.0
                        else:
                            value_C = 0.0

                if value_T > 0:
                    pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
                if value_C > 0:
                    pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)
            print(pred[:5])
            print(np.mean(pred))
            print(np.max(pred))
            print(np.min(pred))

        elif self.way_neighbor == 'item':
            for n in np.arange(len(df)):
                i1 = items[n]
                simil_items = np.fromiter(self.dict_simil_items[i1].keys(), dtype=int)
                u_items_posi = self.dict_users2items[users[n]]  # items that is consumed by u=users[n]
                u_items_treated = self.dict_users2items_treated[users[n]] # items that is treated for u=users[n]
                if n % round(len(df)/10) == 0:
                    print("progress of prediction computation: {:.1f} %".format(100 * n / len(df)))

                # initialize for this u-i1 pair
                value_T = 0.0
                denom_T = 0.0
                value_C = 0.0
                denom_C = 0.0

                if np.any(np.isin(simil_items, u_items_posi)):
                    simil_items = simil_items[np.isin(simil_items, np.unique(np.append(u_items_posi, u_items_treated)))]
                    for i2 in simil_items:
                        if i2 in u_items_treated: # we assume that treated items are less than untreated items
                            denom_T += self.dict_simil_items[i1][i2]
                            if i2 in u_items_posi:
                                value_T += self.dict_simil_items[i1][i2]
                        else:
                            value_C += self.dict_simil_items[i1][i2]
                            # denom_C += self.dict_simil_items[i1][i2]
                            # if i2 in u_items_posi:
                            #     value_C += self.dict_simil_items[i1][i2]
                    denom_C = self.sum_simil[i1] - denom_T  # denom_T + denom_C = sum_simil

                if self.way_self == 'include': # add data of self u-i
                    if i1 in u_items_treated:
                        denom_T += 1.0
                        if i1 in u_items_posi:
                            value_T += 1.0
                    else:
                        denom_C += 1.0
                        if i1 in u_items_posi:
                            value_C += 1.0

                if self.way_self == 'only': # force data to self u-i
                    if i1 in u_items_treated:
                        denom_T = 1.0
                        if i1 in u_items_posi:
                            value_T = 1.0
                        else:
                            value_T = 0.0
                    else:
                        denom_C = 1.0
                        if i1 in u_items_posi:
                            value_C = 1.0
                        else:
                            value_C = 0.0

                if value_T > 0:
                    pred[n] += 2 * self.weight_treated_outcome * value_T / (self.shrinkage_T + denom_T)
                if value_C > 0:
                    pred[n] -= 2 * (1 - self.weight_treated_outcome) * value_C / (self.shrinkage_C + denom_C)

        return pred


    def simil_jaccard(self, x, y):
        return len(np.intersect1d(x, y))/len(np.union1d(x, y))

    def simil_cosine(self, x, y):
        return len(np.intersect1d(x, y))/np.sqrt(len(x)*len(y))

class CausEProd(Recommender):
    def __init__(self, num_users, num_items,
                 metric='logloss',
                 dim_factor=10, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 reg_causal=0.01,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric

        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.reg_causal = reg_causal
        self.sd_init = sd_init
        self.flag_prepared = False

        # user_factors=user_factors_T=user_factors_C for CausE-Prod
        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        # item_factors_T=item_factors
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        self.item_factors_C = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0


    def prepare_dictionary(self, df, colname_time='idx_time'):
        print("start prepare dictionary")
        self.colname_time = colname_time
        self.num_times = np.max(df.loc[:, self.colname_time]) + 1
        self.dict_treatment_positive_sets = dict()
        self.dict_treatment_negative_sets = dict()
        self.dict_treatment_sets = dict()
        self.dict_control_positive_sets = dict()
        self.dict_control_negative_sets = dict()
        # skip control_negative for its volume
        df_train = df.loc[df.loc[:, self.colname_outcome] + df.loc[:, self.colname_treatment] > 0]

        for t in np.arange(self.num_times):
            df_t = df_train.loc[df_train.loc[:, self.colname_time] == t]
            self.dict_treatment_positive_sets[t] = dict()
            self.dict_treatment_negative_sets[t] = dict()
            self.dict_treatment_sets[t] = dict()
            self.dict_control_positive_sets[t] = dict()
            self.dict_control_negative_sets[t] = dict()

            for u in np.unique(df_t.loc[:, self.colname_user]):

                df_tu = df_t.loc[df_t.loc[:, self.colname_user] == u]
                if len(df_tu) < self.num_items:  # check existence of control negatives
                    self.dict_control_negative_sets[t][u] = []

                bool_control = df_tu.loc[:, self.colname_treatment] == 0
                if np.any(bool_control):
                    self.dict_control_positive_sets[t][u] = df_tu.loc[bool_control, self.colname_item].values
                # only treatment
                bool_treatment = np.logical_not(bool_control)
                if np.any(bool_treatment):
                    df_tu = df_tu.loc[bool_treatment]
                    bool_positive = df_tu.loc[:, self.colname_outcome] > 0
                    self.dict_treatment_sets[t][u] = df_tu.loc[:, self.colname_item].values
                    if np.any(bool_positive):
                        self.dict_treatment_positive_sets[t][u] = df_tu.loc[bool_positive, self.colname_item].values
                    bool_negative = np.logical_not(bool_positive)
                    if np.any(bool_negative):
                        self.dict_treatment_negative_sets[t][u] = df_tu.loc[bool_negative, self.colname_item].values
                # else:
                #     self.dict_treatment_sets[t][u] = []

        self.flag_prepared = True
        print("prepared dictionary!")


    # override
    def sample_pair(self):
        t = self.sample_time()
        if random.random() < 0.5: # pick treatment
            flag_treatment = 1
            while True: # pick a user with treatment
                u = random.randrange(self.num_users)
                if u in self.dict_treatment_sets[t]:
                    break

            i = self.sample_treatment(t, u)
            if u in self.dict_treatment_positive_sets[t] and i in self.dict_treatment_positive_sets[t][u]:
                flag_positive = 1
            else:
                flag_positive = 0

        else: # pick control
            flag_treatment = 0
            while True: # pick a user with control
                u = random.randrange(self.num_users)
                if u in self.dict_treatment_sets[t]:
                    len_T = len(self.dict_treatment_sets[t][u])
                else:
                    len_T = 0
                if len_T < self.num_items:
                    break

            if len_T > self.num_items * 0.99:
                # print(len_T)
                i = self.sample_control2(t, u)
            else:
                i = self.sample_control(t, u)

            if u in self.dict_control_positive_sets[t] and i in self.dict_control_positive_sets[t][u]:
                flag_positive = 1
            else:
                flag_positive = 0

        return u, i, flag_positive, flag_treatment


    def train(self, df, iter = 100):

        if not self.flag_prepared: # prepare dictionary
            self.prepare_dictionary(df)

        err = 0
        current_iter = 0
        if self.metric in ['logloss']:
            while True:
                u, i, flag_positive, flag_treatment = self.sample_pair()

                u_factor = self.user_factors[u, :]
                i_factor_T = self.item_factors[i, :]
                i_factor_C = self.item_factors_C[i, :]

                if flag_treatment > 0:
                    rating = np.sum(u_factor * i_factor_T)
                else:
                    rating = np.sum(u_factor * i_factor_C)

                if self.with_bias:
                    rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                if flag_positive > 0:
                    coeff = self.func_sigmoid(-rating)
                else:
                    coeff = -self.func_sigmoid(rating)

                err += np.abs(coeff)


                i_diff_TC = i_factor_T - i_factor_C


                if flag_treatment > 0:
                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor_T - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor_T - self.reg_causal * i_diff_TC)
                    self.item_factors_C[i, :] += \
                        self.learn_rate * (self.reg_causal * i_diff_TC)
                else:
                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor_C - self.reg_factor * u_factor)
                    self.item_factors_C[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor_C + self.reg_causal * i_diff_TC)
                    self.item_factors[i, :] += \
                        self.learn_rate * (-self.reg_causal * i_diff_TC)

                if self.with_bias:
                    self.item_biases[i] += \
                        self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                    self.user_biases[u] += \
                        self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                    self.global_bias += \
                        self.learn_rate * (coeff)

                current_iter += 1
                if current_iter >= iter:
                    return err / iter

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred_T = np.zeros(len(df))
        pred_C = np.zeros(len(df))

        for n in np.arange(len(df)):
            pred_T[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred_T[n] += self.item_biases[items[n]]
                pred_T[n] += self.user_biases[users[n]]
                pred_T[n] += self.global_bias
            pred_C[n] = np.inner(self.user_factors[users[n], :], self.item_factors_C[items[n], :])
            if self.with_bias:
                pred_C[n] += self.item_biases[items[n]]
                pred_C[n] += self.user_biases[users[n]]
                pred_C[n] += self.global_bias

        pred = 1 / (1 + np.exp(-pred_T)) - 1 / (1 + np.exp(-pred_C))

        return pred

class DLMF_MLP(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=64, hidden_dim=128, with_IPS=True, only_treated=False,
                 learn_rate=0.01, reg_factor=0.01,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):

        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)

        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.only_treated = only_treated
        self.learn_rate = learn_rate
        self.reg_factor = reg_factor
        self.dim_factor = dim_factor
        self.hidden_dim = hidden_dim

        self.rng = RandomState(seed=42)
        self.user_factors = self.rng.normal(0, 0.1, size=(self.num_users, dim_factor))
        self.item_factors = self.rng.normal(0, 0.1, size=(self.num_items, dim_factor))

        # MLP параметры
        self.W1 = self.rng.normal(0, 0.1, size=(2 * dim_factor, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0, 0.1, size=(hidden_dim, 1))
        self.b2 = 0.0

    def func_sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def mlp_forward(self, x):
        h = np.dot(x, self.W1) + self.b1
        h = np.tanh(h)
        return np.dot(h, self.W2)[0] + self.b2

    def mlp_backward(self, x, grad_out):
        h = np.dot(x, self.W1) + self.b1                # (hidden_dim,)
        h_act = np.tanh(h)                              # (hidden_dim,)
        grad_h = (1 - h_act ** 2) * (self.W2.flatten() * grad_out)  # (hidden_dim,)

        grad_W2 = np.outer(h_act, np.array([grad_out]))  # (hidden_dim, 1)
        grad_b2 = grad_out
        grad_W1 = np.outer(grad_h, x)                    # (hidden_dim, input_dim)
        grad_b1 = grad_h

        return grad_W1, grad_b1, grad_W2, grad_b2



    def train(self, df, iter=10000):
        df_train = df[df[self.colname_outcome] > 0].copy()
        if self.only_treated:
            df_train = df_train[df_train[self.colname_treatment] == 1]

        if self.capping_T:
            mask_T = (df_train[self.colname_propensity] < self.capping_T) & (df_train[self.colname_treatment] == 1)
            df_train.loc[mask_T, self.colname_propensity] = self.capping_T

        if self.capping_C:
            mask_C = (df_train[self.colname_propensity] > 1 - self.capping_C) & (df_train[self.colname_treatment] == 0)
            df_train.loc[mask_C, self.colname_propensity] = 1 - self.capping_C

        if self.with_IPS:
            df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] / df_train[self.colname_propensity] - \
                              (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome] / (1 - df_train[self.colname_propensity])
        else:
            df_train['ITE'] = df_train[self.colname_treatment] * df_train[self.colname_outcome] - \
                              (1 - df_train[self.colname_treatment]) * df_train[self.colname_outcome]

        users = df_train[self.colname_user].values
        items = df_train[self.colname_item].values
        ITE = df_train['ITE'].values

        current_iter = 0
        while current_iter < iter:
            df_train = df_train.sample(frac=1)
            for n in range(len(df_train)):
                u = users[n]
                i = items[n]
                while True:
                    j = random.randint(0, self.num_items - 1)
                    if j != i:
                        break

                u_vec = self.user_factors[u]
                i_vec = self.item_factors[i]
                j_vec = self.item_factors[j]

                x_pos = np.concatenate([u_vec, i_vec])
                x_neg = np.concatenate([u_vec, j_vec])

                score_pos = self.mlp_forward(x_pos)
                score_neg = self.mlp_forward(x_neg)
                diff = score_pos - score_neg

                ite = ITE[n]
                coeff = 0.0
                if self.metric == 'AR_logi':
                    if ite >= 0:
                        coeff = ite * self.func_sigmoid(-diff)
                    else:
                        coeff = ite * self.func_sigmoid(diff)
                elif self.metric == 'AR_hinge':
                    if ite >= 0 and diff < 1:
                        coeff = ite
                    elif ite < 0 and diff > -1:
                        coeff = ite

                grad_pos = coeff
                grad_neg = -coeff

                gW1_p, gb1_p, gW2_p, gb2_p = self.mlp_backward(x_pos, grad_pos)
                gW1_n, gb1_n, gW2_n, gb2_n = self.mlp_backward(x_neg, grad_neg)

                self.W1 += self.learn_rate * (gW1_p + gW1_n - self.reg_factor * self.W1)
                self.b1 += self.learn_rate * (gb1_p + gb1_n)
                self.W2 += self.learn_rate * (gW2_p + gW2_n - self.reg_factor * self.W2)
                self.b2 += self.learn_rate * (gb2_p + gb2_n)

                # Обновление эмбеддингов
                self.user_factors[u] += self.learn_rate * (coeff * (i_vec - j_vec) - self.reg_factor * u_vec)
                self.item_factors[i] += self.learn_rate * (coeff * u_vec - self.reg_factor * i_vec)
                self.item_factors[j] += self.learn_rate * (-coeff * u_vec - self.reg_factor * j_vec)

                current_iter += 1
                if current_iter >= iter:
                    return

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        for n in range(len(df)):
            u_vec = self.user_factors[users[n]]
            i_vec = self.item_factors[items[n]]
            x = np.concatenate([u_vec, i_vec])
            pred[n] = self.mlp_forward(x)
        return pred

    
if __name__ == "__main__":
    pass
