from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

class PropensityModel(tf.keras.Model):
    def __init__(self, gbdt_model, user_encoder, item_encoder):
        super().__init__()
        self.gbdt_model = gbdt_model
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

    def call(self, inputs, training=False):
        u, i = inputs
        u = u.numpy()
        i = i.numpy()
        u_enc = self.user_encoder.transform(u)
        i_enc = self.item_encoder.transform(i)
        X = np.stack([u_enc, i_enc], axis=1)
        p_pred = self.gbdt_model.predict_proba(X)[:, 1]
        p_tensor = tf.convert_to_tensor(p_pred, dtype=tf.float32)
        return p_tensor, p_tensor, p_tensor, p_tensor


def train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular):
    df = train_df.copy()
    df['attribute'] = 'default'

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_enc'] = user_encoder.fit_transform(df['idx_user'])
    df['item_enc'] = item_encoder.fit_transform(df['idx_item'])

    X = df[['user_enc', 'item_enc']].values
    y = df['outcome'].values

    theta = 0.5
    gamma = np.ones_like(y) * 0.5

    for epoch in range(10):
        weights = []
        for c, g in zip(y, gamma):
            if c == 1:
                weights.append(1.0)
            else:
                denom = 1 - theta * g
                w = (1 - theta) * g / denom if denom > 0 else 0.0
                weights.append(w)
        weights = np.array(weights)
        labels = np.random.binomial(1, weights)

        model = LGBMClassifier(max_depth=3, n_estimators=100, min_child_samples=10)
        model.fit(X, labels, categorical_feature=[0, 1])

        gamma = model.predict_proba(X)[:, 1]

        print(f"--- Epoch {epoch+1} ---")
        print(f"Theta: {theta:.4f}")
        print(f"Labels mean: {labels.mean():.4f}")
        print(f"Gamma mean: {gamma.mean():.4f}, std: {gamma.std():.4f}, min: {gamma.min():.4f}, max: {gamma.max():.4f}")
        print(f"Feature importances: {model.feature_importances_}")
        print()

        theta_new = np.mean(y / np.clip(gamma, 1e-6, 1))
        theta = np.clip(theta_new, 0.01, 0.99)

    tf_model = PropensityModel(model, user_encoder, item_encoder)
    return tf_model
