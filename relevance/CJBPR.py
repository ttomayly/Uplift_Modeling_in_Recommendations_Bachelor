import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

class CJBPR(tf.keras.Model):
    def __init__(self, train_df, vali_df=None, test_df=None, 
                 hidden_dim=100, learning_rate=0.001, reg=2e-5, 
                 alpha=500000, beta=0.5, C=6, neg_samples=5, 
                 batch_size=1024, epochs=20, display_interval=1, **kwargs):
        super(CJBPR, self).__init__(**kwargs)
        
        # Store parameters
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.reg = reg
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.neg = neg_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.display = display_interval
        
        self._init_data_structures()
        
        self._preprocess_data(train_df, vali_df, test_df)
        
        self._build_components()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.compile(optimizer=self.optimizer)

    def _init_data_structures(self):
        """Initialize all required data structures"""
        self.num_users = 0
        self.num_items = 0
        self.user_map = {}
        self.item_map = {}
        self.item_pop = None
        self.item_pop_tensor = None
        self.train_like = []
        self.vali_like = []
        self.test_users = []
        self.test_interact = []
        self.test_like = []
        self.df_list = []

    def _preprocess_data(self, train_df, vali_df, test_df):
        """Preprocess all input data with memory efficiency"""
        train_df = train_df[train_df['outcome'] >= 1].copy()
        train_df = train_df.drop(columns=['outcome'])
        
        all_users = sorted(train_df['idx_user'].unique())
        all_items = sorted(train_df['idx_item'].unique())
        
        self.user_map = {u: i for i, u in enumerate(all_users)}
        self.item_map = {i: idx for idx, i in enumerate(all_items)}
        
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        
        self.train_df = train_df.copy()
        self.train_df['idx_user'] = self.train_df['idx_user'].map(self.user_map)
        self.train_df['idx_item'] = self.train_df['idx_item'].map(self.item_map)
        
        self._split_data_into_components()
        
        self._compute_item_popularity()
        
        self._process_validation_data(vali_df)
        self._process_test_data(test_df)
        
        self._create_training_lists()

    def _split_data_into_components(self):
        """Split training data into C components"""
        len_train = len(self.train_df)
        df_len = int(len_train / self.C)
        left_idx = np.arange(len_train)
        
        for i in range(self.C - 1):
            idx = np.random.choice(left_idx, df_len, replace=False)
            self.df_list.append(self.train_df.iloc[idx].copy())
            left_idx = np.setdiff1d(left_idx, idx)
        
        self.df_list.append(self.train_df.iloc[left_idx].copy())

    def _compute_item_popularity(self):
        """Compute and normalize item popularity"""
        item_counts = self.train_df['idx_item'].value_counts()
        self.item_pop = np.zeros(self.num_items)
        for item, count in item_counts.items():
            self.item_pop[item] = count
        self.item_pop = (self.item_pop / self.item_pop.max()).reshape(-1, 1)
        self.item_pop_tensor = tf.convert_to_tensor(self.item_pop, dtype=tf.float32)

    def _process_validation_data(self, vali_df):
        """Process validation data if provided"""
        if vali_df is not None:
            self.vali_df = vali_df.copy()
            self.vali_df['idx_user'] = self.vali_df['idx_user'].map(self.user_map)
            self.vali_df['idx_item'] = self.vali_df['idx_item'].map(self.item_map)
            
            self.vali_like = [[] for _ in range(self.num_users)]
            for u, i, r in zip(self.vali_df['idx_user'], 
                             self.vali_df['idx_item'], 
                             self.vali_df['outcome']):
                if r >= 1:
                    self.vali_like[u].append(i)

    def _process_test_data(self, test_df):
        """Process test data if provided"""
        if test_df is not None:
            self.test_df = test_df.copy()
            self.test_df['idx_user'] = self.test_df['idx_user'].map(self.user_map)
            self.test_df['idx_item'] = self.test_df['idx_item'].map(self.item_map)
            
            train_pos_counts = self.train_df.groupby('idx_user').size()
            test_pos_counts = self.test_df[self.test_df['outcome'] >= 1].groupby('idx_user').size()
            
            valid_test_users = [
                u for u in test_pos_counts.index
                if test_pos_counts[u] >= 1 and train_pos_counts.get(u, 0) >= 2
            ]
            
            self.test_df = self.test_df[self.test_df['idx_user'].isin(valid_test_users)]
            
            self.test_users = list(self.test_df['idx_user'].unique())
            self.test_interact = []
            self.test_like = []
            
            for u in self.test_users:
                user_df = self.test_df[self.test_df['idx_user'] == u]
                self.test_interact.append(user_df['idx_item'].values)
                self.test_like.append(user_df[user_df['outcome'] >= 1]['idx_item'].values)

    def _create_training_lists(self):
        """Create training like lists"""
        self.train_like = [[] for _ in range(self.num_users)]
        for u, i in zip(self.train_df['idx_user'], self.train_df['idx_item']):
            self.train_like[u].append(i)

    def _build_components(self):
        """Build model components with proper initialization"""
        initializer = tf.keras.initializers.RandomNormal(stddev=0.03)
        
        self.P = [self.add_weight(
            name=f'P_{i}',
            shape=(self.num_users, self.hidden_dim),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]
        
        self.Q = [self.add_weight(
            name=f'Q_{i}',
            shape=(self.num_items, self.hidden_dim),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]
        
        # Exposure parameters
        self.c = [self.add_weight(
            name=f'c_{i}',
            shape=(self.hidden_dim, 1),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]
        
        self.d = [self.add_weight(
            name=f'd_{i}',
            shape=(1, 1),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]
        
        self.a = [self.add_weight(
            name=f'a_{i}',
            shape=(self.hidden_dim, 1),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]
        
        self.b = [self.add_weight(
            name=f'b_{i}',
            shape=(1, 1),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]
        
        self.e = [self.add_weight(
            name=f'e_{i}',
            shape=(self.hidden_dim, 1),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]
        
        self.f = [self.add_weight(
            name=f'f_{i}',
            shape=(1, 1),
            initializer=initializer,
            dtype=tf.float32) for i in range(self.C)]

    def call(self, inputs, training=False):
        """Memory-efficient call implementation"""
        u_batch, i_batch = inputs
        
        u_batch = tf.reshape(tf.cast(u_batch, tf.int32), [-1])  # Flatten to [batch_size]
        i_batch = tf.reshape(tf.cast(i_batch, tf.int32), [-1])  # Flatten to [batch_size]
        
        batch_size = tf.shape(u_batch)[0]
        r_pred = tf.zeros([batch_size], dtype=tf.float32)  # Now 1D tensor
        p_pred = tf.zeros([batch_size], dtype=tf.float32)  # Now 1D tensor
        
        max_chunk_size = 5000  # Adjust based on your GPU memory
        num_chunks = tf.cast(tf.math.ceil(batch_size / max_chunk_size), tf.int32)
        
        for chunk_idx in tf.range(num_chunks):
            start = chunk_idx * max_chunk_size
            end = tf.minimum(start + max_chunk_size, batch_size)
            
            u_chunk = u_batch[start:end]
            i_chunk = i_batch[start:end]
            
            r_chunk, p_chunk = self._process_chunk(u_chunk, i_chunk)
            
            r_pred = tf.tensor_scatter_nd_update(
                r_pred, 
                tf.reshape(tf.range(start, end), [-1, 1]),
                r_chunk
            )
            p_pred = tf.tensor_scatter_nd_update(
                p_pred,
                tf.reshape(tf.range(start, end), [-1, 1]),
                p_chunk
            )
        
        return (
            tf.reshape(r_pred / self.C, [-1, 1]),
            tf.reshape(p_pred / self.C, [-1, 1]) 
        )


    def _process_chunk(self, u_chunk, i_chunk):
        """Process a chunk of data"""
        r_chunk = tf.zeros(tf.shape(u_chunk)[0], dtype=tf.float32) 
        p_chunk = tf.zeros(tf.shape(u_chunk)[0], dtype=tf.float32)
        
        for m in range(self.C):
            p = tf.nn.embedding_lookup(self.P[m], u_chunk)
            q = tf.nn.embedding_lookup(self.Q[m], i_chunk)
            
            r_chunk += tf.reduce_sum(p * q, axis=1)
            
            # Compute w term
            w_term = tf.nn.sigmoid(
                tf.reduce_sum(q * self.a[m][:, 0], axis=1) + tf.squeeze(self.b[m])) 
            
            c_term = tf.nn.sigmoid(
                tf.reduce_sum(q * self.c[m][:, 0], axis=1) + tf.squeeze(self.d[m]))
            
            pop_values = tf.gather(tf.squeeze(self.item_pop_tensor), i_chunk)
            
            first_part = w_term * c_term + (1 - w_term) * pop_values
            
            exponent = tf.nn.sigmoid(
                tf.reduce_sum(q * self.e[m][:, 0], axis=1) + tf.squeeze(self.f[m]))
            
            # Final pop calculation
            pop = tf.pow(first_part, exponent)
            p_chunk += tf.clip_by_value(pop, 0.01, 0.99)
        
        return r_chunk, p_chunk 

    def train_step(self, data):
        """Training step with memory management"""
        u_batch, i_pos_batch, i_neg_batch, _, _ = data
        
        u_batch = tf.cast(u_batch, tf.int32)
        i_pos_batch = tf.cast(i_pos_batch, tf.int32)
        i_neg_batch = tf.cast(i_neg_batch, tf.int32)
        
        with tf.GradientTape() as tape:
            # Process in chunks to avoid OOM
            chunk_size = min(5000, tf.shape(u_batch)[0]) 
            losses = []
            
            for i in range(0, tf.shape(u_batch)[0], chunk_size):
                u_chunk = u_batch[i:i+chunk_size]
                i_pos_chunk = i_pos_batch[i:i+chunk_size]
                i_neg_chunk = i_neg_batch[i:i+chunk_size]
                
                r_pos, p_pos = self((u_chunk, i_pos_chunk), training=True)
                r_neg, p_neg = self((u_chunk, i_neg_chunk), training=True)
                
                rel_loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(r_pos - r_neg)))
                exp_loss = -tf.reduce_mean(tf.math.log(p_pos)) - tf.reduce_mean(tf.math.log(1 - p_neg))
                losses.append((rel_loss, exp_loss))
            
            rel_loss = tf.reduce_mean([l[0] for l in losses])
            exp_loss = tf.reduce_mean([l[1] for l in losses])
            
            reg_loss = self._compute_regularization_loss()
            
            total_loss = rel_loss + exp_loss + reg_loss
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {
            "rel_loss": rel_loss,
            "exp_loss": exp_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss
        }

    def _compute_regularization_loss(self):
        """Compute regularization loss"""
        reg_loss = 0
        for m in range(self.C):
            reg_loss += self.reg * 0.5 * (
                tf.reduce_sum(tf.square(self.P[m])) + 
                tf.reduce_sum(tf.square(self.Q[m]))
            )
            reg_loss += self.alpha * self.reg * 0.5 * (
                tf.reduce_sum(tf.square(self.c[m])) + 
                tf.reduce_sum(tf.square(self.d[m])) +
                tf.reduce_sum(tf.square(self.a[m])) + 
                tf.reduce_sum(tf.square(self.b[m])) +
                tf.reduce_sum(tf.square(self.e[m])) + 
                tf.reduce_sum(tf.square(self.f[m]))
            )
        return reg_loss

    def fit(self):
        """Training loop with progress tracking"""
        for epoch in range(1, self.epochs + 1):
            for m in range(self.C):
                df = self.df_list[m]
                user_list, item_pos_list, item_neg_list = self._negative_sampling(df)
                
                dataset = tf.data.Dataset.from_tensor_slices(
                    (user_list, item_pos_list, item_neg_list, 
                     np.zeros(len(user_list)), np.zeros(len(user_list)))
                ).shuffle(len(user_list)).batch(self.batch_size)
                
                for batch in dataset:
                    metrics = self.train_step(batch)
                
                if epoch % self.display == 0:
                    print(f"Epoch {epoch}, Component {m}: "
                          f"Rel Loss={metrics['rel_loss']:.4f}, "
                          f"Exp Loss={metrics['exp_loss']:.4f}, "
                          f"Reg Loss={metrics['reg_loss']:.4f}, "
                          f"Total Loss={metrics['total_loss']:.4f}")
            
            if epoch % self.display == 0:
                if hasattr(self, 'vali_like') and self.vali_like:
                    self._validate(epoch)
                if hasattr(self, 'test_users') and self.test_users:
                    self._test(epoch)

    def _negative_sampling(self, df):
        """Generate negative samples"""
        pos_users = df['idx_user'].values.reshape((-1, 1))
        pos_items = df['idx_item'].values.reshape((-1, 1))
        
        users = np.tile(pos_users, (self.neg, 1))
        pos = np.tile(pos_items, (self.neg, 1))
        
        neg = np.random.randint(0, self.num_items, size=(len(pos), 1))
        
        mask = (neg != pos).reshape(-1)
        users = users[mask]
        pos = pos[mask]
        neg = neg[mask]
        
        return users.astype(np.int32), pos.astype(np.int32), neg.astype(np.int32)

    def _validate(self, epoch):
        """Memory-efficient validation"""
        # Process validation in batches to avoid OOM
        user_batch_size = 100
        item_batch_size = 1000
        
        recall_sum = 0.0
        valid_users = 0
        
        for u_start in range(0, self.num_users, user_batch_size):
            u_end = min(u_start + user_batch_size, self.num_users)
            u_batch = np.arange(u_start, u_end)
            
            valid_u_batch = [u for u in u_batch if self.vali_like[u]]
            if not valid_u_batch:
                continue
                
            user_scores = np.zeros((len(valid_u_batch), self.num_items))
            
            for i_start in range(0, self.num_items, item_batch_size):
                i_end = min(i_start + item_batch_size, self.num_items)
                i_batch = np.arange(i_start, i_end)
                
                u_tiled = np.repeat(valid_u_batch, len(i_batch))
                i_tiled = np.tile(i_batch, len(valid_u_batch))
                
                r_batch, _ = self((u_tiled, i_tiled), training=False)
                r_batch = r_batch.numpy().reshape(len(valid_u_batch), len(i_batch))
                
                user_scores[:, i_start:i_end] = r_batch
            
            for i, u in enumerate(valid_u_batch):
                scores = user_scores[i]
                train_items = self.train_like[u]
                test_items = self.vali_like[u]
                
                scores[train_items] = -np.inf
                
                top_items = np.argsort(-scores)[:10]
                
                # Calculate recall
                hits = len(set(top_items) & set(test_items))
                recall = hits / len(test_items)
                recall_sum += recall
                valid_users += 1
        
        if valid_users > 0:
            avg_recall = recall_sum / valid_users
            print(f"Validation @ Epoch {epoch}: Recall@10={avg_recall:.4f}")
        else:
            print(f"Validation @ Epoch {epoch}: No valid users with test items")

    def _test(self, epoch):
        """Memory-efficient testing"""
        # Process test users in batches
        user_batch_size = 100
        item_batch_size = 1000
        
        recall_sum = 0.0
        valid_users = 0
        
        for u_start in range(0, len(self.test_users), user_batch_size):
            u_end = min(u_start + user_batch_size, len(self.test_users))
            u_batch = self.test_users[u_start:u_end]
            
            user_scores = np.zeros((len(u_batch), self.num_items))
            
            for i_start in range(0, self.num_items, item_batch_size):
                i_end = min(i_start + item_batch_size, self.num_items)
                i_batch = np.arange(i_start, i_end, dtype=np.int32)
                
                u_tiled = np.repeat(u_batch, len(i_batch))
                i_tiled = np.tile(i_batch, len(u_batch))
                
                r_batch, _ = self((u_tiled, i_tiled), training=False)
                r_batch = r_batch.numpy().reshape(len(u_batch), len(i_batch))
                
                user_scores[:, i_start:i_end] = r_batch
            
            for i, u in enumerate(u_batch):
                idx = u_start + i
                test_items = self.test_interact[idx]
                test_likes = self.test_like[idx]
                
                if len(test_likes) == 0:
                    continue
                    
                test_items = np.array(test_items, dtype=np.int32)
                
                scores = user_scores[i, test_items]
                
                ranked = np.argsort(-scores)[:10]
                
                hits = 0
                for pos in ranked:
                    if test_items[pos] in test_likes:
                        hits += 1
                
                recall = hits / len(test_likes)
                recall_sum += recall
                valid_users += 1
        
        if valid_users > 0:
            avg_recall = recall_sum / valid_users
            print(f"Test @ Epoch {epoch}: Avg Recall@10={avg_recall:.4f}")
        else:
            print(f"Test @ Epoch {epoch}: No valid test users")
            
    def save(self, path):
        """Save model to directory"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        params = {
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'reg': self.reg,
            'alpha': self.alpha,
            'beta': self.beta,
            'C': self.C,
            'neg': self.neg,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'display': self.display,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'item_pop': self.item_pop
        }
        
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)
        
        self.save_weights(os.path.join(path, 'weights'))

    @classmethod
    def load(cls, path):
        """Load model from directory"""
        with open(os.path.join(path, 'params.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        dummy_df = pd.DataFrame({'idx_user': [0], 'idx_item': [0], 'outcome': [1]})
        model = cls(dummy_df, 
                   hidden_dim=params['hidden_dim'],
                   learning_rate=params['lr'],
                   reg=params['reg'],
                   alpha=params['alpha'],
                   beta=params['beta'],
                   C=params['C'],
                   neg_samples=params['neg'],
                   batch_size=params['batch_size'],
                   epochs=params['epochs'],
                   display_interval=params['display'])
        
        model.num_users = params['num_users']
        model.num_items = params['num_items']
        model.user_map = params['user_map']
        model.item_map = params['item_map']
        model.item_pop = params['item_pop']
        model.item_pop_tensor = tf.convert_to_tensor(model.item_pop, dtype=tf.float32)
        
        model.load_weights(os.path.join(path, 'weights'))
        
        return model