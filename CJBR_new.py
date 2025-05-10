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
        
        # Initialize data structures
        self._init_data_structures()
        
        # Preprocess data
        self._preprocess_data(train_df, vali_df, test_df)
        
        # Build model components
        self._build_components()
        
        # Setup optimizer
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
        # Process training data
        train_df = train_df[train_df['outcome'] >= 1].copy()
        train_df = train_df.drop(columns=['outcome'])
        
        # Create mappings
        all_users = sorted(train_df['idx_user'].unique())
        all_items = sorted(train_df['idx_item'].unique())
        
        self.user_map = {u: i for i, u in enumerate(all_users)}
        self.item_map = {i: idx for idx, i in enumerate(all_items)}
        
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        
        # Apply mappings
        self.train_df = train_df.copy()
        self.train_df['idx_user'] = self.train_df['idx_user'].map(self.user_map)
        self.train_df['idx_item'] = self.train_df['idx_item'].map(self.item_map)
        
        # Split data into components
        self._split_data_into_components()
        
        # Compute item popularity
        self._compute_item_popularity()
        
        # Process validation and test data
        self._process_validation_data(vali_df)
        self._process_test_data(test_df)
        
        # Create training like lists
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
        
        # Initialize all components as float32
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
        
        # Convert inputs to proper types and shapes
        u_batch = tf.reshape(tf.cast(u_batch, tf.int32), [-1])  # Flatten to [batch_size]
        i_batch = tf.reshape(tf.cast(i_batch, tf.int32), [-1])  # Flatten to [batch_size]
        
        # Initialize outputs
        batch_size = tf.shape(u_batch)[0]
        r_pred = tf.zeros([batch_size], dtype=tf.float32)  # Now 1D tensor
        p_pred = tf.zeros([batch_size], dtype=tf.float32)  # Now 1D tensor
        
        # Process in smaller chunks if batch is too large
        max_chunk_size = 5000  # Adjust based on your GPU memory
        num_chunks = tf.cast(tf.math.ceil(batch_size / max_chunk_size), tf.int32)
        
        for chunk_idx in tf.range(num_chunks):
            start = chunk_idx * max_chunk_size
            end = tf.minimum(start + max_chunk_size, batch_size)
            
            u_chunk = u_batch[start:end]
            i_chunk = i_batch[start:end]
            
            r_chunk, p_chunk = self._process_chunk(u_chunk, i_chunk)
            
            # Update the corresponding positions in the output tensors
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
            tf.reshape(r_pred / self.C, [-1, 1]),  # Reshape to [batch_size, 1]
            tf.reshape(p_pred / self.C, [-1, 1])   # Reshape to [batch_size, 1]
        )


    def _process_chunk(self, u_chunk, i_chunk):
        """Process a chunk of data"""
        # Initialize outputs as 1D tensors
        r_chunk = tf.zeros(tf.shape(u_chunk)[0], dtype=tf.float32)  # Shape [batch_size]
        p_chunk = tf.zeros(tf.shape(u_chunk)[0], dtype=tf.float32)  # Shape [batch_size]
        
        for m in range(self.C):
            # Get embeddings for this chunk
            p = tf.nn.embedding_lookup(self.P[m], u_chunk)  # Shape [batch_size, hidden_dim]
            q = tf.nn.embedding_lookup(self.Q[m], i_chunk)  # Shape [batch_size, hidden_dim]
            
            # Relevance prediction - sum reduces to [batch_size]
            r_chunk += tf.reduce_sum(p * q, axis=1)
            
            # Exposure prediction - optimized to avoid large intermediate tensors
            # Compute w term
            w_term = tf.nn.sigmoid(
                tf.reduce_sum(q * self.a[m][:, 0], axis=1) + tf.squeeze(self.b[m]))  # Shape [batch_size]
            
            # Compute first part of pop
            c_term = tf.nn.sigmoid(
                tf.reduce_sum(q * self.c[m][:, 0], axis=1) + tf.squeeze(self.d[m]))  # Shape [batch_size]
            
            # Get item popularity values
            pop_values = tf.gather(tf.squeeze(self.item_pop_tensor), i_chunk)  # Shape [batch_size]
            
            # Combine with w term
            first_part = w_term * c_term + (1 - w_term) * pop_values  # Shape [batch_size]
            
            # Compute exponent term
            exponent = tf.nn.sigmoid(
                tf.reduce_sum(q * self.e[m][:, 0], axis=1) + tf.squeeze(self.f[m]))  # Shape [batch_size]
            
            # Final pop calculation
            pop = tf.pow(first_part, exponent)  # Shape [batch_size]
            p_chunk += tf.clip_by_value(pop, 0.01, 0.99)
        
        return r_chunk, p_chunk  # Both shapes [batch_size]

    def train_step(self, data):
        """Training step with memory management"""
        u_batch, i_pos_batch, i_neg_batch, _, _ = data
        
        u_batch = tf.cast(u_batch, tf.int32)
        i_pos_batch = tf.cast(i_pos_batch, tf.int32)
        i_neg_batch = tf.cast(i_neg_batch, tf.int32)
        
        with tf.GradientTape() as tape:
            # Process in chunks to avoid OOM
            chunk_size = min(5000, tf.shape(u_batch)[0])  # Adjust based on your GPU memory
            losses = []
            
            for i in range(0, tf.shape(u_batch)[0], chunk_size):
                u_chunk = u_batch[i:i+chunk_size]
                i_pos_chunk = i_pos_batch[i:i+chunk_size]
                i_neg_chunk = i_neg_batch[i:i+chunk_size]
                
                # Process positive and negative samples separately
                r_pos, p_pos = self((u_chunk, i_pos_chunk), training=True)
                r_neg, p_neg = self((u_chunk, i_neg_chunk), training=True)
                
                # Compute losses for this chunk
                rel_loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(r_pos - r_neg)))
                exp_loss = -tf.reduce_mean(tf.math.log(p_pos)) - tf.reduce_mean(tf.math.log(1 - p_neg))
                losses.append((rel_loss, exp_loss))
            
            # Average losses across chunks
            rel_loss = tf.reduce_mean([l[0] for l in losses])
            exp_loss = tf.reduce_mean([l[1] for l in losses])
            
            # Regularization loss
            reg_loss = self._compute_regularization_loss()
            
            # Total loss
            total_loss = rel_loss + exp_loss + reg_loss
        
        # Compute and apply gradients
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
        user_batch_size = 100  # Process 100 users at a time
        item_batch_size = 1000  # Process 1000 items at a time
        
        recall_sum = 0.0
        valid_users = 0
        
        for u_start in range(0, self.num_users, user_batch_size):
            u_end = min(u_start + user_batch_size, self.num_users)
            u_batch = np.arange(u_start, u_end)
            
            # Only consider users with validation likes
            valid_u_batch = [u for u in u_batch if self.vali_like[u]]
            if not valid_u_batch:
                continue
                
            # Initialize scores for this user batch
            user_scores = np.zeros((len(valid_u_batch), self.num_items))
            
            # Process items in batches
            for i_start in range(0, self.num_items, item_batch_size):
                i_end = min(i_start + item_batch_size, self.num_items)
                i_batch = np.arange(i_start, i_end)
                
                # Create input tensors
                u_tiled = np.repeat(valid_u_batch, len(i_batch))
                i_tiled = np.tile(i_batch, len(valid_u_batch))
                
                # Get predictions
                r_batch, _ = self((u_tiled, i_tiled), training=False)
                r_batch = r_batch.numpy().reshape(len(valid_u_batch), len(i_batch))
                
                # Update scores
                user_scores[:, i_start:i_end] = r_batch
            
            # Evaluate recall for each user in this batch
            for i, u in enumerate(valid_u_batch):
                scores = user_scores[i]
                train_items = self.train_like[u]
                test_items = self.vali_like[u]
                
                # Mask out training items
                scores[train_items] = -np.inf
                
                # Get top 10 items
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
        user_batch_size = 100  # Process 100 users at a time
        item_batch_size = 1000  # Process 1000 items at a time
        
        recall_sum = 0.0
        valid_users = 0
        
        for u_start in range(0, len(self.test_users), user_batch_size):
            u_end = min(u_start + user_batch_size, len(self.test_users))
            u_batch = self.test_users[u_start:u_end]
            
            # Initialize scores for this user batch
            user_scores = np.zeros((len(u_batch), self.num_items))
            
            # Process items in batches
            for i_start in range(0, self.num_items, item_batch_size):
                i_end = min(i_start + item_batch_size, self.num_items)
                i_batch = np.arange(i_start, i_end, dtype=np.int32)  # Ensure integer type
                
                # Create input tensors
                u_tiled = np.repeat(u_batch, len(i_batch))
                i_tiled = np.tile(i_batch, len(u_batch))
                
                # Get predictions
                r_batch, _ = self((u_tiled, i_tiled), training=False)
                r_batch = r_batch.numpy().reshape(len(u_batch), len(i_batch))
                
                # Update scores
                user_scores[:, i_start:i_end] = r_batch
            
            # Evaluate recall for each user in this batch
            for i, u in enumerate(u_batch):
                idx = u_start + i
                test_items = self.test_interact[idx]
                test_likes = self.test_like[idx]
                
                if len(test_likes) == 0:
                    continue
                    
                # Ensure test_items are integers
                test_items = np.array(test_items, dtype=np.int32)
                
                # Get scores for test items only
                scores = user_scores[i, test_items]  # Now test_items are guaranteed to be integers
                
                # Get top 10 items
                ranked = np.argsort(-scores)[:10]
                
                # Calculate recall
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
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import os
# from tqdm import tqdm

# # === Вспомогательные функции === #
# def get_train_instances(train_df, num_negatives=5):
#     """
#     Создаёт обучающие примеры с отрицательными примерами (negative sampling)
#     """
#     user_input, item_input, labels, propensities = [], [], [], []
#     user_item_set = set(zip(train_df['idx_user'], train_df['idx_item']))
#     all_items = train_df['idx_item'].unique()

#     for u, i, p in zip(train_df['idx_user'], train_df['idx_item'], train_df['propensity']):
#         user_input.append(u)
#         item_input.append(i)
#         labels.append(1)
#         propensities.append(p)

#         for _ in range(num_negatives):
#             j = np.random.choice(all_items)
#             while (u, j) in user_item_set:
#                 j = np.random.choice(all_items)
#             user_input.append(u)
#             item_input.append(j)
#             labels.append(0)
#             propensities.append(1.0)

#     return np.array(user_input), np.array(item_input), np.array(labels), np.array(propensities)

# # === Класс модели CJBPR (адаптация из статьи) === #
# class CJBPR:
#     def __init__(self, sess, args, train_df,
#                  train_like=None, vali_like=None,
#                  test_user=None, test_interact=None,
#                  test_like=None, item_pop=None):
#         """
#         Адаптированная версия модели CJBPR из статьи (без TF 1.x сессии)
#         Все параметры кроме train_df можно опустить или передавать как None
#         """
#         self.sess = sess
#         self.args = args
#         self.train_df = train_df
#         self.train_like = train_like
#         self.vali_like = vali_like
#         self.test_user = test_user
#         self.test_interact = test_interact
#         self.test_like = test_like
#         self.item_pop = item_pop

#         self.num_users = args.num_users
#         self.num_items = args.num_items
#         self.hidden = args.hidden
#         self.neg = args.neg
#         self.bs = args.bs
#         self.p_weight = args.p_weight

#         # Эмбеддинги пользователей и товаров
#         self.user_emb = tf.Variable(tf.random.normal([self.num_users, self.hidden]), name="user_emb")
#         self.item_emb = tf.Variable(tf.random.normal([self.num_items, self.hidden]), name="item_emb")

#         # MLP для пропенсити и релевантности
#         self.propensity_mlp = tf.keras.Sequential([
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
#         self.relevance_mlp = tf.keras.Sequential([
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])

#     def __call__(self, inputs, training=False):
#         u, i = inputs
#         u_emb = tf.nn.embedding_lookup(self.user_emb, u)
#         i_emb = tf.nn.embedding_lookup(self.item_emb, i)
#         x = tf.concat([u_emb, i_emb], axis=-1)

#         propensity = self.propensity_mlp(x)
#         relevance = self.relevance_mlp(x)
#         score = tf.reduce_sum(u_emb * i_emb, axis=1, keepdims=True)

#         return score, tf.squeeze(propensity), tf.squeeze(relevance), x

#     def train(self):
#         """
#         Основной цикл обучения модели
#         """
#         user_input, item_input, labels, propensities = get_train_instances(self.train_df, self.neg)
#         dataset = tf.data.Dataset.from_tensor_slices(((user_input, item_input), labels, propensities))
#         dataset = dataset.shuffle(100_000).batch(self.bs)

#         optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)

#         for epoch in range(self.args.epoch):
#             total_loss = 0
#             for (u, i), y, p in dataset:
#                 with tf.GradientTape() as tape:
#                     score, pred_p, pred_r, _ = self((u, i), training=True)
#                     loss_click = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=score[:, 0]))
#                     loss_p = tf.reduce_mean(tf.square(pred_p - p))
#                     loss_r = tf.reduce_mean(tf.square(pred_r - tf.cast(y, tf.float32)))
#                     loss = loss_click + self.p_weight * loss_p + (1 - self.p_weight) * loss_r
#                 grads = tape.gradient(loss, [self.user_emb, self.item_emb] + self.propensity_mlp.trainable_variables + self.relevance_mlp.trainable_variables)
#                 optimizer.apply_gradients(zip(grads, [self.user_emb, self.item_emb] + self.propensity_mlp.trainable_variables + self.relevance_mlp.trainable_variables))
#                 total_loss += loss.numpy()
#             print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

#         # Сохраняем веса эмбеддингов
#         os.makedirs("./saved_weights", exist_ok=True)
#         np.save("./saved_weights/user_emb.npy", self.user_emb.numpy())
#         np.save("./saved_weights/item_emb.npy", self.item_emb.numpy())

#     def predict_scores(self, df):
#         """
#         Предсказание пропенсити и релевантности по батчам
#         """
#         test_users = tf.convert_to_tensor(df['idx_user'].values)
#         test_items = tf.convert_to_tensor(df['idx_item'].values)
#         test_t_data = tf.data.Dataset.from_tensor_slices((test_users, test_items))

#         r_pred_test, p_pred_test = None, None
#         for u, i in test_t_data.batch(5000):
#             _, p_batch, r_batch, _ = self((u, i), training=False)
#             if r_pred_test is None:
#                 r_pred_test = r_batch
#                 p_pred_test = p_batch
#             else:
#                 r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
#                 p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

#         return p_pred_test.numpy(), r_pred_test.numpy()

# # === Аргументы для инициализации === #
# class Args:
#     num_users = 1000
#     num_items = 1000
#     hidden = 64
#     neg = 5
#     bs = 1024
#     epoch = 10
#     lr = 0.001
#     p_weight = 0.4

# # === Пример использования === #
# if __name__ == '__main__':
#     df = pd.read_csv("/mnt/data/data_train.csv")
#     args = Args()
#     sess = tf.compat.v1.Session()

#     # В реальности тебе нужно заранее задать num_users и num_items, либо получить из данных:
#     args.num_users = df['idx_user'].nunique()
#     args.num_items = df['idx_item'].nunique()

#     model = CJBPR(sess, args, train_df=df)
#     model.train()

#     propensity, relevance = model.predict_scores(df)
#     print("First 10 relevance predictions:", relevance[:10])
