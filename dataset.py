# import pandas as pd
# import sqlite3

# campaign_desc = pd.read_csv("dunnhumby_The-Complete-Journey CSV/campaign_desc.csv")
# campaign_table = pd.read_csv("dunnhumby_The-Complete-Journey CSV/campaign_table.csv")
# causal_data = pd.read_csv("dunnhumby_The-Complete-Journey CSV/causal_data.csv")
# coupon_redempt = pd.read_csv("dunnhumby_The-Complete-Journey CSV/coupon_redempt.csv")
# coupon = pd.read_csv("dunnhumby_The-Complete-Journey CSV/coupon.csv")
# hh_demographic = pd.read_csv("dunnhumby_The-Complete-Journey CSV/hh_demographic.csv")
# product = pd.read_csv("dunnhumby_The-Complete-Journey CSV/product.csv")
# transaction_data = pd.read_csv("dunnhumby_The-Complete-Journey CSV/transaction_data.csv")

# merged_data = pd.merge(transaction_data, hh_demographic, on="household_key", how="left")
# merged_data = pd.merge(merged_data, product, on="PRODUCT_ID", how="left")

# campaign_merged = pd.merge(campaign_table, campaign_desc, on="CAMPAIGN", how="left")
# merged_data = pd.merge(merged_data, campaign_merged, on="household_key", how="left")

# # coupon_merged = pd.merge(coupon_redempt, coupon, on="COUPON_UPC", how="left")
# # merged_data = pd.merge(merged_data, coupon_merged, on="household_key", how="left")

# dh_original = merged_data.copy()

# conn = sqlite3.connect(':memory:')

# dh_original.to_sql('dh_personalized', conn, index=False, if_exists='replace')
# causal_data.to_sql('causal_data', conn, index=False, if_exists='replace')

# query = """
# SELECT * FROM dh_personalized
# INNER JOIN causal_data ON dh_personalized.PRODUCT_ID = causal_data.PRODUCT_ID AND dh_personalized.WEEK_NO = causal_data.WEEK_NO
# """

# dh_personalized = pd.read_sql(query, conn)

# # dh_personalized = dh_original.copy()
# # dh_personalized = pd.merge(dh_personalized, causal_data, 
# #                            left_on=['PRODUCT_ID', 'WEEK_NO'], 
# #                            right_on=['PRODUCT_ID', 'WEEK_NO'], 
# #                            how='left')

# user_data = pd.read_csv('u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
# item_data = pd.read_csv('u.item', sep='|', encoding='latin-1', names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
# ratings_data = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# ml_data = pd.merge(pd.merge(ratings_data, user_data, on='user_id'), item_data, on='movie_id')


# dh_original = dh_original[:100000]
# dh_personalized = dh_personalized[:100000]
# ml_data = ml_data[:100000]

# dh_personalized.rename(columns={
#     'household_key': 'idx_user',
#     'PRODUCT_ID': 'idx_item',
#     'DAY': 'idx_time',
#     'sales_value': 'outcome',
#     'propensity': 'propensity',
#     'retail_disc': 'treated'
# }, inplace=True)


# dh_original.rename(columns={
#     'household_key': 'idx_user',
#     'PRODUCT_ID': 'idx_item',
#     'DAY': 'idx_time',
#     'sales_value': 'outcome',
# }, inplace=True)

# ml_data.rename(columns={
#     'user_id': 'idx_user',
#     'movie_id': 'idx_item',
#     'timestamp': 'idx_time',
# }, inplace=True)

# print('i am here')
