# Basket / cross-sell analysis
## 1 ## Look at transactions containing chips and compute most common co-purchased categories (e.g., soda, dip, sandwich fillings).
## 2 ## Use simple co-occurrence:

tx_lines = merged[['transaction_id','category','sku','is_chips']]
chips_txids = tx_lines[tx_lines['is_chips']]['transaction_id'].unique()
co = merged[merged['transaction_id'].isin(chips_txids)].groupby('category')['transaction_id'].nunique().sort_values(ascending=False)
co.to_csv('chips_basket_category_cooccurrence.csv')
