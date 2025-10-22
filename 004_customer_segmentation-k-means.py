# KMeans clustering on behavioral features
## 1 ## Features: frequency, monetary, avg_units_per_tx, avg_price_paid, share_of_wallet (if total customer spend available).
## 2 ##  Preprocess: log-transform skewed variables, scale, try k=3..6, pick with silhouette.

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cust_feat = chips.groupby('customer_id').agg({
    'transaction_id': 'nunique',
    'line_total': 'sum',
    'quantity': 'sum'
}).rename(columns={'transaction_id':'frequency','line_total':'monetary','quantity':'units'})

cust_feat['avg_units_per_tx'] = cust_feat['units'] / cust_feat['frequency']
X = cust_feat[['frequency','monetary','avg_units_per_tx']].fillna(0)
X = np.log1p(X)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

best_k, best_score = None, -1
for k in range(2,7):
    km = KMeans(n_clusters=k, random_state=42).fit(Xs)
    s = silhouette_score(Xs, km.labels_)
    if s > best_score:
        best_k, best_score = k, s
print('best_k', best_k)
km = KMeans(n_clusters=best_k, random_state=42).fit(Xs)
cust_feat['cluster'] = km.labels_
cust_feat.to_csv('chips_customer_clusters.csv')
