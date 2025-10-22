# RFM (simple & powerful)
## 1 ## Compute Recency (days since last chips purchase as of analysis date), Frequency (# chips transactions), Monetary (total spent on chips).
## 2 ## Standardize and create RFM score (1-5 each), then combine (e.g., R*100 + F*10 + M).
## 3 ## Identify segments: Champion, Loyal, Potential, At-risk, Lost.

import numpy as np
snapshot_date = chips['transaction_date'].max() + pd.Timedelta(days=1)
rfm = chips.groupby('customer_id').agg({
    'transaction_date': lambda x: (snapshot_date - x.max()).days,
    'transaction_id': 'nunique',
    'line_total': 'sum'
}).rename(columns={'transaction_date':'recency','transaction_id':'frequency','line_total':'monetary'})

# score
rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1]).astype(int)   # lower recency => higher score
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_Score'] = rfm['r_score'].astype(str)+rfm['f_score'].astype(str)+rfm['m_score'].astype(str)
rfm.to_csv('chips_rfm.csv')
