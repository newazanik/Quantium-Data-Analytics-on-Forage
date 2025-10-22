# Data cleaning & validation — step-by-step (pandas)

# Save as chips_analysis_start.py or use notebook
import pandas as pd
import numpy as np
from datetime import datetime

# --- load ---
tx = pd.read_csv('transactions.csv', parse_dates=['transaction_date'], dayfirst=False)  # adjust parse if needed
cust = pd.read_csv('customers.csv', parse_dates=['signup_date'], dayfirst=False)

# --- basic inspections ---
print(tx.shape, cust.shape)
print(tx.head())
print(tx.dtypes)

# --- common sanity checks on transaction data ---
# 1. Missing critical fields
required_tx = ['transaction_id','customer_id','sku','quantity','price','transaction_date','store_id','category','subcategory']
for c in required_tx:
    if c not in tx.columns:
        print(f'WARNING: {c} missing from transactions')

# 2. Nulls & duplicates
print('tx nulls:\n', tx[required_tx].isnull().sum())
tx = tx.drop_duplicates(subset=['transaction_id','sku'])  # adjust if transaction lines repeated

# 3. Numeric checks
tx['quantity'] = pd.to_numeric(tx['quantity'], errors='coerce')
tx['price'] = pd.to_numeric(tx['price'], errors='coerce')

# Remove or flag negative/zero prices or quantities
bad_qty = tx[tx['quantity'] <= 0]
bad_price = tx[tx['price'] <= 0]
print('bad_qty rows:', len(bad_qty))
print('bad_price rows:', len(bad_price))

# Strategy: remove rows with zero/negative quantity or price, unless flagged for manual review
tx = tx[(tx['quantity'] > 0) & (tx['price'] > 0)]

# 4. Create total line value
tx['line_total'] = tx['quantity'] * tx['price']

# 5. Category identification for chips (adjust condition to your schema)
chips_mask = (
    tx['category'].str.lower().fillna('') == 'chips'
) | (
    tx['subcategory'].str.lower().fillna('').str.contains('chip|crisps')
)
tx['is_chips'] = chips_mask

# 6. Outliers: extremely large quantity or price
qty_q99 = tx['quantity'].quantile(0.99)
price_q99 = tx['price'].quantile(0.99)
print('99th pct quantity, price:', qty_q99, price_q99)
# flag extreme rows
outliers = tx[(tx['quantity'] > qty_q99 * 5) | (tx['price'] > price_q99 * 5)]
print('Potential extreme outliers:', len(outliers))

# --- customers checks ---
print('cust nulls:\n', cust.isnull().sum())
# Normalize key attributes
if 'postal_code' in cust.columns:
    cust['postal_code'] = cust['postal_code'].astype(str).str.strip()

# Remove duplicates on customer_id
cust = cust.drop_duplicates(subset=['customer_id'])

# --- merge ---
merged = tx.merge(cust, how='left', on='customer_id', suffixes=('','_cust'))
print('merged null customer count:', merged['customer_id'].isnull().sum())

# Save cleaned files
tx.to_csv('transactions_clean.csv', index=False)
cust.to_csv('customers_clean.csv', index=False)
merged.to_csv('tx_cust_merged.csv', index=False)
print('Saved cleaned files.')
