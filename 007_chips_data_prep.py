# Task 1: Data preparation and customer analytics

# Python script

# chips_data_prep.py
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# --- paths (adjust if needed) ---
TXN_XLSX = Path('QVI_transaction_data.xlsx')
CUST_CSV = Path('QVI_purchase_behaviour.csv')
OUT_DIR = Path('.')  # saves outputs to current folder (change if desired)

# --- helpers ---
def excel_serial_to_date(x):
    try:
        xi = float(x)
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(int(xi), unit='D')
    except:
        return pd.to_datetime(x, errors='coerce')

def extract_pack_size(desc):
    if pd.isna(desc):
        return np.nan
    s = str(desc).lower()
    m = re.search(r'(\d+\s?g|\d+\s?kg|\d+\s?ml|\d+\s?l|\d+\s?ct|\d+\s?g\w?)', s)
    return m.group(0).replace(' ', '') if m else np.nan

def extract_brand(desc):
    if pd.isna(desc): return np.nan
    tokens = re.split(r'[\s\-\(\)\,\/]+', str(desc))
    return tokens[0].title() if tokens else np.nan

# --- load data ---
xls = pd.ExcelFile(TXN_XLSX)
# first sheet named 'in' in your file — adjust name if different
tx = pd.read_excel(xls, sheet_name='in', dtype=object)

cust = pd.read_csv(CUST_CSV, low_memory=False)

# --- normalize and convert ---
tx.columns = [c.strip().lower().replace(' ', '_') for c in tx.columns]
tx['date'] = tx['date'].apply(excel_serial_to_date) if 'date' in tx.columns else tx.iloc[:,0].apply(excel_serial_to_date)
tx.rename(columns={
    'lylty_card_nbr':'customer_id',
    'prod_qty':'quantity',
    'tot_sales':'line_total',
    'txn_id':'transaction_id',
    'prod_name':'product_description',
    'prod_nbr':'sku',
    'store_nbr':'store_id'
}, inplace=True)

cust.columns = [c.strip().lower().replace(' ', '_') for c in cust.columns]
cust.rename(columns={'lylty_card_nbr':'customer_id'}, inplace=True)

# numeric conversions
tx['quantity'] = pd.to_numeric(tx['quantity'], errors='coerce')
tx['line_total'] = pd.to_numeric(tx['line_total'], errors='coerce')
tx['price'] = tx['line_total'] / tx['quantity']

# derive brand and pack size
tx['pack_size'] = tx['product_description'].apply(extract_pack_size)
tx['brand_guess'] = tx['product_description'].apply(extract_brand)

# merge with customers
merged = tx.merge(cust, how='left', on='customer_id')

# flag chips
merged['is_chips'] = merged['product_description'].str.lower().str.contains('chip|crisp|potato|dorito|doritos', na=False)

# keep only chips for this analysis
chips = merged[merged['is_chips']].copy()

# remove problematic rows
chips = chips[(chips['quantity']>0) & (chips['price']>0) & chips['transaction_id'].notnull() & chips['customer_id'].notnull()]

# Save cleaned chips dataset
chips.to_csv(OUT_DIR / 'tx_chips_clean.csv', index=False)

# --- Customer RFM (chips customers) ---
snapshot_date = chips['date'].max() + pd.Timedelta(days=1)
rfm = chips.groupby('customer_id').agg(
    recency_days = ('date', lambda x: (snapshot_date - x.max()).days),
    frequency = ('transaction_id', 'nunique'),
    monetary = ('line_total', 'sum'),
    units = ('quantity', 'sum'),
    avg_units_per_tx = ('quantity', lambda s: s.sum()/s.nunique())
).reset_index()

# RFM Scoring (quintiles)
rfm['r_score'] = pd.qcut(rfm['recency_days'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['rfm_score'] = rfm['r_score'].astype(str)+rfm['f_score'].astype(str)+rfm['m_score'].astype(str)

rfm.to_csv(OUT_DIR / 'chips_customers_rfm.csv', index=False)

# --- Aggregations for reporting ---
pack_agg = chips.groupby('pack_size').agg(units_sold=('quantity','sum'), revenue=('line_total','sum'), transactions=('transaction_id','nunique')).reset_index().sort_values('units_sold', ascending=False)
brand_agg = chips.groupby('brand_guess').agg(units_sold=('quantity','sum'), revenue=('line_total','sum')).reset_index().sort_values('units_sold', ascending=False)

pack_agg.to_csv(OUT_DIR / 'chips_sales_by_pack_size.csv', index=False)
brand_agg.to_csv(OUT_DIR / 'chips_sales_by_brand_guess.csv', index=False)

# --- Simple plots (matplotlib, one plot per file) ---
daily = chips.groupby(chips['date'].dt.date)['line_total'].sum().reset_index()
daily['date'] = pd.to_datetime(daily['date'])
plt.figure(figsize=(10,4))
plt.plot(daily['date'], daily['line_total'])
plt.title('Daily Chips Revenue')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_daily_chips_revenue.png')
plt.close()

# done
print("Saved: tx_chips_clean.csv, chips_customers_rfm.csv, chips_sales_by_pack_size.csv, chips_sales_by_brand_guess.csv, fig_daily_chips_revenue.png")
