# Exploratory analysis & charts (code snippets)

import matplotlib.pyplot as plt

merged = pd.read_csv('tx_cust_merged.csv', parse_dates=['transaction_date'])

# Filter chips rows
chips = merged[merged['is_chips'] == True].copy()

# 1. Time series: daily sales for chips
daily = chips.groupby(chips['transaction_date'].dt.date)['line_total'].sum().reset_index()
daily['transaction_date'] = pd.to_datetime(daily['transaction_date'])
plt.figure(figsize=(10,4))
plt.plot(daily['transaction_date'], daily['line_total'])
plt.title('Daily Chips Sales')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fig_daily_chips_sales.png')

# 2. Sales by store (top 10)
store_sales = chips.groupby('store_id')['line_total'].sum().sort_values(ascending=False).head(10)
store_sales.plot(kind='bar', figsize=(10,5))
plt.title('Top 10 Stores by Chips Sales')
plt.tight_layout()
plt.savefig('fig_top10_stores.png')

# 3. Packet size distribution (if column exists: 'packet_size')
if 'packet_size' in chips.columns:
    pkt = chips.groupby('packet_size').agg({'line_total':'sum','quantity':'sum'}).sort_values('quantity', ascending=False)
    pkt.to_csv('chips_sales_by_packet_size.csv')
    pkt['quantity'].plot(kind='bar', figsize=(8,4))
    plt.title('Units Sold by Packet Size')
    plt.tight_layout()
    plt.savefig('fig_packet_size_units.png')

# 4. SKU Pareto (top SKUs)
sku = chips.groupby('sku').agg({'quantity':'sum','line_total':'sum'}).sort_values('quantity', ascending=False).head(20)
sku['quantity'].plot(kind='bar', figsize=(10,5))
plt.title('Top 20 SKUs by Units Sold')
plt.tight_layout()
plt.savefig('fig_top_skus.png')

# 5. Basket analysis: distribution of chips-containing basket value
baskets = chips.groupby('transaction_id').agg({'line_total':'sum','quantity':'sum'}).reset_index()
plt.figure(figsize=(8,4))
plt.hist(baskets['line_total'], bins=50)
plt.title('Distribution of Chips Basket Value')
plt.xlabel('Chips Basket Value')
plt.tight_layout()
plt.savefig('fig_chips_basket_dist.png')
