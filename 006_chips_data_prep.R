# Task 1: Data preparation and customer analytics

# R Script

# chips_data_prep.R
library(readxl); library(dplyr); library(lubridate); library(stringr)

tx <- read_excel("QVI_transaction_data.xlsx", sheet = "in", col_types = "text")
cust <- read.csv("QVI_purchase_behaviour.csv", stringsAsFactors = FALSE)

# Convert Excel serial date numbers to actual dates
tx$date <- as.Date(as.numeric(tx$DATE), origin = "1899-12-30")

# rename columns to friendly names
tx <- tx %>%
  rename(customer_id = LYLTY_CARD_NBR, transaction_id = TXN_ID,
         sku = PROD_NBR, product_description = PROD_NAME,
         quantity = PROD_QTY, line_total = TOT_SALES, store_id = STORE_NBR) %>%
  mutate(quantity = as.numeric(quantity), line_total = as.numeric(line_total),
         price = line_total / quantity)

# feature extraction: pack size and brand
extract_pack <- function(s) {
  s <- tolower(as.character(s))
  m <- str_extract(s, "\\d+\\s?g|\\d+\\s?kg|\\d+\\s?ml")
  ifelse(is.na(m), NA, str_replace_all(m, " ", ""))
}
tx$pack_size <- sapply(tx$product_description, extract_pack)
tx$brand_guess <- sapply(tx$product_description, function(x) str_to_title(str_split(x, "[\\s\\-(),/]+")[[1]][1]))

# join customers
cust <- rename(cust, customer_id = LYLTY_CARD_NBR)
merged <- left_join(tx, cust, by = "customer_id")

# chips flag
merged$is_chips <- grepl("chip|crisp|potato|dorito|doritos", tolower(merged$product_description))

# filter chips and drop bad rows
chips <- merged %>% filter(is_chips, quantity > 0, price > 0, !is.na(transaction_id), !is.na(customer_id))

# save cleaned file
write.csv(chips, "tx_chips_clean.csv", row.names = FALSE)

# RFM
snapshot_date <- max(chips$date, na.rm = TRUE) + days(1)
rfm <- chips %>%
  group_by(customer_id) %>%
  summarise(recency_days = as.integer(snapshot_date - max(as.Date(date))),
            frequency = n_distinct(transaction_id),
            monetary = sum(as.numeric(line_total), na.rm = TRUE),
            units = sum(as.numeric(quantity), na.rm = TRUE),
            avg_units_per_tx = units / frequency) %>%
  ungroup()

# Quintile-based RFM scoring
rfm$r_score <- ntile(-rfm$recency_days, 5)  # - recency so more recent => higher score
rfm$f_score <- ntile(rfm$frequency, 5)
rfm$m_score <- ntile(rfm$monetary, 5)
rfm$rfm_score <- paste0(rfm$r_score, rfm$f_score, rfm$m_score)
write.csv(rfm, "chips_customers_rfm.csv", row.names = FALSE)

# Aggregations
pack_agg <- chips %>% group_by(pack_size) %>% summarise(units_sold = sum(as.numeric(quantity)), revenue = sum(as.numeric(line_total)), transactions = n_distinct(transaction_id))
write.csv(pack_agg, "chips_sales_by_pack_size.csv", row.names = FALSE)

brand_agg <- chips %>% group_by(brand_guess) %>% summarise(units_sold = sum(as.numeric(quantity)), revenue = sum(as.numeric(line_total))) %>% arrange(desc(units_sold))
write.csv(brand_agg, "chips_sales_by_brand_guess.csv", row.names = FALSE)

# simple time series plot (base R)
daily <- aggregate(as.numeric(chips$line_total), by = list(date = as.Date(chips$date)), sum)
png("fig_daily_chips_revenue.png", width=1000, height=400)
plot(daily$date, daily$x, type = "l", main = "Daily Chips Revenue", xlab = "", ylab = "Revenue")
dev.off()
