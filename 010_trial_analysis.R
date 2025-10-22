# Task 2: Experimentation and uplift testing

# trial_analysis.R
library(readr); library(dplyr); library(lubridate); library(ggplot2); library(broom)
library(scales)
library(stats)
library(tidyr)

OUT <- "outputs"; dir.create(OUT, showWarnings = FALSE)

tx <- read_csv("tx_chips_clean.csv")  # cleaned chips transactions with date, store_id, line_total, quantity
tx$date <- as.Date(tx$date)

# Parameters
trial_stores <- c(101, 102)  # change
trial_start <- as.Date("2019-03-01")
trial_end <- as.Date("2019-03-31")
pre_weeks <- 8
pre_start <- trial_start - weeks(pre_weeks)

# daily aggregates
daily <- tx %>% group_by(store_id, date) %>% summarise(daily_revenue = sum(line_total), daily_units = sum(quantity), daily_txns = n_distinct(transaction_id)) %>% ungroup()

# pre-period store features
pre <- daily %>% filter(date >= pre_start & date < trial_start)
store_features <- pre %>% group_by(store_id) %>% summarise(pre_mean_rev = mean(daily_revenue, na.rm=TRUE), pre_sd_rev = sd(daily_revenue, na.rm=TRUE), pre_mean_units = mean(daily_units, na.rm=TRUE))

# weekday pattern
pre_wk <- pre %>% mutate(weekday = wday(date,label=TRUE)) %>% group_by(store_id, weekday) %>% summarise(avg_rev = mean(daily_revenue, na.rm=TRUE)) %>% ungroup()
wk_wide <- pre_wk %>% pivot_wider(names_from = weekday, values_from = avg_rev, values_fill = 0)
store_features <- left_join(store_features, wk_wide, by = "store_id")

# build matrix and scale for matching
feat <- store_features %>% select(-store_id) %>% replace(is.na(.),0)
feat_mat <- scale(as.matrix(feat))
rownames(feat_mat) <- store_features$store_id
# use dist & nearest neighbors
distmat <- as.matrix(dist(feat_mat))
# For each trial store find nearest neighbors
match_table <- data.frame(store_id = store_features$store_id, stringsAsFactors = FALSE)
match_table$controls <- NA

for (s in store_features$store_id) {
  dists <- distmat[as.character(s),]
  ord <- order(dists)
  # first one will be itself, pick next N
  n_ctrl <- 2
  ctrls <- as.numeric(names(dists)[ord[ord!=which(names(dists)==as.character(s))][1:n_ctrl]])
  match_table$controls[match_table$store_id==s] <- paste(ctrls, collapse=",")
}

write_csv(match_table, file.path(OUT, "match_table.csv"))

# For each trial store: DiD & bootstrap
results <- list()

for (trial in trial_stores) {
  # get controls
  ctrl_str <- match_table %>% filter(store_id==trial) %>% pull(controls)
  ctrls <- as.numeric(strsplit(ctrl_str, ",")[[1]])
  df <- daily %>% filter(store_id %in% c(trial, ctrls) & date >= pre_start & date <= trial_end)
  df <- df %>% mutate(is_trial = if_else(store_id==trial,1,0), post = if_else(date >= trial_start,1,0))
  # DiD: lm(daily_revenue ~ is_trial*post + factor(store_id))
  m <- lm(daily_revenue ~ is_trial*post + factor(store_id), data=df)
  tidy_m <- tidy(m)
  did_term <- tidy_m %>% filter(term=="is_trial:post")
  coef <- did_term$estimate; pval <- did_term$p.value; se <- did_term$std.error
  # Aggregates
  mean_pre_trial <- df %>% filter(is_trial==1, post==0) %>% summarise(mean=mean(daily_revenue,na.rm=TRUE)) %>% pull(mean)
  mean_post_trial <- df %>% filter(is_trial==1, post==1) %>% summarise(mean=mean(daily_revenue,na.rm=TRUE)) %>% pull(mean)
  mean_pre_ctrl <- df %>% filter(is_trial==0, post==0) %>% summarise(mean=mean(daily_revenue,na.rm=TRUE)) %>% pull(mean)
  mean_post_ctrl <- df %>% filter(is_trial==0, post==1) %>% summarise(mean=mean(daily_revenue,na.rm=TRUE)) %>% pull(mean)
  did_point <- (mean_post_trial - mean_pre_trial) - (mean_post_ctrl - mean_pre_ctrl)
  # bootstrap CI on daily differences
  trial_series <- df %>% filter(is_trial==1) %>% select(date, daily_revenue)
  ctrl_series <- df %>% filter(is_trial==0) %>% group_by(date) %>% summarise(daily_revenue = mean(daily_revenue, na.rm=TRUE))
  merged <- inner_join(trial_series, ctrl_series, by="date", suffix = c("_trial","_ctrl")) %>% mutate(diff = daily_revenue_trial - daily_revenue_ctrl)
  # bootstrap
  nboot <- 2000
  set.seed(42)
  boot_means <- replicate(nboot, mean(sample(merged$diff, replace=TRUE), na.rm=TRUE))
  ci <- quantile(boot_means, probs=c(0.025, 0.975))
  # visualizations
  p1 <- df %>% group_by(date, is_trial) %>% summarise(mean_rev = mean(daily_revenue)) %>% ggplot(aes(date, mean_rev, color=factor(is_trial))) + geom_line() + geom_vline(xintercept=as.numeric(trial_start)) + labs(title=paste("Store",trial,"daily rev"), color="is_trial")
  ggsave(filename=file.path(OUT, paste0(trial,"_timeseries.png")), p1, width=10, height=4)
  p2 <- df %>% group_by(is_trial, post) %>% summarise(mean_rev = mean(daily_revenue, na.rm=TRUE)) %>% mutate(label = if_else(is_trial==1, "Trial","Control"), period = if_else(post==1,"Post","Pre")) %>% ggplot(aes(period, mean_rev, fill=label)) + geom_col(position="dodge") + labs(title=paste("Store",trial,"Pre/Post mean rev"))
  ggsave(filename=file.path(OUT, paste0(trial,"_prepost_bar.png")), p2, width=6, height=4)
  results[[as.character(trial)]] <- tibble(trial_store=trial, controls=list(ctrls), did_coef=coef, did_se=se, did_pval=pval, did_point=did_point, boot_mean=mean(boot_means), boot_ci_low=ci[1], boot_ci_high=ci[2], mean_pre_trial=mean_pre_trial, mean_post_trial=mean_post_trial, mean_pre_ctrl=mean_pre_ctrl, mean_post_ctrl=mean_post_ctrl)
}

res_df <- bind_rows(results)
# adjust p-values BH
res_df$p_adj <- p.adjust(res_df$did_pval, method="BH")
write_csv(res_df, file.path(OUT, "store_results.csv"))
