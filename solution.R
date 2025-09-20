# Load Packages ---- 

pacman::p_load(dplyr, readr, lubridate, slider, ggplot2, tidyr, zoo, CVXR, tictoc)

t0 = Sys.time()
print(paste0('---> R Script Start ', t0))

# Load and prepare data ----
print('---> initial data set up')

# instrument data
df_bonds <- read_csv("data/data_bonds.csv") %>%
  mutate(datestamp = as_date(datestamp))

# albi data
df_albi <- read_csv("data/data_albi.csv") %>%
  mutate(datestamp = as_date(datestamp))

# macro data
df_macro <- read_csv("data/data_macro.csv") %>%
  mutate(datestamp = as_date(datestamp))


print('---> the parameters')

# training and test dates
start_train <- "2005-01-03"
start_test <- "2023-01-03"
end_test <- max(df_bonds$datestamp)

# dates for buy matrix
# we will perform walk forward validation for testing the buys - https://www.linkedin.com/pulse/walk-forward-validation-yeshwanth-n
df_signals <- df_bonds %>%
  filter(datestamp >= start_test & datestamp <= end_test) %>%
  distinct(datestamp) %>%
  arrange(datestamp)

###-----------------------------------------------------------------------------
# This section contains a sample solution
# You are not restricted to the choice of signal, or the portfolio optimisation used to generate weights
# You may modify anything within this section as long as it produces a weight matrix in the required form, and the solution does not violate any of the rules

# Params
n_mom           <- 20
n_vol           <- 20
n_beta          <- 60   
n_res_win       <- 10  
lambda_ridge    <- 10
turnover_lambda <- 0.5
p_active_md     <- 1.5   
kappa_md        <- 1.5  
w_min           <- 0.0
w_max           <- 0.2

zscore <- function(x) { x <- as.numeric(x); (x - mean(x, na.rm = TRUE)) / (sd(x, na.rm = TRUE) + 1e-6) }
nzmed  <- function(x) { ifelse(is.na(x) | x == 0, median(x, na.rm = TRUE), x) }

# One-time features 
# Bonds: lagged returns, momentum, volatility, cheapness
df_bonds_feat <- df_bonds %>%
  group_by(bond_code) %>%
  arrange(datestamp, .by_group = TRUE) %>%
  mutate(
    ret_lag = dplyr::lag(return),
    mom20   = slider::slide_dbl(ret_lag, mean, .before = n_mom-1, .complete = TRUE),
    vol20   = slider::slide_dbl(ret_lag,  sd,   .before = n_vol-1, .complete = TRUE)
  ) %>% ungroup() %>%
  group_by(datestamp) %>%
  mutate(cheap = yield - median(yield, na.rm = TRUE)) %>%
  ungroup()

# Macro (lagged so it's usable at t-1)
df_macro_feat <- df_macro %>%
  arrange(datestamp) %>%
  mutate(
    slope     = us_10y - us_2y,
    top40_l1  = dplyr::lag(top40_return),
    fxvol_l1  = dplyr::lag(fx_vol),
    slope_l1  = dplyr::lag(slope),
    com_l1    = dplyr::lag(comdty_fut)
  ) %>%
  select(datestamp, top40_l1, fxvol_l1, slope_l1, com_l1)

# ALBI (for residual momentum)
df_albi_slim <- df_albi %>% select(datestamp, albi_ret = return)

# Merge panel and build residual momentum with rolling beta 
df_feat <- df_bonds_feat %>%
  left_join(df_macro_feat, by = "datestamp") %>%
  left_join(df_albi_slim,  by = "datestamp") %>%
  arrange(datestamp, bond_code) %>%
  group_by(bond_code) %>%
  mutate(
    albi_lag = dplyr::lag(albi_ret),
    # risk-adjusted momentum 
    s_mom    = mom20 / (vol20 + 1e-6),
    # rolling beta vs ALBI using aligned lagged series:
    cov_ba   = slider::slide2_dbl(ret_lag, albi_lag,
                                  ~ cov(.x, .y, use = "pairwise.complete.obs"),
                                  .before = n_beta - 1, .complete = TRUE),
    var_a    = slider::slide_dbl(albi_lag, ~ var(.x, na.rm = TRUE),
                                 .before = n_beta - 1, .complete = TRUE),
    beta     = dplyr::if_else(var_a > 0, cov_ba / (var_a + 1e-8), 0),
    resid    = ret_lag - beta * albi_lag,
    s_res    = slider::slide_dbl(resid, mean, .before = n_res_win - 1, .complete = TRUE)
  ) %>%
  ungroup()

bond_universe <- df_bonds %>% distinct(bond_code) %>% arrange(bond_code) %>% pull(bond_code)
prev_weights  <- rep(1/length(bond_universe), length(bond_universe))

weight_matrix <- tibble()

#  Walk-forward
for (i in 1:nrow(df_signals)) {
  d <- df_signals$datestamp[i]
  cat(sprintf("---> optimizing for (%s)\n", as.character(d)))
  
  # Training up to t-1
  df_train <- df_feat %>% filter(datestamp < d)
  if (nrow(df_train) == 0) next
  
  # ALBI MD (t-1)
  p_albi_md <- df_albi %>% filter(datestamp < d) %>% pull(modified_duration) %>% tail(1)
  if (length(p_albi_md) == 0 || is.na(p_albi_md)) next
  
  # Current cross-section at last training day
  last_train_day <- max(df_train$datestamp, na.rm = TRUE)
  xsec_now <- df_train %>%
    filter(datestamp == last_train_day) %>%
    select(bond_code, datestamp, return, yield, modified_duration, convexity,
           mom20, vol20, cheap, top40_l1, fxvol_l1, slope_l1, com_l1,
           s_mom, s_res) %>%
    arrange(bond_code)
  
  bonds_today <- xsec_now$bond_code
  n <- length(bonds_today)
  if (n == 0) next
  
  # (2) Ridge regression (cross-sectional forecaster) 
  train_mat <- df_train %>%
    select(bond_code, datestamp, return, yield, modified_duration, convexity,
           mom20, vol20, cheap, top40_l1, fxvol_l1, slope_l1, com_l1) %>%
    tidyr::drop_na()
  if (nrow(train_mat) < 100) next
  
  y <- train_mat$return
  X <- as.matrix(train_mat %>% select(yield, modified_duration, convexity,
                                      mom20, vol20, cheap,
                                      top40_l1, fxvol_l1, slope_l1, com_l1))
  X_scaled <- scale(X)
  p <- ncol(X_scaled)
  beta_r <- solve(t(X_scaled) %*% X_scaled + lambda_ridge * diag(p), t(X_scaled) %*% y)
  
  X0 <- as.matrix(xsec_now %>% select(yield, modified_duration, convexity,
                                      mom20, vol20, cheap,
                                      top40_l1, fxvol_l1, slope_l1, com_l1))
  X0_scaled <- scale(X0, center = attr(X_scaled, "scaled:center"), scale = attr(X_scaled, "scaled:scale"))
  ridge_pred <- as.numeric(X0_scaled %*% beta_r)
  
  #(3) Risk-adjusted momentum + value (cheapness) 
  vol20_now <- nzmed(xsec_now$vol20)
  mom_sig   <- xsec_now$mom20 / (vol20_now + 1e-6)
  cheap_sig <- xsec_now$cheap
  momval_sig <- 0.6 * zscore(mom_sig) + 0.4 * zscore(cheap_sig)
  
  # Ensemble alpha 
  alpha <- 0.4 * zscore(xsec_now$s_res) +
    0.4 * zscore(ridge_pred)     +
    0.2 * zscore(momval_sig)
  
  signals <- as.numeric(alpha)
  
  # Optimisation (CVXR) 
  active_md <- xsec_now$modified_duration - p_albi_md
  
  w <- CVXR::Variable(n)
  
  # L1 turnover: use p_norm to avoid CVXR::abs namespacing issues
  turnover <- CVXR::p_norm(w - prev_weights[seq_len(n)], 1)
  
  # Soft centering penalty WITHOUT abs(): use sum_squares(sum_entries(...))
  md_center_pen <- CVXR::sum_squares(CVXR::sum_entries(w * active_md))
  
  objective <- CVXR::Maximize(t(signals) %*% w - turnover_lambda * turnover - kappa_md * md_center_pen)
  
  constraints <- list(
    sum(w) == 1,
    sum(w * active_md) <=  p_active_md,
    sum(w * active_md) >= -p_active_md,
    w >= w_min,
    w <= w_max
  )
  
  prob <- CVXR::Problem(objective, constraints)
  res  <- CVXR::solve(prob, solver = "ECOS_BB")
  
  if (res$status %in% c("infeasible", "unbounded")) {
    w_opt <- rep(1/n, n)
  } else {
    w_opt <- as.numeric(res$getValue(w))
    w_opt[is.na(w_opt)] <- 0
    w_opt <- pmin(pmax(w_opt, w_min), w_max)
    if (sum(w_opt) == 0) w_opt <- rep(1/n, n)
    w_opt <- w_opt / sum(w_opt)
  }
  
  weight_matrix <- dplyr::bind_rows(
    weight_matrix,
    tibble(
      bond_code = bonds_today,
      weight    = w_opt,
      datestamp = d
    )
  )
  
  prev_weights <- w_opt
}



### Weight generation ends here
### DO NOT MAKE ANY CHANGES BELOW THIS LINE
### ----------------------------------------------------------------------------

# Plotting functions
plot_payoff <- function(weight_matrix, df_bonds, df_albi) {
  port_data <- weight_matrix %>%
    left_join(df_bonds, by = c("bond_code", "datestamp")) %>%
    mutate(port_return = return * weight, port_md = modified_duration * weight) %>%
    group_by(datestamp) %>%
    summarise(port_return = sum(port_return, na.rm = TRUE),
              port_md = sum(port_md, na.rm = TRUE), .groups = "drop") %>%
    left_join(df_albi[, c("datestamp", "return")], by = "datestamp") %>%
    rename(albi_return = return)
  
  df_turnover <- weight_matrix %>%
    group_by(bond_code) %>%
    arrange(datestamp) %>%
    mutate(turnover = abs(weight - lag(weight))/2) %>%
    group_by(datestamp) %>%
    summarise(turnover = sum(turnover, na.rm = TRUE), .groups = "drop")
  
  port_data <- port_data %>%
    left_join(df_turnover, by = "datestamp") %>%
    arrange(datestamp) %>%
    mutate(
      penalty = 0.005 * turnover * lag(port_md, default = NA),
      net_return = port_return - coalesce(penalty, 0),
      portfolio_tri = cumprod(1 + net_return / 100),
      albi_tri = cumprod(1 + albi_return / 100)
    )
  
  tri_data <- port_data %>%
    select(datestamp, portfolio_tri, albi_tri) %>%
    pivot_longer(-datestamp, names_to = "type", values_to = "TRI")
  
  print(ggplot(tri_data, aes(x = datestamp, y = TRI, color = type)) +
          geom_line(size = 1) +
          labs(title = "Portfolio vs ALBI TRI", x = "Date", y = "TRI") +
          theme_minimal())
  
  print(ggplot(port_data, aes(x = datestamp, y = turnover)) +
          geom_line(color = "darkred", size = 1) +
          labs(title = "Daily Turnover", x = "Date", y = "Turnover") +
          theme_minimal())
  
  print(ggplot(weight_matrix, aes(x = datestamp, y = weight, fill = bond_code))+ 
          geom_area() +
          labs(title = "Weights Through Time", x = "Date", y = "Weight") +
          theme_minimal())
  
  cat(sprintf("---> payoff for these buys between %s and %s is %.2f%%\n",
              min(port_data$datestamp), max(port_data$datestamp),
              100 * (tail(port_data$portfolio_tri, 1) - 1)))
  cat(sprintf("---> payoff for ALBI over same period is %.2f%%\n",
              100 * (tail(port_data$albi_tri, 1) - 1)))
}

plot_md <- function(weight_matrix, df_bonds, df_albi) {
  port_data <- weight_matrix %>%
    left_join(df_bonds, by = c("bond_code", "datestamp")) %>%
    mutate(port_md = modified_duration * weight) %>%
    group_by(datestamp) %>%
    summarise(port_md = sum(port_md, na.rm = TRUE), .groups = "drop") %>%
    left_join(df_albi[, c("datestamp", "modified_duration")], by = "datestamp") %>%
    mutate(active_md = port_md - modified_duration)
  
  print(ggplot(port_data, aes(x = datestamp, y = active_md)) +
          geom_line(color = "steelblue", size = 1) +
          labs(title = "Active Modified Duration", x = "Date", y = "Active MD") +
          theme_minimal())
  
  breaches <- port_data %>% filter(abs(active_md) > 1.5)
  if (nrow(breaches) > 0) {
    stop(paste("This portfolio violates the duration constraint on:\n",
               paste(breaches$datestamp, collapse = ", ")))
  } else {
    message("---> The portfolio does not breach the modified duration constraint")
  }
}

# Run visualizations
plot_payoff(weight_matrix, df_bonds, df_albi)
plot_md(weight_matrix, df_bonds, df_albi)

t1 = Sys.time()
elapsed <- as.numeric(difftime(t1, t0, units = "secs"))

cat(sprintf("---> R Script End %s\n", t1))
cat(sprintf("---> Total time taken %02d:%02d\n", floor(elapsed / 60), round(elapsed %% 60)))