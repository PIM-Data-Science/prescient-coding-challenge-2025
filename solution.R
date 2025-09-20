# Load Packages ---- 

#install.packages("pacman")

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


# parameters for optimisation - in this sample solution we use a momentum lookback strategy
n_days <- 2000
prev_weights <- rep(0.1, 10)
p_active_md <- 1.3 # this can be set to your own limit, as long as the portfolio is capped at 1.5 on any given day
weight_bounds <- c(0.0, 0.2)
weight_matrix <- tibble()

# Signal & Weight Generation ----

for(i in 1:nrow(df_signals)){
  print(paste0('---> doing ', df_signals$datestamp[i]))
  
  # iterate training set
  df_train_bonds <- 
    df_bonds %>% 
    filter(datestamp < df_signals$datestamp[i]) 
  
  df_train_albi <- 
    df_albi %>% 
    filter(datestamp < df_signals$datestamp[i]) 
  
  df_train_macro <- 
    df_macro %>% 
    filter(datestamp < df_signals$datestamp[i]) 
  
  df_test_bonds <- 
    df_bonds %>% 
    filter(datestamp >= df_signals$datestamp[i]) 
  
  df_test_albi <- 
    df_albi %>% 
    filter(datestamp >= df_signals$datestamp[i]) 
  
  df_test_macro <- 
    df_macro %>% 
    filter(datestamp >= df_signals$datestamp[i]) 
  
  p_albi_md <- df_train_albi$modified_duration %>% tail(1)
  
  # feature engineering - We want to update this
  df_train_macro <- 
    df_train_macro %>% 
    mutate(steepness = us_10y - us_2y)
  
  features <- c(
    "convexity", "modified_duration", "yield", "md_per_conv",
    "top40_return", "fx_vol", "us_2y", "us_10y", "us_20y",
    "comdty_fut", "steepness"
  )
  
  df_train_bonds <- df_train_bonds %>% mutate(signal = NA)
  
  df_train_bonds <- 
    df_train_bonds %>% 
    group_by(bond_code) %>% 
    mutate(md_per_conv = rollmeanr(return, n_days,na.pad = TRUE)*convexity/modified_duration) %>% 
    left_join(df_train_macro, by = "datestamp") %>%
    mutate(target = lead(return)) %>%  # next day return
    ungroup() %>%
    select(all_of(features), target, datestamp, bond_code) %>%
    na.omit()
  
  dtrain <- xgb.DMatrix(data = as.matrix(df_train_bonds[, features]),
                        label = df_train_bonds$target)
  
  # Create XG Boost model
  if (i == 1)
  {
  xgb_model <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      eta = 0.05,
      max_depth = 6,
      subsample = 0.8,
      colsample_bytree = 0.8,
      eval_metric = "rmse"
    ),
    data = dtrain,
    nrounds = 250,
    verbose = 0
  )
  }
  
  # Predict signal for current date
  dtest <- xgb.DMatrix(data = as.matrix(df_train_bonds %>% select(-c(target, datestamp, bond_code))))
  df_train_bonds <- df_train_bonds %>% mutate(signal = NA)
  df_train_bonds$signal <- predict(xgb_model, dtest)

  df_train_bonds_current <- 
    df_train_bonds %>% 
    filter(datestamp == max(datestamp)) %>% 
    mutate(active_md = modified_duration - p_albi_md)
  
  # --- Optimisation setup ---
  n <- nrow(df_train_bonds_current)
  signals <- df_train_bonds_current$signal
  active_md <- df_train_bonds_current$active_md
  
  # CVXR variable
  w <- Variable(n)
  
  # Parameters
  turnover_lambda <- 0.5             # penalty weight for excess turnover
  
  # Define turnover
  turnover <- sum(abs(w - prev_weights))
  
  # Objective: maximise signal, but penalise excess turnover only
  objective <- Maximize(t(signals) %*% w - turnover_lambda * turnover)
  
  # Constraints
  constraints <- list(
    sum(w) == 1,                        # weights sum to 1
    sum(w * active_md) <=  p_active_md,         # Active MD <= p_active_md
    sum(w * active_md) >= -p_active_md,         # Active MD >= -p_active_md
    w >= 0,                             # min weight 0
    w <= 0.2                            # max weight 0.2
  )
  
  # Define and solve problem
  prob <- Problem(objective, constraints)
  res <- solve(prob, solver = "ECOS_BB")
  
  # Extract optimal weights
  optimal_weights <- as.numeric(res$getValue(w))
  
  weight_matrix <- rbind(weight_matrix, data.frame(
    bond_code = df_train_bonds_current$bond_code,
    weight = optimal_weights,
    datestamp = df_signals$datestamp[i]
  ))
  
  prev_weights <- optimal_weights
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
