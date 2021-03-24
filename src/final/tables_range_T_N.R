###################################
#####     Generate Tables     #####
###################################
library(dplyr)
library(rjson)

args <- commandArgs(trailingOnly = TRUE)
input <- fromJSON(args[1])
simulate_param <- fromJSON(file = input[["simulate"]])
statistic <- read.csv(input[["statistic"]])
produces <- input[["produces"]]


# get parameters #
all_N <- simulate_param[["all_N"]] # Different sample sizes of N
all_T <- simulate_param[["all_T"]] # Different sample sizes of T
beta_true <- simulate_param[["beta_true"]] # Regression coefficients
coefficients <- c("beta1", "beta2", "mu", "gamma", "delta")
# guess number of variables from column names
p = sum(startsWith(colnames(statistic), "mean_interactive."))
select_statistics <-
  list(colName = c("mean", "rmse"),
       presentName = c("Mean", "SD"))

### table for different N and T ###

table_ls <-
  statistic[c("N", "T", paste0(
    rep(select_statistics$colName, p),
    "_interactive.",
    rep(1:p, each = length(select_statistics$colName))
  ))]
colnames(table_ls) <-
  c("N", "T", paste(
    rep(select_statistics$presentName, p),
    rep(coefficients[1:p], each = length(select_statistics$presentName))
  ))

table_fe <-
  statistic[c("N", "T", paste0(
    rep(select_statistics$colName, p),
    "_within.",
    rep(1:p, each = length(select_statistics$colName))
  ))]
colnames(table_fe) <-
  c("N", "T", paste(
    rep(select_statistics$presentName, p),
    rep(coefficients[1:p], each = length(select_statistics$presentName))
  ))

table_combined <-
  bind_rows(
    cbind(method = "Interactive-Effects Estimator", table_ls),
    cbind(method = "Within Estimator", table_fe)
  )
write.csv(table_combined, file = produces, row.names = FALSE)
