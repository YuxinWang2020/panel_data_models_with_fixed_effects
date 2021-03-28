###################################
#####     Generate Plots     ######
###################################
library(dplyr)
library(ggplot2)
library(rjson)

args <- commandArgs(trailingOnly = TRUE)
input <- fromJSON(args[1])
simulate_param <- fromJSON(file = input[["simulate"]])
sim_result <- read.csv(input[["sim_result"]])
out_dir <- input[["out_dir"]]
dir.create(out_dir, showWarnings = F, recursive = TRUE)

# get parameters #
all_N <- simulate_param[["all_N"]] # Different sample sizes of N
all_T <- simulate_param[["all_T"]] # Different sample sizes of T
T_ <- all_T[1]
beta_true <- simulate_param[["beta_true"]] # Regression coefficients
coefficients <- c("beta1", "beta2", "mu", "gamma", "delta")
# guess number of variables from column names
p = sum(startsWith(colnames(sim_result), "beta_interactive."))

### plot beta hat for different N ###
for (select.coef in 1:p) {
  if (anyNA(sim_result[paste0("beta_within.", select.coef)])) {
    next
  }
  # bind result from interactive-effect estimator and within estimator into one
  # data frame for plot
  select.df_beta_hat_ls <-
    select(sim_result, c(1:3, beta = paste0("beta_interactive.", select.coef)))
  select.df_beta_hat_fe <-
    select(sim_result, c(1:3, beta = paste0("beta_within.", select.coef)))
  select.df_beta_hat <-
    bind_rows(
      cbind(method = "interactive-effect estimator", select.df_beta_hat_ls),
      cbind(method = "within estimator", select.df_beta_hat_fe)
    )
  select.df_beta_hat$N <- as.factor(select.df_beta_hat$N)
  select.df_beta_hat$method <- as.factor(select.df_beta_hat$method)

  # jitter point plot
  point_plot <- ggplot(data = select.df_beta_hat, aes(x = N)) +
    geom_jitter(aes(y = beta, color = method, shape = method), height =
                  0) +
    geom_hline(yintercept = beta_true[[coefficients[select.coef]]],
               color = I("black"),
               alpha = 0.5) +
    labs(
      title = paste0("fix T=", T_),
      x = "N",
      y = paste(coefficients[select.coef], "hat")
    ) +
    scale_color_brewer(palette = "Set1") +
    theme_minimal()
  print(point_plot)
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_beta_hat_point.png"),
    width = 7,
    height = 5,
    units = "in",
    dpi = 300
  )

  # box plot
  box_plot <- ggplot(data = select.df_beta_hat, aes(x = N)) +
    geom_boxplot(
      aes(y = beta, fill = N),
      alpha = 0.8,
      color = I("black"),
      width = 0.6
    ) +
    geom_hline(yintercept = beta_true[[coefficients[select.coef]]],
               color = I("black"),
               alpha = 0.5) +
    labs(title = paste0("fix T=", T_),
         y = paste(coefficients[select.coef], "hat")) +
    scale_fill_brewer(palette = "Blues") +
    guides(
      fill = FALSE,
      alpha = FALSE,
      color = FALSE,
      shape = FALSE
    ) +
    theme_minimal() +
    facet_wrap( ~ method)
  print(box_plot)
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_beta_hat_box.png"),
    width = 5,
    height = 5,
    units = "in",
    dpi = 300
  )

  # violin plot
  violin_plot <- ggplot(data = select.df_beta_hat, aes(x = N)) +
    geom_violin(
      aes(y = beta, fill = N),
      color = "black",
      width = 0.9,
      alpha = 0.9
    ) +
    geom_boxplot(
      aes(y = beta),
      alpha = 0.6,
      fill = I("white"),
      width = 0.06
    ) +
    geom_hline(yintercept = beta_true[[coefficients[select.coef]]],
               color = I("black"),
               alpha = 0.5) +
    labs(title = paste0("fix T=", T_),
         y = paste(coefficients[select.coef], "hat")) +
    scale_fill_brewer(palette = "Blues") +
    guides(
      fill = FALSE,
      alpha = FALSE,
      color = FALSE,
      shape = FALSE
    ) +
    theme_minimal() +
    facet_wrap( ~ method)
  print(violin_plot)
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_beta_hat_violin.png"),
    width = 10,
    height = 5,
    units = "in",
    dpi = 300
  )
}
