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
statistic <- read.csv(input[["statistic"]])
out_dir <- input[["out_dir"]]
dir.create(out_dir, showWarnings = F, recursive = TRUE)

# get parameters #
r_N <- statistic$N[1] # sample sizes of N
r_T <- statistic$T[1] # sample sizes of T
beta_true <- simulate_param[["beta_true"]] # Regression coefficients
model <- simulate_param[["model_name"]]
coefficients <- c("beta1", "beta2", "mu", "gamma", "delta")
# guess number of variables from column names
p = sum(startsWith(colnames(sim_result), "beta_interactive."))
### plot beta hat for different N ###
for (select.coef in 1:p) {
  # point plot for rmse #
  select.col <- paste0("rmse_interactive.", select.coef)
  point_plot_r <- ggplot(data = filter(statistic, r != 1), aes(x = r)) +
    geom_jitter(
      aes(y = get(select.col), color = as.factor(r)),
      height = 0,
      width = 0,
      alpha = 0.8
    ) +
    geom_smooth(
      aes(y = get(select.col)),
      method = "loess",
      se = F,
      color = I("azure4"),
      formula = "y~x",
      size = 0.5
    ) +
    labs(
      title = paste0("fix N=", r_N, " T=", r_T, " ", model),
      x = "r",
      y = paste(coefficients[select.coef], "rmse")
    ) +
    scale_color_brewer(palette = "Paired") +
    guides(
      fill = FALSE,
      alpha = FALSE,
      color = FALSE,
      shape = FALSE
    ) +
    theme_minimal()
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_rmse_point_interactive.png"),
    point_plot_r,
    width = 5,
    height = 5,
    units = "in",
    dpi = 300
  )

  # box plot for beta_hat #
  select.col <- paste0("beta_interactive.", select.coef)
  box_plot <- ggplot(data = sim_result, aes(x = as.factor(r))) +
    geom_boxplot(
      aes(y = get(select.col), fill = as.factor(r)),
      alpha = 0.8,
      color = I("black"),
      width = 0.6
    ) +
    geom_hline(yintercept = beta_true[[select.coef]],
               color = I("black"),
               alpha = 0.5) +
    labs(
      title = paste0("fix N=", r_N, " T=", r_T, " ", model),
      y = paste(coefficients[select.coef], " hat"),
      x = "r"
    ) +
    scale_fill_brewer(palette = "Spectral") +
    guides(
      fill = FALSE,
      alpha = FALSE,
      color = FALSE,
      shape = FALSE
    ) +
    theme_minimal()
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_beta_hat_box.png"),
    box_plot,
    width = 5,
    height = 5,
    units = "in",
    dpi = 300
  )

}
