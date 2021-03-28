###################################
#####     Generate Plots     ######
###################################
library(dplyr)
library(ggplot2)
library(rjson)

args <- commandArgs(trailingOnly = TRUE)
input <- fromJSON(args[1])
simulate_param <- fromJSON(file = input[["simulate"]])
statistic <- read.csv(input[["statistic"]])
out_dir <- input[["out_dir"]]
dir.create(out_dir, showWarnings = F, recursive = TRUE)

# get parameters #
all_N <- simulate_param[["all_N"]] # Different sample sizes of N
all_T <- simulate_param[["all_T"]] # Different sample sizes of T
beta_true <- simulate_param[["beta_true"]] # Regression coefficients
coefficients <- c("beta1", "beta2", "mu", "gamma", "delta")
# guess number of variables from column names
p = sum(startsWith(colnames(statistic), "mean_interactive."))

### plot rmse for different N and T ###
for (select.coef in 1:p) {
  if (anyNA(statistic[paste0("rmse_within.", select.coef)])) {
    next
  }
  # restore result from statistics
  stat_ls <-
    select(statistic, c(1:2, rmse = paste0("rmse_interactive.", select.coef)))
  stat_fe <-
    select(statistic, c(1:2, rmse = paste0("rmse_within.", select.coef)))

  # Heatmap for interactive-effect estimator (x=N, y=T, color=rmse)
  heatmap_ls <- ggplot(stat_ls, aes(x = N, y = T, fill = rmse)) +
    geom_tile() +
    scale_fill_distiller(palette = "YlOrBr", direction = 1) +
    labs(
      title = paste0("interactive-effect estimator rmse"),
      x = "N",
      y = "T"
    ) +
    theme_minimal() +
    theme(legend.title = element_blank())
  print(heatmap_ls)
  ggsave(
    paste0(
      out_dir,
      "/",
      coefficients[select.coef],
      "_rmse_heatmap_interactive.png"
    ),
    width = 5,
    height = 5,
    units = "in",
    dpi = 300
  )

  # Heatmap for within estimator (x=N, y=T, color=rmse)
  heatmap_fe <- ggplot(stat_fe, aes(x = N, y = T, fill = rmse)) +
    geom_tile() +
    scale_fill_distiller(palette = "YlOrBr", direction = 1) +
    labs(title = paste0("within estimator rmse"),
         x = "N",
         y = "T") +
    theme_minimal() +
    theme(legend.title = element_blank())
  print(heatmap_fe)
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_rmse_heatmap_within.png"),
    width = 5,
    height = 5,
    units = "in",
    dpi = 300
  )

  # point plot for interactive-effect estimator
  point_plot_N_ls <- ggplot(data = stat_ls, aes(x = N)) +
    geom_jitter(aes(y = rmse, color = as.factor(N)),
                height = 0,
                alpha = 0.8) +
    geom_smooth(
      aes(y = rmse),
      method = "loess",
      se = F,
      color = I("azure4"),
      formula = "y~x",
      size = 0.5
    ) +
    labs(
      title = paste0("interactive-effect estimator
"),
      x = "N",
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
  print(point_plot_N_ls)
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_rmse_point_interactive.png"),
    width = 5,
    height = 5,
    units = "in",
    dpi = 300
  )

  # point plot for within estimator
  point_plot_N_fe <- ggplot(data = stat_fe, aes(x = N)) +
    geom_jitter(aes(y = rmse, color = as.factor(N)),
                height = 0,
                alpha = 0.8) +
    geom_smooth(
      aes(y = rmse),
      method = "loess",
      se = F,
      color = I("azure4"),
      formula = "y~x",
      size = 0.5
    ) +
    labs(
      title = paste0("within estimator"),
      x = "N",
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
  print(point_plot_N_fe)
  ggsave(
    paste0(out_dir, "/", coefficients[select.coef], "_rmse_point_within.png"),
    width = 5,
    height = 5,
    units = "in",
    dpi = 300
  )
}
