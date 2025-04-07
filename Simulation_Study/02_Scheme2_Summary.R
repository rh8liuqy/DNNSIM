rm(list = ls())
library(kableExtra)
library(tidyverse)
library(latticeExtra)

source("./02_Scheme2_Main.R")

ST_GX_D_output <- readRDS("./02_Scheme2/simu_output_MSE_model_ST-GX-D_n_1000.RDS")
ST_GX_B_output <- readRDS("./02_Scheme2/simu_output_MSE_model_ST-GX-B_n_1000.RDS")

# point estimation --------------------------------------------------------

simu_n <- length(ST_GX_D_output)

ST_GX_D_beta1 <- numeric(simu_n)
ST_GX_D_beta2 <- numeric(simu_n)
ST_GX_D_beta3 <- numeric(simu_n)
ST_GX_D_w <- numeric(simu_n)
ST_GX_D_sigma <- numeric(simu_n)
ST_GX_D_delta <- numeric(simu_n)

ST_GX_B_beta1 <- numeric(simu_n)
ST_GX_B_beta2 <- numeric(simu_n)
ST_GX_B_beta3 <- numeric(simu_n)
ST_GX_B_w <- numeric(simu_n)
ST_GX_B_sigma <- numeric(simu_n)
ST_GX_B_delta <- numeric(simu_n)

for (i in 1:simu_n) {
    ST_GX_D_beta1[i] <- ST_GX_D_output[[i]]$DNN_train_output$beta_output[1]
    ST_GX_D_beta2[i] <- ST_GX_D_output[[i]]$DNN_train_output$beta_output[2]
    ST_GX_D_beta3[i] <- ST_GX_D_output[[i]]$DNN_train_output$beta_output[3]
    ST_GX_D_w[i] <- ST_GX_D_output[[i]]$DNN_train_output$w_output
    ST_GX_D_sigma[i] <- ST_GX_D_output[[i]]$DNN_train_output$sigma_output
    ST_GX_D_delta[i] <- ST_GX_D_output[[i]]$DNN_train_output$delta_output
    
    ST_GX_B_beta1[i] <- ST_GX_B_output[[i]]$DNN_train_output$beta_output[1]
    ST_GX_B_beta2[i] <- ST_GX_B_output[[i]]$DNN_train_output$beta_output[2]
    ST_GX_B_beta3[i] <- ST_GX_B_output[[i]]$DNN_train_output$beta_output[3]
    ST_GX_B_w[i] <- ST_GX_B_output[[i]]$DNN_train_output$w_output
    ST_GX_B_sigma[i] <- ST_GX_B_output[[i]]$DNN_train_output$sigma_output
    ST_GX_B_delta[i] <- ST_GX_B_output[[i]]$DNN_train_output$delta_output
}

df_point_estimation <- data.frame(model = rep(c("ST-GX-D","ST-GX-B"),each = 6),
                                  variable = rep(c("$\\beta_1$","$\\beta_2$","$\\beta_3$","$w$","$\\sigma$","$\\delta$"),2),
                                  point_estimation = c(mean(ST_GX_D_beta1),
                                                       mean(ST_GX_D_beta2),
                                                       mean(ST_GX_D_beta3),
                                                       mean(ST_GX_D_w),
                                                       mean(ST_GX_D_sigma),
                                                       mean(ST_GX_D_delta),
                                                       mean(ST_GX_B_beta1),
                                                       mean(ST_GX_B_beta2),
                                                       mean(ST_GX_B_beta3),
                                                       mean(ST_GX_B_w),
                                                       mean(ST_GX_B_sigma),
                                                       mean(ST_GX_B_delta)),
                                  average_bias = c(mean(ST_GX_D_beta1) - 1/sqrt(3),
                                                   mean(ST_GX_D_beta2) - 1/sqrt(3),
                                                   mean(ST_GX_D_beta3) - 1/sqrt(3),
                                                   mean(ST_GX_D_w) - 0.6,
                                                   mean(ST_GX_D_sigma) - 1.5,
                                                   mean(ST_GX_D_delta) - 6.0,
                                                   mean(ST_GX_B_beta1) - 1/sqrt(3),
                                                   mean(ST_GX_B_beta2) - 1/sqrt(3),
                                                   mean(ST_GX_B_beta3) - 1/sqrt(3),
                                                   mean(ST_GX_B_w) - 0.6,
                                                   mean(ST_GX_B_sigma) - 1.5,
                                                   mean(ST_GX_B_delta) - 6.0))

rownames(df_point_estimation) <- NULL
colnames(df_point_estimation) <- c("Model", "Variable", "A.P.E", "Average Bias")
kbl(df_point_estimation,format = "latex",digits=4, escape = FALSE,booktabs = TRUE) %>%
    collapse_rows(columns = c(1))

# MSE of g calculation ----------------------------------------------------

ST_GX_D_MSE <- numeric(simu_n)
ST_GX_B_MSE <- numeric(simu_n)
for (i in 1:simu_n) {
    ST_GX_D_Diff <- true_g_func(ST_GX_D_output[[i]]$g_function_estimation$eta) - ST_GX_D_output[[i]]$g_function_estimation$predicted_value
    ST_GX_B_Diff <- true_g_func(ST_GX_B_output[[i]]$g_function_estimation$eta) - ST_GX_B_output[[i]]$g_function_estimation$predicted_value
    ST_GX_D_MSE[i] <- mean(ST_GX_D_Diff^2)
    ST_GX_B_MSE[i] <- mean(ST_GX_B_Diff^2)
}

df_MSE <- data.frame(model = rep(c("ST-GX-D","ST-GX-B"),each = simu_n),
                     value = c(ST_GX_D_MSE,ST_GX_B_MSE))

p1 <- bwplot(model ~ value, 
             data = df_MSE,
             xlab = "MSE of g(U)",
             main = "Simulation Study || Scheme 2")

pdf("./02_Scheme2/MSE.pdf",width = 5*1.2,height = 3*1.2)
print(p1)
dev.off()

# the plot of g function --------------------------------------------------

set.seed(2)
n <- 1000

df1 <- data_generation(n = n,
                       beta = c(1,1,1))

df_ST_GX_D_g <- data.frame(U = ST_GX_D_output[[2]]$g_function_estimation$eta,
                           value = ST_GX_D_output[[2]]$g_function_estimation$predicted_value)

df_ST_GX_B_g <- data.frame(U = ST_GX_B_output[[2]]$g_function_estimation$eta,
                           value = ST_GX_B_output[[2]]$g_function_estimation$predicted_value)
ST_GX_D_MSE <- mean((ST_GX_D_output[[2]]$g_function_estimation$eta - true_g_func(ST_GX_D_output[[2]]$g_function_estimation$eta))^2)
ST_GX_B_MSE <- mean((ST_GX_B_output[[2]]$g_function_estimation$eta - true_g_func(ST_GX_B_output[[2]]$g_function_estimation$eta))^2)

p2 <- xyplot(value ~ U,
             type = "l",
             lwd = 2,
             data = df_ST_GX_D_g,
             xlim = c(-3.5,3.5),
             ylim = c(-4.5,4.5),
             xlab = "U",
             ylab = "g(U)",
             main = paste0("Simulation Study || Scheme 2 || ST-GX-D || MSE: ",round(ST_GX_D_MSE,4))) +
    latticeExtra::layer(panel.points(x = df_ST_GX_D_g$U,
                                     y = true_g_func(df_ST_GX_D_g$U),
                                     col = rgb(0.8,0,0,0.8))
    )

p3 <- xyplot(value ~ U,
             type = "l",
             lwd = 2,
             data = df_ST_GX_B_g,
             xlim = c(-3.5,3.5),
             ylim = c(-4.5,4.5),
             xlab = "U",
             ylab = "g(U)",
             main = paste0("Simulation Study || Scheme 2 || ST-GX-B || MSE: ",round(ST_GX_B_MSE,4))) +
    latticeExtra::layer(panel.points(x = df_ST_GX_B_g$U,
                                     y = true_g_func(df_ST_GX_B_g$U),
                                     col = rgb(0.8,0,0,0.8))
    )

pdf("./02_Scheme2/Estimated_g.pdf",width = 5*2*1.2,height = 4*1.2)
print(gridExtra::grid.arrange(p2,p3,ncol = 2))
dev.off()
