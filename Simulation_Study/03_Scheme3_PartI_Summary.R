rm(list = ls())
library(lattice)
library(latticeExtra)
library(DNNSIM)
library(tidyverse)
library(reticulate)

ST_GX_D_output <- readRDS("./03_Scheme3/simu_output_beta_model_ST-GX-D_n_1000.RDS")
SN_GX_D_output <- readRDS("./03_Scheme3/simu_output_beta_model_SN-GX-D_n_1000.RDS")
N_GX_D_output <- readRDS("./03_Scheme3/simu_output_beta_model_N-GX-D_n_1000.RDS")

ST_GX_B_output <- readRDS("./03_Scheme3/simu_output_beta_model_ST-GX-B_n_1000.RDS")
SN_GX_B_output <- readRDS("./03_Scheme3/simu_output_beta_model_SN-GX-B_n_1000.RDS")
N_GX_B_output <- readRDS("./03_Scheme3/simu_output_beta_model_N-GX-B_n_1000.RDS")

# summary about beta ------------------------------------------------------

ST_GX_D_beta <- matrix(NA,nrow = length(ST_GX_D_output), ncol = 3)
SN_GX_D_beta <- matrix(NA,nrow = length(ST_GX_D_output), ncol = 3)
N_GX_D_beta <- matrix(NA,nrow = length(ST_GX_D_output), ncol = 3)

ST_GX_B_beta <- matrix(NA,nrow = length(ST_GX_D_output), ncol = 3)
SN_GX_B_beta <- matrix(NA,nrow = length(ST_GX_D_output), ncol = 3)
N_GX_B_beta <- matrix(NA,nrow = length(ST_GX_D_output), ncol = 3)

for (i in 1:length(ST_GX_D_output)) {
    ST_GX_D_beta[i,] <- ST_GX_D_output[[i]]$DNN_train_output$beta_output
    SN_GX_D_beta[i,] <- SN_GX_D_output[[i]]$DNN_train_output$beta_output
    N_GX_D_beta[i,] <- N_GX_D_output[[i]]$DNN_train_output$beta_output
    
    ST_GX_B_beta[i,] <- ST_GX_B_output[[i]]$DNN_train_output$beta_output
    SN_GX_B_beta[i,] <- SN_GX_B_output[[i]]$DNN_train_output$beta_output
    N_GX_B_beta[i,] <- N_GX_B_output[[i]]$DNN_train_output$beta_output
}

colnames(ST_GX_D_beta) <- c("beta1","beta2","beta3")
colnames(SN_GX_D_beta) <- c("beta1","beta2","beta3")
colnames(N_GX_D_beta) <- c("beta1","beta2","beta3")
colnames(ST_GX_B_beta) <- c("beta1","beta2","beta3")
colnames(SN_GX_B_beta) <- c("beta1","beta2","beta3")
colnames(N_GX_B_beta) <- c("beta1","beta2","beta3")

ST_GX_D_beta <- pivot_longer(as.data.frame(ST_GX_D_beta),cols = everything()) %>%
    mutate(model = "ST-GX-D")
SN_GX_D_beta <- pivot_longer(as.data.frame(SN_GX_D_beta),cols = everything()) %>%
    mutate(model = "SN-GX-D")
N_GX_D_beta <- pivot_longer(as.data.frame(N_GX_D_beta),cols = everything()) %>%
    mutate(model = "N-GX-D")

ST_GX_B_beta <- pivot_longer(as.data.frame(ST_GX_B_beta),cols = everything()) %>%
    mutate(model = "ST-GX-B")
SN_GX_B_beta <- pivot_longer(as.data.frame(SN_GX_B_beta),cols = everything()) %>%
    mutate(model = "SN-GX-B")
N_GX_B_beta <- pivot_longer(as.data.frame(N_GX_B_beta),cols = everything()) %>%
    mutate(model = "N-GX-B")

df_beta <- rbind(ST_GX_D_beta,
                 SN_GX_D_beta,
                 N_GX_D_beta,
                 ST_GX_B_beta,
                 SN_GX_B_beta,
                 N_GX_B_beta)

df_beta$model <- factor(df_beta$model,
                        levels = c("N-GX-B",
                                   "SN-GX-B",
                                   "ST-GX-B",
                                   "N-GX-D",
                                   "SN-GX-D",
                                   "ST-GX-D"))

df_beta$name <- factor(df_beta$name,
                       levels = c("beta1",
                                  "beta2",
                                  "beta3"))

df_beta <- as.data.frame(df_beta)

p1 <- bwplot(model ~ value | name, 
             data = df_beta,
             main = "Simulation Study || Scheme 3") +
    latticeExtra::layer(panel.lines(x = 1/sqrt(3),
                                    y = c(-10,10),
                                    col = rgb(0.8,0,0,0.8),
                                    lwd = 2,
                                    lty = 2))

pdf("./03_Scheme3/bwplot_beta.pdf",width = 5*1.6,height = 3*1.6)
print(p1)
dev.off()

# summary about g ---------------------------------------------------------

source("./03_Scheme3_Main.R")

g_MSE_ST_GX_D <- numeric(length(ST_GX_D_output))
g_MSE_SN_GX_D <- numeric(length(ST_GX_D_output))
g_MSE_N_GX_D <- numeric(length(ST_GX_D_output))

g_MSE_ST_GX_B <- numeric(length(ST_GX_D_output))
g_MSE_SN_GX_B <- numeric(length(ST_GX_D_output))
g_MSE_N_GX_B <- numeric(length(ST_GX_D_output))

for (i in 1:length(ST_GX_D_output)) {
    est_g_value <- ST_GX_D_output[[i]]$g_function_estimation$predicted_value
    true_g_value <- true_g_func(ST_GX_D_output[[i]]$g_function_estimation$eta)
    g_MSE_ST_GX_D[i] <- mean((est_g_value - true_g_value)^2)
    
    est_g_value <- SN_GX_D_output[[i]]$g_function_estimation$predicted_value
    true_g_value <- true_g_func(SN_GX_D_output[[i]]$g_function_estimation$eta)
    g_MSE_SN_GX_D[i] <- mean((est_g_value - true_g_value)^2)
    
    est_g_value <- N_GX_D_output[[i]]$g_function_estimation$predicted_value
    true_g_value <- true_g_func(N_GX_D_output[[i]]$g_function_estimation$eta)
    g_MSE_N_GX_D[i] <- mean((est_g_value - true_g_value)^2)
    
    est_g_value <- ST_GX_B_output[[i]]$g_function_estimation$predicted_value
    true_g_value <- true_g_func(ST_GX_B_output[[i]]$g_function_estimation$eta)
    g_MSE_ST_GX_B[i] <- mean((est_g_value - true_g_value)^2)
    
    est_g_value <- SN_GX_B_output[[i]]$g_function_estimation$predicted_value
    true_g_value <- true_g_func(SN_GX_B_output[[i]]$g_function_estimation$eta)
    g_MSE_SN_GX_B[i] <- mean((est_g_value - true_g_value)^2)
    
    est_g_value <- N_GX_B_output[[i]]$g_function_estimation$predicted_value
    true_g_value <- true_g_func(N_GX_B_output[[i]]$g_function_estimation$eta)
    g_MSE_N_GX_B[i] <- mean((est_g_value - true_g_value)^2)
    
}


df_g <- data.frame(MSE = c(g_MSE_ST_GX_D,
                           g_MSE_SN_GX_D,
                           g_MSE_N_GX_D,
                           g_MSE_ST_GX_B,
                           g_MSE_SN_GX_B,
                           g_MSE_N_GX_B),
                   model = rep(c("ST-GX-D",
                                 "SN-GX-D",
                                 "N-GX-D",
                                 "ST-GX-B",
                                 "SN-GX-B",
                                 "N-GX-B"),
                               each = length(ST_GX_D_output)))

df_g$model <- factor(df_g$model,
                     levels = c("N-GX-B",
                                "SN-GX-B",
                                "ST-GX-B",
                                "N-GX-D",
                                "SN-GX-D",
                                "ST-GX-D"))
pdf("./03_Scheme3/bwplot_g_MSE.pdf",width = 5*1.6,height = 3*1.6)
print(bwplot(model ~ MSE,
             data = df_g,
             xlab = "MSE of g(U)",
             main = "Simulation Study || Scheme 3"))
dev.off()

