rm(list = ls())
library(kableExtra)
library(tidyverse)
library(latticeExtra)

# point estimation n = 1000 -----------------------------------------------

point_estimation_n1000 <- readRDS("./01_Scheme1/simu_output_point_estimation_n1000.RDS")
point_estimation_n1000 <- list_rbind(point_estimation_n1000)
point_estimation_n1000 <- as.data.frame(point_estimation_n1000)
# remove one outlier due to numerical instability
index <- point_estimation_n1000$g_func_MSE < 6
point_estimation_n1000 <- point_estimation_n1000[index,]

# bootstrap n = 1000 ------------------------------------------------------

bootstrap_n1000 <- readRDS("./01_Scheme1/ST_GX_D_bootstrap_output_n1000.RDS")
bootstrap_B <- length(bootstrap_n1000$ST_GX_D_Bootstrap_output)

beta_bootstrap_output <- matrix(NA,nrow = bootstrap_B,ncol=3)

w_bootstrap_output <- numeric(bootstrap_B)
sigma_bootstrap_output <- numeric(bootstrap_B)
delta_bootstrap_output <- numeric(bootstrap_B)

for (i in 1:bootstrap_B) {
    beta_bootstrap_output[i,] <- bootstrap_n1000$ST_GX_D_Bootstrap_output[[i]]$beta_output
    w_bootstrap_output[i] <- bootstrap_n1000$ST_GX_D_Bootstrap_output[[i]]$w_output
    sigma_bootstrap_output[i] <- bootstrap_n1000$ST_GX_D_Bootstrap_output[[i]]$sigma_output
    delta_bootstrap_output[i] <- bootstrap_n1000$ST_GX_D_Bootstrap_output[[i]]$delta_output
}

df_bootstrap <- cbind(beta_bootstrap_output,w_bootstrap_output,sigma_bootstrap_output,delta_bootstrap_output)
df_bootstrap <- as.data.frame(df_bootstrap)
colnames(df_bootstrap) <- c("beta1","beta2","beta3","w","sigma","delta")

df_n1000 <- data.frame(sample_size = "$n = 1000$",
                       variable = c("beta1","beta2","beta3","w","sigma","delta"),
                       point_estimation = colMeans(point_estimation_n1000[,1:6]),
                       average_bias = colMeans(point_estimation_n1000[,1:6]) - c(1/sqrt(3),1/sqrt(3),1/sqrt(3),0.6,1.5,6.0),
                       empirical_standard_error = apply(point_estimation_n1000[,1:6],2,sd),
                       bootstrap_standard_error = apply(df_bootstrap, 2, sd))

# point estimation n = 2000 -----------------------------------------------

point_estimation_n2000 <- readRDS("./01_Scheme1/simu_output_point_estimation_n2000.RDS")
point_estimation_n2000 <- list_rbind(point_estimation_n2000)
point_estimation_n2000 <- as.data.frame(point_estimation_n2000)
index <- point_estimation_n2000$delta < 90
point_estimation_n2000 <- point_estimation_n2000[index,]

# bootstrap n = 2000 ------------------------------------------------------

bootstrap_n2000 <- readRDS("./01_Scheme1/ST_GX_D_bootstrap_output_n2000.RDS")
bootstrap_B <- length(bootstrap_n2000$ST_GX_D_Bootstrap_output)

beta_bootstrap_output <- matrix(NA,nrow = bootstrap_B,ncol=3)

w_bootstrap_output <- numeric(bootstrap_B)
sigma_bootstrap_output <- numeric(bootstrap_B)
delta_bootstrap_output <- numeric(bootstrap_B)

for (i in 1:bootstrap_B) {
    beta_bootstrap_output[i,] <- bootstrap_n2000$ST_GX_D_Bootstrap_output[[i]]$beta_output
    w_bootstrap_output[i] <- bootstrap_n2000$ST_GX_D_Bootstrap_output[[i]]$w_output
    sigma_bootstrap_output[i] <- bootstrap_n2000$ST_GX_D_Bootstrap_output[[i]]$sigma_output
    delta_bootstrap_output[i] <- bootstrap_n2000$ST_GX_D_Bootstrap_output[[i]]$delta_output
}

df_bootstrap <- cbind(beta_bootstrap_output,w_bootstrap_output,sigma_bootstrap_output,delta_bootstrap_output)
df_bootstrap <- as.data.frame(df_bootstrap)
colnames(df_bootstrap) <- c("beta1","beta2","beta3","w","sigma","delta")

df_n2000 <- data.frame(sample_size = "$n = 2000$",
                       variable = c("beta1","beta2","beta3","w","sigma","delta"),
                       point_estimation = colMeans(point_estimation_n2000[,1:6]),
                       average_bias = colMeans(point_estimation_n2000[,1:6]) - c(1/sqrt(3),1/sqrt(3),1/sqrt(3),0.6,1.5,6.0),
                       empirical_standard_error = apply(point_estimation_n2000[,1:6],2,sd),
                       bootstrap_standard_error = apply(df_bootstrap, 2, sd))

# combine n = 1000 and n = 2000 -------------------------------------------
df_tab1 <- rbind(df_n1000,df_n2000)
rownames(df_tab1) <- NULL
colnames(df_tab1) <- c("Sample Size", "Variable", "A.P.E",
                       "Average Bias", "Empirical Standard Error",
                       "Bootstrap Standard Error")
df_tab1$Variable <- rep(c("$\\beta_1$","$\\beta_2$","$\\beta_3$","$w$","$\\sigma$","$\\delta$"),2)

kbl(df_tab1,format = "latex",digits=4, escape = FALSE,booktabs = TRUE) %>%
    collapse_rows(columns = c(1))

# g function plot n = 1000 ------------------------------------------------

source("./01_Scheme1_Main.R")

set.seed(135)
df_n1000 <- data_generation(n = 1000,
                       beta = c(1,1,1))

bootstrap_n1000 <- readRDS("./01_Scheme1/ST_GX_D_bootstrap_output_n1000.RDS")
bootstrap_B <- length(bootstrap_n1000$ST_GX_D_Bootstrap_output)
estimated_value <- bootstrap_n1000$g_function_estimation$predicted_value

g_mat_n1000 <- matrix(data = NA, nrow = bootstrap_B, ncol = 100)

for (i in 1:bootstrap_B) {
    g_mat_n1000[i,] <- bootstrap_n1000$ST_GX_D_Bootstrap_output[[i]]$g_output$predicted_value
}

df_g_n1000 <- data.frame(U = bootstrap_n1000$g_function_estimation$eta,
                         estimated_value = apply(g_mat_n1000, 2, function(x){quantile(x,probs = c(0.50))}),
                         true_value = true_g_func(bootstrap_n1000$g_function_estimation$eta))

cover_n1000_ub <- df_g_n1000$true_value < apply(g_mat_n1000, 2, function(x){quantile(x,probs = c(0.95))})
cover_n1000_lb <- df_g_n1000$true_value > apply(g_mat_n1000, 2, function(x){quantile(x,probs = c(0.05))})
mean(cover_n1000_ub & cover_n1000_lb)

MSE_n1000 <- mean((df_g_n1000$true_value - df_g_n1000$estimated_value)^2)

p1 <- xyplot(estimated_value ~ U,
       data = df_g_n1000, 
       type = "l",
       lwd = 2,
       xlab = "U",
       ylab = "Value",
       ylim = c(-3,11),
       xlim = c(-4.3,4.3),
       col = "#0072B2",
       main = paste0("Simulation Study || Scheme 1 || Sample Size: n = 1000 || MSE: ",round(MSE_n1000,4))) + 
    latticeExtra::layer(panel.lines(x = bootstrap_n1000$g_function_estimation$eta,
                                    y = apply(g_mat_n1000, 2, function(x){quantile(x,probs = c(0.05))}),
                                    lwd = 2, 
                                    lty = 4, 
                                    col = rgb(0.8,0,0,0.8))) +
    latticeExtra::layer(panel.lines(x = bootstrap_n1000$g_function_estimation$eta,
                                    y = apply(g_mat_n1000, 2, function(x){quantile(x,probs = c(0.95))}),
                                    lwd = 2, 
                                    lty = 4, 
                                    col = rgb(0.8,0,0,0.8))) + 
    latticeExtra::layer(panel.lines(x = bootstrap_n1000$g_function_estimation$eta, 
                                    y = true_g_func(bootstrap_n1000$g_function_estimation$eta),
                                    lwd = 2, 
                                    lty = 2, 
                                    col = rgb(0.0,0.8,0,0.8))) +
    latticeExtra::layer(panel.points(x = df_n1000$U,
                                     y = df_n1000$y,
                                     col = rgb(0,144,178,60,maxColorValue = 255)))
p1

# g function plot n = 2000 ------------------------------------------------

set.seed(156)
df_n2000 <- data_generation(n = 2000,
                            beta = c(1,1,1))

source("./01_Scheme1_Main.R")
bootstrap_n2000 <- readRDS("./01_Scheme1/ST_GX_D_bootstrap_output_n2000.RDS")
bootstrap_B <- length(bootstrap_n2000$ST_GX_D_Bootstrap_output)
estimated_value <- bootstrap_n2000$g_function_estimation$predicted_value

g_mat_n2000 <- matrix(data = NA, nrow = bootstrap_B, ncol = 100)

for (i in 1:bootstrap_B) {
    g_mat_n2000[i,] <- bootstrap_n2000$ST_GX_D_Bootstrap_output[[i]]$g_output$predicted_value
}

df_g_n2000 <- data.frame(U = bootstrap_n2000$g_function_estimation$eta,
                         estimated_value = apply(g_mat_n2000, 2, function(x){quantile(x,probs = c(0.50))}),
                         true_value = true_g_func(bootstrap_n2000$g_function_estimation$eta))

df_g_n2000 <- data.frame(U = bootstrap_n2000$g_function_estimation$eta,
                         estimated_value = apply(g_mat_n2000, 2, function(x){quantile(x,probs = c(0.50))}),
                         true_value = true_g_func(bootstrap_n2000$g_function_estimation$eta))

cover_n2000_ub <- df_g_n2000$true_value < apply(g_mat_n2000, 2, function(x){quantile(x,probs = c(0.95))})
cover_n2000_lb <- df_g_n2000$true_value > apply(g_mat_n2000, 2, function(x){quantile(x,probs = c(0.05))})
mean(cover_n2000_ub & cover_n2000_lb)

MSE_n2000 <- mean((df_g_n2000$true_value - df_g_n2000$estimated_value)^2)

p2 <- xyplot(estimated_value ~ U,
             data = df_g_n2000, 
             type = "l",
             lwd = 2,
             xlab = "U",
             ylab = "Value",
             ylim = c(-3,11),
             xlim = c(-4.3,4.3),
             col = "#0072B2",
             main = paste0("Simulation Study || Scheme 1 || Sample Size: n = 2000 || MSE: ",round(MSE_n2000,4)),
             key = list(space="bottom",
                        lines=list(col=c("#0072B2",rgb(0.0,0.8,0,0.8),rgb(0.8,0,0,0.8)), 
                                   lty=c(1,2,4), lwd=2),
                        text=list(c("Estimated g(U)",
                                    "True g(U)",
                                    "90% Pointwise Confidence Interval of g(U)")))) + 
    latticeExtra::layer(panel.lines(x = bootstrap_n2000$g_function_estimation$eta,
                                    y = apply(g_mat_n2000, 2, function(x){quantile(x,probs = c(0.05))}),
                                    lwd = 2, 
                                    lty = 4, 
                                    col = rgb(0.8,0,0,0.8))) +
    latticeExtra::layer(panel.lines(x = bootstrap_n2000$g_function_estimation$eta,
                                    y = apply(g_mat_n2000, 2, function(x){quantile(x,probs = c(0.95))}),
                                    lwd = 2, 
                                    lty = 4, 
                                    col = rgb(0.8,0,0,0.8))) + 
    latticeExtra::layer(panel.lines(x = bootstrap_n2000$g_function_estimation$eta, 
                                    y = true_g_func(bootstrap_n2000$g_function_estimation$eta),
                                    lwd = 2, 
                                    lty = 2, 
                                    col = rgb(0.0,0.8,0,0.8))) +
    latticeExtra::layer(panel.points(x = df_n2000$U,
                                     y = df_n2000$y,
                                     col = rgb(0,144,178,60,maxColorValue = 255)))

pdf("./01_Scheme1/Estimated_g.pdf",width = 8.5,height = 11)
print(gridExtra::grid.arrange(p1,p2,nrow = 2))
dev.off()

# plot of MSE -------------------------------------------------------------

df_MSE_n1000 <- data.frame(MSE = point_estimation_n1000$g_func_MSE,
                           x = "n = 1000")
df_MSE_n2000 <- data.frame(MSE = point_estimation_n2000$g_func_MSE,
                           x = "n = 2000")
df_MSE <- rbind(df_MSE_n1000,
                df_MSE_n2000)
p3 <- bwplot(x ~ MSE, 
             data = df_MSE,
             xlab = "MSE of g(U)",
             main = "Simulation Study || Scheme 1")

pdf("./01_Scheme1/MSE.pdf",width = 5*1.2,height = 3*1.2)
print(p3)
dev.off()

median(df_MSE_n1000$MSE)
median(df_MSE_n2000$MSE)
