rm(list = ls())
library(lattice)
library(latticeExtra)
library(DNNSIM)
library(tidyverse)
library(reticulate)

ST_GX_D_CV <- readRDS("./03_Scheme3/simu_output_CV_model_ST-GX-D_n_1000.RDS")
ST_FX_CV <- readRDS("./03_Scheme3/simu_output_CV_model_ST-FX_n_1000.RDS")

df_CV <- data.frame(MSE = c(ST_GX_D_CV[[1]]$CV_function_output$CV_MSE,
                            ST_FX_CV[[1]]$CV_function_output$CV_MSE),
                    model = rep(c("ST-GX-D",
                                  "ST-FX"),
                                each = 10))

bwplot(model ~ MSE, data = df_CV)
