rm(list = ls())
library(lattice)
library(latticeExtra)
library(DNNSIM)
library(tidyverse)
library(reticulate)

reticulate::use_condaenv("~/anaconda3/envs/r-pytorch/python.exe")

source("./02_Scheme2_Main.R")

model <- "ST-GX-B"
simu_n <- 100
n <- 1000

simu_output <- vector("list",simu_n)

current_seed <- 1
count <- 1

while (count <= simu_n) {
    result <- tryCatch({simu_MSE(seed = current_seed, n = n, model = model)},
                       error = function(e){
                           NULL
                       })
    if (! is.null(result)) {
        simu_output[[count]] <- result
        message(sprintf("Simulation with seed %d succeeded.", current_seed))
        current_seed <- current_seed + 1
        count <- count + 1
    } else {
        message(sprintf("Simulation with seed %d failed.", current_seed))
        current_seed <- current_seed + 1
    }
    
}

saveRDS(simu_output,paste0("./02_Scheme2/simu_output_MSE_model_",model,"_n_",n,".RDS"))
