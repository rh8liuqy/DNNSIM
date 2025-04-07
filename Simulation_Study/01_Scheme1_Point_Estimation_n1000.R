rm(list = ls())
library(lattice)
library(latticeExtra)
library(DNNSIM)
library(tidyverse)
library(reticulate)

reticulate::use_condaenv("~/anaconda3/envs/r-pytorch/python.exe")

source("./01_Scheme1_Main.R")

simu_n <- 300
n <- 1000

simu_output <- vector("list",simu_n)

current_seed <- 1
count <- 1

while (count <= simu_n) {
    result <- tryCatch({simu_point_estimation(seed = current_seed, n = n)},
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

saveRDS(simu_output,paste0("./01_Scheme1/simu_output_point_estimation_n",n,".RDS"))
