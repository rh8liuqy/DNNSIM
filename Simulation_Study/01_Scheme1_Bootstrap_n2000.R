rm(list = ls())
library(lattice)
library(latticeExtra)
library(DNNSIM)
library(tidyverse)
library(reticulate)

reticulate::use_condaenv("~/anaconda3/envs/r-pytorch/python.exe")

source("./01_Scheme1_Main.R")

# bootstrap standard error ------------------------------------------------
seed <- 156
n <- 2000

ST_GX_D_bootstrap_output <- simu_bootstrap(seed,n)

saveRDS(ST_GX_D_bootstrap_output,paste0("./01_Scheme1/ST_GX_D_bootstrap_output_","n",n,".RDS"))
