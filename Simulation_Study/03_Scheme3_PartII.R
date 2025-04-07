rm(list = ls())
library(lattice)
library(latticeExtra)
library(DNNSIM)
library(tidyverse)
library(reticulate)

reticulate::use_condaenv("~/anaconda3/envs/r-pytorch/python.exe")

model <- "ST-GX-D"
source("./03_Scheme3_PartII_main.R")

model <- "ST-FX"
source("./03_Scheme3_PartII_main.R")
