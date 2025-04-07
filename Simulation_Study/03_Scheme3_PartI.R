rm(list = ls())
library(lattice)
library(latticeExtra)
library(DNNSIM)
library(tidyverse)
library(reticulate)

reticulate::use_condaenv("~/anaconda3/envs/r-pytorch/python.exe")

model <- "ST-GX-D"
source("./03_Scheme3_PartI_main.R")

model <- "SN-GX-D"
source("./03_Scheme3_PartI_main.R")

model <- "N-GX-D"
source("./03_Scheme3_PartI_main.R")

model <- "ST-GX-B"
source("./03_Scheme3_PartI_main.R")

model <- "SN-GX-B"
source("./03_Scheme3_PartI_main.R")

model <- "N-GX-B"
source("./03_Scheme3_PartI_main.R")

model <- "ST-FX"
source("./03_Scheme3_PartI_main.R")

model <- "SN-FX"
source("./03_Scheme3_PartI_main.R")

model <- "N-FX"
source("./03_Scheme3_PartI_main.R")
