rm(list = ls())

library(latticeExtra)

ST_GX_D_output <- readRDS("./04_Scheme4/simu_output_CV_model_ST-GX-D_n_1000.RDS")
ST_FX_output <- readRDS("./04_Scheme4/simu_output_CV_model_ST-FX_n_1000.RDS")

simu_n <- length(ST_GX_D_output)
ST_GX_D_CV <- vector("list",simu_n)
ST_FX_CV <- vector("list",simu_n)
for (i in 1:simu_n) {
    ST_GX_D_CV[[i]] <- ST_GX_D_output[[i]]$CV_function_output$CV_MSE
    ST_FX_CV[[i]] <- ST_FX_output[[i]]$CV_function_output$CV_MSE
}

df_CV <- data.frame(CV_MSE = c(unlist(ST_GX_D_CV),
                               unlist(ST_FX_CV)),
                    model = rep(c("ST-GX-D",
                                  "ST-FX"),
                                each = 1000))

pdf("./04_Scheme4/bwplot_CV_MSE.pdf",width = 5*1.6,height = 3*1.6)
bwplot(model ~ CV_MSE, 
       data = df_CV,
       xlab = "MSE",
       main = "Simulation Study || Scheme 4")
dev.off()