rm(list = ls())
library(latticeExtra)
library(ald)
library(latex2exp)

xaxis <- seq(-7,7,length.out = 1000)
yaxis <- dALD(y = xaxis,
              mu = 0,
              sigma = 0.5,
              p = 0.6)
df_ald <- data.frame(x = xaxis,
                     y = yaxis)

p1 <- xyplot(y ~ x,
             data = df_ald,
             type = "l",
             lwd = 2,
             ylab = "Density",
             scales = list(x = list(at = c(-7:7))),
             main = TeX("Density Plot of ALD($\\mu = 0, \\sigma = 0.5, p = 0.6$)")) +
    layer(panel.lines(x = c(0,0),
                      y = c(-5,5),
                      lwd = 2,
                      lty = 2,
                      col = rgb(0.8,0,0,0.8)))

pdf("./03_Scheme3/ALD_Density.pdf",width = 5*1.2,height = 3*1.2)
print(p1)
dev.off()
