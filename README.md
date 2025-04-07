# DNNSIM: Deep Monotonic Single-Index Model

This repository hosts supplementary materials and simulation code for the **DNNSIM** R package, available on [CRAN](https://cran.r-project.org/web/packages/DNNSIM/index.html). The package implements a monotonic single-index model using deep neural networks for robust modal regression, especially suited for skewed and heavy-tailed data.

## 📄 Associated Paper

**Title**: *A Monotonic Single-Index Model for Skewed and Heavy-Tailed Data: A Deep Neural Network Approach Applied to Periodontal Studies*  
**Authors**: Qingyang Liu, Shijie Wang, Ray Bai, Dipankar Bandyopadhyay  
**Abstract**: This work proposes a deep monotonic single-index model with a flexible two-piece scale Student-\(t\) distribution, providing a robust alternative for analyzing skewed and heavy-tailed data. The proposed model ensures theoretical guarantees, outlier resistance, and interpretability via parametric index coefficients.

⚠️ **Note**: Due to size constraints, the real data (162MB) used in the paper’s application section is **not included** in this repository. Please contact the authors to request access.

## 🔗 CRAN Package

Install the latest release from CRAN:

```R
install.packages("DNNSIM")
```