% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DNN_model.R
\name{DNN_model}
\alias{DNN_model}
\title{Define and train the DNN-SIM model}
\usage{
DNN_model(
  formula,
  data,
  model,
  num_epochs,
  verbatim = TRUE,
  CV = FALSE,
  CV_K = 10,
  bootstrap = FALSE,
  bootstrap_B = 1000,
  bootstrap_num_epochs = 100,
  U_new = FALSE,
  U_min = -4,
  U_max = 4,
  random_state = 100
)
}
\arguments{
\item{formula}{an object of class "\link[stats]{formula}" (or one that can be coerced to that class): a symbolic description of the model to be fitted.}

\item{data}{a data frame.}

\item{model}{the model type. It must be be one of "N-GX-D","SN-GX-D","ST-GX-D","N-GX-B","SN-GX-B","ST-GX-B","N-FX","SN-FX","ST-FX".}

\item{num_epochs}{an integer. The number of complete passes through the training dataset.}

\item{verbatim}{TRUE/FALSE.If \code{verbatim} is \code{TRUE}, then log information from training the DNN-SIM model will be printed.}

\item{CV}{TRUE/FALSE. Whether use the cross-validation to measure the prediction accuracy.}

\item{CV_K}{an integer. The number of folders K-folder cross-validation.}

\item{bootstrap}{TRUE/FALSE. Whether use the bootstrap method to quantify the uncertainty. The bootstrap option ONLY works for the "ST-GX-D" model.}

\item{bootstrap_B}{an integer. The number of bootstrap iteration.}

\item{bootstrap_num_epochs}{an integer. The number of complete passes through the training dataset in the bootstrap procedure.}

\item{U_new}{TRUE/FALSE. Whether use self defined U for the estimation of single index function, g(U).}

\item{U_min}{a numeric value. The minimum of the self defined U.}

\item{U_max}{a numeric value. The maximum of the self defined U.}

\item{random_state}{an integer. The random seed for initiating the neural network.}
}
\value{
A list consisting of the point estimation, g function estimation (optional), cross-validation results (optional) and bootstrap results(optional).
}
\description{
Define and train the DNN-SIM model
}
\details{
The DNNSIM model is defined as:
\deqn{Y = g(\mathbf{X} \boldsymbol{\beta}) + e.}
The residuals \eqn{e} follow a skewed T distribution, skewed normal distribution, or normal distribution. The single index function \eqn{g} is assumed to be a monotonic increasing function.
}
\examples{

\donttest{
# check python module dependencies
if (reticulate::py_module_available("torch") &
    reticulate::py_module_available("numpy") &
    reticulate::py_module_available("sklearn") &
    reticulate::py_module_available("scipy")) {

  # set the random seed
  set.seed(100)

  # simulate some data
  df1 <- data_simulation(n=100,beta=c(1,1,1),w=0.3,
                         sigma=0.1,delta=10.0,seed=100)

  # the cross-validation and bootstrap takes a long time
  DNN_model_output <- DNN_model(y ~ X1 + X2 + X3 - 1,
                                data = df1,
                                model = "ST-GX-D",
                                num_epochs = 5,
                                verbatim = FALSE,
                                CV = TRUE,
                                CV_K = 2,
                                bootstrap = TRUE,
                                bootstrap_B = 2,
                                bootstrap_num_epochs = 5,
                                U_new = TRUE,
                                U_min = -4.0,
                                U_max = 4.0)
  print(DNN_model_output)
}
}


}
\references{
\insertRef{liu2022bayesian}{DNNSIM}
}
