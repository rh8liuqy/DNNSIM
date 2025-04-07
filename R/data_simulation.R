#' Simulate data for the DNN-SIM model
#'
#' @param n an integer. The sample size.
#' @param beta a vector. The covariate coefficients.
#' @param w a number between 0 and 1. The skewness parameter.
#' @param sigma a number larger than 0. The standard deviation parameter.
#' @param delta a number larger than 0. The degree of freedom parameter.
#' @param seed an integer. The random seed.
#'
#' @return a dataframe of the simulated response variable y and the design matrix X.
#' @details
#' This is a simple data generation function for a simulation study. All elements of the design matrix X follow a uniform distribution from -3.0 and 3.0 independently and identically. The true \eqn{g} function is the standard logistic function.
#'
#' @references
#' \insertRef{liu2022bayesian}{DNNSIM}
#'
#' @export
#'
#' @examples
#'
#' \donttest{
#' # check python module dependencies
#' if (reticulate::py_module_available("torch") &
#'     reticulate::py_module_available("numpy") &
#'     reticulate::py_module_available("sklearn") &
#'     reticulate::py_module_available("scipy")) {
#'   df1 <- data_simulation(n=50,beta=c(1,1,1),w=0.3,
#'                          sigma=0.1,delta=4.0,seed=100)
#'   print(head(df1))
#' }
#' }
#'
data_simulation <- function(n, beta, w, sigma, delta, seed) {
  # ensure the norm of beta is 1
  beta <- beta / norm(beta, type = "2")
  simu_output <- ST_RNG$reg_simu(n, beta, w, sigma, delta, seed)
  output <- cbind(reticulate::py_to_r(simu_output$y),
                  reticulate::py_to_r(simu_output$X))
  output <- as.data.frame(output)
  colnames(output)[1] <- "y"
  # the number of columns in X
  p <- ncol(reticulate::py_to_r(simu_output$X))
  colnames(output)[2:(p+1)] <- paste0("X",1:p)
  return(output)
}
