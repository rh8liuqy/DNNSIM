#' Define and train the DNN-SIM model
#'
#' @param formula an object of class "\link[stats]{formula}" (or one that can be coerced to that class): a symbolic description of the model to be fitted.
#' @param data a data frame.
#' @param num_epochs an integer. The number of complete passes through the training dataset.
#' @param model the model type. It must be be one of "N-GX-D","SN-GX-D","ST-GX-D","N-GX-B","SN-GX-B","ST-GX-B","N-FX","SN-FX","ST-FX".
#' @param verbatim TRUE/FALSE.If `verbatim` is `TRUE`, then log information from training the DNN-SIM model will be printed.
#' @param CV TRUE/FALSE. Whether use the cross-validation to measure the prediction accuracy.
#' @param CV_K an integer. The number of folders K-folder cross-validation.
#' @param bootstrap TRUE/FALSE. Whether use the bootstrap method to quantify the uncertainty. The bootstrap option ONLY works for the "ST-GX-D" model.
#' @param bootstrap_B an integer. The number of bootstrap iteration.
#' @param bootstrap_num_epochs an integer. The number of complete passes through the training dataset in the bootstrap procedure.
#' @param U_new TRUE/FALSE. Whether use self defined U for the estimation of single index function, g(U).
#' @param U_min a numeric value. The minimum of the self defined U.
#' @param U_max a numeric value. The maximum of the self defined U.
#' @param random_state an integer. The random seed for initiating the neural network.
#'
#' @return A list consisting of the point estimation, g function estimation (optional), cross-validation results (optional) and bootstrap results(optional).
#'
#' @details
#' The DNNSIM model is defined as:
#' \deqn{Y = g(\mathbf{X} \boldsymbol{\beta}) + e.}
#' The residuals \eqn{e} follow a skewed T distribution, skewed normal distribution, or normal distribution. The single index function \eqn{g} is assumed to be a monotonic increasing function.
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
#'
#'   # set the random seed
#'   set.seed(100)
#'
#'   # simulate some data
#'   df1 <- data_simulation(n=100,beta=c(1,1,1),w=0.3,
#'                          sigma=0.1,delta=10.0,seed=100)
#'
#'   # the cross-validation and bootstrap takes a long time
#'   DNN_model_output <- DNN_model(y ~ X1 + X2 + X3 - 1,
#'                                 data = df1,
#'                                 model = "ST-GX-D",
#'                                 num_epochs = 5,
#'                                 verbatim = FALSE,
#'                                 CV = TRUE,
#'                                 CV_K = 2,
#'                                 bootstrap = TRUE,
#'                                 bootstrap_B = 2,
#'                                 bootstrap_num_epochs = 5,
#'                                 U_new = TRUE,
#'                                 U_min = -4.0,
#'                                 U_max = 4.0)
#'   print(DNN_model_output)
#' }
#' }
#'
#'
DNN_model <- function(formula,
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
                      U_min = -4.0,
                      U_max = 4.0,
                      random_state = 100) {
  # ensure num_epochs to be an integer
  num_epochs <- as.integer(num_epochs)
  bootstrap_num_epochs <- as.integer(bootstrap_num_epochs)

  # ensure data is an dataframe
  data <- as.data.frame(data)

  # creation of the design matrix
  X_mat <- stats::model.matrix(object = formula, data = data)

  # extract the column names
  coltxt <- colnames(X_mat)

  # define the response variable
  y <- as.numeric(data[,as.character(formula[[2]])])

  # define the number of columns of the design matrix
  p <- ncol(X_mat)

  # initialize the model that need be to trained
  Model_to_Train <- DNN_Model$Models(p = p,
                                     hidden_size = as.integer(512),
                                     model = model,
                                     seed = as.integer(random_state))

  # train the model
  DNN_train_output <- DNN_Model$Train_Model(model = Model_to_Train,
                                            XT = X_mat,
                                            yT = y,
                                            num_epochs = num_epochs,
                                            verbatim = verbatim)

  output <- list(DNN_train_output = DNN_train_output)

  # the prediction of g() function
  if (model %in% c("ST-GX-D",
                   "SN-GX-D",
                   "N-GX-D",
                   "ST-GX-B",
                   "SN-GX-B",
                   "N-GX-B")) {
    if (U_new) {
      g_function_estimation <- DNN_Model$Model_prediction_gX(model = Model_to_Train,
                                                             beta_estimation = DNN_train_output$beta_output$cpu()$detach()$numpy(),
                                                             start = U_min,
                                                             stop = U_max,
                                                             num = 100)
    } else {
      g_function_estimation <- DNN_Model$Model_prediction_gX(model = Model_to_Train,
                                                             beta_estimation = DNN_train_output$beta_output$cpu()$detach()$numpy(),
                                                             start = min(DNN_train_output$U_output$cpu()$detach()$numpy()),
                                                             stop = max(DNN_train_output$U_output$cpu()$detach()$numpy()),
                                                             num = 100)
    }

    output[["g_function_estimation"]] <- g_function_estimation
  }

  if (CV) {
    CV_function_output <- DNN_Model$CV_function(XT = X_mat,
                                                yT = y,
                                                model_type = model,
                                                K = CV_K,
                                                num_epochs = num_epochs,
                                                verbatim = verbatim)
    output[["CV_function_output"]] <- CV_function_output
  }

  if (bootstrap & model == "ST-GX-D") {
    ST_GX_D_Bootstrap_output <- DNN_Model$ST_GX_D_Bootstrap(Trained_Model = Model_to_Train,
                                                            DNN_train_output = DNN_train_output,
                                                            XB = X_mat,
                                                            num_epochs = bootstrap_num_epochs,
                                                            num_bootstrap = bootstrap_B,
                                                            verbatim = verbatim)
    output[["ST_GX_D_Bootstrap_output"]] <- ST_GX_D_Bootstrap_output
  }

  DNN_model_output_names <- names(output$DNN_train_output)
  for (j in DNN_model_output_names) {
    try(output$DNN_train_output[[j]] <- output$DNN_train_output[[j]]$cpu()$detach()$numpy(),
        silent = TRUE)
  }

  return(output)
}
