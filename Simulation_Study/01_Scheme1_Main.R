library(lattice)
library(latticeExtra)
library(DNNSIM)

# true g function ---------------------------------------------------------

true_g_func <- function(U) {
    output <- 10.0 * pnorm(q = 0.25*U, mean = 0, sd = 0.1)
    return(output)
}

# data generation ---------------------------------------------------------

X_generation <- function(n) {
    output <- list(X1 = numeric(n),
                   X2 = numeric(n),
                   X3 = numeric(n))
    for (i in 1:n) {
        X1 <- rbinom(n = 1, size = 1, prob = 0.5)
        if (X1 == 0) {
            X2 <- runif(n = 1, min = -3.0, max = 0)
        } else {
            X2 <- runif(n = 1, min = 0, max = 3.0)
        }
        X3 <- runif(n = 1, min = -3.5, max = 2.5)
        output$X1[i] <- X1
        output$X2[i] <- X2
        output$X3[i] <- X3
    }
    return(output)
}

data_generation <- function(n,beta) {
    beta <- beta / norm(beta, "2")
    noises <- GUD::rTPSC(n = n, w = 0.6, theta = 0.0, sigma = 1.5, delta = 6.0)
    X_generation_output <- X_generation(n = n)
    X1 <- X_generation_output$X1
    X2 <- X_generation_output$X2
    X3 <- X_generation_output$X3
    
    U <- X1 * beta[1] + X2 * beta[2] + X3 * beta[3]
    gU <- true_g_func(U)
    y <- gU + noises
    output <- data.frame(y = y, X1 = X1, X2 = X2, X3 = X3, U = U)
    return(output)
}

# simulation function -----------------------------------------------------

simu_point_estimation <- function(seed,n) {
    set.seed(seed)
    num_epochs <- 1000
    
    df1 <- data_generation(n = n,
                           beta = c(1,1,1))
    
    indication <- TRUE
    random_state <- 100 # Initialize random_state
    ST_GX_D_output <- DNN_model(y ~ X1 + X2 + X3 - 1, 
                                data = df1,
                                model = "ST-GX-D",
                                num_epochs = num_epochs,
                                verbatim = FALSE,
                                random_state = random_state)
    
    g_func_bias <- ST_GX_D_output$g_function_estimation$predicted_value - true_g_func(ST_GX_D_output$g_function_estimation$eta)
    g_func_MSE <- mean(g_func_bias^2)
    
    output <- data.frame(beta1 = ST_GX_D_output$DNN_train_output$beta_output[1],
                         beta2 = ST_GX_D_output$DNN_train_output$beta_output[2],
                         beta3 = ST_GX_D_output$DNN_train_output$beta_output[3],
                         w = ST_GX_D_output$DNN_train_output$w_output,
                         sigma = ST_GX_D_output$DNN_train_output$sigma_output,
                         delta = ST_GX_D_output$DNN_train_output$delta_output,
                         g_func_MSE = g_func_MSE,
                         seed = seed)
    
    return(output)
}

simu_bootstrap <- function(seed,n) {
    set.seed(seed)
    num_epochs <- 1000
    bootstrap_B <- 300
    
    df1 <- data_generation(n = n,
                           beta = c(1,1,1))
    
    ST_GX_D_output <- DNN_model(y ~ X1 + X2 + X3 - 1, 
                                data = df1,
                                model = "ST-GX-D",
                                num_epochs = num_epochs,
                                verbatim = FALSE,
                                bootstrap = TRUE,
                                bootstrap_B = bootstrap_B,
                                bootstrap_num_epochs = num_epochs,
                                random_state = 100)
    
    output <- ST_GX_D_output
    
    return(output)
}
