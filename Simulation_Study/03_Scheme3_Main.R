library(lattice)
library(latticeExtra)
library(DNNSIM)
library(ald)

# Part II of Scheme 3 is abandoned. 
# Simulate data from a single index model is unfair for the ST-FX model.

# true g function ---------------------------------------------------------

true_g_func <- function(U) {
    output <- floor(U)
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
    n1 <- as.integer(n*0.99)
    n2 <- n - n1
    noises <- c(ald::rALD(n=n1,mu=0,sigma = 0.5,p=0.6),rnorm(n=n2,mean=7,sd=0.1))
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

simu_beta <- function(seed,n,model) {
    
    set.seed(seed)
    num_epochs <- 1000
    
    df1 <- data_generation(n = n,
                           beta = c(1,1,1))
    
    indication <- TRUE
    random_state <- 100 # Initialize random_state
    DNN_model_output <- DNN_model(y ~ X1 + X2 + X3 - 1, 
                                  data = df1,
                                  model = model,
                                  num_epochs = num_epochs,
                                  verbatim = FALSE,
                                  random_state = random_state)
    
    output <- DNN_model_output
    
    return(output)
}

simu_CV <- function(seed,n,model) {
    if (! model %in% c("ST-GX-D","ST-GX-B","ST-FX")) {
        stop("Check model type!")
    }
    
    set.seed(seed)
    num_epochs <- 1000
    
    df1 <- data_generation(n = n,
                           beta = c(1,1,1))
    
    indication <- TRUE
    random_state <- 100 # Initialize random_state
    DNN_model_output <- DNN_model(y ~ X1 + X2 + X3 - 1, 
                                  data = df1,
                                  model = model,
                                  num_epochs = num_epochs,
                                  verbatim = FALSE,
                                  random_state = random_state,
                                  CV = TRUE)
    
    output <- DNN_model_output
    
    return(output)
}
