library(lattice)
library(latticeExtra)
library(DNNSIM)

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

data_generation <- function(n) {
    noises <- GUD::rTPSC(n = n, w = 0.6, theta = 0.0, sigma = 1.5, delta = 6.0)
    X_generation_output <- X_generation(n = n)
    X1 <- X_generation_output$X1
    X2 <- X_generation_output$X2
    X3 <- X_generation_output$X3
    
    U <- X1 + X2 + X3
    gU <- U
    y <- gU + noises
    output <- data.frame(y = y, X1 = X1, X2 = X2, X3 = X3, U = U)
    return(output)
}

# simulation function -----------------------------------------------------

simu_CV <- function(seed,n,model) {
    if (! model %in% c("ST-GX-D","ST-GX-B","ST-FX")) {
        stop("Check model type!")
    }
    
    set.seed(seed)
    num_epochs <- 1000
    
    df1 <- data_generation(n = n)
    
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

# run the simulation ------------------------------------------------------

simu_n <- 100
n <- 1000

simu_output <- vector("list",simu_n)

current_seed <- 1
count <- 1

while (count <= simu_n) {
    result <- tryCatch({simu_CV(seed = current_seed, n = n, model = model)},
                       error = function(e){
                           NULL
                       })
    if (! is.null(result)) {
        simu_output[[count]] <- result
        message(sprintf("Simulation with seed %d succeeded.", current_seed))
        current_seed <- current_seed + 1
        count <- count + 1
    } else {
        message(sprintf("Simulation with seed %d failed.", current_seed))
        current_seed <- current_seed + 1
    }
}

saveRDS(simu_output,paste0("./04_Scheme4/simu_output_CV_model_",model,"_n_",n,".RDS"))
