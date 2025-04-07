import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from ST_loss import loss as loss_ST
from SN_loss import loss as loss_SN
from MISC import bernstein_matrix
from sklearn.model_selection import KFold
from ST_RNG import rTPSC

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = "cuda"
else :
    device = "cpu"

# set random seed
np.random.seed(100)
torch.manual_seed(100)

class Models(nn.Module):
    """
    This class contains 9 models.
    Y = g(X*beta) + e.
    Y = f(X) + e.
    Two method to approximate g(), the DNN method and the Bernstein Method.
    One method to approximate f(), the DNN method.
    The residual e follows the N, SN or ST distribution.
    """
    def __init__(self, p, hidden_size, model, seed = None):
        super().__init__()
        
        # set random seed
        if seed is None:
            torch.manual_seed(100)
        else:
            try: 
                seed = int(seed)
            except:
                print("Seed must be an integer or None. Use seed = 100 instead.")
                torch.manual_seed(100)
        
        # the model type, must be one of "N-GX-D","SN-GX-D","ST-GX-D",
        # "N-GX-B","SN-GX-B","ST-GX-B","N-FX","SN-FX","ST-FX".
        self.model = model
        
        # model dimensions
        p = int(p)
        self.p = p
        
        # acitivation function
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        
        # define the model input
        model_input_ones = torch.ones((int(1e2),hidden_size)) * 1e-2
        model_input_ones = model_input_ones.to(device, dtype = torch.float32)
        self.model_input_ones = model_input_ones
        
        # define layers for the estimation of beta
        self.betain = nn.Linear(hidden_size, hidden_size)
        self.beta1 = nn.Linear(hidden_size, hidden_size)
        self.beta2 = nn.Linear(hidden_size, hidden_size)
        self.betaout = nn.Linear(hidden_size, p)
        
        # define layers for the estimation of sigma
        self.sigmain = nn.Linear(hidden_size, hidden_size)
        self.sigma1 = nn.Linear(hidden_size, hidden_size)
        self.sigma2 = nn.Linear(hidden_size, hidden_size)
        self.sigmaout = nn.Linear(hidden_size, 1)
        
        # for SN and ST
        if not (model == "N-GX-D" or model == "N-GX-B" or model == "N-FX"):
            # define layers for the estimation of w
            self.win = nn.Linear(hidden_size, hidden_size)
            self.w1 = nn.Linear(hidden_size, hidden_size)
            self.w2 = nn.Linear(hidden_size, hidden_size)
            self.wout = nn.Linear(hidden_size, 1)
        
        # for ST only
        if model == "ST-GX-D" or model == "ST-GX-B" or model == "ST-FX":
            # define layers for the estimation of delta
            self.deltain = nn.Linear(hidden_size, hidden_size)
            self.delta1 = nn.Linear(hidden_size, hidden_size)
            self.delta2 = nn.Linear(hidden_size, hidden_size)
            self.deltaout = nn.Linear(hidden_size, 1)
        
        # for g() DNN only
        if model == "ST-GX-D" or model == "SN-GX-D" or model == "N-GX-D":
            # define layers for the estimation of the single-index function
            self.SIMin = nn.Linear(1, hidden_size)
            self.SIM1 = nn.Linear(hidden_size, hidden_size)
            self.SIM2 = nn.Linear(hidden_size, hidden_size)
            self.SIMout = nn.Linear(hidden_size, 1)

        # for g() Bernstein only
        if model == "ST-GX-B" or model == "SN-GX-B" or model == "N-GX-B":
            self.L = 50
            # define layers for the estimation of the single-index function
            self.SIMin = nn.Linear(hidden_size, hidden_size)
            self.SIM1 = nn.Linear(hidden_size, hidden_size)
            self.SIM2 = nn.Linear(hidden_size, hidden_size)
            self.SIMout = nn.Linear(hidden_size, self.L + 1)
        
        # for f() only
        if model == "ST-FX" or model == "SN-FX" or model == "N-FX":
            # define layers for the estimation of f(X)
            self.fin = nn.Linear(p, hidden_size)
            self.f1 = nn.Linear(hidden_size, hidden_size)
            self.f2 = nn.Linear(hidden_size, hidden_size)
            self.fout = nn.Linear(hidden_size, 1)
    
    def forward(self, X):
        # for g() only
        if self.model == "ST-GX-D" or self.model == "ST-GX-B" or self.model == "SN-GX-D" or self.model == "SN-GX-B" or self.model == "N-GX-D" or self.model == "N-GX-B":
            # beta estimation
            out1 = self.ReLU(self.betain(self.model_input_ones))
            out1 = self.ReLU(self.beta1(out1))
            out1 = self.ReLU(self.beta2(out1))
            out1 = self.betaout(out1)
            out1 = torch.mean(out1,0)
            out1 = out1 / torch.linalg.norm(out1)
        
        # for g()-DNN only
        if self.model == "ST-GX-D" or self.model == "SN-GX-D" or self.model == "N-GX-D":
            # the single index function estimation
            eta = X @ out1
            X_nrow = X.shape[0]
            eta = eta.reshape((X_nrow,1))
            out5 = self.Tanh(self.SIMin(eta))
            out5 = self.Tanh(self.SIM1(out5))
            out5 = self.Tanh(self.SIM2(out5))
            out5 = self.SIMout(out5)
            out5 = out5.flatten()
        
        # for g()-Bernstein only
        if self.model == "ST-GX-B" or self.model == "SN-GX-B" or self.model == "N-GX-B":
            eta = X @ out1
            eta = eta.flatten()
            B_mat = bernstein_matrix(L = self.L, x = eta)
            out7 = self.ReLU(self.SIMin(self.model_input_ones))
            out7 = self.ReLU(self.SIM1(out7))
            out7 = self.ReLU(self.SIM2(out7))
            out7 = self.SIMout(out7)
            out7 = torch.mean(out7,axis = 0)
            out7 = out7.flatten()
            # ensure monotocity
            out7, _ = torch.sort(out7)
            
            # the single index function estimation
            out5 = (B_mat @ out7).flatten()
        
        # for f() only
        if self.model == "ST-FX" or self.model == "SN-FX" or self.model == "N-FX":
            out6 = self.ReLU(self.fin(X))
            out6 = self.ReLU(self.f1(out6))
            out6 = self.ReLU(self.f2(out6))
            out6 = self.fout(out6).flatten()
        
        # for SN and ST
        if not (self.model == "N-GX-D" or self.model == "N-GX-B" or self.model == "N-FX"):
            # w estimation
            out2 = self.ReLU(self.win(self.model_input_ones))
            out2 = self.ReLU(self.w1(out2))
            out2 = self.ReLU(self.w2(out2))
            out2 = self.wout(out2)
            out2 = torch.mean(out2)
            out2 = 1.0 / (1.0 + torch.exp(-1.0 * out2))
        
        # sigma estimation
        out3 = self.ReLU(self.sigmain(self.model_input_ones))
        out3 = self.ReLU(self.sigma1(out3))
        out3 = self.ReLU(self.sigma2(out3))
        out3 = self.sigmaout(out3)
        out3 = torch.mean(out3)
        out3 = torch.exp(out3)
        
        # for ST only
        if self.model == "ST-GX-D" or self.model == "ST-GX-B" or self.model == "ST-FX":
            # delta estimation
            out4 = self.ReLU(self.deltain(self.model_input_ones))
            out4 = self.ReLU(self.delta1(out4))
            out4 = self.ReLU(self.delta2(out4))
            out4 = self.deltaout(out4)
            out4 = torch.mean(out4)
            # use expit function to ensure numerical stability
            # ensure the degree of freedom is no less than 2
            out4 = torch.special.expit(out4) * 198.0 + 2.0
        
        # out1:beta
        # out2:skewness
        # out3:standard deviation
        # out4:degree of freedom
        # out5:g()
        # out6:f()
        # out7:coefficients associated with the Bernstein polynomials
        
        # use theta to denote the mode(fixed effect)
        if self.model == "ST-GX-D":
            output = {"beta":out1,
                      "w":out2,
                      "sigma":out3,
                      "delta":out4,
                      "theta":out5}
        
        if self.model == "SN-GX-D":
            output = {"beta":out1,
                      "w":out2,
                      "sigma":out3,
                      "theta":out5}
            
        if self.model == "N-GX-D":
            output = {"beta":out1,
                      "sigma":out3,
                      "theta":out5}
            
        if self.model == "ST-GX-B":
            output = {"beta":out1,
                      "w":out2,
                      "sigma":out3,
                      "delta":out4,
                      "theta":out5,
                      "Bernstein polynomials coefficients":out7}
            
        if self.model == "SN-GX-B":
            output = {"beta":out1,
                      "w":out2,
                      "sigma":out3,
                      "theta":out5,
                      "Bernstein polynomials coefficients":out7}
            
        if self.model == "N-GX-B":
            output = {"beta":out1,
                      "sigma":out3,
                      "theta":out5,
                      "Bernstein polynomials coefficients":out7}
        
        if self.model == "ST-FX":
            output = {"w":out2,
                      "sigma":out3,
                      "delta":out4,
                      "theta":out6}
            
        if self.model == "SN-FX":
            output = {"w":out2,
                      "sigma":out3,
                      "theta":out6}
        
        if self.model == "N-FX":
            output = {"sigma":out3,
                      "theta":out6}
        
        return output

def Train_Model(model,XT,yT,num_epochs,verbatim):
    """
    The program to train the proposed DNN model (no generative bootstrap).

    Parameters
    ----------
    model : Models
        The models that need training.
    XT : torch.Tensor
        The design matrix.
    yT : torch.Tensor
        The response vector.
    num_epochs : int
        The number of times the optimazation algroithm sees the entire data set.
    verbatim : bool
        The indication variable that decides whether print log messages to the console.

    Returns
    -------
    output : dict
        A dictionary containing the point estimations and residuals.

    """
    
    # set random set
    torch.manual_seed(100)
    # make copy for X and y
    X = copy.deepcopy(XT)
    y = copy.deepcopy(yT)
    if type(y) == list:
        y = np.array(y)
    
    # convert model to an appropriate device, CPU or GPU
    model = model.to(device)
    
    # ensure X,y are torch.Tensor
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
        X = X.to(device = device, dtype = torch.float32)
    if type(y) == np.ndarray:
        y = torch.from_numpy(y)
        y = y.to(device = device, dtype = torch.float32)
        
    # ensure num_epochs is an integer
    num_epochs = int(num_epochs)
    
    # create the training dataset
    train_dataset = TensorDataset(X, y)
    
    # mini-batch creation
    if X.shape[0] <= 25000:
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=X.shape[0],
                                      shuffle=True)
    else :
        train_dataloader = DataLoader(train_dataset,
                                      batch_size = 25000,
                                      shuffle=True)
    
    # initilize the optimizer
    lr0 = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = lr0)
    decayRate = 0.999
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, 
                                                             gamma=decayRate)
    
    # record loss function
    loss_list = [None] * num_epochs
    for epoch in range(1,num_epochs+1):
        loss_epoch = np.zeros(1)
        if verbatim:
            print(f"Epoch {epoch}\n-------------------------------")
        for batch, (XB, yB) in enumerate(train_dataloader):
            model_output = model(XB)
            
            if model.model == "ST-GX-D":
                beta_output = model_output["beta"]
                w_output = model_output["w"]
                sigma_output = model_output["sigma"]
                delta_output = model_output["delta"]
                SIM_output = model_output["theta"]
        
                # define loss_value
                loss_value = loss_ST(x = yB,
                                     w = w_output,
                                     theta = SIM_output,
                                     sigma = sigma_output,
                                     delta = delta_output)
            
            if model.model == "SN-GX-D":       
                beta_output = model_output["beta"]
                w_output = model_output["w"]
                sigma_output = model_output["sigma"]
                SIM_output = model_output["theta"]
        
                # define loss_value
                loss_value = loss_SN(x = yB,
                                     w = w_output,
                                     theta = SIM_output,
                                     sigma = sigma_output)
            
            if model.model == "N-GX-D":
                beta_output = model_output["beta"]
                sigma_output = model_output["sigma"]
                SIM_output = model_output["theta"]
        
                # define loss_value
                loss_value = (0.5 * torch.square(yB - SIM_output)) / torch.square(sigma_output) + torch.log(sigma_output)
            
            if model.model == "ST-GX-B":
                w_output = model_output["w"]
                delta_output = model_output["delta"]
                beta_output = model_output["beta"]
                sigma_output = model_output["sigma"]
                SIM_output = model_output["theta"]
                Bernstein_output = model_output["Bernstein polynomials coefficients"]
        
                # define loss_value
                loss_value = loss_ST(x = yB,
                                     w = w_output,
                                     theta = SIM_output,
                                     sigma = sigma_output,
                                     delta = delta_output)
            
            if model.model == "SN-GX-B":
                w_output = model_output["w"]
                beta_output = model_output["beta"]
                sigma_output = model_output["sigma"]
                SIM_output = model_output["theta"]
                Bernstein_output = model_output["Bernstein polynomials coefficients"]
        
                # define loss_value
                loss_value = loss_SN(x = yB,
                                     w = w_output,
                                     theta = SIM_output,
                                     sigma = sigma_output)
            
            if model.model == "N-GX-B":
                beta_output = model_output["beta"]
                sigma_output = model_output["sigma"]
                SIM_output = model_output["theta"]
                Bernstein_output = model_output["Bernstein polynomials coefficients"]
        
                # define loss_value
                loss_value = (0.5 * torch.square(yB - SIM_output)) / torch.square(sigma_output) + torch.log(sigma_output)
            
            if model.model == "ST-FX":
                delta_output = model_output["delta"]
                w_output = model_output["w"]
                sigma_output = model_output["sigma"]
                fX_output = model_output["theta"]
        
                # define loss_value
                loss_value = loss_ST(x = yB,
                                     w = w_output,
                                     theta = fX_output,
                                     sigma = sigma_output,
                                     delta = delta_output)
            
            if model.model == "SN-FX":
                w_output = model_output["w"]
                sigma_output = model_output["sigma"]
                fX_output = model_output["theta"]
        
                # define loss_value
                loss_value = loss_SN(x = yB,
                                     w = w_output,
                                     theta = fX_output,
                                     sigma = sigma_output)
            
            if model.model == "N-FX":
                sigma_output = model_output["sigma"]
                fX_output = model_output["theta"]
        
                # define loss_value
                loss_value = (0.5 * torch.square(yB - fX_output)) / torch.square(sigma_output) + torch.log(sigma_output)
            
            loss_sum = torch.sum(loss_value)
            # cumulate loss in this batch
            loss_epoch = loss_epoch + loss_sum.detach().cpu().numpy()
            # back propogation
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            my_lr_scheduler.step()
            
            if model.model == "ST-GX-D" or model.model == "SN-GX-D" or model.model == "N-GX-D":
                # ensure weights associated with the single index functoin are non-negative
                for name, param in model.named_parameters():
                    if ("weight" in name) and ("SIM" in name):
                        param.data.clamp_(min=0.0)
        
        # record loss in this epoch
        loss_list[epoch-1] = loss_epoch
    
    if model.model in {"ST-GX-D", "SN-GX-D", "N-GX-D", "ST-GX-B", "SN-GX-B", "N-GX-B"}:
        # calculate the SIM using all X
        SIM_output = model(X)["theta"]
    
    if not(model.model == "ST-FX" or model.model == "SN-FX" or model.model == "N-FX"):
        # calculate U = X*beta, the index
        U_output = (X @ beta_output).flatten()
    
        # calculate the residuals
        residuals_output = (y - SIM_output).flatten()
    
    if model.model == "ST-FX" or model.model == "SN-FX" or model.model == "N-FX":
        # calculate the f(X) using all X
        fX_output = model(X)["theta"]
        fX_output = fX_output.flatten()
        
        # calculate the residuals
        residuals_output = (y - fX_output).flatten()
    
    # defind loss output
    loss_output = np.array(loss_list).flatten()
    
    if model.model == "ST-GX-D":
        # define the output
        output = {"beta_output":beta_output,
                  "w_output":w_output,
                  "sigma_output":sigma_output,
                  "delta_output":delta_output,
                  "SIM_output":SIM_output,
                  "U_output": U_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "SN-GX-D":
        # define the output
        output = {"beta_output":beta_output,
                  "w_output":w_output,
                  "sigma_output":sigma_output,
                  "SIM_output":SIM_output,
                  "U_output": U_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "N-GX-D":
        # define the output
        output = {"beta_output":beta_output,
                  "sigma_output":sigma_output,
                  "SIM_output":SIM_output,
                  "U_output": U_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "ST-GX-B":
        # define the output
        output = {"beta_output":beta_output,
                  "w_output":w_output,
                  "sigma_output":sigma_output,
                  "delta_output":delta_output,
                  "SIM_output":SIM_output,
                  "Bernstein_output":Bernstein_output,
                  "U_output": U_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "SN-GX-B":
        # define the output
        output = {"beta_output":beta_output,
                  "w_output":w_output,
                  "sigma_output":sigma_output,
                  "SIM_output":SIM_output,
                  "Bernstein_output":Bernstein_output,
                  "U_output": U_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "N-GX-B":
        # define the output
        output = {"beta_output":beta_output,
                  "sigma_output":sigma_output,
                  "SIM_output":SIM_output,
                  "Bernstein_output":Bernstein_output,
                  "U_output": U_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "ST-FX":
        output = {"w_output":w_output,
                  "sigma_output":sigma_output,
                  "delta_output":delta_output,
                  "fX_output":fX_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "SN-FX":
        output = {"w_output":w_output,
                  "sigma_output":sigma_output,
                  "fX_output":fX_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    if model.model == "N-FX":
        output = {"sigma_output":sigma_output,
                  "fX_output":fX_output,
                  "residuals_output": residuals_output,
                  "loss_output": loss_output}
    
    output["model_type"] = model.model
    
    return output

def Model_prediction_gX(model,beta_estimation,start,stop,num):
    """
    The function to estimate the single index function using a trained DNN model.
    
    Parameters
    ----------
    model : Models
        The trained y = g(X * beta) + e model.
    beta_estimation : torch.Tensor
        The point estimation of beta.
    start : float
        The starting value of the eta vector.
    stop : float
        The ending value of the eta vector.
    num : int
        Number of etas to generate..

    Returns
    -------
    output : dict
        A dictionary consisting of eta vector and associated predicted value.

    """
    p = model.p
    num = int(num)
    X = np.zeros(shape = (num,p))
    if type(beta_estimation) == torch.Tensor :
        beta_estimation = beta_estimation.cpu().detach().numpy()
    beta0 = beta_estimation[0]
    X[:,0] = (X[:,0] + 1.0) / beta0
    X[:,0] = X[:,0] * np.linspace(start = start, stop = stop, num = num)
    X = torch.from_numpy(X)
    X = X.to(device, dtype = torch.float32)
    model_output = model(X)
    
    if model.model in {"ST-GX-D", "SN-GX-D", "N-GX-D", "ST-GX-B", "SN-GX-B", "N-GX-B"}:
        SIM_output = model_output["theta"]

    output = {"eta": np.linspace(start = start, stop = stop, num = num),
              "predicted_value": SIM_output.cpu().detach().numpy()}
    return output

def CV_function(XT,yT,model_type,K,num_epochs,verbatim):
    """
    Function for K-folder Cross-Validation.

    Parameters
    ----------
    XT : torch.Tensor
        The design matrix.
    yT : torch.Tensor
        The response vector.
    model_type : str
        The type of model. Must be one of ST-GX-D, SN-GX-D, N-GX-D, ST-GX-B, SN-GX-B, N-GX-B, ST-FX, SN-FX, N-FX.
    K: int
        The K of the K-folder Cross-Validation
    num_epochs: int
        The number of epochs for training the model.
    verbatim : bool
        The indication variable that decides whether print log messages to the console.
    
    Returns
    -------
    output : dict
        The dictionary of the output consisting of MSE and MAE.

    """
    # ensure model type is correct
    valid_model_types = {"ST-GX-D", "SN-GX-D", "N-GX-D", "ST-GX-B", "SN-GX-B", "N-GX-B", "ST-FX", "SN-FX", "N-FX"}
    if model_type not in valid_model_types:
        print("model_type must be one of ST-GX-D, SN-GX-D, N-GX-D, ST-GX-B, SN-GX-B, N-GX-B, ST-FX, SN-FX, N-FX.")
        return
    
    X = copy.deepcopy(XT)
    y = copy.deepcopy(yT)
    y = np.array(y)
    
        # ensure X,y are torch.Tensor
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
        X = X.to(device = device, dtype = torch.float32)
    if type(y) == np.ndarray:
        y = torch.from_numpy(y)
        y = y.to(device = device, dtype = torch.float32)
    
    # setting of the DNN models
    hidden_size = 512
    p = X.shape[1]
    
    # initiate the function's output
    K = int(K)
    MSE_output = [None] * K
    MAE_output = [None] * K
    
    kf = KFold(n_splits=K,shuffle=True,random_state=100)

    for CV_iter, (train_index, test_index) in enumerate(kf.split(X)):
        print("Cross Validation: ", CV_iter + 1, "of ", K)
        model_DNN = Models(p,hidden_size,model_type).to(device)

        X_train = X[train_index,:]
        X_test = X[test_index,:]
        y_train = y[train_index]
        y_test = y[test_index]
        
        ## model_DNN is trained with the code below and 
        ## the output `DNN_train_output` will not be used later
        DNN_train_output = Train_Model(model = model_DNN, 
                                       XT = X_train,
                                       yT = y_train,
                                       num_epochs = num_epochs,
                                       verbatim = verbatim)

        model_DNN_output = model_DNN(X_test)

        fixed_effect_prediction = model_DNN_output["theta"]

        residual_test_set = fixed_effect_prediction.detach().cpu().numpy() - y_test.detach().cpu().numpy()
        residual_test_MSE = np.mean(np.square(residual_test_set))
        residual_test_MAE = np.mean(np.abs(residual_test_set))
        MSE_output[CV_iter] = residual_test_MSE
        MAE_output[CV_iter] = residual_test_MAE

    output = {"CV_MSE":np.array(MSE_output),
              "CV_MAE":np.array(MAE_output)}
    
    return output 

def ST_GX_D_Bootstrap_Single_Iteration(Trained_Model,DNN_train_output,XB,num_epochs,seed,verbatim):
    """
    The function for the parametric bootstrap procedure for one iteration. 
    This function is ONLY suitable for the ST-GX-D model.

    Parameters
    ----------
    Trained_Model: DNN_Model.Models
        The trained DNN model.
    DNN_train_output : dict
        The trained ST-GX-D model's output.
    XB : torch.Tensor
        The design matrix.
    num_epochs : int
        The number of epochs for training the model.
    seed : int
        The random seed.
    verbatim : bool
        The indication variable that decides whether print log messages to the console.

    Returns
    -------
    output : dict
        The dictionary containing the output (point estimation) of one bootstrap iteration.

    """
    ## ensure seed is an integer
    seed = int(seed)
    print("Bootstrap seed: ", str(seed))
    ## extract DNN_train_output, which is based on the original sample/data, not on the bootstrap sample.
    U_output = DNN_train_output["U_output"].detach().cpu().numpy()
    SIM_output = DNN_train_output["SIM_output"].detach()
    w_output = DNN_train_output["w_output"].detach().cpu().numpy()
    sigma_output = DNN_train_output["sigma_output"].detach().cpu().numpy()
    delta_output = DNN_train_output["delta_output"].detach().cpu().numpy()
    ## generate noise
    ### the sample size
    n = SIM_output.shape[0]
    ### generate noises
    noises = rTPSC(n = n, w = w_output, theta = np.zeros((n,)), 
                   sigma = sigma_output, delta = delta_output, 
                   seed = seed)
    noises = torch.from_numpy(noises).to(device = device, dtype = torch.float32)
    ## generate the response variable    
    yB = SIM_output + noises
    ## option 1 (abandoned): used the pre-trained model as the initial model.
    ## model_DNN = copy.deepcopy(Trained_Model)
    ## option 2: initilize a different model
    hidden_size = 512
    p = XB.shape[1]
    model_DNN = Models(p,hidden_size,"ST-GX-D").to(device)

    Train_Model_output = Train_Model(model = model_DNN, 
                                     XT = XB,
                                     yT = yB,
                                     num_epochs = num_epochs,
                                     verbatim = verbatim)
    ## estimation of g() function based on the bootstrap sample
    g_output = Model_prediction_gX(model = model_DNN,
                                   beta_estimation = Train_Model_output["beta_output"].detach().cpu().numpy(),
                                   start = np.min(U_output),
                                   stop = np.max(U_output),
                                   num = 100)

    output = {"SIM_output":Train_Model_output["SIM_output"].detach().cpu().numpy(),
              "beta_output":Train_Model_output["beta_output"].detach().cpu().numpy(),
              "w_output":Train_Model_output["w_output"].detach().cpu().numpy(),
              "sigma_output":Train_Model_output["sigma_output"].detach().cpu().numpy(),
              "delta_output":Train_Model_output["delta_output"].detach().cpu().numpy(),
              "loss_output":Train_Model_output["loss_output"],
              "g_output":g_output}
    return output

def ST_GX_D_Bootstrap(Trained_Model,DNN_train_output,XB,num_epochs,num_bootstrap,verbatim):
    """
    The function for the parametric bootstrap procedure for the ST-GX-D model.

    Parameters
    ----------
    Trained_Model: DNN_Model.Models
        The trained DNN model.
    DNN_train_output : dict
        The trained ST-GX-D model's output.
    XB : torch.Tensor
        The design matrix.
    num_epochs : int
        The number of epochs for training the model.
    num_bootstrap : int
        The number of bootstrap iterations.
    verbatim : bool
        The indication variable that decides whether print log messages to the console.

    Returns
    -------
    output : list
        The list containing all bootstrap point estimation.

    """
    ## ensure the following two are integers
    num_epochs = int(num_epochs)
    num_bootstrap = int(num_bootstrap)
    
    ## define output
    output = [None] * num_bootstrap
    
    count = 0
    seed = 0
    ## begin the bootstrap
    while count < num_bootstrap:
        try:
            output[count] = ST_GX_D_Bootstrap_Single_Iteration(Trained_Model = Trained_Model,
                                                               DNN_train_output = DNN_train_output,
                                                               XB = XB,
                                                               num_epochs = num_epochs,
                                                               seed = seed,
                                                               verbatim = verbatim)
            count = count + 1
            seed = seed + 1
        except:
            print("Bootstrap Failed ! Seed: ", str(seed))
            seed = seed + 1
    
    return output
