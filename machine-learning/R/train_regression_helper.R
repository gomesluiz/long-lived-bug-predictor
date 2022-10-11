library(caret)            # knn
library(elasticnet)       # enet 
library(earth)            # earth 
library(gbm)              # gbm
library(kernlab)          # svm 
library(MASS)             # rlm  
library(nnet)             # nnet and avNNet
library(pls)              # pls
library(plyr)             # plyr
library(rpart)            # rpart2 
library(RWeka)            # M5 
library(randomForest)     # rf 

library(futile.logger)

KNN <- 'knn'
ENE <- 'enet' 
EAR <- 'earth' 
GBM <- 'gbm'
SVM <- 'svm' 
RLM <- 'rlm'  
NNE <- 'nnet' 
AVN <- 'avNNet'
PLS <- 'pls'
RP2 <- 'rpart2' 
M5_ <- 'M5' 
LM_ <- 'lm' 
RF_ <- 'rf' 

train_regressors <- c(KNN, ENE, EAR, GBM, SVM, RLM, NNE, AVN, PLS, RP2, M5_, RF_)
names(train_regressors) <- train_regressors
DEFAULT_CONTROL <- trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")

#' Training regression model with KNN. 
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained knn regression model.
train_regressor_with_knn <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  library(caret)            # knn
  flog.trace("[train_regressor_with_knn] Training regressor model with %s", KNN)
  
  grid <- expand.grid(
    k = c(seq(5, 10, by=1))
  )

  result <- train(
    x = .x,
    y = .y,
    method = "knn",
    trControl = .control,
    tuneGrid  = grid
  )

  return(result)
}

#' Training regression model with ENET. 
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained knn regression model.
train_regressor_with_ene <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  library(elasticnet)       # enet 
  flog.trace("[train_with_ene] Training model with %s", ENE)
  
  grid <- expand.grid(
    .lambda   = c(0, 0.01, 1),
    .fraction = seq(.05, 1, length = 20) 
  )
  
  result <- train(
    x = .x,
    y = .y,
    method = "enet",
    trControl = .control,
    tuneGrid  = grid
  )
  
  return(result)
}

#' Training regression model with EARTH
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained regression model.
train_regressor_with_ear <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  library(earth)            # earth 
  flog.trace("[train_with_ear] Training model with %s", EAR)
  
  grid <- expand.grid(
    .degree = 1:2,
    .nprune = 2:38
  )  
  
  set.seed(100)
  result <- train(
    x = .x,
    y = .y,
    method    = "earth",
    trControl = .control,
    tuneGrid  = grid
  )

  return(result)
}

#' Training regression model with GBM
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained regression model.
train_regressor_with_gbm <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  library(gbm) # gbm             
  flog.trace("[train_with_gbm] Training model with %s", GBM)
  
  grid <- expand.grid(
    .interaction.depth = seq(1, 7, by=2), 
    .n.trees   = seq(100, 1000, by=50), 
    .shrinkage = c(0.01, 0.1),
    .n.minobsinnode = seq(10, 20, by=1)
  )  
  
  result <- train(
    x = .x,
    y = .y,
    method    = "gbm",
    trControl = .control,
    tuneGrid  = grid,
    verbose   = FALSE
  )
  
  return(result)
}

#' Training model with SVM RBF
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_regressor_with_svm <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  flog.trace("[train_with_svm] Training model with SVM")
  
  grid <- expand.grid(
    C = c(
      2**(-5), 2**(0), 2**(5), 2**(10)
    ),
    sigma = c(
      2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)
    )
  )
  
  result <- train(
    x = .x,
    y = .y,
    method    = "svmRadial",
    trControl = .control,
    tuneGrid  = grid
  )

  return(result)
}
#' Training model with SVM RBF
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_regressor_with_yyyy <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  flog.trace("[train_with_svm] Training model with SVM")
  
  grid <- expand.grid(
    C = c(
      2**(-5), 2**(0), 2**(5), 2**(10)
    ),
    sigma = c(
      2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)
    )
  )
  
  result <- train(
    x = .x,
    y = .y,
    method    = "svmRadial",
    trControl = .control,
    tuneGrid  = grid
  )

  return(result)
}

#' Training model with SVM RBF
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_regressor_with_zzzz <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  flog.trace("[train_with_svm] Training model with SVM")
  
  grid <- expand.grid(
    C = c(
      2**(-5), 2**(0), 2**(5), 2**(10)
    ),
    sigma = c(
      2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)
    )
  )
  
  result <- train(
    x = .x,
    y = .y,
    method    = "svmRadial",
    trControl = .control,
    tuneGrid  = grid
  )

  return(result)
}

#' Training model with neural network.
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_with_nnet <- function(.x, .y, .control=DEFAULT_CONTROL) {
  flog.trace("[train_with_nnet] Training model with NNET")
  
  grid <- expand.grid(
    size  = c(10, 20, 30, 40), 
    decay = (0.5)
  )
  
  result <- train(
    x = .x,
    y = .y,
    method = "nnet",
    trControl = .control,
    tuneGrid  = grid,
    MaxNWts   = 5000,
    verbose   = FALSE
  )
  
  return(result)
}

#' Training model with neural network.
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_with_rf <- function(.x, .y, .control=DEFAULT_CONTROL) {
  flog.trace("[train_with_rf] Training model with RF")
  
  grid <- expand.grid(
    mtry = c(10, 15, 20, 25)
  )

  result <- train(
    x = .x,
    y = .y,
    method = "rf",
    trControl = .control,
    tuneGrid  = grid,
    ntree   = 200,
    verbose = FALSE
  )
  
  return(result)
}


#' Training model with Näive Bayes. 
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_with_nb <- function(.x, .y, .control=DEFAULT_CONTROL) {
  
  flog.trace("[train_with_nb] Training model with NB")
  
  grid <- expand.grid(
    fL        = c(0, 0.5, 1.0),
    usekernel = TRUE, 
    adjust    = c(0, 0.5, 1.0)
  )

  result <- train(
    x = .x,
    y = .y,
    method = "nb",
    trControl = .control,
    tuneGrid  = grid
  )

  return(result)
}

#' Training model with a classifier pass as parameter. 
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_regressor_with <- function(.x, .y, .regressor, .control=DEFAULT_CONTROL) {
  if (!.regressor %in% train_regressors)
  {
    stop(sprintf("%s unknown regressor!", .regressor))
    
  }
  
  if (.regressor == KNN) {
    return(train_regressor_with_knn(.x, .y, .control))
  } else if (.regressor == ENE) {
    return(train_regressor_with_ene(.x, .y, .control))
  } else if (.regressor == EAR) {
    return(train_regressor_with_ear(.x, .y, .control))
  } else if (.regressor == GBM) {
    return(train_regressor_with_gbm(.x, .y, .control))
  } else if (.regressor == SVM) {
    return(train_regressor_with_svm(.x, .y, .control))
  } 
}
