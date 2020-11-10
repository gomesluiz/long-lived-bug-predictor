library(futile.logger)

# algorithms
KNN  <- "knn"
NB   <- "nb"
NNET <- "nn"
RF   <- "rf"
SVM  <- "svm"
XB   <- "xgbTree"

train_classifiers <- c(
  KNN, 
  NB,
  NNET,
  RF, 
  SVM,
  XB 
)
names(train_classifiers) <- train_classifiers

# metrics 
ACC <- "Accuracy"
KPP <- "Kappa"
ROC <- "ROC"
DST <- "Dist"

train_metrics <- c(
  ACC,
  KPP,
  ROC,
  DST
)

DEFAULT_SEED    <- 144
DEFAULT_CONTROL <- trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
DEFAULT_PREPROC <- c("BoxCox","center", "scale", "pca")
# "nter": subtract mean from values.
# "scale" : divide values by standard deviation.
# ("center", "scale"): combining the scale and center transforms will 
#   standarize your data. Attributes will have a mean value of 0 and standard
#   deviation of 1.
# "range" : normalize values. Data values can be scaled in the range of [0, 1]
#    which is called normalization.

fourStats <- function(data, lev=levels(data$obs), model=NULL)
{
  # This code will get use the area under the ROC curve and the sensitivity and 
  # specifity values using the current candidate value of the probability threshold.
  out <- c(twoClassSummary(data, lev=levels(data$obs), model=NULL))
  
  # the best possible model has sensitivity of 1 and specificity of 1.
  # How far are we from that value.
  coords <- matrix(c(1, 1, out["Spec"], out["Sens"]),
                   ncol=2,
                   byrow=TRUE)
  colnames(coords) <- c("Spec", "Sens")
  rownames(coords) <- c("Best", "Current")
  
  return (c(out, Dist=dist(coords)[1]))
}
#' Training model with knn. 
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return a trained model knn.
train_with_knn <- function(.x, .y, .control=DEFAULT_CONTROL, .metric=ACC, .seed=DEFAULT_SEED, .grid) {
  
  flog.trace("[train_with_knn] Training model with KNN and %s", .metric)
  
  # k: neighbors 
  if (is.na(.grid))
    .grid <- expand.grid(k = c(5, 11, 15, 21, 25, 33))
 
  set.seed(.seed) 
  result <- train(
    x = .x,
    y = .y,
    method = KNN,
    trControl = .control,
    tuneGrid  = .grid,
    metric=.metric
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
train_with_nb <- function(.x, .y, .control=DEFAULT_CONTROL, .metric=ACC
                        , .seed=DEFAULT_SEED, .grid=NA) {
  flog.trace("[train_with_nb] Training model with NB and %s", .metric)
  
  if (is.na(.grid))
      .grid <- expand.grid(
        fL        = 0:5,               # laplace correction
        usekernel = c(TRUE, FALSE),    # distribution type
        adjust    = seq(0, 5, by = 1)  # bandwidth adjustment
      )
  
  set.seed(.seed) 
  result <- train(
    x = .x,
    y = .y,
    method = NB,
    trControl = .control,
    tuneGrid  = .grid,
    preProcess = c("pca"),
    metric=.metric
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
train_with_nnet <- function(.x, .y, .control=DEFAULT_CONTROL, .metric=ACC
                          , .seed=DEFAULT_SEED, .grid=NA) {
  
  flog.trace("[train_with_nnet] Training model with NNET and %s", .metric)
  
  if (is.na(.grid))
    .grid <- expand.grid(
      size  = c(10, 20, 30, 40, 50),  # Hidden units 
      decay = (0.5)                   # Weight decay
    )
  
  set.seed(.seed) 
  result <- train(
    x = .x,
    y = .y,
    method = "nnet",
    trControl = .control,
    tuneGrid  = .grid,
    MaxNWts   = 5000,
    verbose   = FALSE,
    metric = .metric
  )
  
  return(result)
}

#' Training model with random forest (ranger).
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_with_rf <- function(.x, .y, .control=DEFAULT_CONTROL, .metric=ACC
                        , .seed=DEFAULT_SEED, .grid=NA) {
  
  flog.trace("[train_with_rf] Training model with RF and %s", .metric)

  if (is.na(.grid)) 
    .grid <- expand.grid(
      mtry = c(25, 50, 75, 100)
    )
  
  set.seed(.seed) 
  result <- train(
    x = .x,
    y = .y,
    method = RF,
    trControl = .control,
    tuneGrid  = .grid,
    ntree   = 200,
    verbose = FALSE,
    metric = .metric
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
train_with_svm <- function(.x, .y, .control=DEFAULT_CONTROL, .metric=ACC
                         , .seed=DEFAULT_SEED, .grid=NA) {
  
  flog.trace("[train_with_svm] Training model with SVM and %s", .metric)

  if (is.na(.grid)) 
    .grid <- expand.grid(
      C = c(
        2**(-5), 2**(0), 2**(5), 2**(10)
      ),
      sigma = c(
        2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)
      )
    )
  
  set.seed(.seed) 
  result <- train(
    x = .x,
    y = .y,
    method    = "svmRadial",
    trControl = .control,
    tuneGrid  = .grid,
    metric = .metric
  )

  return(result)
}

#' Training model with XgBoost. 
#'
#' @param .x A dataframe with independable variables
#' @param .y A dataframe with dependable variable
#' @param .control A control Caret parameter
#'
#' @return A trained model.
train_with_xb <- function(.x, .y, .control=DEFAULT_CONTROL, .metric=ACC
                        , .seed=DEFAULT_SEED, .grid=NA) {
  
  flog.trace("[train_with_xb] Training model with XgBoost and %s", .metric)
  
  if (is.na(.grid))
    .grid <- expand.grid(
      nrounds = c(100, 200, 300, 400), # Boosting iterations
      max_depth = c(3:7),              # Max tree depth
      eta = c(0.05, 1),                # Shrinkage
      gamma = c(0.01),                 # Minimum loss reduction 
      colsample_bytree = c(0.75),      # Subsamble ratio of columns
      subsample = c(0.50),             # Subsample percentage
      min_child_weight = c(0)          # Minimum sum of instance weight
    )

  set.seed(.seed) 
  result <- train(
    x = .x,
    y = .y,
    method = "xgbTree",
    trControl = .control,
    tuneGrid  = .grid,
    metric = .metric
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
train_with <- function(.x, .y, .classifier, .control=DEFAULT_CONTROL, 
                       .metric=ACC, .seed=DEFAULT_SEED, .grid=NA) {
  if (!.classifier %in% train_classifiers)
    stop(sprintf("%s unknown classifier!", .classifier))
 
  if (!.metric %in% train_metrics)
    stop(sprintf("%s unknown metric!", .metric))
  
  #if ((.metric == ROC) & (is.na(.grid))){ 
  #  flog.trace("Change control for ROC")
  #  .control <- trainControl(method = "repeatedcv", 
  #                           number =  5,
  #                           repeats = 2, 
  #                           classProbs = TRUE,  
  #                           summaryFunction = twoClassSummary, 
  #                           search = "grid")
  #}
  
  #if (.metric == DST){ 
  #  flog.trace("Change control for DST")
  #  .control <- trainControl(method = "repeatedcv", 
  #                           repeats = 5, 
  #                           classProbs = TRUE, 
  #                           summaryFunction = fourStats, 
  #                           search = "grid")
  #}
  
  if (.classifier == KNN) {
    return(train_with_knn(.x, .y, .control, .metric, .seed, .grid))
  } else if (.classifier == NB) {
    return(train_with_nb(.x, .y, .control, .metric, .seed))
  } else if (.classifier == NNET) {
    return(train_with_nnet(.x, .y, .control, .metric, .seed, .grid))
  } else if (.classifier == RF) {
    return(train_with_rf(.x, .y, .control, .metric, .seed, .grid))
  } else if (.classifier == SVM) {
    return(train_with_svm(.x, .y, .control, .metric, .seed, .grid))
  } else if (.classifier == XB) {
    return(train_with_xb(.x, .y, .control, .metric, .seed, .grid))
  }
}