#' @description
#' Predict if a bug will be long-lived or not using using Eclipse dataset.
#'
#' @author:
#' Luiz Alberto (gomes.luiz@gmail.com)
#'
#' @usage:
#' $ nohup Rscript ./predict_long_lived_bug_e1_train_metric_rev1.R > 
#'    predict_long_lived_bug_e1_train_metric_rev1.log 2>&1 &
#'
#' @details:
#'
#'         17/09/2019 16:00 pre-process in the train method removed
#'
# clean R Studio session.
rm(list = ls(all.names = TRUE))
options(readr.num_columns = 0)
timestamp <- format(Sys.time(), "%Y%m%d")

# Constants --------------------------------------------------------------------
debug_on     <- FALSE
force_create <- TRUE
processors   <- ifelse(debug_on, 3, 24)
base_path   <- file.path("~", "Workspace", "long-lived-bug-predictor-w-ml")
lib_path    <- file.path(base_path, "R")
data_path   <- file.path(base_path, "data")
output_path <- file.path(base_path, "output")
output_data_path <- file.path(output_path, "data")
# Constants --------------------------------------------------------------------


# Libraries --------------------------------------------------------------------
if (!require("caret")) install.packages("caret")
if (!require("doParallel")) install.packages("doParallel")
if (!require("e1071")) install.packages("e1071")
if (!require("klaR")) install.packages("klaR")
if (!require("kernlab")) install.packages("kernlab")
if (!require("futile.logger")) install.packages("futile.logger")
if (!require("nnet")) install.packages("nnet")
if (!require("qdap")) install.packages("qdap")
if (!require("randomForest")) install.packages("randonForest")
if (!require("SnowballC")) install.packages("SnowballC")
if (!require("smotefamily")) install.packages("smotefamily")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("tidytext")) install.packages("tidytext")
if (!require("tm")) install.packages("tm")

library(caret)
library(dplyr)
library(doParallel)
library(e1071)
library(klaR)
library(kernlab)
library(futile.logger)
library(nnet)
library(qdap)
library(randomForest)
library(SnowballC)
library(tidyverse)
library(tidytext)
library(tm)

source(file.path(lib_path, "balance_dataset.R"))
source(file.path(lib_path, "calculate_metrics.R"))
source(file.path(lib_path, "convert_to_term_matrix.R"))
source(file.path(lib_path, "clean_corpus.R"))
source(file.path(lib_path, "clean_text.R"))
source(file.path(lib_path, "format_file_name.R"))
source(file.path(lib_path, "get_metrics_file.R"))
source(file.path(lib_path, "get_next_parameter.R"))
source(file.path(lib_path, "get_resampling_method.R"))
source(file.path(lib_path, "insert_one_evaluation_data.R"))
source(file.path(lib_path, "make_dtm.R"))
source(file.path(lib_path, "train_helper.R"))
source(file.path(lib_path, "balance_dataset.R"))
# Libraries --------------------------------------------------------------------

# Experimental parameters ------------------------------------------------------
r_cluster    <- makePSOCKcluster(processors)
registerDoParallel(r_cluster)
resampling <- c("repeatedcv5x10")
class_label <- "long_lived"
prefix_reports <- "20200731"

classifier <- c(KNN, NB, RF)
feature    <- c("long_description")
balancing  <- c(SMOTEMETHOD)
train_metric <- c(ACC)
max_term   <- c(150)
threshold  <- c(365)
seeds <- c(DEFAULT_SEED)

if (debug_on) {
  exec_mode  <- "debug"
  classifier <- c(KNN)
  projects   <- c("winehq")
} else {
  exec_mode  <- "final"
  projects   <- c("eclipse", "freedesktop", "gcc", "gnome", "mozilla", "winehq")
  flog.appender(
    appender.file(
      file.path(
        output_path, 
        sprintf("logs/%s_r4_4b_predict_long_lived.log", timestamp)
      )
    )
  )
}
#
flog.threshold(TRACE)
flog.trace("Long live prediction Research Question 4 - Experiment 4b")
flog.trace("Evaluation metrics ouput path: %s", output_data_path)

results.train.file <- sprintf(
  "%s_r4_4b_predict_long_lived_bug_train_%s.csv", 
  timestamp, 
  exec_mode
)

parameters <- crossing(
  feature  , max_term  , classifier  ,
  balancing, resampling, train_metric, 
  threshold
)

results.started <- FALSE
for (project_name in projects)
{
  flog.trace("Current project name : %s", project_name)
  reports.file <- file.path(data_path, sprintf("%s_%s_bug_report_data.csv", prefix_reports, project_name))
  flog.trace("Bug report file name: %s", reports.file)

  reports <- read_csv(reports.file, na = c("", "NA"))
  if (debug_on) {
    flog.trace("DEBUG_MODE: Sample bug reports dataset")
    set.seed(DEFAULT_SEED)
    reports <- sample_n(reports, 1000)
  }
  
  # feature engineering.
  reports <- reports[, c("bug_id", "short_description", "long_description", "bug_fix_time")]
  reports <- reports[complete.cases(reports), ]
  #
  flog.trace("Clean text feature")
  reports$short_description <- clean_text(reports$short_description)
  reports$long_description  <- clean_text(reports$long_description)
  
  for (row in 1:nrow(parameters)) {
    parameter <- parameters[row, ]
    flog.trace("Converting dataframe to term matrix")
    reports.dataset <- convert_to_term_matrix(reports, parameter$feature, parameter$max_term )
    flog.trace("Text mining: extracted %d terms from %s",  ncol(reports.dataset) - 2, parameter$feature )
    for (seed in seeds) {
      set.seed(seed)
      flog.trace("SEED <%d>", seed)
      
      flog.trace("Current parameters:\n N.Terms...: [%d]\n Classifier: [%s]\n Metric...: [%s]\n Feature...: [%s]\n Threshold.: [%s]\n Balancing.: [%s]\n Resampling: [%s]",
        parameter$max_term, parameter$classifier, parameter$train_metric
        , parameter$feature, parameter$threshold, parameter$balancing
        , parameter$resampling)


      flog.trace("Partitioning dataset in training and testing")
      reports.dataset$long_lived <- as.factor(ifelse(reports.dataset$bug_fix_time <= parameter$threshold, "N", "Y"))
      in_train <- createDataPartition(reports.dataset$long_lived, p = 0.75, list = FALSE)
      train.dataset <- reports.dataset[in_train, ]
      test.dataset  <- reports.dataset[-in_train, ]

      test.file <- file.path(data_path, sprintf("%s_%s_bug_report_test_data_%s.csv"
                                                 , timestamp
                                                 , project_name
                                                 , exec_mode))
      write_csv(test.dataset, test.file)
      
      flog.trace("Balancing training dataset")
      train.balanced.dataset = balance_dataset( 
        train.dataset, 
        class_label, 
        c("bug_id", "bug_fix_time"), 
        parameter$balancing
      )

      X_train <- subset(train.balanced.dataset, select = -c(long_lived))
      y_train <- train.balanced.dataset[, class_label]

      X_test <- subset(test.dataset, select = -c(bug_id, bug_fix_time, long_lived))
      y_test <- test.dataset[, class_label]

      flog.trace("Training prediction model ")
      if (parameter$classifier == KNN) {
        grid    <- expand.grid(k=c(5))
      } else if (parameter$classifier == NB) {
        grid    <- expand.grid(fL=c(0), usekernel=c('adjust'))
      } else if (parameter$classifier == NNET) {
        grid = expand.grid(size = c(20), decay = c(0.5))
      } else if (parameter$classifier == RF) {
        grid    <- expand.grid(mtry=c(25))
      } else if (parameter$classifier == SVM) {
        grid    <- expand.grid(C = c(2**(5)), sigma = c(2**(-5)))
      }

      fit_control <- get_resampling_method(parameter$resampling)
      fit_model   <- train_with(
        .x = X_train, 
        .y = y_train, 
        .classifier = parameter$classifier,
        .control    = fit_control, 
        .metric     = parameter$train_metric, 
        .seed       = seed, 
        .grid       = grid
      )
      
      model.file <- file.path(output_data_path, sprintf("%s_%s_long_lived_predict_model_%s.rds"
                                                 , timestamp
                                                 , project_name
                                                 , exec_mode))
      saveRDS(fit_model, model.file)
      
      flog.trace("Calculating training results metrics")
      train.results <- fit_model$resampledCM
      train.results <- train.results[, !(names(train.results) %in% names(fit_model$bestTune))]

      names(train.results)[1] <- "tn"
      names(train.results)[2] <- "fp"
      names(train.results)[3] <- "fn"
      names(train.results)[4] <- "tp"

      train.results$acc <- (train.results$tp + train.results$tn) / 
        (train.results$tp + train.results$fp + train.results$tn + train.results$fn)

      positives <- train.results$tp + train.results$fn
      train.results$sensitivity <- ifelse(positives == 0, 0, train.results$tp/positives)

      negatives <- train.results$tn + train.results$fp
      train.results$specificity <- ifelse(negatives == 0, 0, train.results$tn/negatives)

      train.results$balanced_acc <- (train.results$sensitivity + train.results$specificity)/2
  
      train.results$project    <- project_name 
      train.results$classifier <- parameter$classifier
      train.results$balancing  <- parameter$balancing
      train.results$resampling <- parameter$resampling
      train.results$metric  <- parameter$train_metric
      train.results$max_term  <- parameter$max_term
      train.results$feature <- parameter$feature
      train.results$threshold   <-  parameter$threshold
      
      if (parameter$classifier == KNN) {
        train.results$hyper1  <- "k"
        train.results$value1  <- grid$k
        train.results$hyper2  <- "" 
        train.results$value2  <- 0 
      } else if (parameter$classifier == NB) {
        train.results$hyper1  <- "fL"
        train.results$value1  <- grid$fL
        train.results$hyper2  <- "usekernel" 
        train.results$value2  <- grid$usekernel
      } else if (parameter$classifier == NNET) {
        train.results$hyper1  <- "size"
        train.results$value1  <- grid$size
        train.results$hyper2  <- "decay"
        train.results$value2  <- grid$decay 
      } else if (parameter$classifier == RF) {
        train.results$hyper1  <- "mtry"
        train.results$value1  <- grid$mtry
      } else if (parameter$classifier == SVM) {
        train.results$hyper1  <- "sigma"
        train.results$value1  <- grid$sigma
        train.results$hyper2  <- "C" 
        train.results$value2  <- grid$C 
      } else {
        stop("Hyperparameters is required to record metrics")
      }
      train.results$seed    <- seed
      
      if (!results.started) {
        all_train.results <- train.results[FALSE, ]
        results.started   <- TRUE
      }
      all_train.results <- rbind(all_train.results, train.results)
    }
  }
  write_csv(all_train.results, file.path(output_data_path, results.train.file))
  flog.trace("Training results recorded on CSV file.")
}
flog.trace("Experiment 4 - 4b Finished")
