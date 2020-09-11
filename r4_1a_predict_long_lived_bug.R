#' @description
#' Predict if a bug will be long-lived in Eclipse dataset.
#'
#' @author:
#' Luiz Alberto (gomes.luiz@gmail.com)
#'
#' @usage:
#' $ nohup Rscript ./01a_predict_long_lived_bug_e1.R > \\
#'   01a_predict_long_lived_bug_e1.log 2>&1 &
#'
#' @details:
#'
#'         05/08/2020 16:30 code refactored
#'         17/09/2019 16:00 pre-process in the train method removed
#'

# clean R Studio session.
rm(list = ls(all.names = TRUE))
options(readr.num_columns = 0)
timestamp <- format(Sys.time(), "%Y%m%d")

# Constants --------------------------------------------------------------------
debug_on     <- FALSE
force_create <- TRUE
processors   <- ifelse(debug_on, 3, 30)
base_path   <- file.path("~", "Workspace", "long-lived-bug-predictor-ml-in-r")
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
threshold  <- c(365)
resampling <- c("cv10")
max_term   <- c(100)
class_label <- "long_lived"
prefix_reports <- "20200731"
projects      <- c("eclipse")

if (debug_on) {
  classifier   <- c(KNN, SVM)
  feature      <- c("short_description")
  balancing    <- c(UNBALANCED)
  train_metric <- c(ACC)
} else {
  classifier <- c(KNN, NB, NNET, RF, SVM)
  feature    <- c("short_description", "long_description")
  balancing  <- c(UNBALANCED, SMOTEMETHOD)
  train_metric <- c(ACC)

  flog.appender(
    appender.file(
      file.path(
        output_path, "logs",
        sprintf("%s_r4_1a_predict_long_lived_bug.log", timestamp)
      )
    )
  )
}

parameters <- crossing(
  feature, max_term, classifier,
  balancing, resampling,
  train_metric, threshold
)
# Experimental parameters ------------------------------------------------------
flog.threshold(TRACE)
flog.trace("Long live prediction Research Question 4 - Experiment 1")
flog.trace("Evaluation metrics ouput path: %s", output_path)

results_started <- FALSE
results_file  <- sprintf( "%s_r4_1a_predict_long_lived_bug_results_%s.csv", 
                          timestamp, ifelse(debug_on, "debug", "final")) 
for (project_name in projects) {
  flog.trace("Project name: %s", project_name)
  flog.trace("Loading or create metrics file")

  # create or load yielded metrics files ---------------------------------------
  start_parameter <- 1 
  
  flog.trace("Starting in parameter number: %d", start_parameter)
  reports_file <- file.path(data_path, sprintf("%s_%s_bug_report_data.csv", 
                                               prefix_reports,  project_name))
  flog.trace("Bug report file name: %s", reports_file)
  reports_data <- read_csv(reports_file, na = c("", "NA"))
  
  if (debug_on) {
    flog.trace("DEBUG_MODE: Sample bug reports dataset")
    set.seed(DEFAULT_SEED)
    reports_data <- sample_n(reports_data, 1000)
  }

  # feature engineering --------------------------------------------------------
  reports_data <- reports_data[, c("bug_id", "short_description", "long_description",  "bug_fix_time")]
  reports_data <- reports_data[complete.cases(reports_data), ]

  flog.trace("Clean text feature")
  reports_data$short_description <- clean_text(reports_data$short_description)
  reports_data$long_description  <- clean_text(reports_data$long_description)
  # feature engineering --------------------------------------------------------
  
  last_feature  <- "@"
  best_accuracy <- 0
  for (i in start_parameter:nrow(parameters)) {
    set.seed(DEFAULT_SEED)
    parameter <- parameters[i, ]

    flog.trace("Processing feature         : %s", parameter$feature)
    flog.trace("Processing number of terms : %d", parameter$max_term)
    flog.trace("Processing classifier      : %s", parameter$classifier)
    flog.trace("Processing training metric : %s", parameter$train_metric)
    flog.trace("Processing threshold       : %d", parameter$threshold)
    flog.trace("Processing balancing       : %s", parameter$balancing)
    flog.trace("Processing resampling      : %s", parameter$resampling)
    
    if (parameter$feature != last_feature) {
      flog.trace("Converting dataframe to term matrix")
      reports_matrix <- convert_to_term_matrix(reports_data, parameter$feature, 
                                               parameter$max_term)
      flog.trace("Text mining: extracted %d terms from %s", 
                  ncol(reports_matrix) - 2, parameter$feature)
      last_feature <- parameter$feature
    }

    flog.trace("Partitioning dataset in training and testing")
    reports_matrix$long_lived <- as.factor(
      ifelse(reports_matrix$bug_fix_time > parameter$threshold, 1, 0)
    )
    
    #in_train <- createDataPartition(reports_matrix$long_lived, p = 0.80, list = FALSE)
    #X_train <- reports_matrix[in_train, ]
    X_train <- reports_matrix
    #X_test  <- reports_matrix[-in_train, ]

    flog.trace("Balancing training dataset")
    reports_balanced <- balance_dataset(
      X_train,
      class_label,
      c("bug_id", "bug_fix_time"),
      parameter$balancing
    )

    X_train <- subset(reports_balanced, select = -c(long_lived))
    y_train <- reports_balanced[, class_label]
    #X_test <- subset(test.dataset, select = -c(bug_id, bug_fix_time, long_lived))
    #y_test <- test.dataset[, class_label]

    flog.trace("Training model ")
    fit_control <- get_resampling_method(parameter$resampling)
    fit_model <- train_with(
      .x = X_train,
      .y = y_train,
      .classifier = parameter$classifier,
      .control = fit_control,
      .metric = parameter$train_metric
    )
    
    flog.trace("Recording best tune hyperparameters in CSV file")
    train.results <- parameters[i, ]
    train.results$train_size         <- nrow(X_train)
    train.results$train_size_class_0 <- length(subset(y_train, y_train == '0'))
    train.results$train_size_class_1 <- length(subset(y_train, y_train == '1'))
    train.results$hyper1 <- "" 
    train.results$value1 <- 0 
    train.results$hyper2 <- "" 
    train.results$value2 <- 0 
    train.results$hyper3 <- "" 
    train.results$value3 <- 0 
    train.results$seed  <- DEFAULT_SEED
    if (parameter$classifier == KNN)
    {
      train.results$hyper1 <-  'k'
      train.results$value1 <- fit_model$bestTune$k
    }
    if (parameter$classifier == NB)
    {
      train.results$hyper1 <-  'fL'
      train.results$value1 <- fit_model$bestTune$fL
      train.results$hyper2 <-  'usekernel'
      train.results$value2 <- fit_model$bestTune$usekernel
      train.results$hyper3 <-  'adjust'
      train.results$value3 <- fit_model$bestTune$adjust
    }
    if (parameter$classifier == NNET)
    {
      train.results$hyper1 <-  'size'
      train.results$value1 <- fit_model$bestTune$size
      train.results$hyper2 <-  'decay'
      train.results$value2 <- fit_model$bestTune$decay
    }
    if (parameter$classifier == RF)
    {
      train.results$hyper1 <-  'mtry'
      train.results$value1 <- fit_model$bestTune$mtry
    }
    if (parameter$classifier == SVM)
    {
      train.results$hyper1 <-  'sigma'
      train.results$value1 <- fit_model$bestTune$sigma
      train.results$hyper2 <-  'C'
      train.results$value2 <- fit_model$bestTune$C
    }
    if (!results_started) {
      all_train.results = train.results[FALSE, ]
      results_started <- TRUE
    }
    all_train.results <- rbind(all_train.results, train.results)
  }
  write_csv(all_train.results, file.path(output_data_path, results_file))
  flog.trace("Training results recorded on CSV file.")
}
