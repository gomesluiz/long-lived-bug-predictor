#' @description
#' Predict if a bug will be long-lived in Eclipse dataset.
#'
#' @author:
#' Luiz Alberto (gomes.luiz@gmail.com)
#'
#' @usage:
#' $ nohup Rscript ./04a_predict_long_lived_bug_e1.R > \\
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
debug_on     <- TRUE
force_create <- TRUE
processors   <- ifelse(debug_on, 3, 24)
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
resamplings <- c("cv10")
class_label <- "long_lived"
prefix_reports <- "20200731"

classifiers <- c(SVM)
features    <- c("long_description")
balancings  <- c(SMOTEMETHOD)
train_metrics <- c(ACC)
max_terms     <- c(150)
thresholds    <- c(365)

if (debug_on) {
  projects    <- c("gcc", "winehq")
} else {
  projects    <- c("eclipse", "freedesktop", "gcc", "gnome", "mozilla", "winehq")
  flog.appender(
    appender.file(
      file.path(
        output_path, "logs",
        sprintf("%s_r4_4a_predict_long_lived_bug.log", timestamp)
      )
    )
  )
}

parameters <- crossing(
  features, max_terms, classifiers,
  balancings, resamplings,
  train_metrics, thresholds
)
# Experimental parameters ------------------------------------------------------
flog.threshold(TRACE)
flog.trace("Long live prediction Research Question 4 - Experiment 3")
flog.trace("Evaluation metrics ouput path: %s", output_path)

results_started <- FALSE
results_file  <- sprintf( "%s_r4_4a_predict_long_lived_bug_results_%s.csv", 
                          timestamp, ifelse(debug_on, "debug", "final")) 
start_parameter <- 1 
for (project_name in projects) {
  flog.trace("Project name: %s", project_name)
  flog.trace("Loading or create metrics file")
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
  reports_data <- reports_data[, c("bug_id", "long_description",  "bug_fix_time")]
  reports_data <- reports_data[complete.cases(reports_data), ]

  flog.trace("Clean text features")
  reports_data$long_description  <- clean_text(reports_data$long_description)
  # feature engineering --------------------------------------------------------
  
  best_accuracy <- 0
  for (i in start_parameter:nrow(parameters)) {
    set.seed(DEFAULT_SEED)
    parameter <- parameters[i, ]

    flog.trace("Processing feature         : %s", parameter$features)
    flog.trace("Processing number of terms : %d", parameter$max_terms)
    flog.trace("Processing classifier      : %s", parameter$classifiers)
    flog.trace("Processing training metric : %s", parameter$train_metrics)
    flog.trace("Processing threshold       : %d", parameter$thresholds)
    flog.trace("Processing balancing       : %s", parameter$balancings)
    flog.trace("Processing resampling      : %s", parameter$resamplings)
    
    flog.trace("Converting dataframe to term matrix")
    reports_matrix <- convert_to_term_matrix(reports_data, parameter$features, 
                                               parameter$max_terms)
    flog.trace("Text mining: extracted %d terms from %s", 
                  ncol(reports_matrix) - 2, parameter$features)

    flog.trace("Partitioning dataset in training and testing")
    reports_matrix$long_lived <- as.factor(
      ifelse(reports_matrix$bug_fix_time > parameter$thresholds, 1, 0)
    )
    
    flog.trace("Balancing training dataset")
    reports_balanced <- balance_dataset(
      reports_matrix,
      class_label,
      c("bug_id", "bug_fix_time"),
      parameter$balancings
    )

    X_train <- subset(reports_balanced, select = -c(long_lived))
    y_train <- reports_balanced[, class_label]

    flog.trace("Training model ")
    fit_control <- get_resampling_method(parameter$resamplings)
    fit_grid    <- expand.grid(C = c(2**(5)),sigma = c(2**(-5)))
    fit_model   <- train(
      x = X_train,
      y = y_train,
      method   = "svmRadial",
      trControl = fit_control,
      tuneGrid  = fit_grid,
      metric    = parameter$train_metrics
    )
    
    flog.trace("Recording training resultas in CSV file")
    train.results <- fit_model$resampledCM
    train.results <- train.results[, !(names(train.results) %in% names(fit_model$bestTune))]
    
    names(train.results)[1] <- "tn"
    names(train.results)[2] <- "fn"
    names(train.results)[3] <- "fp"
    names(train.results)[4] <- "tp"
    
    train.results$acc <- (train.results$tp + train.results$tn) / 
      (train.results$tp + train.results$fp + train.results$tn + train.results$fn)
    
    positives <- train.results$tp + train.results$fn
    train.results$sensitivity <- ifelse(positives == 0, 0, train.results$tp/positives)
    
    negatives <- train.results$tn + train.results$fp
    train.results$specificity <- ifelse(negatives == 0, 0, train.results$tn/negatives)
    
    train.results$balanced_acc <- (train.results$sensitivity + train.results$specificity)/2
    
    metrics_record = train.results %>% 
      top_n(1, balanced_acc)
    
    metrics_record = metrics_record[1, ]
    
    metrics_record$project <- project_name
    metrics_record$classifier <- parameter$classifiers
    metrics_record$balancing  <- parameter$balancings
    metrics_record$resampling <- parameter$resamplings
    metrics_record$train_metric  <- parameter$train_metrics
    metrics_record$max_term <- parameter$max_terms
    metrics_record$feature <- parameter$features
    metrics_record$threshold <- parameter$thresholds
    metrics_record$train_size <- nrow(X_train)
    metrics_record$train_size_class_0 <- length(subset(y_train, y_train == '0'))
    metrics_record$train_size_class_1 <- length(subset(y_train, y_train == '1'))
    metrics_record$seed    <- DEFAULT_SEED
    if (!results_started) {
      all_train.results = metrics_record[FALSE, ]
      results_started <- TRUE
    }
    all_train.results <- rbind(all_train.results, metrics_record)
    write_csv(all_train.results, file.path(output_data_path, results_file))
    flog.trace("Training results recorded on CSV file.")
  }
}
