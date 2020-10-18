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

resampling <- c("repeatedcv5x10")
feature    <- "long_description"
balancing  <- SMOTEMETHOD
train_metric <- ACC
max_term   <- 150
threshold  <- 365
seed <- DEFAULT_SEED
class_label <- "long_lived"

if (debug_on) {
  exec_mode  <- "debug"
  classifier <- KNN
  projects   <- c("winehq")
} else {
  exec_mode  <- "final"
  classifier <- SVM
  projects   <- c("eclipse", "freedesktop", "gcc", "gnome", "mozilla", "winehq")
  flog.appender(
    appender.file(
      file.path(
        output_path, 
        sprintf("logs/%s_r5_1b_predict_long_lived.log", timestamp)
      )
    )
  )
}
#
flog.threshold(TRACE)
flog.trace("Long live prediction Research Question 4 - Experiment 1b")
flog.trace("Evaluation metrics ouput path: %s", output_data_path)

results.test.file <- sprintf(
  "%s_r5_1b_predict_long_lived_bug_test_%s.csv", 
  timestamp, 
  exec_mode
)
results.hat.file <- sprintf(
  "%s_r5_1b_predict_long_lived_bug_hat_%s.csv", 
  timestamp, 
  exec_mode
)

results.started <- FALSE
for (project_name in projects)
{
  flog.trace("Current project name : %s", project_name)
  
  test_file    <- file.path(data_path, sprintf("%s_%s_bug_report_test_data_%s.csv", timestamp, project_name, exec_mode))
  flog.trace("Reading test dataset : %s", test_file)
  test_dataset <- read_csv(test_file, na = c("", "NA"))
  
  model_file   <- file.path(output_data_path, sprintf("%s_%s_long_lived_predict_model_%s.rds" , timestamp , project_name , exec_mode))
  flog.trace("Reading model file: %s", model_file)
  final_model  <- readRDS(model_file)

  flog.trace("Testing predicting model ")
  X_test <- subset(test_dataset, select = -c(bug_id, bug_fix_time, long_lived))
  y_test <- as.factor(test_dataset$long_lived)

  y_hat <- predict(object = final_model, X_test)
  test.metrics <- calculate_metrics(y_hat, y_test)
  
  flog.trace("Save test results ")
  test.results <-
    data.frame(
        project    = project_name,
        feature    = feature,
        max_term   = max_term,
        classifier = classifier,
        balancing  = balancing,
        resampling = resampling,
        metric     = train_metric,
        threshold  = threshold,
        tp = test.metrics$tp,
        fp = test.metrics$fp,
        tn = test.metrics$tn,
        fn = test.metrics$fn,
        sensitivity = test.metrics$sensitivity,
        specificity = test.metrics$specificity,
        balanced_acc = test.metrics$balanced_acc,
        balanced_acc_manual = test.metrics$balanced_acc_manual,
        precision = test.metrics$precision,
        recall = test.metrics$recall,
        fmeasure = test.metrics$fmeasure,
        seed     = seed
    )
      
  hat.results <- cbind(test_dataset[, c("bug_id", "bug_fix_time", "long_lived")], y_hat, project_name, classifier)

  if (!results.started) {
    all_test.results  <- test.results[FALSE,]
    all_hat.results   <- hat.results[FALSE, ]
    results.started   <- TRUE
  }
  all_test.results  <- rbind(all_test.results,  test.results)
  all_hat.results   <- rbind(all_hat.results, hat.results)
  
  write_csv(all_test.results, file.path(output_data_path, results.test.file))
  write_csv(all_hat.results, file.path(output_data_path, results.hat.file))
  flog.trace("Test results recorded on CSV file.")
}
flog.trace("Experiment 5 - 1b Finished")
