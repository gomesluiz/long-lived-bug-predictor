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
thresholds  <- c(365)
resamplings <- c("cv10")
max_terms   <- c(100)
class_label <- "long_lived"
prefix_reports <- "20200731"

if (debug_on) {
  projects      <- c("eclipse")
  classifiers   <- c(KNN, NB, NNET, RF, SVM)
  features      <- c("short_description")
  balancings    <- c(SMOTEMETHOD)
  train_metrics <- c(ACC)
} else {
  #projects    <- c("eclipse", "freedesktop", "gcc", "gnome", "mozilla", "winehq")
  projects    <- c("eclipse")
  classifiers <- c(KNN, NB, NNET, RF, SVM)
  features    <- c("short_description", "long_description")
  balancings  <- c(UNBALANCED, SMOTEMETHOD)
  train_metrics <- c(ACC)

  flog.appender(
    appender.file(
      file.path(
        output_path, "logs",
        sprintf("%s_r3_1a_predict_long_lived_bug.log", timestamp)
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
flog.trace("Long live prediction Research Question 3 - Experiment 1")
flog.trace("Evaluation metrics ouput path: %s", output_path)

results_started <- FALSE
results_file  <- sprintf( "%s_r3_1a_predict_long_lived_bug_results_%s.csv", 
                          timestamp, ifelse(debug_on, "debug", "final")) 
for (project_name in projects) {
  flog.trace("Project name: %s", project_name)
  flog.trace("Loading or create metrics file")

  # create or load yielded metrics files ---------------------------------------
  start_parameter <- 1 
  mask_file       <- sprintf("r3_1a_results_%s.csv" , project_name)
  metrics_file    <- get_metrics_file(output_path, mask_file)

  if (!is.na(metrics_file)) {
    all_metrics    <- read_csv(metrics_file)
    next_parameter <- get_next_parameter_number(all_metrics, parameters)
    if (next_parameter != -1) {
      start_parameter <- next_parameter
    }
  }

  # A new metric file have to be generated.
  if ((start_parameter == 1) || (force_create == TRUE)) {
    #start_parameter <- 1
    metrics_file    <- format_file_name(output_data_path, timestamp, mask_file)
    flog.trace("Metrics file create: %s", metrics_file)
  } else {
    flog.trace("Metrics file: %s", metrics_file)
  }
  # create or load yielded metrics files ---------------------------------------
  
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

  flog.trace("Clean text features")
  reports_data$short_description <- clean_text(reports_data$short_description)
  reports_data$long_description  <- clean_text(reports_data$long_description)
  # feature engineering --------------------------------------------------------
  
  last_feature  <- "@"
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
    
    if (parameter$features != last_feature) {
      flog.trace("Converting dataframe to term matrix")
      reports_matrix <- convert_to_term_matrix(reports_data, parameter$features, 
                                               parameter$max_terms)
      flog.trace("Text mining: extracted %d terms from %s", 
                  ncol(reports_matrix) - 2, parameter$features)
      last_feature <- parameter$features
    }

    flog.trace("Partitioning dataset in training and testing")
    reports_matrix$long_lived <- as.factor(
      ifelse(reports_matrix$bug_fix_time > parameter$thresholds, 1, 0)
    )
    
    #in_train <- createDataPartition(reports.dataset$long_lived, p = 0.75, list = FALSE)
    #train.dataset <- reports.dataset[in_train, ]
    #test.dataset <- reports.dataset[-in_train, ]

    flog.trace("Balancing training dataset")
    reports_balanced <- balance_dataset(
      reports_matrix,
      class_label,
      c("bug_id", "bug_fix_time"),
      parameter$balancings
    )

    X_train <- subset(reports_balanced, select = -c(long_lived))
    y_train <- reports_balanced[, class_label]
    #X_test <- subset(test.dataset, select = -c(bug_id, bug_fix_time, long_lived))
    #y_test <- test.dataset[, class_label]

    flog.trace("Training model ")
    fit_control <- get_resampling_method(parameter$resamplings)
    fit_model <- train_with(
      .x = X_train,
      .y = y_train,
      .classifier = parameter$classifiers,
      .control = fit_control,
      .metric = parameter$train_metrics
    )
    
    #flog.trace("Recording best tune hyperparameters in CSV file")
    
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
    
    train.results$hyper1 = "" 
    train.results$value1 = 0 
    train.results$hyper2 = "" 
    train.results$value2 = 0 
    train.results$hyper3 = "" 
    train.results$value3 = 0 
    if (parameter$classifiers == KNN)
    {
      train.results$hyper1 <-  'k'
      train.results$value1 <- fit_model$resampledCM$k
    }
    if (parameter$classifiers == NB)
    {
      train.results$hyper1 <-  'fL'
      train.results$value1 <- fit_model$resampledCM$fL
      train.results$hyper2 <-  'usekernel'
      train.results$value2 <- fit_model$resampledCM$usekernel
      train.results$hyper3 <-  'adjust'
      train.results$value3 <- fit_model$resampledCM$adjust
    }
    if (parameter$classifiers == NNET)
    {
      train.results$hyper1 <-  'size'
      train.results$value1 <- fit_model$resampledCM$size
      train.results$hyper2 <-  'decay'
      train.results$value2 <- fit_model$resampledCM$decay
    }
    if (parameter$classifiers == RF)
    {
      train.results$hyper1 <-  'mtry'
      train.results$value1 <- fit_model$resampledCM$mtry
    }
    if (parameter$classifiers == SVM)
    {
      train.results$hyper1 <-  'sigma'
      train.results$value1 <- fit_model$resampledCM$sigma
      train.results$hyper2 <-  'C'
      train.results$value2 <- fit_model$resampledCM$C
    }
    
    metrics_record = train.results %>% 
      top_n(1, balanced_acc)
    
    metrics_record = metrics_record[1, ]
    
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
    # saving model plot to file
    #plot_file_name <- sprintf(
    #  "%s_rq3e1_%s_%s_%s_%s_%s_%s.jpg", timestamp, project.name,
    #  parameter$classifier, parameter$balancing, parameter$feature,
    #  parameter$n_term, parameter$metric.type
    #)
    #plot_file_name_path <- file.path(DATADIR, plot_file_name)
    #jpeg(plot_file_name_path)
    #print(plot(fit_model))
    #dev.off()

    #flog.trace("Testing model ")
    #y_hat <- predict(object = fit_model, X_test)

    #flog.trace("Calculating evaluating metrics")
    #metrics <- calculate_metrics(y_hat, y_test)

    #flog.trace("Evaluating metrics calculated!")
    #if (!is.na(metrics$balanced_acc)) {
    #  if (metrics$balanced_acc > best.accuracy) {
    #    test.result <- cbind(test.dataset[, c("bug_id", "bug_fix_time", "long_lived")], y_hat)
    #    write_csv(test.result, file.path(DATADIR, sprintf("%s_rq3e1_%s_test_results.csv", timestamp, project.name)))
    #    best.accuracy <- metrics$balanced_acc
    #  }
    #}
    #flog.trace(
    #  "Metrics: sensitivity [%f], specificity [%f], b_acc [%f]",
    #  metrics$sensitivity, metrics$specificity, metrics$balanced_acc
    #)

    #metrics_record <-
    #  data.frame(
    #    project = project_name,
    #    feature = parameter$features,
    #    n_term = parameter$max_terms,
    #    classifier = parameter$classifiers,
    #    balancing = parameter$balancings,
    #    resampling = parameter$resamplings,
    #    metric = parameter$train_metrics,
    #    threshold = parameter$thresholds,
    #    train_size = nrow(X_train),
    #    train_size_class_0 = length(subset(y_train, y_train == '0')),
    #    train_size_class_1 = length(subset(y_train, y_train == '1'))
        #test_size = nrow(X_test),
        #test_size_class_0 = length(subset(y_test, y_test == "N")),
        #test_size_class_1 = length(subset(y_test, y_test == "Y")),
        #tp = metrics$tp,
        #fp = metrics$fp,
        #tn = metrics$tn,
        #fn = metrics$fn,
        #sensitivity = metrics$sensitivity,
        #specificity = metrics$specificity,
        #balanced_acc = metrics$balanced_acc,
        #balanced_acc_manual = metrics$balanced_acc_manual,
        #precision = metrics$precision,
        ##recall = metrics$recall,
        #fmeasure = metrics$fmeasure
    #  )
    #flog.trace("Recording evaluation results on CSV file.")
    #insert_one_evaluation_data(metrics.file, metrics.record)
    #flog.trace("Evaluation results recorded on CSV file.")
  }
}
