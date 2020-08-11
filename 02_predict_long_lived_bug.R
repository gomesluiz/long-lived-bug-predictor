#' @description 
#' Predict if a bug will be long-lived or not using using Eclipse dataset.
#'
#' @author: 
#' Luiz Alberto (gomes.luiz@gmail.com)
#'
#' @usage: 
#' $ nohup Rscript ./predict_long_lived_bug_e2.R > predict_long_lived_bug_e2.log 2>&1 &
#' 
#' @details:
#' 
#'         17/09/2019 16:00 pre-process in the train method removed
#' 

# clean R Studio session.
rm(list = ls(all.names = TRUE))
options(readr.num_columns = 0)
timestamp       <- format(Sys.time(), "%Y%m%d%H%M%S")

# setup project folders.
IN_DEBUG_MODE  <- FALSE
FORCE_NEW_FILE <- TRUE
BASEDIR <- file.path("~","Workspace", "doctorate", "projects")
LIBDIR  <- file.path(BASEDIR, "lib", "R")
PRJDIR  <- file.path(BASEDIR, "long-lived-bug-prediction")
SRCDIR  <- file.path(PRJDIR, "R")
DATADIR <- file.path(PRJDIR, "notebooks", "datasets")

if (!require('caret')) install.packages("caret")
if (!require('doParallel')) install.packages("doParallel")
if (!require('e1071')) install.packages("e1071")
if (!require("klaR")) install.packages("klaR") # naive bayes package.
if (!require("kernlab")) install.packages("kernlab")
if (!require("futile.logger")) install.packages("futile.logger")
if (!require("nnet")) install.packages("nnet")
if (!require("qdap")) install.packages("qdap")
if (!require("randomForest")) install.packages("randomForest")
if (!require("SnowballC")) install.packages("SnowballC")
if (!require('smotefamily')) install.packages('smotefamily')
if (!require("tidyverse")) install.packages("tidyverse")
if (!require('tidytext')) install.packages('tidytext')
if (!require("tm")) install.packages("tm")
if (!require("xgboost")) install.packages("xgboost")

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

source(file.path(LIBDIR, "balance_dataset.R"))
source(file.path(LIBDIR, "calculate_metrics.R"))
source(file.path(LIBDIR, "convert_to_term_matrix.R"))
source(file.path(LIBDIR, "clean_corpus.R"))
source(file.path(LIBDIR, "clean_text.R"))
source(file.path(LIBDIR, "format_file_name.R"))
source(file.path(LIBDIR, "get_last_evaluation_file.R"))
source(file.path(LIBDIR, "get_next_parameter_number.R"))
source(file.path(LIBDIR, "get_resampling_method.R"))
source(file.path(LIBDIR, "insert_one_evaluation_data.R"))
source(file.path(LIBDIR, "make_dtm.R"))
source(file.path(LIBDIR, "train_helper.R"))

# main function
processors <- ifelse(IN_DEBUG_MODE, 3, 3)
r_cluster <-  makePSOCKcluster(processors)
registerDoParallel(r_cluster)

class_label     <- "long_lived"

# setup experimental parameters.
projects    <- c("eclipse")
n_term      <- c(100, 150, 200, 250, 300)
classifier  <- c(NB)
feature     <- c("short_description")
threshold   <- c(365)
balancing   <- c(UNBALANCED)
resampling  <- c("repeatedcv")
metric.type <- c(ROC)
parameters  <- crossing(feature, n_term, classifier, balancing, resampling, metric.type, threshold)

flog.threshold(TRACE)
flog.trace("Long live prediction Research Question 4 - Experiment 1")
flog.trace("Evaluation metrics ouput path: %s", DATADIR)

for (project.name in projects){
  flog.trace("Current project name : %s", project.name)
  
  parameter.number  <- 1
  metrics.mask      <- sprintf("rq3e2_%s_predict_long_lived_metrics.csv", project.name)
  metrics.file      <- get_last_evaluation_file(DATADIR, metrics.mask)
  
  # get last parameter number and metrics file. 
  if (!is.na(metrics.file)) {
    all.metrics      <- read_csv(metrics.file)
    next.parameter   <- get_next_parameter_number(all.metrics, parameters)
    # There are parameters to be processed.
    if (next.parameter != -1) {
       parameter.number = next.parameter
    }
  }
 
  # A new metric file have to be generated. 
  if ((parameter.number == 1) || (FORCE_NEW_FILE == TRUE)) {
    parameter.number = 1
    metrics.file <- format_file_name(DATADIR, timestamp, metrics.mask)
    flog.trace("New Evaluation file: %s", metrics.file)
  } else {
    flog.trace("Current Evaluation file: %s", metrics.file)
  }
  
  flog.trace("Starting in parameter number: %d", parameter.number)

  reports.file <- file.path(DATADIR, sprintf("20190917_%s_bug_report_data.csv", project.name))
  flog.trace("Bug report file name: %s", reports.file)
  
  reports <- read_csv(reports.file, na  = c("", "NA"))
  if (IN_DEBUG_MODE){
    flog.trace("DEBUG_MODE: Sample bug reports dataset")
    set.seed(DEFAULT_SEED)
    reports <- sample_n(reports, 1000) 
  }

  # feature engineering. 
  reports <- reports[, c('bug_id', 'short_description', 'long_description' , 'bug_fix_time')]
  reports <- reports[complete.cases(reports), ]
  
  flog.trace("Clean text features")
  reports$short_description <- clean_text(reports$short_description)
  reports$long_description  <- clean_text(reports$long_description)
 
  last.feature     <- "" 
  best.accuracy    <- 0
  best.sensitivity <- 0
  for (i in parameter.number:nrow(parameters)) {
    parameter = parameters[i, ]
    
    flog.trace("Current parameters:\n N.Terms...: [%d]\n Classifier: [%s]\n Metric...: [%s]\n Feature...: [%s]\n Threshold.: [%s]\n Balancing.: [%s]\n Resampling: [%s]"
             , parameter$n_term , parameter$classifier , parameter$metric.type, parameter$feature
             , parameter$threshold , parameter$balancing , parameter$resampling)

    #if (parameter$feature != last.feature){
      flog.trace("Converting dataframe to term matrix")
      flog.trace("Text mining: extracting %d terms from %s", parameter$n_term, parameter$feature)
      reports.dataset <- convert_to_term_matrix(reports, parameter$feature, parameter$n_term)
      flog.trace("Text mining: extracted %d terms from %s", ncol(reports.dataset) - 2, parameter$feature)
      last.feature = parameter$feature
    #}
    
    flog.trace("Partitioning dataset in training and testing")
    set.seed(DEFAULT_SEED)
    reports.dataset$long_lived <- as.factor(ifelse(reports.dataset$bug_fix_time <= parameter$threshold, "N", "Y"))
    in_train <- createDataPartition(reports.dataset$long_lived, p = 0.75, list = FALSE)
    train.dataset <- reports.dataset[in_train, ]
    test.dataset  <- reports.dataset[-in_train, ]
     
    flog.trace("Balancing training dataset")
    balanced.dataset = balance_dataset(train.dataset, 
                                       class_label, 
                                       c("bug_id", "bug_fix_time"), 
                                       parameter$balancing)
    
    X_train <- subset(balanced.dataset, select=-c(long_lived))
    y_train <- balanced.dataset[, class_label]
    
    X_test  <- subset(test.dataset, select=-c(bug_id, bug_fix_time, long_lived))
    y_test  <- test.dataset[, class_label]

    flog.trace("Training model ")
    fit_control <- get_resampling_method(parameter$resampling)
    fit_model   <- train_with (.x=X_train, 
                               .y=y_train, 
                               .classifier=parameter$classifier,
                               .control=fit_control,
                               .metric=parameter$metric.type)
    # saving model plot to file 
    plot_file_name = sprintf("%s_rq3e2_%s_%s_%s_%s_%s_%s.jpg", timestamp, project.name
                             , parameter$classifier, parameter$balancing, parameter$feature
                             , parameter$n_term, parameter$metric.type)
    plot_file_name_path = file.path(DATADIR, plot_file_name)
    jpeg(plot_file_name_path)
    print(plot(fit_model))
    dev.off()
    
    flog.trace("Testing model ")
    y_hat <- predict(object = fit_model, X_test)

    flog.trace("Calculating evaluating metrics")
    metrics <- calculate_metrics(y_hat, y_test)
    
    flog.trace("Evaluating metrics calculated!")
    if (!is.na(metrics$balanced_acc)){
      if (metrics$balanced_acc > best.accuracy){
        test.result = cbind(test.dataset[, c("bug_id", "bug_fix_time", "long_lived")], y_hat)
        write_csv( test.result , file.path(DATADIR, sprintf("%s_rq3e2_%s_test_results_balanced_acc.csv", timestamp, project.name)))
        best.accuracy = metrics$balanced_acc
      }
    }
    
    if (!is.na(metrics$sensitivity)){
      if (metrics$sensitivity > best.sensitivity){
        test.result = cbind(test.dataset[, c("bug_id", "bug_fix_time", "long_lived")], y_hat)
        write_csv( test.result , file.path(DATADIR, sprintf("%s_rq3e2_%s_test_results_sensitivity.csv", timestamp, project.name)))
        best.sensitivity = metrics$sensitivity
      }
    }
    flog.trace("Metrics: sensitivity [%f], specificity [%f], b_acc [%f]"
                , metrics$sensitivity, metrics$specificity, metrics$balanced_acc)
    
    metrics.record<-
      data.frame(
        project    = project.name,
        feature    = parameter$feature,
        n_term     = parameter$n_term,
        classifier = parameter$classifier,
        balancing  = parameter$balancing,
        resampling = parameter$resampling,
        metric     = parameter$metric.type,
        threshold  = parameter$threshold,
        train_size = nrow(X_train),
        train_size_class_0 = length(subset(y_train, y_train == "N")),
        train_size_class_1 = length(subset(y_train, y_train == "Y")),
        test_size = nrow(X_test),
        test_size_class_0 = length(subset(y_test, y_test == "N")),
        test_size_class_1 = length(subset(y_test, y_test == "Y")),
        tp = metrics$tp,
        fp = metrics$fp,
        tn = metrics$tn,
        fn = metrics$fn,
        sensitivity = metrics$sensitivity,
        specificity = metrics$specificity,
        balanced_acc = metrics$balanced_acc,
        balanced_acc_manual = metrics$balanced_acc_manual,
        precision = metrics$precision,
        recall = metrics$recall,
        fmeasure = metrics$fmeasure
      )
      flog.trace("Recording evaluation results on CSV file.")
      insert_one_evaluation_data(metrics.file, metrics.record)
      flog.trace("Evaluation results recorded on CSV file.")
  }
}
