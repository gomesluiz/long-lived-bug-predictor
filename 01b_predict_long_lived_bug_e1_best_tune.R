#' @description 
#' Predict if a bug will be long-lived or not using using Eclipse dataset.
#'
#' @author: 
#' Luiz Alberto (gomes.luiz@gmail.com)
#'
#' @usage: 
#' $ nohup Rscript ./predict_long_lived_bug_e1.R > predict_long_lived_bug_e1.log 2>&1 &
#' 
#' @details:
#' 
#'         17/09/2019 16:00 pre-process in the train method removed
#' 

# clean R Studio session.
rm(list = ls(all.names = TRUE))
options(readr.num_columns = 0)
timestamp       <- format(Sys.time(), "%Y%m%d")

# setup project folders.
IN_DEBUG_MODE  <- FALSE
FORCE_NEW_FILE <- TRUE
BASEDIR <- file.path("~","Workspace", "doctorate", "projects")
PRJDIR  <- file.path(BASEDIR, "long-lived-bug-prediction", "machine-learning")
SRCDIR  <- file.path(PRJDIR, "R")
LIBDIR  <- file.path(SRCDIR, "lib")
DATADIR <- file.path(PRJDIR, "source", "datasets")

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
processors <- ifelse(IN_DEBUG_MODE, 3, 30)
r_cluster <-  makePSOCKcluster(processors)
registerDoParallel(r_cluster)
project.name    <- "eclipse"
class_label     <- "long_lived"
if(IN_DEBUG_MODE)
{
  #classifiers  <- c(KNN)
  classifiers  <- c(KNN, NB, NNET, RF, SVM)
} else {
  classifiers  <- c(KNN, NB, NNET, RF, SVM)
  flog.appender(
    appender.file(
      file.path(
        DATADIR, 
        sprintf("%s_predict_long_lived_bug_e1_best_tune_rev1.log", timestamp)
      )
    )
  )
}

flog.threshold(TRACE)
flog.trace("Long live prediction Research Question 3 - Experiment 1")
flog.trace("Evaluation metrics ouput path: %s", DATADIR)
flog.trace("Current project name : %s", project.name)

metrics.file   = sprintf("20190918180550_rq4e1_%s_predict_long_lived_metrics.csv", project.name)
metrics.path   = file.path(DATADIR, metrics.file)
metrics.data   = read_csv(metrics.path)
all.best.metrics   = metrics.data[FALSE, ]

flog.trace("Reading the best model by classifier")
for(predictor in classifiers){
  best.metrics = metrics.data %>% 
        filter(classifier == predictor & metric != "Kappa") %>%
        top_n(1, balanced_acc)
  all.best.metrics <- rbind(all.best.metrics, best.metrics)
}
all.best.metrics$hyper1 <- ""
all.best.metrics$value1 <- 0 
all.best.metrics$hyper2 <- ""
all.best.metrics$value2 <- 0  
all.best.metrics$hyper3 <- ""
all.best.metrics$value3 <- 0  

reports.file <- file.path(DATADIR, sprintf("20190917_%s_bug_report_data.csv", project.name))
flog.trace("Bug report file name: %s", reports.file)

reports <- read_csv(reports.file, na  = c("", "NA"))
if ( IN_DEBUG_MODE ){
  flog.trace("DEBUG_MODE: Sample bug reports dataset")
  set.seed(DEFAULT_SEED)
  reports <- sample_n(reports, 1000) 
}

# feature engineering. 
reports <- reports[, c('bug_id', 'short_description', 'long_description' , 'bug_fix_time')]
reports <- reports[complete.cases(reports), ]
#
flog.trace("Clean text features")
reports$short_description <- clean_text(reports$short_description)
reports$long_description  <- clean_text(reports$long_description)

for (row in 1:nrow(all.best.metrics)) {
  set.seed(DEFAULT_SEED)
  
  parameter <- all.best.metrics[row, ]
  flog.trace("Current parameters:\n N.Terms...: [%d]\n Classifier: [%s]\n Metric...: [%s]\n Feature...: [%s]\n Threshold.: [%s]\n Balancing.: [%s]\n Resampling: [%s]"
           , parameter$n_term , parameter$classifier , parameter$metric, parameter$feature
           , parameter$threshold , parameter$balancing , parameter$resampling)
  
  flog.trace("Converting dataframe to term matrix")
  flog.trace("Text mining: extracting %d terms from %s", parameter$n_term, parameter$feature)
  reports.dataset <- convert_to_term_matrix(reports, parameter$feature, parameter$n_term)
  flog.trace("Text mining: extracted %d terms from %s", ncol(reports.dataset) - 2, parameter$feature)
  
  flog.trace("Partitioning dataset in training and testing")
  reports.dataset$long_lived <- as.factor(ifelse(reports.dataset$bug_fix_time <= parameter$threshold, "N", "Y"))
  in_train <- createDataPartition(reports.dataset$long_lived, p = 0.75, list = FALSE)
  train.dataset <- reports.dataset[in_train, ]
  test.dataset  <- reports.dataset[-in_train, ]
   
  flog.trace("Balancing training dataset")
  balanced.dataset = balance_dataset(
    train.dataset,  
    class_label,  
    c("bug_id", "bug_fix_time"), 
    parameter$balancing
  )
  
  X_train <- subset(balanced.dataset, select=-c(long_lived))
  
  y_train <- balanced.dataset[, class_label]
  X_test  <- subset(test.dataset, select=-c(bug_id, bug_fix_time, long_lived))
  y_test  <- test.dataset[, class_label]

  flog.trace("Training the prediction model ")
  fit_control <- get_resampling_method(parameter$resampling)
  fit_model   <- train_with (.x=X_train, 
                             .y=y_train, 
                             .classifier=parameter$classifier,
                             .control=fit_control,
                             .metric=parameter$metric)
  
  flog.trace("Recording best tune hyperparameters in CSV file")
  all.best.metrics[row, "hyper1"] = names(fit_model$bestTune)[1]
  all.best.metrics[row, "value1"] = fit_model$bestTune[[1]] 

  if ((parameter$classifier == SVM) | 
      (parameter$classifier == NB)  | 
      (parameter$classifier == NNET))
  {
    all.best.metrics[row, "hyper2"] = names(fit_model$bestTune)[2]
    all.best.metrics[row, "value2"] = fit_model$bestTune[[2]] 
  } 

  if (parameter$classifier == NB) 
  {
    all.best.metrics[row, "hyper3"] = names(fit_model$bestTune)[3]
    all.best.metrics[row, "value3"] = fit_model$bestTune[[3]] 
  }
}

results.file  <- sprintf( "%s_rq3e1_all_best_train_tunes_%s.csv", timestamp, ifelse(IN_DEBUG_MODE, "debug", "final")) 

write_csv(all.best.metrics,  file.path(DATADIR, results.file))
flog.trace("Training best tune hyperparameters recorded on CSV file.")