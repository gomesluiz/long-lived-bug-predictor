import logging
import os
import sys

import numpy as np
import pandas as pd
import sklearn

from scipy.sparse.construct import random
from sklearn.metrics import balanced_accuracy_score

sys.path.append('src/bert/2-feature-extraction/')
import train_test_helpers as tth
import save_load_helpers  as slh

today     = '20210305'
OUTPUT_LOGS_PATH = 'output/logs/'
INPUT_DATA_PATH  = 'data/dtms/'
OUTPUT_DATA_PATH = 'data/metrics/feature-extraction'
#PROJECTS = ['eclipse', 'gcc', 'gnome', 'mozilla', 'winehq']
PROJECTS = ['gnome']
DEBUG_ON = 1
REFERENCE = '20210305'

# setup logging
if DEBUG_ON:
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s'
    , level=logging.INFO)
else:
    logging.basicConfig(filename=os.path.join(OUTPUT_LOGS_PATH, 'predict-long-lived-w-tfidf.log')
        , filemode='w'
        , format='%(asctime)s - %(levelname)s: %(message)s'
        , level=logging.INFO)

logging.info('Starting predicting long-lived bugs')
logging.info('Python version: {}'.format(sys.version))
logging.info('Nunpy version: {}'.format(np.__version__))
logging.info('Pandas version: {}'.format(pd.__version__))
logging.info('Sklearn version: {}'.format(sklearn.__version__))

for project in PROJECTS:
    logging.info('Loading tensor data for {}'.format(project))
    
    X_train, y_train, ids_train = slh.load_dtms_data(today, project, INPUT_DATA_PATH, 'train')
    X_test, y_test, ids_test    = slh.load_dtms_data(today, project, INPUT_DATA_PATH, 'test')
    logging.info(f'Training data shape: {X_train.shape}, {y_train.shape}')
    logging.info(f'Testing  data shape: {X_test.shape}, {y_test.shape}')

    y_train_classes = tth.count_bug_classes(y_train)
    y_test_classes  = tth.count_bug_classes(y_test)

    #logging.info('Loading best parameters')
    #params_path  = os.path.join(OUTPUT_DATA_PATH, f"20210302_{project}_feature_extraction_tfidf_train_params.csv")
    #best_params  = pd.read_csv(params_path)

    logging.info('Setup predictor algorithms.')
    
    train_scores = []
    test_scores  = []
    test_predictions = []
    classifiers  =  tth.build_classifiers(params=None)
    for classifier_name, classifier_estimator, parameters in classifiers:
        logging.info(f"Training {classifier_name} classifier.")
        model = tth.make_model(X_train, y_train, classifier_estimator, parameters)
        (scores, mean, stddev) = tth.score_model(X_train, y_train, model)
        
        for fold, score in enumerate(scores):
            train_scores.append([project, classifier_name, fold, score])

        logging.info(f"Balanced accuracy Mean = {mean:.5f} and Std = {stddev:.5f}")

        logging.info(f"Evaluating {classifier_name} classifier.")
        (test_acc, test_balanced_acc, y_hat) = tth.evaluate_model(X_test, y_test, model)        
        test_scores.append([project, classifier_name, y_train_classes[0], 
            y_train_classes[1], y_test_classes[0], y_test_classes[1], 
            test_acc, test_balanced_acc])
        
        for i in range(len(y_hat)):
            test_predictions.append([project, ids_test[i], classifier_name, y_test[i], y_hat[i]])

        logging.info(f"Accuracy = {test_acc:.5f}, Balanced accuracy= {test_balanced_acc:.5f}")

    logging.info('Saving metrics into csv file.')
    train_scores_filename = os.path.join(OUTPUT_DATA_PATH, 
        '{}_{}_feature_extraction_tfidf_train_scores.csv'.format(today, project)
    )
    train_scores_df = pd.DataFrame(train_scores, columns=['project', 'classifier', 'fold', 'balanced_acc'])
    train_scores_df['feature_extraction'] = 'tf-idf' 
    train_scores_df.to_csv(train_scores_filename, index=False)
    
    test_scores_filename = os.path.join(OUTPUT_DATA_PATH, 
        '{}_{}_feature_extraction_tfidf_test_scores.csv'.format(today, project)
    )
    test_scores_df = pd.DataFrame(test_scores, columns=['project', 'classifier', 'train_class_0', 'train_class_1'
        , 'test_class_0', 'test_class_1', 'acc', 'balanced_acc'])
    test_scores_df['feature_extraction'] = 'tf-idf' 
    test_scores_df.to_csv(test_scores_filename, index=False)

    # predictions
    test_predictions_filename = os.path.join(OUTPUT_DATA_PATH, 
        '{}_{}_feature_extraction_tfidf_test_predictions.csv'.format(today, project)
    )
    test_predictions_df = pd.DataFrame(test_predictions, columns=['project', 'bug_id', 'classifier', 'y_test', 'y_hat'])
    test_predictions_df['feature_extraction'] = 'tfidf' 
    test_predictions_df.to_csv(test_predictions_filename, index=False)

    logging.info('Finishing predicting long-lived bugs')
