import logging
import os
import sys
from datetime import date

import pandas as pd

sys.path.append('src/bert/2-feature-extraction/')
import save_load_helpers as slh
import train_test_helpers as tth

today = date.today().strftime('%Y%m%d')
OUTPUT_LOGS_PATH = 'output/logs/'
INPUT_DATA_PATH = 'data/dtms/'
OUTPUT_DATA_PATH = 'data/metrics/feature-extraction'
PROJECTS = ['eclipse', 'freedesktop', 'gcc', 'gnome', 'mozilla', 'winehq']
DEBUG_ON = 0

# setup logging
if DEBUG_ON:
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
else:
    logging.basicConfig(filename=os.path.join(OUTPUT_LOGS_PATH, 'predict-long-lived-w-tfidf-save-best-param.log'),
                        filemode='w', format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

logging.info('Starting predicting long-lived bugs')
logging.info('Python version: {}'.format(sys.version))

for project in PROJECTS:
    logging.info('Loading dtm data for {}'.format(project))

    X_train, y_train = slh.load_dtms_data(project, INPUT_DATA_PATH, 'train')
    X_test, y_test = slh.load_dtms_data(project, INPUT_DATA_PATH, 'test')

    logging.info('Setup predictor algorithms.')

    best_params = []
    classifiers = tth.build_classifiers()
    for classifier_name, classifier_estimator, parameters in classifiers:
        logging.info(f"Training {classifier_name} classifier.")
        model = tth.make_model(
            X_train, y_train, classifier_estimator, parameters)
        logging.info("The best params: {}".format(model.best_params_))
        best_params.append(
            [project, classifier_name, 'tf-idf', model.best_params_])

    logging.info('Saving params into csv file.')
    best_params_filename = os.path.join(OUTPUT_DATA_PATH,
                                        '{}_{}_feature_extraction_tfidf_train_params.csv'.format(
                                            today, project)
                                        )
    best_params_df = pd.DataFrame(
        best_params, columns=['project', 'classifier', 'feature-extraction', 'params'])
    best_params_df.to_csv(best_params_filename, index=False)

    logging.info('Finishing predicting long-lived bugs')
