import logging
import os
from datetime import date

import numpy as np
from sklearn.model_selection import train_test_split

from load_data import load_bug_report_data

today = date.today().strftime('%Y%m%d')

PROJECTS    = ['eclipse', 'freedesktop', 'gnome', 'gcc', 'mozilla', 'winehq']
INPUT_DATA_PATH  = 'data/raw/'
OUTPUT_DATA_PATH = 'data/clean/'
LOG_PATH    = 'output/logs/'

# setup logging
#logging.basicConfig(filename=os.path.join(LOG_PATH, 'prepare-data-no-stop-words.log')
#    , filemode='w'
#    , format='%(asctime)s - %(levelname)s: %(message)s'
#    , level=logging.INFO)

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s'
    , level=logging.INFO)

# processing data for each project.
for project in PROJECTS:
    logging.info('Starting processing data of %s project', project)
    filename = '{}_bug_report_data.csv'.format(project)
    filepath = os.path.join(INPUT_DATA_PATH, filename)
    full_set = load_bug_report_data(filepath)
    median   = np.median(full_set['bug_fix_time'].values)
    logging.info('Bug fixing time median: {}'.format(median))

    # label bug report 
    full_set['label'] = full_set['bug_fix_time'].apply(
        lambda t: 1 if t > median else 0
    )

    label_counts = full_set['label'].value_counts()

    logging.info('Bugs with label 0={},  1={}'.format(label_counts[0], label_counts[1]))

    train_set, test_set = train_test_split(
        full_set[['bug_id', 'long_description', 'label']], 
        stratify=full_set['label'],
        random_state=42
    )
    logging.info('Train shape: {}'.format(train_set.shape))
    label_counts = train_set['label'].value_counts()
    logging.info('Bugs with label 0={},  1={}'.format(label_counts[0], label_counts[1]))

    logging.info('Test  shape: {}'.format(test_set.shape))
    label_counts = test_set['label'].value_counts()
    logging.info('Bugs with label 0={},  1={}'.format(label_counts[0], label_counts[1])) 

    train_set.to_csv(os.path.join(OUTPUT_DATA_PATH, '{}_{}_bug_report_train_data.csv'.format(today, project)), index=False)
    test_set.to_csv(os.path.join(OUTPUT_DATA_PATH, '{}_{}_bug_report_test_data.csv'.format(today, project)), index=False)
    logging.info('Finishing processing data of %s project', project)
    
    