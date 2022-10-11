import logging
import os

import numpy as np
import pandas as pd
import sklearn

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

REFERENCE  = '20210305'
PROJECTS   = ['eclipse', 'freedesktop', 'gnome', 'gcc', 'mozilla', 'winehq']
INPUT_DATA_PATH     = 'data/clean/'
OUTPUT_DATA_PATH    = 'data/dtms/'
OUTPUT_LOGS_PATH    = 'output/logs/'

def extract_features(reports, max_features=128):
    """Extract features from bug reports using tf-idf 
    
       Args:
            reports(dataframe): bug reports dataframe.
            max_features(int) : maximum number os features.

       Returns:
            features(numpy array): a tf-idf array of features.
            labels(num array): a numpy array of labels.
    """
    class StemmTokenizer:
        def __init__(self):
            self.ss = SnowballStemmer('english', ignore_stopwords=True)
        def __call__(self, doc):
            return [self.ss.stem(t) for t in word_tokenize(doc)]

    vectorizer = TfidfVectorizer(tokenizer=StemmTokenizer(), max_features=max_features, stop_words='english')
    ids        = reports[0].values
    features   = vectorizer.fit_transform(reports[1]).toarray() 
    labels     = reports[2].values

    return (ids, features, labels)

def read_reports(reference, project, step):
    """Read bugs reports from file path.
    
       Args:
            full_filename(string): filename with filepath 

       Returns:
            reports(dataframe): a dataframe with bug reports.
    """

    reports_path = os.path.join(INPUT_DATA_PATH, '{}_{}_bug_report_{}_data.csv'.format(reference, project, step))
    reports = pd.read_csv(reports_path, header=None, skiprows=1, low_memory=False)
    return reports

def save_features(reference, project, step, ids, features, labels):
    dtm_path = os.path.join(OUTPUT_DATA_PATH, '{}_{}_bug_report_{}_data'.format(reference, project, step))
    np.save(dtm_path, np.column_stack((ids, features, labels))) 

# setup logging
#logging.basicConfig(filename=os.path.join(OUTPUT_LOGS_PATH, 'extract-feature-w-tfidf.log')
#    , filemode='w'
#    , format='%(asctime)s - %(levelname)s: %(message)s'
#    , level=logging.INFO)
#
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s'
    , level=logging.INFO)

logging.info('Sklearn version: {}'.format(sklearn.__version__))

for project in PROJECTS:
    logging.info('Starting processing {} dataset'.format(project))

    for step in ['train', 'test']:
        logging.info(f'>> Processing {step} step')
        logging.info(f'Loading reports data')
        reports = read_reports(REFERENCE, project, step)

        logging.info('Extracting features from reports')
        ids, features, labels  = extract_features(reports)
        logging.info(f'Features and labels shapes: {features.shape}, {labels.shape}')

        logging.info('Saving features into file')
        save_features(REFERENCE, project, step, ids, features, labels)

    logging.info('Processing {} dataset finished.'.format(project))

logging.info('Finishing extraction')
