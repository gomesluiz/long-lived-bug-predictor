import re
import unicodedata

import numpy as np
import pandas as pd
from nltk.corpus import stopwords


def unicode_to_ascii(s):
    """Converte a string from unicode to ascii

        Args:
            s(str): string to be converted.

        Returns:
            a string converted.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    #w = unicode_to_ascii(w.lower().strip())
    w = w.lower()
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    #w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w


def load_bug_report_data(filepath):
    """Read a bug report data set.

    Args:
        filepath (str): a complete filename path.

    Returns:
        result (dataframe): a bug report dataframe.

    """
    reports = pd.read_csv(filepath, encoding='utf8', sep=',', parse_dates=True
      ,low_memory=False)

    reports.dropna(inplace=True)
    reports['long_description'] = reports['long_description'].map(preprocess_sentence)
    reports['long_description'] = reports['long_description'].replace('', np.nan)
    
    result = reports.loc[:, ('long_description', 'bug_fix_time', 'bug_id')]
    result.dropna(inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result

