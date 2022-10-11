import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def make_test(metrics, classifiers=['knn', 'nb', 'nn', 'rf', 'svm']):
    """Make statistical test on metrics dataset
    
    Args:
        metrics (dataframe): metrics dataframe.

    Returns:
        statistics(dataframe): statistical test dataframe.

    """
    statistics = pd.DataFrame(
        columns=['classifier1', 'mean1', 'classifier2', 'mean2', 'condition', 'direction'])
    alpha = 0.05
    p = 1.0
    for classifier1 in classifiers:
        sample1 = metrics.loc[:, (slice('balanced_acc'), classifier1)].values
        mean1 = np.mean(sample1)
        for classifier2 in classifiers:
            sample2 = metrics.loc[:, (slice('balanced_acc'), classifier2)].values
            mean2 = np.mean(sample2)
            condition = True
            if (classifier1 != classifier2):
                _, p = wilcoxon(sample1.flatten(), sample2.flatten())
                condition = p > alpha

            statistics = statistics.append(
                {'classifier1': classifier1,
                 'mean1': mean1,
                 'classifier2': classifier2,
                 'mean2': mean2,
                 'condition': condition,
                 'direction': '$-$' if condition else '$\\leftarrow$' if mean1 > mean2 else '$\\uparrow$'}, ignore_index=True
            )
    return statistics
