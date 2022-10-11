import os
import glob 

import numpy as np
import torch

def load_tensors_data(reference, project, input_data_path, step="train"):
    """Load pytorch tensor data from file

    Args:
        filepath (str): a full filename path.

    Returns:
        X (array): a numpy array of features.
        y (array): a numpy array of labels.
    """
    tensors_path  = os.path.join(input_data_path, "{}_{}_bug_report_{}_data_*.pt".format(reference, project, step))
    tensors_files = sorted(glob.glob(tensors_path))
    tensors_data  = None 
    for file in tensors_files:
        if tensors_data is None:
            tensors_data = torch.load(file)
        else:
            tensors_data = np.vstack((tensors_data, torch.load(file)))

    X = tensors_data[:, 1:-1]
    y = tensors_data[:, -1].astype(int)
    ids = tensors_data[:, 0]

    return (X, y, ids)

def load_dtms_data(reference, project, input_path, step="train"):
    """Load a dtm numpy array from file

    Args:
        project (str): name of project.

    Returns:
        X (array): a numpy array of features.
        y (array): a numpy array of labels.
    """
    dtm_path = os.path.join(input_path, "{}_{}_bug_report_{}_data.npy".format(reference,project, step))
    dtm_data = np.load(dtm_path, allow_pickle=True)


    ids = dtm_data[:, 0]
    X = dtm_data[:, 1:-1]
    y = dtm_data[:, -1].astype(int)


    return (X, y, ids)