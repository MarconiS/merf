import copy as _copy
import os as _os

import numpy as _np
from sklearn import calibration as _calibration
from sklearn import ensemble as _ensemble
from sklearn import utils as _utils

_path = _os.path.realpath(__file__)


def match_species_ids(crown_id, label_id, labels):
    """Matches the crown IDs from the training data with the labels associated with species IDs
    
    Args:
        crown_id - an array of crown IDs from the training data
        label_id - an array of crown IDs associated with the labeled data
        labels   - an array of labels (e.g., species names, species codes)
                   label_id and labels should be of the same size
                 
    Returns:
        [unique_labels, unique_crowns, crown_labels]
        unique_labels - a list of the unique entities from the input labels variable
        unique_crowns - a list of the unique crown entities from the crown_id variable
        crown_labels  - an array with the labels aligned with the original shape of crown_id
    """
    # get the unique labels and crown id's
    unique_labels = _np.unique(labels)
    unique_crowns = _np.unique(crown_id)
    n_crowns = len(unique_crowns)

    # set up the output array
    nchar = _np.max([len(label) for label in unique_labels])
    crown_labels = _np.chararray(len(crown_id), itemsize=nchar)

    for i in range(n_crowns):
        index_crown = crown_id == unique_crowns[i]
        index_label = label_id == unique_crowns[i]
        crown_labels[index_crown] = labels[index_label]

    return [unique_labels, unique_crowns, crown_labels]


def get_sample_weights(y):
    """Calculates the balanced sample weights for a set of unique classes
    
    Args:
        y - the input class labels
        
    Returns:
        weights_sample - an array of length (y) with the per-class weights per sample
    """
    # get the unique classes in the array
    classes = _np.unique(y)
    n_classes = len(classes)

    # calculate the per-class weights
    weights_class = _utils.class_weight.compute_class_weight('balanced', classes, y)

    # create and return an array the same dimensions as the input y vector
    weights_sample = _np.zeros(len(y))
    for i in range(n_classes):
        ind_y = y == classes[i]
        weights_sample[ind_y] = weights_class[i]

    return weights_sample

