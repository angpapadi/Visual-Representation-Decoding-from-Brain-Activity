from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd

import bdpy
import sklearn.metrics as metrics

import god_config as config

SIM_METRICS = ['pearson', 'euclidean', 'cosine']
SIMETRIC = SIM_METRICS[1]


# Main #################################################################
def main():
    output_file = config.results_file

    image_feature_file = config.image_feature_file

    # Load results -----------------------------------------------------
    print('Loading %s' % output_file)
    with open(output_file, 'rb') as f:
        results = pickle.load(f)

    data_feature = bdpy.BData(image_feature_file)

    # Category identification ------------------------------------------
    print('Running pair-wise category identification')

    feature_list = results['feature']
    pred_percept = results['predicted_feature_averaged_percept']        #same shape
    pred_imagery = results['predicted_feature_averaged_imagery']        #same shape
    cat_label_percept = results['category_label_set_percept']
    cat_label_imagery = results['category_label_set_imagery']
    cat_feature_percept = results['category_feature_averaged_percept']  #same shape
    cat_feature_imagery = results['category_feature_averaged_imagery']  #same shape

    ind_cat_other = (data_feature.select('FeatureType') == 4).flatten()

    pwident_cr_pt = []  # Prop correct in pair-wise identification (perception)
    pwident_cr_im = []  # Prop correct in pair-wise identification (imagery)

    for f, fpt, fim, pred_pt, pred_im in zip(feature_list, cat_feature_percept, cat_feature_imagery,
                                             pred_percept, pred_imagery):
        # for all configurations (subjects, cnn layer features, regions of interest) = 400 iterations
        feat_other = data_feature.select(f)[ind_cat_other, :]

        n_unit = fpt.shape[1]
        feat_other = feat_other[:, :n_unit]

        feat_candidate_pt = np.vstack([fpt, feat_other])    # feat_candidate_pt.shape = (15372,1000) = (synsets, features)
        feat_candidate_im = np.vstack([fim, feat_other])

        simmat_pt = corrmat(pred_pt, feat_candidate_pt, simetric = SIMETRIC)     # similarity matrix (50,15372) = (samples, synsets)
        simmat_im = corrmat(pred_im, feat_candidate_im, simetric = SIMETRIC)

        cr_pt = get_pwident_correctrate(simmat_pt)          # mysterious function. cr_pt.shape = (50,1)
        cr_im = get_pwident_correctrate(simmat_im)

        pwident_cr_pt.append(np.mean(cr_pt))
        pwident_cr_im.append(np.mean(cr_im))

    results['catident_correct_rate_percept'] = pwident_cr_pt    # pwident_cr_pt.shape = (50,1)
    results['catident_correct_rate_imagery'] = pwident_cr_im

    # Save the merged dataframe ----------------------------------------
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print('Saved %s' % output_file)

    # Show results -----------------------------------------------------
    tb_pt = pd.pivot_table(results, index=['roi'], columns=['feature'],
                           values=['catident_correct_rate_percept'], aggfunc=np.mean)
    tb_im = pd.pivot_table(results, index=['roi'], columns=['feature'],
                           values=['catident_correct_rate_imagery'], aggfunc=np.mean)

    print(tb_pt)
    print(tb_im)


# Functions ############################################################

def get_pwident_correctrate(simmat):
    '''
    Returns correct rate in pairwise identification

    Parameters
    ----------
    simmat : numpy array [num_prediction * num_category]
        Similarity matrix

    Returns
    -------
    correct_rate : correct rate of pair-wise identification
    '''

    num_pred = simmat.shape[0]
    labels = range(num_pred)

    correct_rate = []
    for i in xrange(num_pred):
        pred_feat = simmat[i, :]
        correct_feat = pred_feat[labels[i]]
        pred_num = len(pred_feat) - 1
        correct_rate.append((pred_num - np.sum(pred_feat > correct_feat)) / float(pred_num))

    return correct_rate


def corrmat(x, y, simetric ='pearson', var='row'):
    """
    Returns correlation matrix between `x` and `y`
    Parameters
    ----------
    x, y : array_like
        Matrix or vector
    var : str, 'row' or 'col'
        Specifying whether rows (default) or columns represent variables
    Returns
    -------
    rmat
        Correlation matrix
    """

    if simetric == 'pearson':
        # Fix x and y to represent variables in each row
        if var == 'row':
            pass
        elif var == 'col':
            x = x.T
            y = y.T
        else:
            raise ValueError('Unknown var parameter specified')

        nobs = x.shape[1]

        # Subtract mean(a, axis=1) from a
        submean = lambda a: a - np.matrix(np.mean(a, axis=1)).T

        cmat = (np.dot(submean(x), submean(y).T) / (nobs - 1)) / np.dot(np.matrix(np.std(x, axis=1, ddof=1)).T,
                                                                    np.matrix(np.std(y, axis=1, ddof=1)))

    elif simetric == 'euclidean':
        cmat = metrics.pairwise.euclidean_distances(x,y)
        cmat = 1 / (1 + cmat)

    elif simetric == 'cosine':
        cmat = metrics.pairwise.cosine_similarity(x,y)

    else:
        raise ValueError('Similarity metric not known')

    return np.array(cmat)


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
