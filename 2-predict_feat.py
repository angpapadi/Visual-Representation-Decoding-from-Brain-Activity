"""
Visual Representation Decoding from Human Brain Activity

Based on the code from KamitaniLab/GenericObjectDecoding

"""

from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import os
import pickle
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
import bdpy
from bdpy.bdata import concat_dataset
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.distcomp import DistComp
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras

import god_config as config

#____________________________FLAGS________________________________#
CAFFEflag = False                   # if true use features extracted from caffe model, otherwise use the saved features from the ImageFeatures.h5 file
cboflag = False

#_______________________HYPERPARAMETERS___________________________#
MODELS = ['lineareg', 'ridge', 'mlp', 'knn', 'kernelreg']
MODELOPTION = MODELS[3]             # model that regresses brain data to target feature vectors

# keras mlp parameters
NEURONSPERLAYER = 300
NEURONSOUTPUT = 1000
BATCH = 128
EPOCHS = 100
OPTIMIZER = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# scikit-learn models parameters
NEIGHBORS = 5                                       # for knn regression
KERNEL = 'polynomial'                               # kernel for kernel ridge regression
DEGREE = 2                                          # degree for kernel ridge regression for polynomial kernels
ALPHAS = [0.1,10]                                   # alpha parameters for ridge regression


def main():
    # Settings ---------------------------------------------------------

    # Data settings
    subjects = config.subjects
    rois = config.rois
    num_voxel = config.num_voxel

    if CAFFEflag :
        if cboflag:
            image_feature1 = '/home/akpapadim/Desktop/RemoteThesis/cbof-kamitani/data/ImageFeatures_caffe_cbof.pkl'
        else:
            image_feature1 = '/home/akpapadim/Desktop/RemoteThesis/cbof-kamitani/data/ImageFeatures_caffe.pkl'

    image_feature = config.image_feature_file
    features = config.features
    results_dir = config.results_dir

    # Misc settings
    analysis_basename = os.path.basename(__file__)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)

    data_feature = bdpy.BData(image_feature)

    # check which features file to open
    if CAFFEflag:
        data_feature1 = pd.read_pickle(image_feature1)
        print('From file ', image_feature1)
    elif cboflag == False:
        print('From file ', image_feature)

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for sbj, roi, feat in product(subjects, rois, features):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])
        print('Feature:    %s' % feat)

        # Distributed computation
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        if os.path.exists(results_file):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        dist = DistComp(lockdir='tmp', comp_id=analysis_id)
        if dist.islocked():
            print('%s is already running. Skipped.' % analysis_id)
            continue

        dist.lock()

        # Prepare data
        print('Preparing data')
        dat = data_all[sbj]

        x = dat.select(rois[roi])  # Brain data
        datatype = dat.select('DataType')  # Data type
        labels = dat.select('Label')  # Image labels in brain data

        if CAFFEflag:
            yold = data_feature.select(feat)  # Image features

            y = data_feature1[feat]
            y_label = data_feature1['ImageID']

            if cboflag == False:
                y = np.concatenate(y).reshape(y.shape[0],y[0].shape[0]) # reshape to 1250, 1000
                y_label = y_label.reshape(y.shape[0],1)

        else:
            y = data_feature.select(feat)  # Image features
            y_label = data_feature.select('ImageID')  # Image labels

        y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

        # alternative sorting method is the same as get_refdata
        """
        object_map = {}
        for i in range(len(y)):
            key = y_label[i][0]
            object_map[key]= y[i]

        y_sorted2 = [object_map[id[0]] for id in labels]
        """

        # Get training and test dataset
        i_train = (datatype == 1).flatten()  # Index for training
        i_test_pt = (datatype == 2).flatten()  # Index for perception test
        i_test_im = (datatype == 3).flatten()  # Index for imagery test
        i_test = i_test_pt + i_test_im

        x_train = x[i_train, :]
        x_test = x[i_test, :]

        y_train = y_sorted[i_train, :]
        y_test = y_sorted[i_test, :]

        # Feature prediction
        pred_y, true_y = feature_prediction(x_train, y_train,
                                            x_test, y_test,
                                            modeloption= MODELOPTION)
        print('Model: ',MODELOPTION)

        #pred_y = true_y # suppose ideal regression

        # Separate results for perception and imagery tests
        i_pt = i_test_pt[i_test]  # Index for perception test within test
        i_im = i_test_im[i_test]  # Index for imagery test within test

        pred_y_pt = pred_y[i_pt, :]
        pred_y_im = pred_y[i_im, :]

        true_y_pt = true_y[i_pt, :]
        true_y_im = true_y[i_im, :]

        # Get averaged predicted feature
        test_label_pt = labels[i_test_pt, :].flatten()
        test_label_im = labels[i_test_im, :].flatten()

        pred_y_pt_av, true_y_pt_av, test_label_set_pt \
            = get_averaged_feature(pred_y_pt, true_y_pt, test_label_pt)
        pred_y_im_av, true_y_im_av, test_label_set_im \
            = get_averaged_feature(pred_y_im, true_y_im, test_label_im)

        # Get category averaged features
        catlabels_pt = np.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
        catlabels_im = np.vstack([int(n) for n in test_label_im])  # Category labels (imagery test)
        catlabels_set_pt = np.unique(catlabels_pt)  # Category label set (perception test)
        catlabels_set_im = np.unique(catlabels_im)  # Category label set (imagery test)

        
        if CAFFEflag:

            yold_catlabels = data_feature.select('CatID')  # Category labels in image features
            ind_catave = (data_feature.select('FeatureType') == 3).flatten()  # boolean mask of featuretype

            y_catave_pt = get_refdata(yold[ind_catave, :], yold_catlabels[ind_catave, :], catlabels_set_pt)
            y_catave_im = get_refdata(yold[ind_catave, :], yold_catlabels[ind_catave, :], catlabels_set_im)

        else:
            y_catlabels = data_feature.select('CatID')  # Category labels in image features
            ind_catave = (data_feature.select('FeatureType') == 3).flatten() #boolean mask of featuretype

            y_catave_pt = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_pt)
            y_catave_im = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_im)


        # Prepare result dataframe
        results = pd.DataFrame({'subject': [sbj, sbj],
                                'roi': [roi, roi],
                                'feature': [feat, feat],
                                'test_type': ['perception', 'imagery'],
                                'true_feature': [true_y_pt, true_y_im],
                                'predicted_feature': [pred_y_pt, pred_y_im],
                                'test_label': [test_label_pt, test_label_im],
                                'test_label_set': [test_label_set_pt, test_label_set_im],
                                'true_feature_averaged': [true_y_pt_av, true_y_im_av],
                                'predicted_feature_averaged': [pred_y_pt_av, pred_y_im_av],
                                'category_label_set': [catlabels_set_pt, catlabels_set_im],
                                'category_feature_averaged': [y_catave_pt, y_catave_im]})

        # Save results
        makedir_ifnot(os.path.dirname(results_file))
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print('Saved %s' % results_file)

        dist.unlock()


#___________________________FUNCTIONS_____________________________#

def model_predictions(modeloption, x_train, y_train, x_test, y_test):

    if modeloption == 'lineareg':
        model = LinearRegression()
        model.fit(x_train, y_train)  # Training
        y_predicted = model.predict(x_test)  # Test

    elif modeloption == 'ridge':
        model = RidgeCV(alphas = ALPHAS)
        model.fit(x_train, y_train)
        print(model.alpha_)
        y_predicted = model.predict(x_test)

    elif modeloption == 'mlp':
        # Build Keras model
        model = Sequential()
        """
        #model.add(keras.layers.Dropout(0.2, input_shape=(x_train.shape[1],)))
        model.add(Dense(NEURONSPERLAYER, input_shape =(x_train.shape[1],)))
        model.add(Activation('relu'))
        #model.add(keras.layers.Dropout(0.2))
        #model.add(Dense(NEURONSPERLAYER))
        #model.add(Activation('relu'))
        #model.add(keras.layers.Dropout(0.2))
        model.add(Dense(NEURONSOUTPUT))
        model.add(Activation('linear'))
        """
        #trial11
        #model.add(keras.layers.Dropout(0.3, input_shape=(x_train.shape[1],)))
        model.add(Dense(NEURONSPERLAYER, activation='sigmoid', input_shape=(x_train.shape[1],)))
        model.add(Dense(1000, activation=None))
        model.compile(loss='mean_squared_error', optimizer=OPTIMIZER)

        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose =0)
        y_predicted = model.predict(x_test, batch_size=BATCH, verbose=0, steps=None)

        # show training loss and test loss
        print(history.history['loss'])
        print(model.evaluate(x_test, y_test, batch_size=BATCH, verbose =0))

    elif modeloption == 'knn':
        model = KNeighborsRegressor(n_neighbors= NEIGHBORS, weights='distance')
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)

    elif modeloption == 'kernelreg':
        model = KernelRidge(kernel= KERNEL, degree=DEGREE, alpha=0.001, coef0=10)
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)


    return y_predicted


def feature_prediction(x_train, y_train, x_test, y_test, modeloption):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''

    # Normalize brain data
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    print('Running feature prediction')

    # Normalize labels
    norm_mean_y=[]
    norm_scale_y=[]
    for i in range(y_train.shape[1]):
        column = y_train[:, i]

        mean_y = np.mean(column,axis=0)
        norm_mean_y.append(mean_y)

        std_y = np.std(column,axis=0,ddof=1)
        scale_y = 1 if std_y == 0 else std_y
        norm_scale_y.append(scale_y)

    y_train = (y_train - norm_mean_y) / norm_scale_y

    # Fit the model and get y_predicted
    y_predicted = model_predictions(modeloption, x_train, y_train, x_test, y_test)

    # Denormalize predicted features
    y_predicted = y_predicted * norm_scale_y + norm_mean_y

    y_true = y_test
    return y_predicted, y_true


def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


#________________________RUN AS A SCRIPT__________________________#
if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
