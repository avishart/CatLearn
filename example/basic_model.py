""" Script to test the ML model. Takes a database of candidates from a GA
    search with target values set in atoms.info['key_value_pairs'][key] and
    returns the errors for a random test and training dataset.
"""
from __future__ import print_function
from __future__ import absolute_import

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train
from atoml.fingerprint_setup import normalize, return_fpv
from atoml.particle_fingerprint import ParticleFingerprintGenerator
from atoml.predict import GaussianProcess
from atoml.feature_select import clean_zero

# Decide whether to remove output and print graph.
cleanup = True
plot = False

# Define starting guess for hyperparameters.
width = 0.5
reg = 0.001

# Connect database generated by a GA search.
db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
print('Getting candidates from the database')
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=500, key='raw_score')
trainset = get_train(candidates=all_cand, trainsize=500,
                     taken_cand=testset['taken'], key='raw_score')

# Get the list of fingerprint vectors and normalize them.
print('Getting the fingerprint vectors')
fpv = ParticleFingerprintGenerator(get_nl=False, max_bonds=13)
test_fp = return_fpv(testset['candidates'], [fpv.nearestneighbour_fpv])
train_fp = return_fpv(trainset['candidates'], [fpv.nearestneighbour_fpv])

c = clean_zero(train=train_fp, test=test_fp)
test_fp = c['test']
train_fp = c['train']


def do_predict(train, test, train_target, test_target, hopt=False):
    # Scale features.
    nfp = normalize(train=train, test=test)

    # Do the predictions.
    pred = gp.get_predictions(train_fp=nfp['train'],
                              test_fp=nfp['test'],
                              train_target=train_target,
                              test_target=test_target,
                              get_validation_error=True,
                              get_training_error=True,
                              optimize_hyperparameters=hopt)

    if plot:
        pred['actual'] = test_target
        index = [i for i in range(len(test_fp))]
        df = pd.DataFrame(data=pred, index=index)
        with sns.axes_style("white"):
            sns.regplot(x='actual', y='prediction', data=df)
        plt.title('Validation RMSE: {0:.3f}'.format(
            pred['validation_rmse']['average']))
        plt.show()

    return pred


# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 0.5}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

print('Original parameters')
a = do_predict(train=train_fp, test=test_fp, train_target=trainset['target'],
               test_target=testset['target'], hopt=False)

# Print the error associated with the predictions.
print('Training error:', a['training_rmse']['average'])
print('Model error:', a['validation_rmse']['average'])

# Try with hyperparameter optimization.
print('Optimized parameters')
a = do_predict(train=train_fp, test=test_fp, train_target=trainset['target'],
               test_target=testset['target'], hopt=True)

# Print the error associated with the predictions.
print('Training error:', a['training_rmse']['average'])
print('Model error:', a['validation_rmse']['average'])