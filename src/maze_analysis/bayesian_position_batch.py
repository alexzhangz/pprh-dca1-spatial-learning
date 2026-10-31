"""Manuscript definitions collected from bayesian_position_batch.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import numpy as np

from scipy.sparse import csr_matrix

from sklearn.model_selection import KFold, StratifiedKFold

from .bayesian_position import GaussianBinDecoder

def make_trial_folds(trials, *, seed, n_splits=5):
    """Split trial IDs, not frames; balance actual choice when counts permit."""
    if len(trials)<n_splits:
        raise ValueError('not enough trials for requested folds')
    sides=np.array([t.actual_left for t in trials],int)
    stratified=np.bincount(sides,minlength=2).min()>=n_splits
    splitter=(StratifiedKFold if stratified else KFold)(n_splits=n_splits,shuffle=True,random_state=int(seed))
    folds=np.full(len(trials),-1,int)
    for fold,(_,test) in enumerate(splitter.split(np.arange(len(trials)),sides)):
        folds[test]=fold
    return folds,('actual-choice stratified trials' if stratified else 'shuffled trials; insufficient minority-side trials for stratification')

def confusion_counts(truth, prediction, n_bins=98):
    truth, prediction = np.asarray(truth, int), np.asarray(prediction, int)
    if truth.shape != prediction.shape or np.any((truth < 0) | (truth >= n_bins)) or np.any((prediction < 0) | (prediction >= n_bins)):
        raise ValueError('invalid confusion inputs')
    return np.bincount(truth*n_bins+prediction, minlength=n_bins*n_bins).reshape(n_bins,n_bins)

def normalize_confusion(counts):
    counts = np.asarray(counts, float)
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts,total,out=np.full_like(counts,np.nan),where=total>0)

class TrialGaussianCV:
    """Cache neural data, recomputing only label-dependent binned sums per shift.

    Session-wide centering is purely a numerically stable coordinate change.
    All model moments and shrinkage targets exclude the held-out trial and
    reproduce direct training-only fits. It is not fitted preprocessing.
    """
    def __init__(self, trials, *, folds=None):
        self.trials = trials
        self.x = np.concatenate([t.neural for t in trials])
        self.y = np.concatenate([t.bins for t in trials])
        self.lengths = np.array([len(t.bins) for t in trials])
        self.bounds = np.r_[0,np.cumsum(self.lengths)]
        folds=np.arange(len(trials)) if folds is None else np.asarray(folds,int)
        if folds.shape!=(len(trials),) or not np.array_equal(np.unique(folds),np.arange(len(np.unique(folds)))) or len(np.unique(folds))<2:
            raise ValueError('invalid trial fold assignments')
        self.n_folds=len(np.unique(folds))
        self.fold_index = np.repeat(folds,self.lengths)
        self.offset = self.x.mean(axis=0)
        self.centered = self.x-self.offset
        self.squared = self.centered**2
        self.query = []
        self.global_var = []
        self.train_center = []
        # Compute training global variance directly once per fold. This avoids
        # cancellation when a neuron is constant in training but not in test.
        for i in range(self.n_folds):
            train = self.x[self.fold_index != i]
            center = train.mean(axis=0)
            variance = (train-center).var(axis=0)
            self.global_var.append(variance)
            self.train_center.append(center)

    def predict(self, bins=None, *, return_posterior=False):
        bins = self.y if bins is None else np.asarray(bins,int)
        if bins.shape != self.y.shape or np.any((bins<0)|(bins>=98)):
            raise ValueError('invalid shifted labels')
        n_trials,n_features = self.n_folds, self.x.shape[1]
        assignment = csr_matrix((np.ones(len(bins)),(self.fold_index*98+bins,np.arange(len(bins)))),shape=(n_trials*98,len(bins)))
        counts = np.asarray(assignment.sum(axis=1)).reshape(n_trials,98)
        sums = (assignment@self.centered).reshape(n_trials,98,n_features)
        squares = (assignment@self.squared).reshape(n_trials,98,n_features)
        total_count,total_sum,total_square = counts.sum(axis=0),sums.sum(axis=0),squares.sum(axis=0)
        predicted = np.empty(len(bins),np.int16)
        posterior = np.empty((len(bins),98),np.float32) if return_posterior else None
        training_counts=[]
        for i in range(self.n_folds):
            count = total_count-counts[i]
            mean = (total_sum-sums[i])/np.maximum(count[:,None],1)
            variance = np.maximum((total_square-squares[i])/np.maximum(count[:,None],1)-mean*mean,0)
            global_var=self.global_var[i]
            variance=.9*variance+.1*global_var
            variance=np.maximum(variance,np.maximum(global_var*1e-6,1e-12))
            model=GaussianBinDecoder(mean+self.offset,variance,count,self.train_center[i],global_var>0)
            test=self.fold_index==i
            probability=model.predict_proba(self.x[test])
            predicted[test]=probability.argmax(axis=1)
            if posterior is not None:
                posterior[test]=probability
            training_counts.append(count.astype(np.int64))
        return predicted,posterior,np.asarray(training_counts)
