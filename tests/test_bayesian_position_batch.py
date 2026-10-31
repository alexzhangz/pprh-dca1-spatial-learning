import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from maze_analysis.bayesian_position import BinnedTrial, GaussianBinDecoder
from maze_analysis.bayesian_position_batch import TrialGaussianCV, confusion_counts, normalize_confusion, make_trial_folds


def test_cached_cv_matches_training_only_direct_fits_with_shifted_labels():
    rng=np.random.default_rng(6)
    trials=[]
    for i,n in enumerate([40,45,39]):
        x=rng.normal(size=(n,5))+1000
        x[:,-1]=7
        if i==0: x[:,-2]=12
        else: x[:,-2]=8
        y=rng.choice([0,13,14,55,56,97],size=n)
        trials.append(BinnedTrial(i,bool(i%2),True,x,y,np.zeros((n,2)),np.arange(n)))
    cached=TrialGaussianCV(trials)
    for y in [cached.y,np.concatenate([np.roll(t.bins,9) for t in trials])]:
        prediction,posterior,counts=cached.predict(y,return_posterior=True)
        for i,t in enumerate(trials):
            train=cached.fold_index!=i
            model=GaussianBinDecoder.fit(cached.x[train],y[train])
            p=model.predict_proba(t.neural)
            np.testing.assert_allclose(posterior[~train],p,atol=3e-8)
            np.testing.assert_array_equal(prediction[~train],p.argmax(axis=1))
            np.testing.assert_array_equal(counts[i],model.counts)


def test_confusion_rows_and_missing_bins():
    counts=confusion_counts([0,0,1],[0,1,1])
    normalized=normalize_confusion(counts)
    np.testing.assert_allclose(normalized[0,:2],[.5,.5])
    assert normalized[1,1]==1
    assert np.isnan(normalized[2:]).all()


def test_five_folds_keep_trials_whole_and_reproduce_direct_fit():
    rng=np.random.default_rng(42)
    trials=[BinnedTrial(i,bool(i%2),True,rng.normal(size=(20,3)),rng.integers(0,98,20),np.zeros((20,2)),np.arange(20)) for i in range(10)]
    folds,method=make_trial_folds(trials,seed=17)
    np.testing.assert_array_equal(folds,make_trial_folds(trials,seed=17)[0])
    assert 'stratified' in method and len(np.unique(folds))==5
    cv=TrialGaussianCV(trials,folds=folds)
    prediction,p,counts=cv.predict(return_posterior=True)
    for i in range(5):
        test=cv.fold_index==i
        model=GaussianBinDecoder.fit(cv.x[~test],cv.y[~test])
        direct=model.predict_proba(cv.x[test])
        np.testing.assert_allclose(p[test],direct,atol=3e-8)
        np.testing.assert_array_equal(prediction[test],direct.argmax(axis=1))
        assert set(folds[np.arange(10)%2==0])==set(range(5))
        assert set(folds[np.arange(10)%2==1])==set(range(5))
