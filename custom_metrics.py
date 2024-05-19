import numpy as np
from sklearn import metrics
from scipy import stats
import warnings 



def MSE_score(targets, preds, mask):
    assert targets.ndim == preds.ndim == 2
    assert targets.shape == preds.shape, (targets.shape, mask.shape)
    values = []
    
    for i in range(len(targets)):
        m = mask[i]
        if np.all(~m):
            break
        t = targets[i][m]
        p = preds[i][m]
        if t.size == 0:
            values.append(0.0)
        else:
            values.append(np.sqrt(metrics.mean_squared_error(t, p)))
    
    return np.array(values)


def NSE_score(targets, preds, mask):
    assert targets.ndim == preds.ndim == 2
    assert targets.shape == preds.shape, (targets.shape, mask.shape)
    values = []
    
    for i in range(len(targets)):
        m = mask[i]
        t = targets[i][m]
        p = preds[i][m]
        
        if t.size == 0:
            values.append(1)
        else:
            values.append(metrics.r2_score(t, p)) 
    
    return 1 / ( 2 - np.array(values))


def pearson(targets, preds, mask):
    assert targets.ndim == preds.ndim == 2
    assert targets.shape == preds.shape, (targets.shape, preds.shape)
    values = []
    
    valid = 0
    for i in range(len(targets)):
        m = mask[i]
        t = targets[i][m]
        p = preds[i][m]
        
        if t.size == 0:
            values.append(1)
        else:
            with warnings.catch_warnings(record=True) as caught_list:
                v = stats.pearsonr(t, p)[0] ## Usually prediction is Nan if target is constant
                increment_valid = False
                for c in caught_list:
                    if isinstance(c.message, stats.ConstantInputWarning):
                        v = 0 
                        increment_valid = True
                if increment_valid: valid += 1
            values.append(float(v))
    
    values = np.array(values)
    
    return values


def CSI_score(targets, preds, mask, threshold=0.001):
    assert targets.ndim == preds.ndim == 2
    assert targets.shape == preds.shape, (targets.shape, mask.shape)
    values = []
    
    threshold = np.log(1 + threshold/1e-2)
    
    y_true = targets > threshold
    y_preds = preds > threshold 
    
    for i in range(len(targets)):
        t = y_true[i]
        p = y_preds[i]

        
        tp = t * p 
        fp = (~t) * p 
        fn = t * (~p)
        
        tp = tp.sum() 
        fp = fp.sum()
        fn = fn.sum()
        
        csi = tp / (tp + fp + fn + 1e-8)
        
        values.append(csi)
    
    return np.array(values)
