import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def sensitivity(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)


def auc(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    return metrics.auc(fpr, tpr)

def mcc(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.matthews_corrcoef(y_true, y_prob)

def accuracy(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.accuracy_score(y_true, y_prob)

def cutoff(y_true,y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob,drop_intermediate=False)
    return thresholds[np.argmax(np.array(tpr) - np.array(fpr))],np.array(tpr) - np.array(fpr), fpr, tpr, thresholds

def precision(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.precision_score(y_true, y_prob)

def recall(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.recall_score(y_true, y_prob)

def f1(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.f1_score(y_true, y_prob)

def AUPRC(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.average_precision_score(y_true, y_prob)

def cofusion_matrix(y_true,y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()

    return tn, fp, fn, tp


