# numerics and data packages
import pandas as pd
import numpy as np

# sklearn
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

# imbalanced
from imblearn.metrics import classification_report_imbalanced 
from imblearn.metrics import sensitivity_specificity_support

# depreciated package be careful
try:
    import scikitplot as skplt
except ModuleNotFoundError:
    print("Scikit plot not avaliable not all plotting features are available")

from . import plotting_sklearn as pltsk

# matplotlib
import matplotlib.pyplot as plt

# logging
import logging 


def get_confusion_matrix(df, predicted_column_name="prediction", known_column_name="known", labels=(0,1), return_dict=False):
    """
    Function to produce get the confusion matrix from a classification result
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param predicted_column_name: str - the column name containing the predicted classes
    :param known_column_name: str - the column name containing the ground truth known classes
    :param labels: tuple - the labels used for the classes
    :param return_dict: bool - return the confusion matrix as a dictionary with keys tp, tn, fp, fn rather than an array
    """

    log = logging.getLogger(__name__)

    c_matrix = confusion_matrix(df[known_column_name].values, df[predicted_column_name].values, labels=labels)
    
    log.debug(c_matrix)

    if return_dict is True:
        c_matrix = confusion_matrix_to_dict(c_matrix)

    return c_matrix


def confusion_matrix_to_dict(cm):
    """
    Convert confusion matrix to dict
    :param cm: confusion matrix from sklearn
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # 
    # TN | FP
    #----|----
    # FN | TP
    #

    cm_d = {"tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1]
           }

    return cm_d


def accuracy_percentage(df, prediction_column="prediction", known_column="known"):
    """
    Function to calculate the accuracy of a classification
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param prediction_column: str - the column name containing the predicted classes
    :param known_column: str - the column name containing the ground truth known classes
    """
    log = logging.getLogger(__name__)

    log.debug(df)

    df_tmp = df[df[prediction_column] == df[known_column]]
    acc = (len(df_tmp) / len(df.index)) * 100.0

    return acc

def g_mean(cm_d):
    """
    """
    
    sensitivity = tpr(cm_d)
    specificity = tnr(cm_d)
    
    g = np.sqrt(sensitivity * specificity)
    
    return g

def accuracy(cm):
    """
    Function to calculate the accuracy from a classification using a confusion matrix
    :param cm: confusion matrix dataframe from sklearn or dictionary from get_confusion_matrix(return_dict=True)
    """
    if isinstance(cm, dict):
        ac = (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])
    else:
        cm_acc = cm.diagonal()/matrix.sum(axis=1)
        ac = cm_acc.diagonal()

    return ac

def tpr(cm_d):
    """
    Function to calculate the true postive rate also known as sensitivity
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tp"] / (cm_d["tp"] + cm_d["fn"])

def tnr(cm_d):
    """
    Function to calculate the true negative rate
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tn"] / (cm_d["tn"] + cm_d["fp"])

def fpr(cm_d):
    """
    Function to calculate the false postive rate
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["fp"] / (cm_d["fp"] + cm_d["tn"])

def fnr(cm_d):
    """
    Function to calculate the false negative rate also known as sensitivity
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["fn"] / (cm_d["fn"] + cm_d["tp"])

def precision(cm_d):
    """
    Function to calculate the precision
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tp"] / (cm_d["tp"] + cm_d["fp"])

def recall(cm_d):
    """
    Function to calculate the recall
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    """
    return cm_d["tp"] / (cm_d["tp"] + cm_d["fn"])

def generalized_f(cm_d, beta=1.0):
    """
    Function to calculate generalized f score
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    :param beta: float - coefficent to weight recall and precision trade off
    """
    beta2 = beta * beta

    return ((1 + beta2) * precision(cm_d) * recall(cm_d)) / ((precision(cm_d) * beta2) + recall(cm_d))

def matthews_correlation_coefficient(cm_d, ytest=None, ypredicted=None):
    """
    Function to calculate Matthew's correlation coefficient
    :param cm_d: confusion matrix dictionary from get_confusion_matrix(return_dict=True)
    :param ytest: iterable - to test the local function against sklean, this is the known ground truth values
    :param ypredicted: iterable - to test the local function against sklean, this is the predicted values
    """
    log = logging.getLogger(__name__)

    verbose = False

    if ytest is not None and ypredicted is not None:
        from  sklearn.metrics  import matthews_corrcoef
        mcc_sklearn = matthews_corrcoef(ytest, ypredicted)
        verbose = True
        log.info("Matthew's correlation coefficent from sklearn = {}".format(mcc_sklearn))

    mcc_function = (cm_d["tp"] * cm_d["tn"] - cm_d["fp"] * cm_d["fn"]) / np.sqrt((cm_d["tp"] + cm_d["fp"]) * (cm_d["tp"] + cm_d["fn"]) * (cm_d["tn"] + cm_d["fp"]) * (cm_d["tn"] + cm_d["fn"]))

    if verbose is True:
        log.info("Matthew's correlation coefficent from function = {}".format(mcc_function))

    return mcc_function


def auc(metric1, metric2):
    """
    Function to calculate the area under the curve
    :param metric1: iterable - x axis metric values as an iterble
    :param metric2: iterable - y axis metric values as an iterble
    """

    return sklearn.metrics.auc(metric1, metric2)


def calculate_confusion_based_metrics(cmtx=None, df=None, predicted_column_name="prediction", known_column_name="known", probabilities_column_name=None, 
        probabilities=None, labels=(0,1), positive_label=1, imbalanced=False, verbose=False, key=1, plt_filename=None, all_classes=True, get_roc_curve=True,
        get_pr_curve=True, vmin=None, vmax=None, col_map="viridis", annotate=True, title=None, font_scale=1.4, fontsize=20):
    """
    :param cmtx: np array or dict - confusion matrix from sklearn or confusion_matrix_to_dict()
    :param df: pandas dataframe - datafrom of two columns one predicted data the other ground truth known classes
    :param predicted_column_name: str - the column name containing the predicted classes
    :param known_column_name: str - the column name containing the ground truth known classes
    :param probabilities_column_name: str - the column name containing a classes predicted probabilities (sklearn predict_proba() output)
    :param probabilities: np array - class probability prediction from sklearn predict_proba() output
    :param labels: tuple - the labels used for the classes
    :param positive_label: str/int/float - label applied for the positive class
    :param imbalanced: bool True/False - use the imbalanced learn scorers where possible rather than normal sklearn
    :param verbose: bool True/False - more log output than normal
    :param key: int - class key to use to get the support from sklearn classification report
    :param plt_filename: str - filename to save plot
    :param font_scale: float - for seaborn how much to scale the font by
    :param fontsize: float - matplotlib fontsize
    """

    log = logging.getLogger(__name__)

    log.info("Attempting to calculate confusion based metrics")

    if cmtx is not None:
        if isinstance(cmtx, dict):
            log.info("Using provided confusion matrix: {}".format(cmtx))
        else:
            log.info("Looks like provided confusion matrix is not a dictionary converting")
            cmtx = confusion_matrix_to_dict(cmtx)
            log.info("converted: {}".format(cmtx))
    elif df is not None:
        if isinstance(df, pd.DataFrame):
            cmtx = get_confusion_matrix(df, predicted_column_name=predicted_column_name, known_column_name=known_column_name, labels=labels, return_dict=True)
            log.info(cmtx)
            if plt_filename is not None:
                pltsk.plot_metrics(df, predicted_column_name=predicted_column_name, known_column_name=known_column_name, probabilities=probabilities, 
                 positive_label=positive_label, labels=labels, name=plt_filename, all_classes=all_classes, roc_curve=get_roc_curve, pr_curve=get_pr_curve,
                                  col_map=col_map, annotate=annotate, vmin=vmin, vmax=vmax, title=title, font_scale=font_scale, fontsize=fontsize)
        else:
            log.info("Looks like provided data is not a pandas dataframe - ERROR")
            return ()
    else:
        log.info("Neither confusion matrix or data given - ERROR")
        return ()

    output_metrics = {}
    ac = accuracy(cmtx)
    output_metrics["accuracy"] = ac
    true_pos_rate = tpr(cmtx)
    output_metrics["tpr"] = true_pos_rate
    false_pos_rate = fpr(cmtx)
    output_metrics["fpr"] = false_pos_rate
    true_neg_rate = tnr(cmtx)
    output_metrics["tnr"] = true_neg_rate
    false_neg_rate = fnr(cmtx)
    output_metrics["fnr"] = false_neg_rate
    gmean = g_mean(cmtx)
    output_metrics["g-mean"] = gmean
    f_half = generalized_f(cmtx, beta=0.5)
    output_metrics["f_half"] = f_half
    f_one =  generalized_f(cmtx, beta=1.0)
    output_metrics["f1"] = f_one
    f_two =  generalized_f(cmtx, beta=2.0)
    output_metrics["f2"] = f_two
    mcc = matthews_correlation_coefficient(cmtx)
    output_metrics["matthews_correlation_coefficient"] = mcc
    # Own function calls only calculate for one class classification report does it for both directly
    precis = precision(cmtx)
    output_metrics["precision"] = precis
    rec = recall(cmtx)
    output_metrics["recall"] = rec

    if df is not None:
        if imbalanced is False:
            rep = sklearn.metrics.classification_report(df[known_column_name], df[predicted_column_name], output_dict=True, labels=labels)
            try:
                output_metrics["support"] = rep[str(float(key))]["support"]
            except KeyError:
                log.warning("Support cannot be gotten from classification report")
        else:
            rep = classification_report_imbalanced(df[known_column_name], df[predicted_column_name], labels=labels)
            output_metrics["imbalenced-str"] = rep

        
        if verbose is True:
            tmp_rep = sklearn.metrics.classification_report(df[known_column_name], df[predicted_column_name])
            log.info(tmp_rep)

    if probabilities_column_name is not None and df is not None:
        fposr, tposr, roc_thresholds = roc_curve(df[known_column_name].values,
                                         df[probabilities_column_name].values,
                                         pos_label=positive_label)
        if verbose is True:
            log.info("ROC curve fpr and tpr data:\n{}\n{}".format(fposr, tposr))
        output_metrics["tpr-roc"] = tposr
        output_metrics["fpr-roc"] = fposr
        output_metrics["roc_thresholds"] = roc_thresholds

        p, r, pr_thresholds = precision_recall_curve(df[known_column_name].values,
                                                         df[probabilities_column_name].values,
                                                         pos_label=positive_label)
        if verbose is True:
            log.info("Precision recall curve precision and recall data:\n{}\n{}".format(p, r))
        output_metrics["precision-pr"] = p
        output_metrics["recall-pr"] = r
        output_metrics["pr_thresholds"] = pr_thresholds

        roc_auc = auc(fposr, tposr)
        output_metrics["roc_auc"] = roc_auc
        pr_auc = auc(r, p)
        output_metrics["pr_auc"] = pr_auc

    elif probabilities is not None:
        log.debug("known_column_name: {}\n\n".format(df[known_column_name].values))
        log.debug("probabilities[:,1] {}\n\n".format(probabilities[:,1]))
        log.debug("pos lab {}\n\n".format(positive_label))
        
        fposr, tposr, roc_thresholds = roc_curve(df[known_column_name].values,
                                         probabilities[:,1],
                                         pos_label=positive_label)
        if verbose is True:
            log.info("ROC curve fpr and tpr data:\n{}\n{}".format(fposr, tposr))
        output_metrics["tpr-roc"] = tposr
        output_metrics["fpr-roc"] = fposr
        output_metrics["roc_thresholds"] = roc_thresholds

        p, r, pr_thresholds = precision_recall_curve(df[known_column_name].values,
                                                         probabilities[:,1],
                                                         pos_label=positive_label)
        
        if verbose is True:
            log.info("Precision recall curve precision and recall data:\n{}\n{}".format(p, r))
        output_metrics["precision-pr"] = p
        output_metrics["recall-pr"] = r
        output_metrics["pr_thresholds"] = pr_thresholds

        #roc_auc = sklearn.metrics.roc_auc_score(df[known_column_name].values, df[probabilities_column_name].values)
        # problematic but general
        roc_auc = auc(fposr, tposr)
        output_metrics["roc_auc"] = roc_auc
        pr_auc = auc(r, p)
        output_metrics["pr_auc"] = pr_auc
    

    if verbose is True:
        log.info("{}".format("\n".join(["{}:\n{}".format(k, v) for k, v in output_metrics.items()])))

    return output_metrics
