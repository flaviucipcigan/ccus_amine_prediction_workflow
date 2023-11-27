from cgi import test
import os
import logging
from datetime import datetime
import pandas as pd
import re
import numpy as np
import scipy
import pickle
import csv

#RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import DataStructs

# scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV,  KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import r2_score

# scikit-imbalenced learn
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support

# shap
import shap

# Own modules
from . import classification_metrics as cmetrics

# stats and plotting
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend
from mlxtend.evaluate import permutation_test as pt
from IPython.display import Image, display
from statistics import mean
from operator import itemgetter

def autoscale(df):
    """
    scale a pandas dataframe using autoscaling/z-scaling
    :param df: pandas dataframe - data frame to be returned scaled
    """
    df_tmp = df.copy()
    normalized_df = (df_tmp-df_tmp.mean())/df_tmp.std()
    return normalized_df

def minmaxscale(df):
    """
    scale a pandas dataframe using min max scaling
    :param df: pandas dataframe - data frame to be returned scaled
    """
    
    df_tmp = df.copy()
    normalized_df = (df_tmp-df_tmp.min())/(df_tmp.max()-df_tmp.min())
    return normalized_df

def get_mol_from_smiles(smiles, canonicalize=True):
    """ 
    Function to make a mol object based on smiles
    :param smiles: str - SMILES string
    :param canonicalize: True/False - use RDKit canonicalized smile or the input resprectively
    """

    log = logging.getLogger(__name__)

    if canonicalize is True:
        s = Chem.CanonSmiles(smiles, useChiral=1)
    else:
        s = smiles
    mol = Chem.MolFromSmiles(s)
    log.debug("Input smiles: {} RDKit Canonicalized smiles {} (Note RDKit does not use "
              "general canon smiles rules https://github.com/rdkit/rdkit/issues/2747)".format(smiles, s))
    Chem.rdmolops.SanitizeMol(mol)
    Chem.rdmolops.Cleanup(mol)

    return mol

def smiles_to_molcule(s, addH=True, canonicalize=True, threed=True, add_stereo=False, remove_stereo=False, random_seed=10459, verbose=False):
    """ 
    :param s: str - smiles string
    :param addH: bool - Add Hydrogens or not
    """

    log = logging.getLogger(__name__)

    mol = get_mol_from_smiles(s, canonicalize=canonicalize)
    Chem.rdmolops.Cleanup(mol)
    Chem.rdmolops.SanitizeMol(mol)
    
    if remove_stereo is True:
        non_isosmiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=False)
        mol = get_mol_from_smiles(non_isosmiles, canonicalize=canonicalize)
        Chem.rdmolops.Cleanup(mol)
        Chem.rdmolops.SanitizeMol(mol)
    
        if verbose is True:
            for atom in mol.GetAtoms():
                log.info("Atom {} {} in molecule from smiles {} tag will be cleared. "
                        "Set properties {}.".format(atom.GetIdx(), atom.GetSymbol(), s, atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))

    if addH is True:
        mol = Chem.rdmolops.AddHs(mol)

    if add_stereo is True:
        rdCIPLabeler.AssignCIPLabels(mol)


    if threed:
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)

    return mol 

def df_corr(x, y, corr_method):
    """
    """
    df = pd.DataFrame(data=np.array([x, y]).T)
    corr = df.corr(method=corr_method)
    
    return corr

def find_correlating_features(features, targets, thresh=0.4, plot=True, process_non_numeric=True, corr_method="spearman", sig_metric=None, significance=False, 
                              random_seed=105791, n_sample=10000, sig_level=0.05):
    """
    Pulls out features which correlate with a property bay at least the threshold as measure by the Pearson correlation
    coefficent (R) [-1, 1].
    :param features: pandas dataframe - pandas data frame of features to correlate against a property
    :param targets: pandas series - pandas series of property target values
    :param thresh: float - Pearson correlation coefficent threshold to be equal to or more than
    :param plot: true/false - plot the features with a suitably high correlation
    :param process_non_numeric: true/false - Proces or not features which mordred could not calcualte for all molecules
    :param corr_method: str - any correlation method that can be given pandas .corr(method=arg)
    :param significance: true/false - one tail significance of the corr_method: how many times out of the n_sample do I obtain a correlation coefficient that is greater than the observed value
    :param random_seed: int - random seed for sampling in the significance test
    :param n_sample: int - number of resamples/shuffle of the data significance test
    :param sig_level: float - level to consider the results significant 0.05 is 95% significance level here
    """
    
    log = logging.getLogger(__name__)
    
    log.info(targets)
    usable_features = []
    significant_features = []
    
    if significance is True:
        log.info("Significance will be calculated for the correlations")
        if sig_metric == "pearson":
            fcorr = lambda x, y: abs(scipy.stats.pearsonr(x, y)[0])
        elif sig_metric == "spearman":
            fcorr = lambda x, y: abs(scipy.stats.spearmanr(x, y)[0])
        elif sig_metric == "kendalltau":
            fcorr = lambda x, y: abs(scipy.stats.kendalltau(x, y)[0])
        elif sig_metric == "onetail_gt_pearson":
            fcorr = lambda x, y: scipy.stats.pearsonr(x, y)[0]
        elif sig_metric == "onetail_gt_spearman":
            fcorr = lambda x, y: scipy.stats.spearmanr(x, y)[0]
        elif sig_metric == "onetail_gt_kendalltau":
            fcorr = lambda x, y: scipy.stats.kendalltau(x, y)[0]
        elif sig_metric == "onetail_lt_pearson":
            fcorr = lambda x, y: scipy.stats.pearsonr(y, x)[0]
        elif sig_metric == "onetail_lt_spearman":
            fcorr = lambda x, y: scipy.stats.spearmanr(y, x)[0]
        elif sig_metric == "onetail_lt_kendalltau":
            fcorr = lambda x, y: scipy.stats.kendalltau(y, x)[0]
        elif sig_metric == "notequal":
            fcorr = "x_mean != y_mean"
        elif sig_metric== "greaterthan":
            fcorr = "x_mean > y_mean"
        elif sig_metric == "lessthan":
            fcorr = "x_mean < y_mean"
        elif sig_metric is None:
            fcorr = "x_mean != y_mean"
        else:
            log.error("ERROR - Correlation method unrecognised")
            return
    
    for ent in features.columns:

        tf = True if ent in features.columns else False
        log.debug("{} {}".format(ent, tf))
        series = features[ent].copy()

        if series.isnull().sum().sum() > 0:
            #log.info("NaN identifed")
            if series.isnull().sum() < 0.7*len(features[ent].values):
                log.info("filling NaNs")
                series.fillna(features[ent].mean())
            else:
                log.warning("{} - Too many NaN for filling".format(ent))
                continue
        else:
            pass
        
        # Drop any rows with NA
        feat = series.values
        tmp_df = pd.DataFrame(data=np.array([feat, targets]).T, columns=[ent, "property"])
        tmp_df.dropna(inplace=True)

        try:
            correlations = tmp_df.corr(method=corr_method)

            if abs(correlations.loc[ent, "property"]) > thresh:
                #print("mia xara")
                usable_features.append(ent)
                if significance is True:
                    p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
                    significant = True if p_value < sig_level else False
                    log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))
                    if significant is True:
                        significant_features.append(ent)
                else:
                    log.info("{}: {:.4f}" .format(ent, correlations.loc[ent, "property"]))

                if plot is True:
                    tmp_df.plot.scatter(ent, "property", grid=True)
            else:
                log.debug("Feature {} does not correlate to the threshold {}".format(ent, thresh))
            
            del tmp_df
            
        except KeyError:
            if process_non_numeric is True:
                # print("oxi")
                # If non-numeric error messages occur from mordred the correlation matrix is empy and has a key error
                log.debug("WARNING - some molecules do not have this feature ({}) thus the correlation is for a sub-set.".format(ent))

                tmp_df = tmp_df[pd.to_numeric(tmp_df[ent], errors='coerce').notnull()]
                tmp_df[ent] = pd.to_numeric(tmp_df[ent])
                tmp_df["property"] = pd.to_numeric(tmp_df["property"])

                # correlations = tmp_df.corr(method=corr_method)

                # if abs(correlations.loc[ent, "property"]) > thresh:
                #     print("gamiesai")
                #     usable_features.append(ent)
                #     if significance is True:
                #         # print(feat)
                #         p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
                #         significant = True if p_value < sig_level else False
                #         log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))
                #     else:
                #         log.info("{}: {:.4f}" .format(ent, correlations.loc[ent, "property"]))

                del tmp_df
            else:
                del tmp_df
                
    if significance is False:
        return usable_features
    else:
        return usable_features, significant_features
    
def permutation_test(features, targets, n_sample=10000, corr_method="pearson", random_seed=105791,  sig_level=0.05):
    """
    :param features: pandas dataframe - pandas data frame of features to correlate against a property
    :param targets: pandas series - pandas series of property target values
    :param process_non_numeric: true/false - Process or not features which mordred could not calcualte for all molecules
    :param corr_method: str - any correlation method that can be given pandas .corr(method=arg)
    :param significance: true/false - one tail significance of the corr_method: how many times out of the n_sample do I obtain a correlation coefficient that is greater than the observed value
    :param random_seed: int - random seed for sampling in the significance test
    :param n_sample: int - number of resamples/shuffle of the data significance test
    :param sig_level: float - level to consider the results significant 0.05 is 95% significance level here
    """
    log = logging.getLogger(__name__)
    
    
    if significance is True:
        log.info("Significance will be calculated for the correlations")
        if corr_method == "pearson":
            fcorr = lambda x, y: scipy.stats.pearsonr(x, y)[0]
        elif corr_method == "spearman":
            fcorr = lambda x, y: scipy.stats.spearmanr(x, y)[0]
        elif corr_method == "kendalltau":
            fcorr = lambda x, y: scipy.stats.kendalltau(x, y)[0]
        else:
            log.error("ERROR - Correlation method unrecognised")
            return
    
    feat = series.values
    tmp_df = pd.DataFrame(data=np.array([feat, targets]).T, columns=[ent, "property"])
    tmp_df.dropna(inplace=True)
    
    try:
            correlations = tmp_df.corr(method=corr_method)
            p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
            significant = True if p_value < sig_level else False
            log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))
            
    except KeyError:
            
            # If non-numeric error messages occur from mordred the correlation matrix is empy and has a key error
            log.warning("WARNING - some molecules do not have this feature thus the correlation is for a sub-set.")
            tmp_df = tmp_df[pd.to_numeric(tmp_df[ent], errors='coerce').notnull()]
            tmp_df[ent] = pd.to_numeric(tmp_df[ent])
            tmp_df["property"] = pd.to_numeric(tmp_df["property"])

            correlations = tmp_df.corr(method=corr_method) 
            p_value = pt(feat, targets, method="approximate", num_rounds=n_sample, func=fcorr, seed=random_seed)
            significant = True if p_value < sig_level else False
            log.info("{}: {:.4f} P: {:.4f} Significant at {:.4f} level? {}" .format(ent, correlations.loc[ent, "property"], p_value, sig_level, significant))
    
    return correlations, p_value, significant




def amine_and_aromatic_nsp2_checker(smiles, ammonia="[NH3]", primary="[#7X3;H2][#6X4]", secondary="[#7X3;H1]([#6X4])[#6X4]",
                  tertiary="[#7X3]([#6X4])([#6X4])[#6X4]",
                  alipathic_amines="[$([NH3]),$([NH2][CX4]),$([NH]([CX4])[CX4]),$([NX3]([CX4])([CX4])[CX4])]",
                  aromatic_sp2_n="[$([nX3](:*):*),$([nX2](:*):*)]"):
    """
    Function to check if the smiles are amines or conatin a aromatic N sp2
    :param smiles: str - smiles string to look for substructure
    :param ammonia: str - SMARTS pattern to look for ammonia
    :param primary: str - SMARTS pattern to look for primary amines
    :param secondary: str - SMARTS pattern to look for secondary amines
    :param tertiary: str - SMARTS pattern to look for tertiary amines
    :param alipathic_amines: str - SMARTS pattern to look for aliphatic amines
    :param aromatic_sp2_n: str - SMARTS pattern to look for aromatic N sp2 hybrid
    :return: tuple smiles and True/False for the possible structures
    """

    log = logging.getLogger(__name__)

    mol = smiles_to_molcule(smiles)

    nh3 = Chem.MolFromSmarts(ammonia)
    amine1 = Chem.MolFromSmarts(primary)
    amine2 = Chem.MolFromSmarts(secondary)
    amine3 = Chem.MolFromSmarts(tertiary)
    aliphatic_amine = Chem.MolFromSmarts(alipathic_amines)
    aromatic_Nsp2 = Chem.MolFromSmarts(aromatic_sp2_n)

    is_ammonia = 0
    has_primary_amine = 0
    has_secondary_amine = 0
    has_tertiary_amine = 0
    has_mixed_amine = 0
    has_an_amine = 0
    has_aromatic_Nsp2 = 0
    

    if mol.HasSubstructMatch(aliphatic_amine):
        has_an_amine = 1

    if mol.HasSubstructMatch(aromatic_Nsp2):
        has_aromatic_Nsp2 = 1

    if mol.HasSubstructMatch(nh3):
        is_ammonia = 1

    if mol.HasSubstructMatch(amine1):
        has_primary_amine = 1

    if mol.HasSubstructMatch(amine2):
        has_secondary_amine = 1

    if mol.HasSubstructMatch(amine3):
        has_tertiary_amine = 1
    
    n_amine_types = has_primary_amine + has_secondary_amine + has_tertiary_amine + has_aromatic_Nsp2
    log.debug("Number of amine types: {}".format(n_amine_types))
    if n_amine_types > 1:
        has_mixed_amine = 1

    log.debug("SMILES {}\n\tIs ammonia? {}\n\tPrimary amine? {}\n\tSecondary amine? {}\n\tTertiary amine? {}\n\t"
             "Has an amine? {}\n\tHas sp2 aromatic N? {}\n".format(
        smiles,
        is_ammonia,
        has_primary_amine,
        has_secondary_amine,
        has_tertiary_amine,
        has_an_amine,
        has_aromatic_Nsp2))

    return (smiles, is_ammonia, has_primary_amine, has_secondary_amine, has_tertiary_amine, has_mixed_amine, has_an_amine, has_aromatic_Nsp2)

def vector_amine_types(smiles, ammonia="[NH3]", primary="[NX3;H2][CX4]", secondary="[NX3;H1]([CX4])[CX4]",
                  tertiary="[NX3]([CX4])([CX4])[CX4]", quaternary="[NX4+]", aromatic_sp2_n="[$([nX3](:*):*),$([nX2](:*):*)]"):
    """
    Function to check if the smiles are amines or conatin a aromatic N sp2
    :param smiles: str - smiles string to look for substructure
    :param ammonia: str - SMARTS pattern to look for ammonia
    :param primary: str - SMARTS pattern to look for primary amines
    :param secondary: str - SMARTS pattern to look for secondary amines
    :param tertiary: str - SMARTS pattern to look for tertiary amines
    :param aromatic_sp2_n: str - SMARTS pattern to look for aromatic N sp2 hybrid
    :return: tuple smiles and True/False for the possible structures
    """

    log = logging.getLogger(__name__)

    mol = smiles_to_molcule(smiles)

    nh3 = Chem.MolFromSmarts(ammonia)
    amine1 = Chem.MolFromSmarts(primary)
    amine2 = Chem.MolFromSmarts(secondary)
    amine3 = Chem.MolFromSmarts(tertiary)
    ammonium = Chem.MolFromSmarts(quaternary)
    aromatic_Nsp2 = Chem.MolFromSmarts(aromatic_sp2_n)
    
    
    amine_type = -1
    has_primary_amine = 0
    has_secondary_amine = 0
    has_tertiary_amine = 0
    has_quaternary_ammonium = 0
    has_aromatic_Nsp2 = 0

    if mol.HasSubstructMatch(nh3):
        amine_type = 0
        has_primary_amine = 1

    if mol.HasSubstructMatch(amine1):
        amine_type = 1
        has_primary_amine = 1
        
    if mol.HasSubstructMatch(amine2):
        amine_type = 2
        has_secondary_amine = 1

    if mol.HasSubstructMatch(amine3):
        amine_type = 3
        has_tertiary_amine = 1
    
    if mol.HasSubstructMatch(ammonium):
        amine_type = 4
        has_quaternary_ammonium = 1
        
    if mol.HasSubstructMatch(aromatic_Nsp2):
        amine_type = 5
        has_aromatic_Nsp2 = 1
        
    if has_primary_amine + has_secondary_amine + has_tertiary_amine + has_quaternary_ammonium + has_aromatic_Nsp2 > 1:
        amine_type = 6

    log.debug("Amine/ammonium type: {}".format(amine_type))

    return amine_type

def amine_aromatic_none_aromatic(smiles, 
                  none_aromatic_amines="[$([NH3]),$([NH2][CX4]),$([NH]([CX4])[CX4]),$([NX3]([CX4])([CX4])[CX4])]",
                  aromatic_sp2_n="[$([nX3](:*):*),$([nX2](:*):*)]"):
    """
    Function to check if the smiles are amines or conatin a aromatic N sp2
    :param smiles: str - smiles string to look for substructure
    :param alipathic_amines: str - SMARTS pattern to look for aliphatic amines
    :param aromatic_sp2_n: str - SMARTS pattern to look for aromatic N sp2 hybrid
    :return: tuple smiles and True/False for the possible structures
    """

    log = logging.getLogger(__name__)

    mol = smiles_to_molcule(smiles)
    
    aliphatic_amine = Chem.MolFromSmarts(none_aromatic_amines)
    aromatic_Nsp2 = Chem.MolFromSmarts(aromatic_sp2_n)


    aromatic_none_aromatic = -1

    if mol.HasSubstructMatch(aliphatic_amine):
        aromatic_none_aromatic = 0

    if mol.HasSubstructMatch(aromatic_Nsp2):
        aromatic_none_aromatic = 1
        
    log.debug("Amine is aromatic (0 is no 1 is yes) ? {}".format(aromatic_none_aromatic))

    return aromatic_none_aromatic


def general_two_class(ent, bound=0.5, **kwarg):
    """
    function to perform a basic classification
    :param ent: float - value to perform classification against
    """
    
    if ent < bound:
        c = 0
    elif ent >= bound:
        c = 1
    
    return c

def count_amine_types(smi, primary = "[NX3;H2][CX4]", secondary ="[NX3;H1]([CX4])[CX4]",tertiary ="[NX3]([CX4])([CX4])[CX4]", aromatic_sp2_n="[$([nX3](:*):*),$([nX2](:*):*)]",
                      show=False, show_primary=False, show_secondary=False, show_tertiary=False, show_aromaticsp2=False):
    """
    Function to count the sub-structuer matches to the 4 amine types
    :param smi: str - smiles string
    :param primary: str - Smarts string for primary aliphatic (chain) amine identifying sub-structures
    :param secondary: str - Smarts string for identifying identifying secondary aliphatic (chain) amine sub-structures
    :param tertiary: str - Smarts string for identifying tertiary aliphatic (chain) amine sub-structures
    :param aromatic_sp2_n: str - Smarts string for identifying aromatic sp2 hybridized nitrogen atoms sub-structures
    :param show: True/False - boolean to plot the molecule graphs and overlaps
    :param show_primary: True/False - boolean to plot the molecule graphs and overlaps for primary amine matches
    :param show_secondary: True/False - boolean to plot the molecule graphs and overlaps for secondary amine matches
    :param show_tertiary: True/False - boolean to plot the molecule graphs and overlaps for tertiary amine matches
    :param show_aromaticsp2=False : True/False - boolean to plot the molecule graphs and overlaps for aromatic sp2 hybridized nitrogen atom matches
    """
    
    log = logging.getLogger(__name__)
    
    primary_substructure = Chem.MolFromSmarts(primary)
    secondary_substructure = Chem.MolFromSmarts(secondary)
    tertiary_substructure = Chem.MolFromSmarts(tertiary)
    aromsp2_substructure = Chem.MolFromSmarts(aromatic_sp2_n)

    mol = smiles_to_molcule(smi, threed=False)
    matches = mol.GetSubstructMatches(primary_substructure)
    n_primary = len(matches)
    if show is True or show_primary is True:
        log.info("\n----- Primary -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))

    mol = smiles_to_molcule(smi, threed=False)
    matches = mol.GetSubstructMatches(secondary_substructure)
    n_secondary = len(matches)
    if show is True or show_secondary is True:
        log.info("\n----- Secondary -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))

    mol = smiles_to_molcule(smi, threed=False)
    matches = mol.GetSubstructMatches(tertiary_substructure)
    n_tertiary = len(matches)
    if show is True or show_tertiary is True:
        log.info("\n----- Tertiary -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))
        
    mol = smiles_to_molcule(smi, threed=False)
    matches = mol.GetSubstructMatches(aromsp2_substructure)
    n_aromaticsp2 = len(matches)
    if show is True or show_aromaticsp2 is True:
        log.info("\n----- Atomatic sp2 hybridized nitrogen atoms -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))
    
    return n_primary, n_secondary, n_tertiary, n_aromaticsp2 

def capacity_classes(n_primary, n_secodnary, n_tertiary, n_aromatic_sp2, capacity, units="molco2_moln", number_N_atoms=None, amines_mr=None, co2_mass=44.009,
                    primary_secondary_base=0.5, tertiary_base=1.0, nsp2_base=0.5):
    """
    Function to output a suggested threshold for 'good' or 'bad' classification of amine molecules based on carbon capture capacity
    in the given units
    :param n_primary: int - number of primary amine groups in the molecule
    :param n_secondary: int - number of secondary amine groups in the molecule
    :param n_tertiary: int - number of tertiary amine groups in the molecule
    :param n_aromatic_sp2: int - number of aromatic sp2 hybridized nitrogen atoms in the molecule
    :param capacity: list - amine capacity values in the appropiate units
    :param units: str - Three accepted unit "molco2_moln", "molco2_molamine", "gco2_gamine" classes are consistent across tehse units
    :param number_N_atoms: list - The number of N atoms in the amine for each smiles 
    :param amine_mr: list - The molar mass to each amine for each smiles
    :param co2_mass: float - mass of a co2 molecule
    :param primary_secondary_base: float - the base value expected for a single N atom in a 1' or 2' amine to contribute to the property
    :param tertiary_base: float - the base value expected for a single N atom in a 3' amine to contribute to the property
    :param nsp2_base: float - the base value expected for a single sp2 hydridized N to contribute to the property
    """
    
    log = logging.getLogger(__name__)
    
    classes = []
    
    molar_ratios = [ent/co2_mass for ent in amines_mr]
    df = pd.DataFrame(data=np.array([n_primary, n_secodnary, n_tertiary, n_aromatic_sp2]).T,
                      columns=["primary_amine_counts","secondary_amine_counts", "tertiary_amine_counts", "aromatic_sp2_n" ])
    
    for indx, row in df.iterrows():
        ret = classify(*row.values, primary_secondary_base=primary_secondary_base, tertiary_base=tertiary_base, nsp2_base=nsp2_base)

        # N molar capacity
        if units == "molco2_moln":

            if capacity[indx] < ret:
                classes.append(0)
            else:
                classes.append(1)
            log.info("{} N molar capacity (mol co2 / mol N) threshold {:.2f} capacity {:.2f} class {}".format(indx, ret, capacity[indx], classes[-1]))

        elif units == "molco2_molamine":
            # amine molar capacity
            if capacity[indx] < ret*number_of_N_atoms[indx]:
                classes.append(0)
            else:
                classes.append(1)
            log.info("{} Amine molar capacity (mol co2 / mol amine) threshold {:.2f} capacity {:.2f} class {}".format(indx, ret*number_of_N_atoms[indx], capacity[indx], classes[-1]))
        
        elif units == "gco2_gamine":
            # mass capacity
            if capacity[indx] < (ret*number_of_N_atoms[indx])/molar_ratios[indx]:
                classes.append(0)
            else:
                classes.append(1)    
            log.info("{} Mass capacity (co2 (g) / amine (g)) threshold {:.2f} capacity {:.2f} class {}\n----\n".format(
                indx, (ret*number_of_N_atoms[indx])/molar_ratios[indx],capacity[indx],classes[-1]))
    
    return classes

def classify(n_primary, n_secodnary, n_tertiary, n_aromatic_sp2, primary_secondary_base=0.5, tertiary_base=1.0, nsp2_base=0.5):
    """
    Function to output a suggested threshold for 'good' or 'bad' classification of amine molecules based on carbon capture capacity
    :param n_primary: int - number of primary amine groups in the molecule
    :param n_secondary: int - number of secondary amine groups in the molecule
    :param n_tertiary: int - number of tertiary amine groups in the molecule
    :param n_aromatic_sp2: int - number of aromatic sp2 hybridized nitrogen atoms in the molecule
    :param primary_secondary_base: float - the base value expected for a single N atom in a 1' or 2' amine to contribute to the property
    :param tertiary_base: float - the base value expected for a single N atom in a 3' amine to contribute to the property
    :param nsp2_base: float - the base value expected for a single sp2 hydridized N to contribute to the property
    """
    
    log = logging.getLogger(__name__)
    
    if n_primary + n_secodnary > 0 and n_tertiary > 0:
        log.debug("{} {} {} {}".format(n_primary, n_secodnary, n_tertiary, n_aromatic_sp2))
        n = (0.5 * (n_primary + n_secodnary)) + n_tertiary
        d = 2.0 * n_tertiary
        log.debug("{}/{} = {}".format(n, d, n/d))
        return n / d
    
    elif n_primary >= 1 and n_secodnary >= 1:
        return primary_secondary_base * (n_primary + n_secodnary)
    
    elif n_primary > 1:
        return primary_secondary_base * (n_primary)
    
    elif n_secodnary > 1:
        return primary_secondary_base * (n_secodnary)
    
    elif n_primary == 1:
        return primary_secondary_base
    
    elif n_secodnary == 1:
        return primary_secondary_base
    
    elif n_tertiary > 0:
        return tertiary_base
    elif n_primary == 0 and n_secodnary == 0 and n_tertiary == 0:
        return nsp2_base
    
def test_imbalenced_classifiers_with_optimization(df, classes, classifiers, clf_options, scale=True, cv=5, clf_names=None, 
                                                  class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                  random_seed=1057091):
    """
    function to run classification test over classifiers using imbalenced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(data, classes, test_size=test_set_size, random_state=random_seed, shuffle=True)
    log.info("Xtrain: {}\nXtest: {}\nytrain: {}\nytest: {}".format(Xtrain, Xtest, ytrain, ytest))
    log.info("{} {}".format(Xtest.index, ytest))
    log.info("Test set is made up of:\n{}".format("\n".join(["name {} smiles {} class {}".format(names[j], smiles[j], c) for j, c in zip(Xtest.index, ytest["classes"].values)])))
    
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, false negatives is "
             "element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        log.info("Search for optimal parameters for {}".format(name))
        
        # Grid search model optimizer
        clf = classf
        log.info("{} {} {} ".format(clf_options, clf_names, iteration))
        parameters = clf_options[clf_names[iteration]]
        log.debug("\tname: {} parameters: {}".format(name, parameters))
        optparam_search = GridSearchCV(clf, parameters, cv=cv, error_score=np.nan)
        log.debug("CV xtrain: {}".format(Xtrain))
        optparam_search.fit(Xtrain, ytrain.values.ravel())
        opt_parameters = optparam_search.best_params_
        if no_train_output is False:
            log.info("\t{}".format(pd.DataFrame(data=optparam_search.cv_results_)))
            log.info("\tBest parameters; {}".format(opt_parameters))
        else:
            pass
        
        # Fit final model using optimized parameters
        log.info("\n----- {} -----".format(name))
        log.debug("Xtrain: {}\nXtest: {}\nytrain: {}\nytest: {}".format(Xtrain, Xtest, ytrain, ytest))
        clf.fit(Xtrain, ytrain.values.ravel())
        clf_score = clf.score(Xtest, ytest)
        predicted_clf = clf.predict(Xtest)
        c_matix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matix,display_labels=class_labels)
        disp.plot()
        probs = clf.predict_proba(Xtest)
        fp, tp, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
        log.info(clf.classes_)
        roc_auc = auc(fp, tp)
        log.info("ROC analysis:\n\tTrue positives:\n\t{}\n\tFalse positives:\n\t{}".format(tp, fp))
        plt.figure(figsize=(10,10))
        plt.plot(fp, tp, color="red",
                 lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
        plt.plot([0, 1], [0, 1], "k:")
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
        
        # output metrics for consideration
        log.info("Confusion matrix ({}):\n{}\n".format(name, c_matix))
        log.info("\nscore ({}): {}".format(name, clf_score))
        log.info("{}".format(classification_report(ytest, predicted_clf)))
        log.info("Imbalence reports:")
        log.info("Imbalence classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
        sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
        log.info("{} {} {}".format(sensitvity, specificity, support))
        log.info("Predicted | Label\n------------------")
        log.info("{}\n-----\n".format("\n".join(["{}   |   {}".format(p, k) for p, k in zip(predicted_clf, ytest["classes"].values)])))
        
        iteration = iteration + 1
        
def constant_value_columns(df):
    """
    Function to find constant value columns
    :param df: Pandas dataframe - dataframe to find non unique columns
    """
    
    log = logging.getLogger(__name__)
    
    cols = [name for name in df.columns if df[name].nunique() == 1]
    
    return cols


def grid_search_classifier_parameters(clf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=5, name=None, scoring=("roc_auc", "precision", "recall")):
    """
    Grid search calssifer hyperparams and find the best report metrics if requested
    """
    log = logging.getLogger(__name__)
    
    # Grid search model optimizer
    parameters = clf_options[clf_names[iteration]]
    log.info("\tname: {} parameters: {}".format(name, parameters))
    
    optparam_search = GridSearchCV(clf, parameters, cv=cv, error_score=np.nan, scoring=scoring, refit=scoring[0], return_train_score=True)
    log.debug("\tCV xtrain: {}".format(Xtrain))
    
    optparam_search.fit(Xtrain, ytrain.values.ravel())
    opt_parameters = optparam_search.best_params_
    
    if no_train_output is False:
        reported_metrics = pd.DataFrame(data=optparam_search.cv_results_)
        reported_metrics.to_csv("{}/{}_grid_search_metrics.csv".format(name,name))
        log.info("\tBest parameters; {}".format(opt_parameters))
        for mean, std, params in zip(optparam_search.cv_results_["mean_test_{}".format(scoring[0])], 
                                     optparam_search.cv_results_["std_test_{}".format(scoring[0])], 
                                     optparam_search.cv_results_['params']):
            log.info("\t{:.4f} (+/-{:.4f}) for {}".format(mean, std, params))
    else:
        pass
    
    return opt_parameters

def kfold_test_imbalenced_classifiers_with_optimization(df, classes, classifiers, clf_options, scale=True, cv=5, n_repeats=20, clf_names=None, 
                                                        class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                        random_seed=107901, overwrite=False, smote=False, smote_catagorical_indexes=None,
                                                        smote_continuous_indexes=None):
    """
    function to run classification test over classifiers using imbalenced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    log_df = pd.DataFrame()
    
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix = [],[],[],[],[]
    predictions = []
    important = []
    shap_vals = []
    test_idxs = []
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.
    #rkf = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_seed)
    kf = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, "
             "false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        if not n_repeats % 2:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)))
            plt_rows = int(n_repeats/2.0)
        else:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)+1))
            plt_rows = int(n_repeats/2.0)+1
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
        
        # Loop over  Kfold here 
        for train_indx, test_indx in kf.split(df, classes):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = classes.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = classes.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            if smote is True:
                log.info("Using SMOTE to balence data")
                min_class = None
                number_in_min_class = len(ytrain)
                for clabel in class_labels:
                    n_for_class = len(ytrain[ytrain["classes"] == clabel].index)
                    if n_for_class < number_in_min_class:
                        number_in_min_class = n_for_class
                        min_class = clabel
                
                log.info(f"iteration {kf_iteration} min class is label {min_class}")

                nneighbours = min(5, len(ytrain[ytrain["classes"]==clabel].index)-1)
                log.info("Number of neighbours used for SMOTE: {}".format(nneighbours))

                # feature_types is set when the fingerprints are calculated

                # No catagorical
                if smote_catagorical_indexes is None and smote_continuous_indexes is not None:
                    log.info("Feature types: Continous")
                    sm = SMOTE(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                # Some catagorical - Need to provide the indexes
                elif smote_catagorical_indexes is not None and smote_continuous_indexes is not None:
                    log.info("Feature types: Mixed")
                    sm = SMOTENC(sampling_strategy="not majority", categorical_features=smote_catagorical_indexes, random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                # All catagorical
                elif smote_catagorical_indexes is not None and smote_continuous_indexes is None:
                    log.info("Feature types: Catagorical")
                    sm = SMOTEN(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                else:
                    log.info("Assuming feature types: Catagorical")
                    sm = SMOTEN(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)

                log.debug(f"{Xtrain.to_numpy()}\n{ytrain.to_numpy()}")
                features_smote, classes_smote = sm.fit_resample(Xtrain.to_numpy(), ytrain.to_numpy())
                features_smote = pd.DataFrame(features_smote, columns=Xtrain.columns)
                classes_smote = pd.DataFrame(classes_smote, columns=["classes"])
                log.debug(f"{features_smote}\n{classes_smote}")
                n_sythetic_points = features_smote.shape[0] - len(Xtrain.index)
                log.info("SMOTE has added {} sythetic features to balence the classes".format(n_sythetic_points))

            # Grid search model optimizer
            opt_param = grid_search_classifier_parameters(classf, features_smote, classes_smote, clf_options, clf_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())
            
            if name.strip() == "ExtraTreesClassifier":
                important.append(clf.feature_importances_)
            if re.search(r"^logistic\wregression$", name.strip().lower()):
                log.info("Saving importance LR model")
                important.append(clf.coef_[0])

                explainer = shap.explainers.Linear(clf, Xtrain, feature_names=df.columns)
                shap_vs = explainer.shap_values(Xtest)
                log.info(shap_vs)
                shap_vals.append(shap_vs)
                test_idxs.append(test_indx)


            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
            sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
            sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            
            ## Get the confusion matrices 
            c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
            confusion_matrices.append(c_matrix)
            
            ## Calculate the roc area under the curve
            probs = clf.predict_proba(Xtest)
            fpr, tpr, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            list_roc_auc.append(roc_auc)
            
            roc_aucs.append(roc_auc)
            log.info("\tROC analysis area under the curve: {}".format(roc_auc))
            
            # output metrics for consideration
            log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))
            
            list_c_matrix.append(c_matrix)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   

            list_score.append(clf_score)
        
            log.info("\tImbalence reports:")
            log.info("\tImbalence classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
            output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)
            
            ## Plot the roc curves
            ax = plt.subplot(2, plt_rows, kf_iteration+1)
            ax.plot(fpr, tpr, color="red",
                     lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
            
                # ugliest legend i ve made in my life - maybe one under the other?
            
            try:
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "pre_class0 = {:.2f}\n".format(output_dict[0]['pre'])+"pre_class1 = {:.2f}".format(output_dict[1]['pre']))
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "f1_class0 = {:.2f}\n".format(output_dict[0]['f1'])+ "f1_class1 = {:.2f}".format(output_dict[1]['f1']))
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "rec_class0 = {:.2f}\n".format(output_dict[0]['rec'])+ "rec_class1 = {:.2f}".format(output_dict[1]['rec']))
            except KeyError as kerr:
                log.warning("Key error found plotting ROC curves {}".format(kerr))

            ax.plot([0, 1], [0, 1], "k:")
            ax.set_xlim(xmin=0.0, xmax=1.01)
            ax.set_ylim(ymin=0.0, ymax=1.01)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            
            log.info("Appending classification report")
            list_report.append(classification_report_imbalanced(ytest, predicted_clf))
            
            log.info("Getting sensitvity, specificity, support")
            sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
            log.debug("\t{} {} {}".format(sensitvity, specificity, support))

            log.info("Print predictions")
            log.info("\t Index | Predicted | Label\n\t------------------")
            log.info("\t{}\n-----\n".format("\n\t".join(["{}   |   {}   |   {}".format(i, p, k) for i, p, k in zip(test_indx, predicted_clf, ytest["classes"].values)])))
    
            pred = [list(test_indx),list(ytest["classes"].values),list(predicted_clf), list(probs[:,0]), list(probs[:,1])]
            
            pred = pd.DataFrame(pred)
            log.info("Save data")
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
            log.info("\nNext iteration\n")
        plt.close()
        
        del predictions[:]
        
        if any(x not in tmp for x in [y for y in range(len(classes.index))]):
             log.info("WARNING there appears to be left over indexes which have not been used for testing: {}".format())
        else:
            log.info("All points have been used in a test case over all fold as they should have been")
        
        # Plot and assess classifier over all folds
        
        # NOTE - rows are scores columns are classes
        log.info("Getting means and standard deviations")
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        average_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)
        
        log.info("{} {} {} {}".format(average_scores, std_scores, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][0], 
                                                                                                                                   std_scores[0][0], 
                                                                                                                                   average_scores[1][0], 
                                                                                                                                   std_scores[1][0], 
                                                                                                                                   average_scores[2][0], 
                                                                                                                                   std_scores[2][0], 
                                                                                                                                   average_scores[3][0], 
                                                                                                                                   std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][1], 
                                                                                                                                   std_scores[0][1], 
                                                                                                                                   average_scores[1][1], 
                                                                                                                                   std_scores[1][1], 
                                                                                                                                   average_scores[2][1], 
                                                                                                                                   std_scores[2][1], 
                                                                                                                                   average_scores[3][1], 
                                                                                                                                   std_scores[3][1])
        #score_str3 ="Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n".format(score_str1, score_str2) #, score_str3)
        plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points', size=12, ha='center', va='bottom')
        figure.tight_layout()
        plt.savefig("{0}/{0}_roc_curves.png".format(name))
        plt.show()

        iteration = iteration + 1

        if name.strip() =='ExtraTreesClassifier':
            imp = pd.DataFrame(list(map(np.ravel, important)))
            log.info("\n\t----- Important features: {} -----".format(imp))
            log.info("\n\t----- max feature in each row: {} -----".format(imp.idxmax(axis=1)))
            imp.to_csv("importance.csv")
            top = pd.DataFrame({n: imp.T[col].nlargest(10).index.tolist()
                          for n, col in enumerate(imp.T)}).T
            top.to_csv('topimportance.csv')
            box = imp.plot.box()
            plt.savefig("boxplot_feature_imp.png")
        elif re.search(r"^logistic\wregression$", name.strip().lower()):
            imp = pd.DataFrame(list(map(np.ravel, important)))
            log.info("\n\t----- Important features: {} -----".format(imp))
            log.info("\n\t----- max feature in each row: {} -----".format(imp.idxmax(axis=1)))
            imp.to_csv("importance_lr.csv", index=False)
            top = pd.DataFrame({n: imp.T[col].nlargest(10).index.tolist()
                          for n, col in enumerate(imp.T)}).T
            top.to_csv('topimportance_lr.csv', index=False)
            box = imp.plot.box()
            plt.savefig("boxplot_feature_imp_lr.png")

            shap_vals.append(shap_vs)
            test_idxs.append(test_indx)

            log.debug(f"N folds {len(test_idxs)} {len(shap_vals)}")

            log.debug(f"lengths {' '.join([str(ent.shape) for ent in test_idxs])} {' '.join([str(elt.shape) for elt in shap_vals])}")

            if not all(len(ent) == len(test_idxs) for ent in test_idxs):
                newlength = min([len(ent) for ent in test_idxs])
                for i, (ti, sv) in enumerate(zip(test_idxs, shap_vals)):
                    if len(ti) != newlength:
                        dropn = len(ti) - newlength
                        test_idxs[i] = np.delete(test_idxs[i], [x for x in range(len(ti) - dropn, len(ti))], axis=0)
                        shap_vals[i] = np.delete(shap_vals[i], [x for x in range(len(ti) - dropn, len(ti))], axis=0)

                log.debug(f"New lengths {' '.join([str(ent.shape) for ent in test_idxs])} {' '.join([str(elt.shape) for elt in shap_vals])}")

            for sa in shap_vals:
                log.debug(f"SHAP array:\n{np.array(sa)}\n-----\n")

            test_set = test_idxs[0]
            shap_values = np.array(shap_vals[0])
            for i in range(1, len(test_idxs)-1):
                test_set = np.concatenate((test_set,test_idxs[i]),axis=0)
                log.debug(f"shap iteration {i}")
                shap_values = np.concatenate((shap_values, np.array(shap_vals[i])),axis=0)
            
            log.info("using sample of {} points".format(shap_values.shape[0]))
            X_test = pd.DataFrame(df.iloc[test_set, :], columns=[" ".join(ent.split("_") for ent in df.columns)])
            # X_test.to_csv("shap_x_test.csv", index=False)
            # shap_df = pd.DataFrame(shap_vals)
            # shap_df.to_csv("shap.csv", index=False)
            
            log.info(X_test.shape)
            shap.initjs()
            log.info(plt.get_fignums())
            figure = plt.figure()
            log.info(plt.get_fignums())
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig("shap.png", dpi=200, bbox_inches="tight")

    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)
    
    log_df.to_csv("logs2.csv")

    



def kfold_test_imbalenced_classifiers_with_optimization_tmp(df, classes, classifiers, clf_options, scale=True, cv=5, n_repeats=20, clf_names=None, 
                                                        class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                        random_seed=107901, overwrite=False, smote=False, smote_catagorical_indexes=None,
                                                        smote_continuous_indexes=None):
    """
    function to run classification test over classifiers using imbalenced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    log_df = pd.DataFrame()
    labelpredictions = pd.DataFrame()
    
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix = [],[],[],[],[]
    list_average_roc_auc,list_average_scores = [],[]
    disp,predictions,actual,index,same = [], [], [],[],[]
    probablity = []
    important = []
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.
    #rkf = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_seed)
    kf = KFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, "
             "false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        if not n_repeats % 2:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)))
            plt_rows = int(n_repeats/2.0)
        else:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)+1))
            plt_rows = int(n_repeats/2.0)+1
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
        
        # Loop over  Kfold here 
        for train_indx, test_indx in kf.split(df):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = classes.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = classes.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            if smote is True:
                log.info("Using SMOTE to balence data")
                min_class = None
                number_in_min_class = len(ytrain)
                for clabel in class_labels:
                    n_for_class = len(ytrain[ytrain["classes"] == clabel].index)
                    if n_for_class < number_in_min_class:
                        number_in_min_class = n_for_class
                        min_class = clabel
                
                log.info(f"iteration {kf_iteration} min class is label {min_class}")

                nneighbours = min(5, len(ytrain[ytrain["classes"]==clabel].index)-1)
                log.info("Number of neighbours used for SMOTE: {}".format(nneighbours))

                # feature_types is set when the fingerprints are calculated

                # No catagorical
                if smote_catagorical_indexes is None and smote_continuous_indexes is not None:
                    log.info("Feature types: Continous")
                    sm = SMOTE(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                # Some catagorical - Need to provide the indexes
                elif smote_catagorical_indexes is not None and smote_continuous_indexes is not None:
                    log.info("Feature types: Mixed")
                    sm = SMOTENC(sampling_strategy="not majority", categorical_features=smote_catagorical_indexes, random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                # All catagorical
                elif smote_catagorical_indexes is not None and smote_continuous_indexes is None:
                    log.info("Feature types: Catagorical")
                    sm = SMOTEN(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                else:
                    log.info("Assuming feature types: Catagorical")
                    sm = SMOTEN(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)

                log.debug(f"{Xtrain.to_numpy()}\n{ytrain.to_numpy()}")
                features_smote, classes_smote = sm.fit_resample(Xtrain.to_numpy(), ytrain.to_numpy())
                features_smote = pd.DataFrame(features_smote, columns=Xtrain.columns)
                classes_smote = pd.DataFrame(classes_smote, columns=["classes"])
                log.debug(f"{features_smote}\n{classes_smote}")
                n_sythetic_points = features_smote.shape[0] - len(Xtrain.index)
                log.info("SMOTE has added {} sythetic features to balence the classes".format(n_sythetic_points))

            # Grid search model optimizer
            opt_param = grid_search_classifier_parameters(classf, features_smote, classes_smote, clf_options, clf_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())

            if name == "ExtraTreesClassifier":
                important.append(clf.feature_importances_)
            # if name == "Logistic_Regression":
            #     important.append(clf.coef_)


            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
            sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
            sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            
            ## Get the confusion matrices 
            c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
            confusion_matrices.append(c_matrix)
            
            ## Calculate the roc area under the curve
            probs = clf.predict_proba(Xtest)
            fpr, tpr, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            list_roc_auc.append(roc_auc)
            
            roc_aucs.append(roc_auc)
            log.info("\tROC analysis area under the curve: {}".format(roc_auc))
            
            # output metrics for consideration
            log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))
            
            list_c_matrix.append(c_matrix)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   

            list_score.append(clf_score)
        
            log.info("\tImbalence reports:")
            log.info("\tImbalence classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
            output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)
            
            ## Plot the roc curves
            ax = plt.subplot(2, plt_rows, kf_iteration+1)
            ax.plot(fpr, tpr, color="red",
                     lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
            
                # ugliest legend i ve made in my life - maybe one under the other?
            
            try:
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "pre_class0 = {:.2f}\n".format(output_dict[0]['pre'])+"pre_class1 = {:.2f}".format(output_dict[1]['pre']))
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "f1_class0 = {:.2f}\n".format(output_dict[0]['f1'])+ "f1_class1 = {:.2f}".format(output_dict[1]['f1']))
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "rec_class0 = {:.2f}\n".format(output_dict[0]['rec'])+ "rec_class1 = {:.2f}".format(output_dict[1]['rec']))
            except KeyError as kerr:
                log.warning("Key error found plotting ROC curves {}".format(kerr))

            ax.plot([0, 1], [0, 1], "k:")
            ax.set_xlim(xmin=0.0, xmax=1.01)
            ax.set_ylim(ymin=0.0, ymax=1.01)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            
            log.info("Appending classification report")
            list_report.append(classification_report_imbalanced(ytest, predicted_clf))
            
            log.info("Getting sensitvity, specificity, support")
            sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
            log.debug("\t{} {} {}".format(sensitvity, specificity, support))

            log.info("Print predictions")
            log.info("\t Index | Predicted | Label\n\t------------------")
            log.info("\t{}\n-----\n".format("\n\t".join(["{}   |   {}   |   {}".format(i, p, k) for i, p, k in zip(test_indx, predicted_clf, ytest["classes"].values)])))
    
            pred = [list(test_indx),list(ytest["classes"].values),list(predicted_clf), list(probs[:,0]), list(probs[:,1])]
            
            pred = pd.DataFrame(pred)
            log.info("Save data")
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
            log.info("\nNext iteration\n")
        
        del predictions[:]
        
        if any(x not in tmp for x in [y for y in range(len(classes.index))]):
             log.info("WARNING there appears to be left over indexes which have not been used for testing: {}".format())
        else:
            log.info("All points have been used in a test case over all fold as they should have been")
        
        # Plot and assess classifier over all folds
        
        # NOTE - rows are scores columns are classes
        log.info("Getting means and standard deviations")
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        # average_roc_auc = np.mean(roc_aucs, axis=0)
        # std_roc_auc = np.std(roc_aucs, axis=0)
        
        log.info("{} {} {} {}".format(average_scores, std_scores)) #, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][0], 
                                                                                                                                   std_scores[0][0], 
                                                                                                                                   average_scores[1][0], 
                                                                                                                                   std_scores[1][0], 
                                                                                                                                   average_scores[2][0], 
                                                                                                                                   std_scores[2][0], 
                                                                                                                                   average_scores[3][0], 
                                                                                                                                   std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][1], 
                                                                                                                                   std_scores[0][1], 
                                                                                                                                   average_scores[1][1], 
                                                                                                                                   std_scores[1][1], 
                                                                                                                                   average_scores[2][1], 
                                                                                                                                   std_scores[2][1], 
                                                                                                                                   average_scores[3][1], 
                                                                                                                                   std_scores[3][1])
        #score_str3 ="Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n".format(score_str1, score_str2) #, score_str3)
        plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points', size=12, ha='center', va='bottom')
        figure.tight_layout()
        plt.savefig("{0}/{0}_roc_curves.png".format(name))
        plt.show()

        iteration = iteration + 1
        best=[]
        if name =='ExtraTreesClassifier':
            imp = pd.DataFrame(list(map(np.ravel, important)))
            log.info("\n\t----- Important features: {} -----".format(imp))
            log.info("\n\t----- max feature in each row: {} -----".format(imp.idxmax(axis=1)))
            imp.to_csv("importance.csv")
            top = pd.DataFrame({n: imp.T[col].nlargest(10).index.tolist()
                          for n, col in enumerate(imp.T)}).T
            top.to_csv('topimportance.csv')
            box = imp.plot.box()
            plt.savefig("boxplot_feature_imp.png")


    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)
    
    log_df.to_csv("logs2.csv")

def kfold_test_imbalenced_classifiers_with_optimization_no_smote(df, classes, classifiers, clf_options, scale=True, cv=5, n_repeats=20, clf_names=None, 
                                                        class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                        random_seed=107901, overwrite=False):
    """
    function to run classification test over classifiers using imbalenced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    log_df = pd.DataFrame()
    labelpredictions = pd.DataFrame()
    
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix = [],[],[],[],[]
    list_average_roc_auc,list_average_scores = [],[]
    disp,predictions,actual,index,same = [], [], [],[],[]
    probablity = []
    important = []
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.
    #rkf = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_seed)
    kf = KFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, "
             "false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        if not n_repeats % 2:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)))
            plt_rows = int(n_repeats/2.0)
        else:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)+1))
            plt_rows = nt(n_repeats/2.0)+1
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
        
        # Loop over  Kfold here 
        for train_indx, test_indx in kf.split(df):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = classes.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = classes.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            # Grid search model optimizer
            opt_param = grid_search_classifier_parameters(classf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())

            if name == "ExtraTreesClassifier":
                important.append(clf.feature_importances_)
            # if name == "Logistic_Regression":
            #     important.append(clf.coef_)


            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
            sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
            sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            
            ## Get the confusion matrices 
            c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
            confusion_matrices.append(c_matrix)
            
            ## Calculate the roc area under the curve
            probs = clf.predict_proba(Xtest)
            fpr, tpr, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            list_roc_auc.append(roc_auc)
            
            roc_aucs.append(roc_auc)
            log.info("\tROC analysis area under the curve: {}".format(roc_auc))
            
            # output metrics for consideration
            log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))
            
            list_c_matrix.append(c_matrix)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   

            list_score.append(clf_score)
        
            log.info("\tImbalence reports:")
            log.info("\tImbalence classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
            output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)
            
            ## Plot the roc curves
            ax = plt.subplot(2, plt_rows, kf_iteration+1)
            ax.plot(fpr, tpr, color="red",
                     lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
            
                # ugliest legend i ve made in my life - maybe one under the other?
            
            ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "pre_class0 = {:.2f}\n".format(output_dict[0]['pre'])+"pre_class1 = {:.2f}".format(output_dict[1]['pre']))
            ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "f1_class0 = {:.2f}\n".format(output_dict[0]['f1'])+ "f1_class1 = {:.2f}".format(output_dict[1]['f1']))
            ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "rec_class0 = {:.2f}\n".format(output_dict[0]['rec'])+ "rec_class1 = {:.2f}".format(output_dict[1]['rec']))

            ax.plot([0, 1], [0, 1], "k:")
            ax.set_xlim(xmin=0.0, xmax=1.01)
            ax.set_ylim(ymin=0.0, ymax=1.01)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            
                   
            list_report.append(classification_report_imbalanced(ytest, predicted_clf))
            
            sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
            log.debug("\t{} {} {}".format(sensitvity, specificity, support))
            log.info("\t Index | Predicted | Label\n\t------------------")
            log.info("\t{}\n-----\n".format("\n\t".join(["{}   |   {}   |   {}".format(i, p, k) for i, p, k in zip(test_indx, predicted_clf, ytest["classes"].values)])))
    
            pred = [list(test_indx),list(ytest["classes"].values),list(predicted_clf), list(probs[:,0]), list(probs[:,1])]
            
            pred = pd.DataFrame(pred)
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
        
        del predictions[:]
        
        if any(x not in tmp for x in [y for y in range(len(classes.index))]):
             log.info("WARNING there appears to be left over indexes which have not been used for testing: {}".format())
        else:
            log.info("All points have been used in a test case over all fold as they should have been")
        
        # Plot and assess classifier over all folds
        
        # NOTE - rows are scores columns are classes
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        average_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)
        
        log.info("{} {} {} {}".format(average_scores, std_scores, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][0], 
                                                                                                                                   std_scores[0][0], 
                                                                                                                                   average_scores[1][0], 
                                                                                                                                   std_scores[1][0], 
                                                                                                                                   average_scores[2][0], 
                                                                                                                                   std_scores[2][0], 
                                                                                                                                   average_scores[3][0], 
                                                                                                                                   std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][1], 
                                                                                                                                   std_scores[0][1], 
                                                                                                                                   average_scores[1][1], 
                                                                                                                                   std_scores[1][1], 
                                                                                                                                   average_scores[2][1], 
                                                                                                                                   std_scores[2][1], 
                                                                                                                                   average_scores[3][1], 
                                                                                                                                   std_scores[3][1])
        score_str3 ="Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n{}".format(score_str1, score_str2, score_str3)
        plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points', size=12, ha='center', va='bottom')
        figure.tight_layout()
        plt.savefig("{0}/{0}_roc_curves.png".format(name))
        plt.show()

        iteration = iteration + 1
        best=[]
        if name =='ExtraTreesClassifier':
            imp = pd.DataFrame(list(map(np.ravel, important)))
            log.info("\n\t----- Important features: {} -----".format(imp))
            log.info("\n\t----- max feature in each row: {} -----".format(imp.idxmax(axis=1)))
            imp.to_csv("importance.csv")
            top = pd.DataFrame({n: imp.T[col].nlargest(10).index.tolist()
                          for n, col in enumerate(imp.T)}).T
            top.to_csv('topimportance.csv')
            # for col in imp.columns:
            #     # hist = imp[col].hist(bins=10)
            box = imp.plot.box()
            #     print("Plotting for column {}".format(col))
            #     plt.savefig("histplot_%s.png" %col)
            plt.savefig("boxplot_feature_imp.png")
            #     plt.show()


        # info = imp.mask(imp == 0).describe()
        # info.to_csv("ETCfeaturesinfo.csv")
        # log.info("\n\t----- Describe features: {} -----".format(info))

    # if name == "Logistic_Regression":
    #     # print(important)
    #     imp = pd.DataFrame(list(map(np.ravel, important)))
    #     log.info("\n\t----- Important features: {} -----".format(imp))
    #     info = imp.describe()
    #     info.to_csv("LRfeaturesinfo.csv")
    #     log.info("\n\t----- Describe features: {} -----".format(info))

    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)
    
    log_df.to_csv("logs2.csv")
    
def kfold_test_imbalenced_classifiers_with_optimization_and_explanation(df, classes, classifiers, clf_options, scale=True, cv=5, n_repeats=20, clf_names=None, 
                                                        class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                        random_seed=107901, overwrite=False, generic_shap=False, ):
    """
    function to run classification test over classifiers using imbalenced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    log_df = pd.DataFrame()
    labelpredictions = pd.DataFrame()
    
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix = [],[],[],[],[]
    list_average_roc_auc,list_average_scores = [],[]
    disp,predictions,actual,index,same = [], [], [],[],[]
    probablity = []
    important = []
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.
    #rkf = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_seed)
    kf = KFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, "
             "false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        if not n_repeats % 2:
            figure = plt.figure(1, figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)))
            plt_rows = int(n_repeats/2.0)
        else:
            figure = plt.figure(1, figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)+1))
            plt_rows = nt(n_repeats/2.0)+1
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
        
        # Loop over  Kfold here 
        fold = 0
        test_set_index_list = []
        shap_values_list = []
        for train_indx, test_indx in kf.split(df):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = classes.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = classes.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            # Grid search model optimizer
            opt_param = grid_search_classifier_parameters(classf, Xtrain, ytrain, clf_options, clf_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())
            
            #shap
            # Kernel explainer is slow but suitable for any model so we use it here where we are testing different model types
            if generic_shap is False:
                explainer = shap.KernelExplainer(clf.predict, Xtest)
                shap_values = explainer.shap_values(Xtest)
            elif generic_shap is True:
                func = lambda x: clf.predict_proba(x)[:,1]
                medians = Xtrain.median().values.reshape((1,Xtrain.shape[1]))
                explainer = shap.Explainer(func, medians)
                shap_values = explainer(Xtest)
                
            test_set_index_list.append(test_indx)
            shap_values_list.append(shap_values)
            
            #log.info("SHAP values:{}\n".format(shap_values))
            #figg = plt.figure(2, figsize=(40,20))
            #ax1 = figg.add_subplot(121)
            #shap.plots.waterfall(shap_values, max_display=10, show=False)
            #ax = plt.subplot(3, 1, 2)
            #shap.summary_plot(shap_values, Xtest, show=False)
            #ax2 = figg.add_subplot(122)
            #shap.force_plot(explainer.expected_value, shap_values, Xtest, show=False)
            #plt.tight_layout()
            #plt.savefig("shap_summary_{}_{}.png".format(name, kf_iteration))
            #plt.close()
            
            # switch back to the kf roc curves
            #plt.figure(1)

            
            
#             plt.figure(2)
#             plot = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
#             log.info(type(plot))
#             plt.tight_layout()
#             plt.savefig("shap_waterfalls_{}_{}.png".format(name, kf_iteration))
#             plt.close()
#             plt.figure(1)

            if name == "ExtraTreesClassifier":
                important.append(clf.feature_importances_)
            # if name == "Logistic_Regression":
            #     important.append(clf.coef_)


            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
            sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
            sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            
            ## Get the confusion matrices 
            c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
            confusion_matrices.append(c_matrix)
            
            ## Calculate the roc area under the curve
            probs = clf.predict_proba(Xtest)
            fpr, tpr, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            list_roc_auc.append(roc_auc)
            
            roc_aucs.append(roc_auc)
            log.info("\tROC analysis area under the curve: {}".format(roc_auc))
            
            # output metrics for consideration
            log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))
            
            list_c_matrix.append(c_matrix)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   

            list_score.append(clf_score)
        
            log.info("\tImbalence reports:")
            log.info("\tImbalence classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
            output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)
            
            ## Plot the roc curves
            ax = plt.subplot(2, plt_rows, kf_iteration+1)
            ax.plot(fpr, tpr, color="red",
                     lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
            
                # ugliest legend i ve made in my life - maybe one under the other?
            
            ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "pre_class0 = {:.2f}\n".format(output_dict[0]['pre'])+"pre_class1 = {:.2f}".format(output_dict[1]['pre']))
            ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "f1_class0 = {:.2f}\n".format(output_dict[0]['f1'])+ "f1_class1 = {:.2f}".format(output_dict[1]['f1']))
            ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "rec_class0 = {:.2f}\n".format(output_dict[0]['rec'])+ "rec_class1 = {:.2f}".format(output_dict[1]['rec']))

            ax.plot([0, 1], [0, 1], "k:")
            ax.set_xlim(xmin=0.0, xmax=1.01)
            ax.set_ylim(ymin=0.0, ymax=1.01)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            
                   
            list_report.append(classification_report_imbalanced(ytest, predicted_clf))
            
            sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
            log.debug("\t{} {} {}".format(sensitvity, specificity, support))
            log.info("\t Index | Predicted | Label\n\t------------------")
            log.info("\t{}\n-----\n".format("\n\t".join(["{}   |   {}   |   {}".format(i, p, k) for i, p, k in zip(test_indx, predicted_clf, ytest["classes"].values)])))
    
            pred = [list(test_indx),list(ytest["classes"].values),list(predicted_clf), list(probs[:,0]), list(probs[:,1])]
            
            pred = pd.DataFrame(pred)
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
            
            
        if generic_shap is False:
            explainer = shap.KernelExplainer(clf.predict, Xtest)
            shap_values = explainer.shap_values(Xtest)
        elif generic_shap is True:
            func = lambda x: clf.predict_proba(x)[:,1]
            medians = Xtrain.median().values.reshape((1,Xtrain.shape[1]))
            explainer = shap.Explainer(func, medians)
            shap_values = explainer(Xtest)
                
        test_set_index_list.append(test_indx)
        shap_values_list.append(shap_values)
        
        test_set = test_set_index_list[0]
        shap_values = np.array(shap_values_list[0])
        for i in range(1,len(test_set_index_list)):
            test_set = np.concatenate((test_set, test_set_index_list[i]), axis=0)
            shap_values = np.concatenate((shap_values,np.array(shap_values_list[i])),axis=1)
        #bringing back variable names     
        Xtest_tmp = pd.DataFrame(X[test_set],columns=columns)
        
        fig = plt.figure(2, figsize=(20,20))
        shap.summary_plot(shap_values[1], X_test, show=False)
        plt.savefig("shap_{}_kfold.png".format(name))
        plt.figure(1)
        
        del predictions[:]
        
        if any(x not in tmp for x in [y for y in range(len(classes.index))]):
             log.info("WARNING there appears to be left over indexes which have not been used for testing: {}".format())
        else:
            log.info("All points have been used in a test case over all fold as they should have been")
        
        # Plot and assess classifier over all folds
        
        # NOTE - rows are scores columns are classes
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        average_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)
        
        log.info("{} {} {} {}".format(average_scores, std_scores, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][0], 
                                                                                                                                   std_scores[0][0], 
                                                                                                                                   average_scores[1][0], 
                                                                                                                                   std_scores[1][0], 
                                                                                                                                   average_scores[2][0], 
                                                                                                                                   std_scores[2][0], 
                                                                                                                                   average_scores[3][0], 
                                                                                                                                   std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][1], 
                                                                                                                                   std_scores[0][1], 
                                                                                                                                   average_scores[1][1], 
                                                                                                                                   std_scores[1][1], 
                                                                                                                                   average_scores[2][1], 
                                                                                                                                   std_scores[2][1], 
                                                                                                                                   average_scores[3][1], 
                                                                                                                                   std_scores[3][1])
        score_str3 ="Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n{}".format(score_str1, score_str2, score_str3)
        plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points', size=12, ha='center', va='bottom')
        figure.tight_layout()
        plt.savefig("{0}/{0}_roc_curves.png".format(name))
        plt.show()

        iteration = iteration + 1
        best=[]
        if name =='ExtraTreesClassifier':
            imp = pd.DataFrame(list(map(np.ravel, important)))
            log.info("\n\t----- Important features: {} -----".format(imp))
            log.info("\n\t----- max feature in each row: {} -----".format(imp.idxmax(axis=1)))
            imp.to_csv("importance.csv")
            top = pd.DataFrame({n: imp.T[col].nlargest(10).index.tolist()
                          for n, col in enumerate(imp.T)}).T
            top.to_csv('topimportance.csv')
            # for col in imp.columns:
            #     # hist = imp[col].hist(bins=10)
            box = imp.plot.box()
            #     print("Plotting for column {}".format(col))
            #     plt.savefig("histplot_%s.png" %col)
            plt.savefig("boxplot_feature_imp.png")
            #     plt.show()


        # info = imp.mask(imp == 0).describe()
        # info.to_csv("ETCfeaturesinfo.csv")
        # log.info("\n\t----- Describe features: {} -----".format(info))

    # if name == "Logistic_Regression":
    #     # print(important)
    #     imp = pd.DataFrame(list(map(np.ravel, important)))
    #     log.info("\n\t----- Important features: {} -----".format(imp))
    #     info = imp.describe()
    #     info.to_csv("LRfeaturesinfo.csv")
    #     log.info("\n\t----- Describe features: {} -----".format(info))

    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)
    
    log_df.to_csv("logs2.csv")
        
    

    
def directory_names_from_classfier_names(classifier_names):
    """
    Function to get the names of the directories from the classifier scripts
    :param classifier_names: list - list of classifier names
    """
    
    names = ["{}".format("_".join(name.split())) for name in classifier_names]
    
    return names
    
def build_data_from_directory(data_directory, max_folds=10):
    """
    Fucntion to build a set of data from csv files names K.csv where K is the fold number and the csv
    is the predictions for the test data from that fold
    :param directory: str - name of the directory to read the csv files from
    :param max_fold: int - the number of folds run in the Kfold cv
    """
    
    log = logging.getLogger(__name__)
    
    for i in range(0, max_folds):
        log.info("Reading {}.csv".format(i))
        data_fold = pd.read_csv(os.path.join(data_directory, "{}.csv".format(i)), header=0)
        #log.info(data_fold)
        if i == 0:
            data = data_fold.copy()
        else:
            data = pd.concat([data, data_fold])

    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.columns = ["m_index", "known", "prediction", "prob0", "prob1"]
    data["m_index"] = [int(ent) for ent in data["m_index"].values]
    data.set_index("m_index", inplace=True, drop=True, verify_integrity=True)
    data.sort_index(inplace=True)
    
    return data


def build_data_from_directory_regr(data_directory, max_folds=10):
    """
    Fucntion to build a set of data from csv files names K.csv where K is the fold number and the csv
    is the predictions for the test data from that fold
    :param directory: str - name of the directory to read the csv files from
    :param max_fold: int - the number of folds run in the Kfold cv
    """

    log = logging.getLogger(__name__)

    for i in range(0, max_folds):
        log.info("Reading {}.csv".format(i))
        data_fold = pd.read_csv(os.path.join(data_directory, "{}.csv".format(i)), header=0)
        # log.info(data_fold)
        if i == 0:
            data = data_fold.copy()
        else:
            data = pd.concat([data, data_fold])

    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.columns = ["index", "known", "prediction"]
    data["index"] = [int(ent) for ent in data["index"].values]
    data.set_index("index", inplace=True, drop=True, verify_integrity=True)
    data.sort_index(inplace=True)

    return data

def metrics_for_regression(directories=('LassoCV',
 'KNeighborsRegressor',
 'Decision_Tree_Regressor',
 'SVR',
 'Bayesian_Regr'), max_folds=10,
                            names=None, smiles=None):
    log = logging.getLogger(__name__)

    for directory in directories:

        log.info("\n-----\nAnalyzing predictions for model {}\n-----".format(directory))
        data = build_data_from_directory_regr(directory, max_folds=max_folds)

        if names is not None:
            data["names"] = names
        if smiles is not None:
            data["smiles"] = smiles

        variance = explained_variance_score(data['known'], data['prediction'])
        MAE = mean_absolute_error(data['known'], data['prediction'])
        MSE = mean_squared_error(data['known'], data['prediction'])
        R2 = r2_score(data['known'], data['prediction'])
        log.info("\n-----\n Scores for Regressor: Explained Variance: {}, MAE: {}, MSE: {},R2: {}\n-----".format(variance,MAE,MSE,R2))
        f = open("{}/metrics.txt".format(directory), "w")
        f.write(str(variance))
        f.write("\n")
        f.write(str(MAE))
        f.write("\n")
        f.write(str(MSE))
        f.write("\n")
        f.write(str(R2))
        f.write("\n")
        f.close()
        plt.scatter(data['known'], data['prediction'], color='blue',marker='x')
        plt.ylabel('Prediction',fontsize=20)
        plt.xlabel('Actual',fontsize=20)
        plt.plot([-1, 4], [-1, 4], "k:")

        plt.xticks(np.arange(-1, 4, step=0.5))
        plt.yticks(np.arange(-1, 4, step=0.5))
        plt.savefig("{}/regression.png".format(directory))
        plt.show()

def metrics_for_all_classes(directories=("AdaBoost", "Decision_Tree", "ExtraTreesClassifier", "Gaussian_Process", "Logistic_Regression", "Nearest_Neighbors"), max_folds=10, 
                            names=None, smiles=None):
    """
    Function to run over all directories and build the outputs over the folds in folds ML training scheme.
    :param directories: iterable - all of the directories to loop over
    :param max_folds: int - The number of folds which the data was passed over and predictions saved to $(fold_number).csv
    """
    
    log = logging.getLogger(__name__)

    for directory in directories:
          
        log.info("\n-----\nAnalyzing predictions for model {}\n-----".format(directory))
        data = build_data_from_directory(directory, max_folds=max_folds)
        if names is not None:
            data["names"] = names
        if smiles is not None:
            data["smiles"] = smiles
            
        if "names" or "smiles" in data.keys():
            misclassed = which_are_misclassified(data)
        elif "names" and "smiles" in data.keys():
            misclassed = which_are_misclassified(data)
        else:
            log.info("Ideally give at least one of names or smiles to find out which molecules were misclassified")
            misclassed = which_are_misclassified(data, return_indx=True)
        
        with open(os.path.join(directory, "misclassed_molecules.csv"), "w") as fout:
            misclassed.to_csv(fout)
        
        probs = data[["prob0", "prob1"]].to_numpy()
        
        #cmetrics.plot_metrics(data, probabilities=probs, name="{}/metrics_plot.png".format(directory))
        c = cmetrics.get_multi_label_confusion_matrix(df=data, return_dict=True)
        multi_metrics = cmetrics.calculate_multi_label_confusion_based_metrics(df=data, probabilities=probs, positive_label=1, imbalanced=True,
                                                                               plt_filename=os.path.join(directory, "metrics.png"), verbose=False)
        f = open("{}/multi_metrics.txt".format(directory), "w")
        f.write( str(multi_metrics) )
        f.close()

        a_file = open("{}/multi_metrics.pkl".format(directory), "wb")

        pickle.dump(multi_metrics, a_file)

        a_file.close()

        metrics0=[]
        metrics = pd.DataFrame(columns=['tpr','fpr','tnr','fnr','f_half','f1','f2','MCC','precision','recall','roc_auc','pr_auc'], index = [0,1])
        metrics0.append((round(multi_metrics[0]['tpr'], 2)))
        metrics0.append((round(multi_metrics[0]['fpr'], 2)))
        metrics0.append((round(multi_metrics[0]['tnr'], 2)))
        metrics0.append((round(multi_metrics[0]['fnr'], 2)))
        metrics0.append((round(multi_metrics[0]['f_half'], 2)))
        metrics0.append((round(multi_metrics[0]['f1'], 2)))
        metrics0.append((round(multi_metrics[0]['f2'], 2)))
        metrics0.append((round(multi_metrics[0]['matthews_correlation_coefficient'], 2)))
        metrics0.append((round(multi_metrics[0]['precision'], 2)))
        metrics0.append((round(multi_metrics[0]['recall'], 2)))
        metrics0.append((round(multi_metrics[0]['roc_auc'], 2)))
        metrics0.append((round(multi_metrics[0]['pr_auc'], 2)))

        metrics.loc[0] = metrics0

        metrics1=[]
        metrics1.append((round(multi_metrics[1]['tpr'], 2)))
        metrics1.append((round(multi_metrics[1]['fpr'], 2)))
        metrics1.append((round(multi_metrics[1]['tnr'], 2)))
        metrics1.append((round(multi_metrics[1]['fnr'], 2)))
        metrics1.append((round(multi_metrics[1]['f_half'], 2)))
        metrics1.append((round(multi_metrics[1]['f1'], 2)))
        metrics1.append((round(multi_metrics[1]['f2'], 2)))
        metrics1.append((round(multi_metrics[1]['matthews_correlation_coefficient'], 2)))
        metrics1.append((round(multi_metrics[1]['precision'], 2)))
        metrics1.append((round(multi_metrics[1]['recall'], 2)))
        metrics1.append((round(multi_metrics[1]['roc_auc'], 2)))
        metrics1.append((round(multi_metrics[1]['pr_auc'], 2)))

        metrics.loc[1] = metrics1

        metrics.to_csv("{}/metrics.csv".format(directory))
        metrics.to_latex("{}/metric.tex".format(directory), index=True)
        log.info("Over all data points including smote points")
        display(Image(os.path.join(directory, "metrics.png")))
        
        log.info("Over all data points including smote points:\n{}".format(metrics))

        ####
        """
        IMPORTANT BE CAREFUL HERE! This index needs to change depending on how many SMOTE are created!!!
        89-110 are for OPERA & mordred with conditions classification.
        
        When only Mordred are used we have 101 original data!
        """
        # data.drop(data.index[101:134], axis=0, inplace=True)
        data.drop(data.index[131:166], axis=0, inplace=True)
        # the above is with added data
        # data.drop(data.index[89:110], axis=0,inplace=True)
        probs = data[["prob0", "prob1"]].to_numpy()

        metrics = cmetrics.calculate_confusion_based_metrics(df=data, probabilities=probs, positive_label=1)

        #cmetrics.plot_metrics_skplt(data, probabilities=probs, name="{}/_original_data_metrics_plot.png".format(directory))
        c = cmetrics.get_multi_label_confusion_matrix(df=data, return_dict=True)

        multi_metrics = cmetrics.calculate_multi_label_confusion_based_metrics(df=data, probabilities=probs, positive_label=1, imbalanced=True, 
                                                                               plt_filename=os.path.join(directory, "metrics_real_only.png"), verbose=False)

        f = open("{}/multi_metrics_original.txt".format(directory),"w")
        f.write( str(multi_metrics) )
        f.close()

        a_file = open("{}/multi_metrics_original.pkl".format(directory), "wb")

        pickle.dump(multi_metrics, a_file)

        a_file.close()

        metrics0=[]
        metrics = pd.DataFrame(columns=['tpr','fpr','tnr','fnr','f_half','f1','f2','MCC','precision','recall','roc_auc','pr_auc'], index = [0,1])
        metrics0.append((round(multi_metrics[0]['tpr'], 2)))
        metrics0.append((round(multi_metrics[0]['fpr'], 2)))
        metrics0.append((round(multi_metrics[0]['tnr'], 2)))
        metrics0.append((round(multi_metrics[0]['fnr'], 2)))
        metrics0.append((round(multi_metrics[0]['f_half'], 2)))
        metrics0.append((round(multi_metrics[0]['f1'], 2)))
        metrics0.append((round(multi_metrics[0]['f2'], 2)))
        metrics0.append((round(multi_metrics[0]['matthews_correlation_coefficient'], 2)))
        metrics0.append((round(multi_metrics[0]['precision'], 2)))
        metrics0.append((round(multi_metrics[0]['recall'], 2)))
        metrics0.append((round(multi_metrics[0]['roc_auc'], 2)))
        metrics0.append((round(multi_metrics[0]['pr_auc'], 2)))

        metrics.loc[0] = metrics0


        metrics1=[]
        metrics1.append((round(multi_metrics[1]['tpr'], 2)))
        metrics1.append((round(multi_metrics[1]['fpr'], 2)))
        metrics1.append((round(multi_metrics[1]['tnr'], 2)))
        metrics1.append((round(multi_metrics[1]['fnr'], 2)))
        metrics1.append((round(multi_metrics[1]['f_half'], 2)))
        metrics1.append((round(multi_metrics[1]['f1'], 2)))
        metrics1.append((round(multi_metrics[1]['f2'], 2)))
        metrics1.append((round(multi_metrics[1]['matthews_correlation_coefficient'], 2)))
        metrics1.append((round(multi_metrics[1]['precision'], 2)))
        metrics1.append((round(multi_metrics[1]['recall'], 2)))
        metrics1.append((round(multi_metrics[1]['roc_auc'], 2)))
        metrics1.append((round(multi_metrics[1]['pr_auc'], 2)))

        metrics.loc[1] = metrics1

        metrics.to_csv("{}/metrics_original.csv".format(directory))
        log.info("Over all REAL data points NOT including smote points")
        display(Image(os.path.join(directory, "metrics_real_only.png")))
        
        log.info("Over all REAL data points NOT including smote points:\n{}".format(metrics))
    
def which_are_misclassified(data, known_column_label="known", predicted_column_label="prediction", return_indx=False):
    """
    Function to get the molecules which are misclassified
    :param data: pandas dataframe - columns of at least known_column_label and predicted_column_label good to have smiles and name
    """
    log = logging.getLogger(__name__)
    
    log.info(data)
    
    if return_indx is False:
        df = data[data[known_column_label] != data[predicted_column_label]]
        if "names" in df.keys() and "smiles" in df.keys():
            log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{} {}".format(name, smile) for name, smile in zip(df["names"].values,df["smiles"].values)])))
        elif "names" in df.keys():
            log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{}".format(name) for name in df["names"].values])))
        elif "smiles" in df.keys():
            log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{}".format(smile) for smile in df["smiles"].values])))
    
    else:
        df = data[data[known_column_label] != data[predicted_column_label]]
        log.info("The molecules which are misclassified are:\n{}".format("\n".join(["{}".format(ith) for ith in df.index])))
    
    return df

def get_feature_names_from_column_transformers(ctransformer):
    """
    Function to get feature names from sklearn column transofrmers see
    https://towardsdatascience.com/extracting-plotting-feature-names-importance-from-scikit-learn-pipelines-eb5bfa6a31f4
    :param ctransformer: columnTransformer - sklearn transformer
    """
    
    log = logging.getLogger(__name__)

    log.info("\n-----\nGetting feature names from column transformer\n-----\n")
    
    new_feature_names = []
    trans_list = []
    
    for ith, ent in enumerate(ctransformer.transformers_):
        trans_name, trans, original_feature_name = ent
        log.info("Transformer index: {}\nTranformer name: {}\nTransformer: {}\nOriginal feature names: {}\n".format(ith, trans_name, trans, original_feature_name))
        if hasattr(trans, "get_feature_names_out"):
            if isinstance(trans, OneHotEncoder):
                names = list(trans.get_feature_names_out(original_feature_name))
            else:
                names = list(trans.get_feature_names_out())
            
        elif hasattr(trans, "features_"):
            missing_indicator_indices = trans.indicator_.features_
            missing_features = [original_feature_name[ith] + "_missing_flag" for ith in missing_indicator_indices]
            names = original_feature_name + missing_features

        else:
            names = original_feature_name
                
        new_feature_names = new_feature_names + names
        trans_list = trans_list + [trans_name]*len(names)
        
    return new_feature_names


############### NOT VERY USEFUL #################


def test_classifiers(df, classes, classifiers, smiles=None, names=None, scale=True, clf_names=None, class_labels=(0,1), random_seed=1059701):
    """
    function to run classification test over classifiers
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.warning("IT IS NOT RECOMMENDED TO USE THIS FUNCTION!!!!!")
    
    log.info("Features: {}".format(df.columns))
    
    iteration = 1
    data = df.copy()
    data.reset_index(inplace=True)
    
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    # We will need to add scaling
#     if scale is True:
#         data = minmaxscale(data)
#         log.info("Scaled data:\n{}".format(data))
#     else:
#         log.info("Using unscaled features")
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(data, classes, test_size=0.1, random_state=random_seed, shuffle=True)
    log.debug("Xtrain: {}\nXtest: {}\nytrain: {}\nytest: {}".format(Xtrain, Xtest, ytrain, ytest))
    log.info("Test set is made up of:\n{}".format("\n".join(["name {} smiles {} class {}".format(names[j], smiles[j], c) for j, c in zip(Xtest.index, ytest)])))
    
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for clf in classifiers:
        log.info("\n----- {} -----".format(names))
        clf.fit(Xtrain, ytrain)
        clf_score = clf.score(Xtest, ytest)
        predicted_clf = clf.predict(Xtest)
        c_matix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matix,display_labels=class_labels)
        disp.plot() 
        log.info("Confusion matrix:\n{}\n".format(c_matix))
        log.info("\nscore: {}".format(clf_score))
        log.info("{}".format(classification_report(ytest, predicted_clf)))
        log.info("Predicted | Label\n------------------")
        log.info("{}".format("\n".join(["{}   |   {}".format(p, k) for p, k in zip(predicted_clf, ytest)])))

def test_classifiers_with_optimization(df, classes, classifiers, clf_options, smiles=None, names=None, scale=True, cv=5, 
                                       clf_names=None, class_labels=(0,1), no_train_output=False, test_set_size=0.2, random_seed=1059701):
    """
    function to run classification test over classifiers
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.warning("IT IS NOT RECOMMENDED TO USE THIS FUNCTION!!!!!")
    
    log.info("Features: {}".format(df.columns))
    
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
#     if scale is True:
#         data = minmaxscale(data)
#         log.info("Scaled data:\n{}".format(data))
#     else:
#         log.info("Using unscaled features")
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(data, classes, test_size=test_set_size, random_state=random_seed, shuffle=True)
    log.info("Xtrain: {}\nXtest: {}\nytrain: {}\nytest: {}".format(Xtrain, Xtest, ytrain, ytest))
    log.info("{} {}".format(Xtest.index, ytest))
    log.info("Test set is made up of:\n{}".format("\n".join(["name {} smiles {} class {}".format(names[j], smiles[j], c) for j, c in zip(Xtest.index, ytest["classes"].values)])))
    
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        log.info("Search for optimal parameters for {}".format(name))
        
        # Grid search model optimizer
        clf = classf
        parameters = clf_options[clf_names[iteration]]
        log.debug("\tname: {} parameters: {}".format(name, parameters))
        optparam_search = GridSearchCV(clf, parameters, cv=cv, error_score=np.nan)
        log.debug("CV xtrain: {}".format(Xtrain))
        optparam_search.fit(Xtrain, ytrain.values.ravel())
        opt_parameters = optparam_search.best_params_
        if no_train_output is False:
            log.info("\t{}".format(pd.DataFrame(data=optparam_search.cv_results_)))
            log.info("\tBest parameters; {}".format(opt_parameters))
        else:
            pass
        
        # Fit final model using optimized parameters
        log.info("\n----- {} -----".format(name))
        log.debug("Xtrain: {}\nXtest: {}\nytrain: {}\nytest: {}".format(Xtrain, Xtest, ytrain, ytest))
        clf.fit(Xtrain, ytrain.values.ravel())
        clf_score = clf.score(Xtest, ytest)
        predicted_clf = clf.predict(Xtest)
        c_matix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
        probs = clf.predict_proba(Xtest)
#         print("probablity of a class 1 ",probs[:,0])
#         log.info("probablity of a class".format(probs))
        fp, tp, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
        roc_auc = auc(fp, tp)
        log.info("ROC analysis:\n\tTrue positives:\n\t{}\n\tFalse positives:\n\t{}".format(tp, fp))
        plt.figure(figsize=(10,10))
        plt.plot(fp, tp, color="red",
                 lw=1.5, label="ROC curve (auc={:.2f})".format(roc_auc))
        plt.plot([0, 1], [0, 1], "k:")
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
        
        
        # output metrics for consideration
        log.info("Confusion matrix ({}):\n{}\n".format(name, c_matix))
        log.info("\nscore ({}): {}".format(name, clf_score))
        log.info("{}".format(classification_report(ytest, predicted_clf)))
        log.info("Predicted | Label\n------------------")
        log.info("{}\n-----\n".format("\n".join(["{}   |   {}".format(p, k) for p, k in zip(predicted_clf, ytest["classes"].values)])))
        
        iteration = iteration + 1


def kfold_test_imbalenced_classifiers_with_optimization(df, classes, classifiers, clf_options, scale=True, cv=5, n_repeats=20, clf_names=None, 
                                                        class_labels=(0,1), no_train_output=False, test_set_size=0.2, smiles=None, names=None,
                                                        random_seed=107901, overwrite=False, smote=False, smote_catagorical_indexes=None,
                                                        smote_continuous_indexes=None):
    """
    function to run classification test over classifiers using imbalenced resampling
    inspired from https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    :param df: dataframe - data frame of features and identifers (smiles and/or names)
    :param classes: iterable - list of classes/labels
    :param classifiers: list - list of classifier methods
    :param plot: true/false - plot the results or not
    """
    
    log = logging.getLogger(__name__)
    
    log.info("Features: {}".format(df.columns))
    
    log_df = pd.DataFrame()
    
    list_report, list_roc_auc, list_opt_param, list_score, list_c_matrix = [],[],[],[],[]
    predictions = []
    important = []
    iteration = 0
    pd.set_option('display.max_columns', 20)
    data = df.copy()
    data.reset_index(inplace=True)
        
    if clf_names is None:
        clf_names = [i for i in range(0, len(classifiers))]
    
    if scale is True:
        data = minmaxscale(data)
        log.info("Scaled data:\n{}".format(data))
    else:
        log.info("Using unscaled features")
    
    # Kfold n_repeats is the number of folds to run.
    # Setting the random seed determines the degree of randomness. This means run n_repeats of 
    # independent cross validators.
    #rkf = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_seed)
    kf = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=random_seed)
    log.info("Starting classification: NOTE on confusion matrix - In binary classification, true negatives is element 0,0, "
             "false negatives is element 1,0, true positives is element 1,1 and false positives is element 0,1")
    for name, classf in zip(clf_names, classifiers):
        log.info("\n-----\nBegin {}\n-----\n".format(name))
        
        kf_iteration = 0
        if not n_repeats % 2:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)))
            plt_rows = int(n_repeats/2.0)
        else:
            figure = plt.figure(figsize=(2 * 20.0, 5.0 * int(n_repeats/2.0)+1))
            plt_rows = int(n_repeats/2.0)+1
        scores = []
        confusion_matrices = []
        roc_aucs = []
        score_list = []
        tmp = []
        name = "{}".format("_".join(name.split()))
        
        # Make directory for each classifier
        if not os.path.isdir(name):
            os.makedirs(name, exist_ok = True)
        elif overwrite is False and os.path.isdir(name) is True:
            log.warning("Directory already exists and overwrite is False will stop before overwriting.".format(name))
            return None
        else:
            log.info("Directory {} already exists will be overwritten".format(name))
        
        # Loop over  Kfold here 
        for train_indx, test_indx in kf.split(df, classes):
            log.info("----- {}: Fold {} -----".format(name, kf_iteration))
            
            tmp = tmp + test_indx.tolist()
            log.info(test_indx.tolist())
            
            # Set the training and testing sets by train test index
            log.info("\tTrain indx {}\n\tTest indx: {}".format(train_indx, test_indx))
            
            # Train
            Xtrain = df.iloc[train_indx]
            log.debug("Train X\n{}".format(Xtrain))
            ytrain = classes.iloc[train_indx]
            log.debug("Train Y\n{}".format(ytrain))
            
            # Test
            Xtest = df.iloc[test_indx]
            log.debug("Test X\n{}".format(Xtest))
            ytest = classes.iloc[test_indx]
            log.debug("Test Y\n{}".format(ytest))
            
            # way to calculate the test indexes
            #test_i = np.array(list(set(df.index) - set(train_indx)))

            if smote is True:
                log.info("Using SMOTE to balence data")
                min_class = None
                number_in_min_class = len(ytrain)
                for clabel in class_labels:
                    n_for_class = len(ytrain[ytrain["classes"] == clabel].index)
                    if n_for_class < number_in_min_class:
                        number_in_min_class = n_for_class
                        min_class = clabel
                
                log.info(f"iteration {kf_iteration} min class is label {min_class}")

                nneighbours = min(5, len(ytrain[ytrain["classes"]==clabel].index)-1)
                log.info("Number of neighbours used for SMOTE: {}".format(nneighbours))

                # feature_types is set when the fingerprints are calculated

                # No catagorical
                if smote_catagorical_indexes is None and smote_continuous_indexes is not None:
                    log.info("Feature types: Continous")
                    sm = SMOTE(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                # Some catagorical - Need to provide the indexes
                elif smote_catagorical_indexes is not None and smote_continuous_indexes is not None:
                    log.info("Feature types: Mixed")
                    sm = SMOTENC(sampling_strategy="not majority", categorical_features=smote_catagorical_indexes, random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                # All catagorical
                elif smote_catagorical_indexes is not None and smote_continuous_indexes is None:
                    log.info("Feature types: Catagorical")
                    sm = SMOTEN(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)
                else:
                    log.info("Assuming feature types: Catagorical")
                    sm = SMOTEN(sampling_strategy="not majority", random_state=random_seed, k_neighbors=nneighbours, n_jobs=-1)

                log.debug(f"{Xtrain.to_numpy()}\n{ytrain.to_numpy()}")
                features_smote, classes_smote = sm.fit_resample(Xtrain.to_numpy(), ytrain.to_numpy())
                features_smote = pd.DataFrame(features_smote, columns=Xtrain.columns)
                classes_smote = pd.DataFrame(classes_smote, columns=["classes"])
                log.debug(f"{features_smote}\n{classes_smote}")
                n_sythetic_points = features_smote.shape[0] - len(Xtrain.index)
                log.info("SMOTE has added {} sythetic features to balence the classes".format(n_sythetic_points))

            # Grid search model optimizer
            opt_param = grid_search_classifier_parameters(classf, features_smote, classes_smote, clf_options, clf_names, iteration, no_train_output, cv=cv, name=name)
            
            list_opt_param.append(opt_param)
            
            # Fit final model using optimized parameters
            clf = classf
            clf.set_params(**opt_param)
            log.info("\n\t----- Predicting using: {} -----".format(name))
            log.debug("\tXtrain: {}\n\tXtest: {}\n\tytrain: {}\n\tytest: {}".format(Xtrain, Xtest, ytrain, ytest))
            clf.fit(Xtrain, ytrain.values.ravel())
            
            if name.strip() == "ExtraTreesClassifier":
                important.append(clf.feature_importances_)
            if re.search(r"^logistic\wregression$", name.strip().lower()):
                log.info("Saving importance LR model")
                important.append(clf.coef_[0])


            # Evaluate the model
            ## evaluate the model on multiple metric score as list for averaging
            predicted_clf = clf.predict(Xtest)
            sc = precision_recall_fscore_support(ytest, predicted_clf, average=None)
            sc_df = pd.DataFrame(data=np.array(sc).T, columns=["precision", "recall", "f1score", "support"])
            sc_df.to_csv(os.path.join(name, "fold_{}_score.csv".format(kf_iteration)))
            score_list.append(sc)
            
            ## evaluate the principle score metric only (incase different to those above although this is unlikely)
            clf_score = clf.score(Xtest, ytest)
            scores.append(clf_score)
            
            ## Get the confusion matrices 
            c_matrix = confusion_matrix(ytest, predicted_clf, labels=class_labels)
            confusion_matrices.append(c_matrix)
            
            ## Calculate the roc area under the curve
            probs = clf.predict_proba(Xtest)
            fpr, tpr, thresholds = roc_curve(ytest, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            list_roc_auc.append(roc_auc)
            
            roc_aucs.append(roc_auc)
            log.info("\tROC analysis area under the curve: {}".format(roc_auc))
            
            # output metrics for consideration
            log.info("\tConfusion matrix ({}):\n{}\n".format(name, c_matrix))
            
            list_c_matrix.append(c_matrix)
            log.info("\n\tscore ({}): {}".format(name, clf_score))   

            list_score.append(clf_score)
        
            log.info("\tImbalence reports:")
            log.info("\tImbalence classification report:\n{}".format(classification_report_imbalanced(ytest, predicted_clf)))
            output_dict = classification_report_imbalanced(ytest, predicted_clf, output_dict=True)
            
            ## Plot the roc curves
            ax = plt.subplot(2, plt_rows, kf_iteration+1)
            ax.plot(fpr, tpr, color="red",
                     lw=1.5, label="ROC curve (auc = {:.2f})".format(roc_auc))
            
                # ugliest legend i ve made in my life - maybe one under the other?
            
            try:
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "pre_class0 = {:.2f}\n".format(output_dict[0]['pre'])+"pre_class1 = {:.2f}".format(output_dict[1]['pre']))
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "f1_class0 = {:.2f}\n".format(output_dict[0]['f1'])+ "f1_class1 = {:.2f}".format(output_dict[1]['f1']))
                ax.plot(fpr, tpr, alpha=0.0,color="white", lw=1.5,label= "rec_class0 = {:.2f}\n".format(output_dict[0]['rec'])+ "rec_class1 = {:.2f}".format(output_dict[1]['rec']))
            except KeyError as kerr:
                log.warning("Key error found plotting ROC curves {}".format(kerr))

            ax.plot([0, 1], [0, 1], "k:")
            ax.set_xlim(xmin=0.0, xmax=1.01)
            ax.set_ylim(ymin=0.0, ymax=1.01)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            
            log.info("Appending classification report")
            list_report.append(classification_report_imbalanced(ytest, predicted_clf))
            
            log.info("Getting sensitvity, specificity, support")
            sensitvity, specificity, support = sensitivity_specificity_support(ytest, predicted_clf)
            log.debug("\t{} {} {}".format(sensitvity, specificity, support))

            log.info("Print predictions")
            log.info("\t Index | Predicted | Label\n\t------------------")
            log.info("\t{}\n-----\n".format("\n\t".join(["{}   |   {}   |   {}".format(i, p, k) for i, p, k in zip(test_indx, predicted_clf, ytest["classes"].values)])))
    
            pred = [list(test_indx),list(ytest["classes"].values),list(predicted_clf), list(probs[:,0]), list(probs[:,1])]
            
            pred = pd.DataFrame(pred)
            log.info("Save data")
            pred.T.to_csv("{}/{}.csv".format(name, kf_iteration))
            kf_iteration = kf_iteration + 1
            log.info("\nNext iteration\n")
        
        del predictions[:]
        
        if any(x not in tmp for x in [y for y in range(len(classes.index))]):
             log.info("WARNING there appears to be left over indexes which have not been used for testing: {}".format())
        else:
            log.info("All points have been used in a test case over all fold as they should have been")
        
        # Plot and assess classifier over all folds
        
        # NOTE - rows are scores columns are classes
        log.info("Getting means and standard deviations")
        average_scores = np.mean(score_list, axis=0)
        std_scores = np.std(score_list, axis=0)
        average_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)
        
        log.info("{} {} {} {}".format(average_scores, std_scores, average_roc_auc, std_roc_auc))

        # precision_recall_fscore_support
        score_str1 = "Class 0: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][0], 
                                                                                                                                   std_scores[0][0], 
                                                                                                                                   average_scores[1][0], 
                                                                                                                                   std_scores[1][0], 
                                                                                                                                   average_scores[2][0], 
                                                                                                                                   std_scores[2][0], 
                                                                                                                                   average_scores[3][0], 
                                                                                                                                   std_scores[3][0])
        score_str2 = "Class 1: Pre: {:.2f} +/- {:.2f} Rec: {:.2f} +/- {:.2f} Fsc: {:.2f} +/- {:.2f} Sup: {:.2f} +/- {:.2f}".format(average_scores[0][1], 
                                                                                                                                   std_scores[0][1], 
                                                                                                                                   average_scores[1][1], 
                                                                                                                                   std_scores[1][1], 
                                                                                                                                   average_scores[2][1], 
                                                                                                                                   std_scores[2][1], 
                                                                                                                                   average_scores[3][1], 
                                                                                                                                   std_scores[3][1])
        #score_str3 ="Average ROC AUCs: {:.2f} +/- {:.2f}".format(average_roc_auc, std_roc_auc)
        score_text = "{}\n{}\n".format(score_str1, score_str2) #, score_str3)
        plt.annotate(score_text, xy=(0.5, 0), xytext=(0, 0), xycoords="figure fraction", textcoords='offset points', size=12, ha='center', va='bottom')
        figure.tight_layout()
        plt.savefig("{0}/{0}_roc_curves.png".format(name))
        plt.show()

        iteration = iteration + 1
        best=[]
        if name.strip() =='ExtraTreesClassifier':
            imp = pd.DataFrame(list(map(np.ravel, important)))
            log.info("\n\t----- Important features: {} -----".format(imp))
            log.info("\n\t----- max feature in each row: {} -----".format(imp.idxmax(axis=1)))
            imp.to_csv("importance.csv")
            top = pd.DataFrame({n: imp.T[col].nlargest(10).index.tolist()
                          for n, col in enumerate(imp.T)}).T
            top.to_csv('topimportance.csv')
            box = imp.plot.box()
            plt.savefig("boxplot_feature_imp.png")
        elif re.search(r"^logistic\wregression$", name.strip().lower()):
            imp = pd.DataFrame(list(map(np.ravel, important)))
            log.info("\n\t----- Important features: {} -----".format(imp))
            log.info("\n\t----- max feature in each row: {} -----".format(imp.idxmax(axis=1)))
            imp.to_csv("importance_lr.csv", index=False)
            top = pd.DataFrame({n: imp.T[col].nlargest(10).index.tolist()
                          for n, col in enumerate(imp.T)}).T
            top.to_csv('topimportance_lr.csv', index=False)
            box = imp.plot.box()
            plt.savefig("boxplot_feature_imp_lr.png")


    log_df["opt_param"] = pd.Series(list_opt_param)
    log_df["roc_auc"] = pd.Series(list_roc_auc)

    log_df["report"] = pd.Series(list_report)
    log_df["score"] = pd.Series(list_score)

    log_df["c_matrix"] = pd.Series(list_c_matrix)
    
    log_df.to_csv("logs2.csv")
