[![DOI](https://zenodo.org/badge/722976777.svg)](https://zenodo.org/doi/10.5281/zenodo.10213103)

Machine learning prediction workflow for carbon capture amines. Reproduces the classifiers from [Chemical Space Analysis and Property Prediction for Carbon Capture Amine Molecules](https://chemrxiv.org/engage/chemrxiv/article-details/6465d217f2112b41e9bebcc8).

In folder "data", you will find the 98 molecules used for training and testing in a csv format.

To keep the Jupyter Notebooks uncluttered all functions have been moved to different modules in the folder "utils".

In folder "utils", you should find five python modules, "classification_metrics.py" contains all functions used to calculate the different classification metrics.
"classification_workflow_functions", contains all functions used for training and testing, as well as functions used to generate Mordred descriptors and identify correlating features. There are also functions used to generate the ccus fingerprints and separate the capacity classes.
"finger_prints.py", is the module that generates the ccus fingerprints and the MACCkeys. "molecules_and_images.py", contains functions to convert SMILES strings to molecules, inchies, and images. Finally, "plotting_sklear.py", contains all the functions used for plotting the confusion matrix, and ROC/PR curves.

The three notebooks, "ccus_fingerprints_workflow.ipynb", "maccs_workflow.ipynb", "mordred_workflow.ipynb" can be used to reproduce the results in the paper for the ccus fingerprints, the maccs keys and the Mordred descriptors accordingly.