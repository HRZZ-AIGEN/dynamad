# Dynamic applicability domain (dAD)
Dynamic applicability domain is a method for defining the applicability domain of a sample x based on the prelocated conformity regions in the training set. The method is based on the conformal prediction framework, which is a framework for constructing prediction intervals with confidence estimates. The method is described in detail in the preprint [Dynamic applicability domain (dAD) for compound-target binding affinity prediction task with confidence guarantees](https://doi.org/10.1101/2022.08.22.504786).

<img width="1023" alt="infographic_dad_clean" src="https://github.com/HRZZ-AIGEN/dynamad/assets/75166378/9bc95c15-1179-4ed4-a0e1-45c56aa0b4ef">


# Documentation for the dynamad library
## I) Conformity region retrieval functions
The conf_region_e1 and conf_region_e2 functions are used to locate the most similar samples in the training set, or in other words, locate the conformity regions of e1 and e2 based on predefined number of neighbours, k and q respectively.
The functions take these arguments:
* `pairs` - list of interacting pairs for a given dataset, usually compounds and targets
* `e1_sim` - similarity matrix for e1 samples (n) toward the traning samples (m) of shape n*m
* `e2_sim` - similarity matrix for e2 samples (n) toward the traning samples (m) of shape n*m
* `k` - number of neighbours for e1
* `q` - number of neighbours for e2
* `id1` - column name for e1 samples
* `id2` - column name for e2 samples

## II) Calibration function
The calibration function is used to retrieve the calibration set of experimentally measured interactions in the traning set for sample x, based on the prelocated conformity regions. It takes these arguments:
* `pairs` - interacting pairs for a given dataset, usually compounds and targets (pandas dataframe)
* `c_conf` - conformity region for e1
* `t_conf` - conformity region for e2
* `i` - index of sample x in the dataset

## III) Prediction function - Defining the applicability domain based on neares neighbours
The prediction function dad_nn is used to predict the interaction of sample x based on the prelocated conformity regions. It takes these arguments:
* `x` - array of sample x features
* `pairs` - list of interacting pairs for a given dataset, usually compounds and targets
* `tr_pairs` - list of interacting pairs for a the training set, usually compounds and targets 
* `cv` - cross-validation predictions fort the traning samples
* `tr_y` - experimental values for the training set
* `model` - pretrained model
* `e1_sim` - similarity matrix for e1 samples (n) toward the traning samples (m) of shape n*m
* `e2_sim` - similarity matrix for e2 samples (n) toward the traning samples (m) of shape n*m
* `ispace` - interaction space of the traning data
* `k` - number of neighbours for e1
* `q` - number of neighbours for e2
* `id1` - column name for e1 samples
* `id2` - column name for e2 samples
* `type_m` - type of model used for prediction, can be 'x_conf' for finding nonconformity scores based on putative test noncoformity, or 'cal_conf' for finding nonconformity scores just based on calibration set nonconformity.


