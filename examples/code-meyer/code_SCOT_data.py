import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa

import sys
sys.path.append('./SCOT-code/src/')
import scot2 as sc



path = 'SCOT-code/'

#####################   scGEM Dataset   ###########################
X_trans = np.genfromtxt(path + "data/scGEM_methylation.csv", delimiter=",")
y_trans = np.genfromtxt(path + "data/scGEM_expression.csv", delimiter=",")
print("Dimensions of input datasets are: ", "X= ", X_trans.shape, " y= ", y_trans.shape)

num_samples = np.shape(X_trans)[0]
## Normalize row by row the datasets
scot=sc.SCOT(X_trans, y_trans)
scot.normalize(norm='l2')
X = scot.X
Y = scot.y[0]

## Set the marginal to uniform distributions
a,b = np.ones(np.shape(X)[0])/np.shape(X)[0], np.ones(np.shape(Y)[0])/np.shape(Y)[0]

## Compute the distance matrices based on shortest path distance
k = 35
scot.init_distances(k)
D1 = scot.Cx[0]
D2 = scot.Cy[0]



#####################    Dataset SNAREseq    ########################
X_trans = np.load(path + "data/scatac_feat.npy")
y_trans = np.load(path + "data/scrna_feat.npy")
print("Dimensions of input datasets are: ", "X= ", X_trans.shape, " y= ", y_trans.shape)


num_samples = np.shape(X_trans)[0]
## Normalize row by row the datasets
scot=sc.SCOT(X_trans, y_trans)
scot.normalize(norm='l2')
X = scot.X
Y = scot.y[0]

## Set the marginal to uniform distributions
a,b = np.ones(np.shape(X)[0])/np.shape(X)[0], np.ones(np.shape(Y)[0])/np.shape(Y)[0]

## Compute the distance matrices based on shortest path distance
k = 50
scot.init_distances(k)
D1 = scot.Cx[0]
D2 = scot.Cy[0]
