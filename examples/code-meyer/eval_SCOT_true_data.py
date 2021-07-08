import numpy as np
import sys
import FastGromovWass
import LinSinkhorn
import time
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append('./EXP_GW/SCOT/SCOT-code/src/')
import scot2 as sc
import evals as evals


# Proj of X into Y
def proj_bary_X_to_Y(P,X,Y,a,b):
    return (np.dot(P,Y).T / a).T

# Proj of Y into X
def proj_bary_Y_to_X(P,X,Y,a,b):
    return (np.dot(P.T,X).T / b).T


def new_eval_func(P):
    frac = []
    n, m = np.shape(P)

    if n != m:
        print( 'n and m are not the same')
        return 'Error'

    for k in range(n):
        P_trans = P[k,:]
        ind_sorted = np.argsort(-P_trans)
        rank_1 = np.where(ind_sorted == k)[0][0]

        P_trans = P[:,k]
        ind_sorted = np.argsort(-P_trans)
        rank_2 = np.where(ind_sorted == k)[0][0]

        rank = rank_1 + rank_2
        frac.append(rank / (2 * (n-1)))

    return frac



## To load the file containing the matrix
path = 'EXP_GW/SCOT/results/'

with open(path + 'Couplings_Sin_exp2.npy', 'rb') as f:
    Couplings_Sin = np.load(f)

with open(path + 'Couplings_LR_exp2.npy', 'rb') as f:
    Couplings_LR = np.load(f)


path = 'EXP_GW/SCOT/SCOT-code/'
#### SNARE-seq DATASET #####
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

# Arr_Coupling_Sin = np.zeros((num_gammas_init,num_samples,num_samples))
# Arr_Coupling_LR = np.zeros((num_gammas_init,mod,num_samples,num_samples))

ranks = [10, 50, 100]
mod = len(ranks)

gammas_init = [250,100,50,10] #[1000,500,250,100,50,30,10,1]
num_gammas_init = len(gammas_init)


linestyles = ["-", "-.", ":", "--", "-"]
colors = ["g", "r", "b"]
linewidth = 3
labels_LR_free = ["GW-LR, r = " + str(num) for num in ranks]
fig, ax = plt.subplots(nrows=1, ncols=num_gammas_init)

v_min = 10
v_max = 0
for ind_gamma, gamma in enumerate(gammas_init):
    P_Sin = Couplings_Sin[ind_gamma+2,:,:]
    y_aligned = proj_bary_Y_to_X(P_Sin,X,Y,a,b)
    fracs = evals.calc_domainAveraged_FOSCTTM(X, y_aligned)
    print('ok')
    #fracs = new_eval_func(P_Sin)

    v_min = min(v_min,np.min(fracs))
    v_max = max(v_max,np.max(fracs))
    ax[ind_gamma].plot(np.arange(len(fracs)), np.sort(fracs),
        linestyle=linestyles[2],
        color="b",
        label="Entropic-GW",
        linewidth=linewidth + 2,
    )



    for ind_rank, rank in enumerate(ranks):
        P_LR = Couplings_LR[ind_gamma+2,ind_rank,:,:]
        y_aligned = proj_bary_Y_to_X(P_LR,X,Y,a,b)
        fracs = evals.calc_domainAveraged_FOSCTTM(X, y_aligned)
        print('ok')
        #fracs = new_eval_func(P_LR)

        v_min = min(v_min,np.min(fracs))
        v_max = max(v_max,np.max(fracs))
        ax[ind_gamma].plot(np.arange(len(fracs)), np.sort(fracs),
            linestyle=linestyles[1],
            color=plt.cm.autumn(ind_rank / mod),
            label=labels_LR_free[ind_rank],
            linewidth=linewidth + 1,
        )

for j in range(num_gammas_init):

    if j == 0:
        ax[j].set_ylabel("Sorted Error", fontsize=28)
    ax[j].set_xlabel("Cells", fontsize=28)
    ax[j].set_ylim([0.9 * v_min, 1.1 * v_max])
    ax[j].tick_params(axis="y", labelsize=20)
    ax[j].tick_params(axis="x", labelsize=15)

    ax[j].set_title("$\gamma$: " + str(gammas_init[j]), size=34)

handles, labels = ax[num_gammas_init - 1].get_legend_handles_labels()

ax[num_gammas_init - 1].legend(
    handles,
    labels,
    loc=(-3.6, -0.5),
    borderaxespad=0.1,
    frameon=False,
    fontsize=28,
    ncol=5,
)
fig.tight_layout()
fig.set_size_inches(35.5, 5)
plt.show()
path = 'EXP_GW/SCOT/results/'
fig.savefig(path+'plot_accuracy_FOSCTTM_not_all.pdf', bbox_inches="tight")
