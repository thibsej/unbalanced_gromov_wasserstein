# Positive Unlabeled learning experiments

We propose in the paper to apply UGW to domain adaptation data in a PU learning setting. 
The unbalanced plan perform a partial matching of the data, which allows to predict which samples
should be in the same class as the source dataset.

The code is only available on the repo and uses extra packages. 
To reproduce the experiments, run the package,
install the dependencies and go into the folder.
```train
git clone https://github.com/thibsej/unbalanced_gromov_wasserstein
cd unbalanced_gromov_wasserstein/experiments_pu
pip install -r requirements.txt
```

Do not forget to install the package to run UGW solver, 
after installing numpy, matplotlib and pytorch. 
```setup
pip install unbalancedgw
```

## Download the data
The data is available [here](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code).
You should download the 'surf features' and 'decaf features of the Office dataset.
Then store it in a folder located at /unbalanced_gromov_wasserstein/experiments_pu/data.

## Compute the prediction and accuracies
To compute the prediction and convert the accuracies in a pandas dataframe, run:

```train
python compute_prediction.py
python compute_accuracy.py
```

## Observe the accuracy results
Then you can run the notebook 
[display_performance.ipynb](https://github.com/thibsej/unbalanced_gromov_wasserstein/blob/master/experiments_pu/display_performance.ipynb)
which displays the accuracy for all tasks.
The reproduction of the results from 
[Chapel et al.](https://arxiv.org/abs/2002.08276)
is available in display_results_pgw.ipynb.
