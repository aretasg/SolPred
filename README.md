# Organic compound aqueous solubility prediction using the Delaney (ESOL) dataset

This is an attempt to model the Delany aqueous solubility dataset published in 2004 with contemporary ML approaches using Pycaret package and utilising ensemble stacking of top 3 best performing models.

Two approaches are taken to model the dataset
1. Using all the descriptors as initially proposed in the Delaney publication
2. Using VSA and other descriptors as proposed in other publications

## Installation
```bash
git clone https://github.com/aretasg/sol_pred.git
cd sol_pred
conda env create -f environment.yml
conda activate sol_pred
```

# Validation (Unseen data) Set Metrics
| Model | MAE | RMSE | R2 |
| --- | ---- | ---- | ---- |
| 1 | 0.4572 | 0.4537 | 0.8886 |
| 2 | 0.4558 | 0.4587 | 0.8874 |

* Both methods performed evidently better then the orignal ESOL and comparably between each other with the former (2) having a better residual plot with less outliers
* For all metrics and other please refer to the end of each jupyter notebook

# Availability
Both methods are distributed as `.pkl` files and an example script to run them

## References
* Delaney, 2004
* Avdeef, 2020

## License
MIT license
