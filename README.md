<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png width=135/>  eXtreme Gradient Boosting

# XGBoost
XGBoost: A Scalable Tree Boosting System

## Gradient Boosting
Gradient boosting refers to a class of ensemble machine learning algorithms that can be used for classification or regression predictive modeling problems.

Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models. This is a type of ensemble machine learning model referred to as boosting.

Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.

## Extreme Gradient Boosting
Extreme Gradient Boosting, or XGBoost for short, is an efficient open-source implementation of the gradient boosting algorithm. As such, XGBoost is an algorithm, an open-source project, and a Python library.

It was initially developed by Tianqi Chen and was described by Chen and Carlos Guestrin in their 2016 paper titled “XGBoost: A Scalable Tree Boosting System.”

It is designed to be both computationally efficient (e.g. fast to execute) and highly effective, perhaps more effective than other open-source implementations.

The two main reasons to use XGBoost are execution speed and model performance.

XGBoost dominates structured or tabular datasets on classification and regression predictive modeling problems. The evidence is that it is the go-to algorithm for competition winners on the Kaggle competitive data science platform.

-------
### Most Commonly Configured Hyperparameters
- n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
- max_depth: The maximum depth of each tree, often values are between 1 and 10.
- eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
- subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.
- colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.

For all parameters click [this](https://xgboost.readthedocs.io/en/latest/parameter.html).

Good hyperparameter values can be found by trial and error for a given dataset, or systematic experimentation such as using a grid search across a range of values.

When using machine learning algorithms that have a stochastic learning algorithm, it is good practice to evaluate them by averaging their performance across multiple runs or repeats of cross-validation. When fitting a final model, it may be desirable to either increase the number of trees until the variance of the model is reduced across repeated evaluations, or to fit multiple final models and average their predictions.

### Further

- Tutorials
  - [Extreme Gradient Boosting (XGBoost) Ensemble in Python](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/)
  - [Gradient Boosting with Scikit-Learn, XGBoost, LightGBM, and CatBoost](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
  - [Best Results for Standard Machine Learning Datasets](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
  - [How to Use XGBoost for Time Series Forecasting](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)
- Papers
  - [XGBoost: A Scalable Tree Boosting System, 2016.](https://arxiv.org/abs/1603.02754)


### Installation Guide for Python
Pre-built binary are uploaded to PyPI (Python Package Index) for each release. Supported platforms are Linux (x86_64, aarch64), Windows (x86_64) and MacOS (x86_64).

```
pip install xgboost
```
You might need to run the command with --user flag or use virtualenv if you run into permission errors. Python pre-built binary capability for each platform:

| Platform | GPU | Multi-Node-Multi-GPU |
| ------------- | ------------- | ------------- |
| Linux x86_64  | ✔ | ✔ |
| Linux aarch64  | ✘ | ✘ |
| MacOS  | ✘ | ✘ |
| Windows  | ✔ | ✘ |

Installation guide for other platforms click [this](https://xgboost.readthedocs.io/en/latest/install.html).

### Requirements
Requires: Python >=3.6

[![PyPI version](https://badge.fury.io/py/xgboost.svg)](https://pypi.python.org/pypi/xgboost/)

[![Conda version](https://img.shields.io/conda/vn/conda-forge/py-xgboost.svg)](https://anaconda.org/conda-forge/py-xgboost)

-------
#### Resources 
[XGBoost for Regression by Jason Brownlee on March 12, 2021](https://machinelearningmastery.com/xgboost-for-regression/)

[Udemy Course by M. Vahit Keskin](https://www.udemy.com/course/python-egitimi/) 

-------
## Example
- Dataset: Hitters.csv
 - Preprocessing: 
   - Cleaning - missing data
   - Normalization - One hot encoding
 - Evaluation Metric: RMSE
 - Hiperparameter Optimization: GridSearchCV

## Dataset: Hitters
Context

This dataset is part of the R-package ISLR and is used in the related book by G. James et al. (2013) "An Introduction to Statistical Learning with applications in R" to demonstrate how Ridge regression and the LASSO are performed using R.

Content

This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University. This is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.
