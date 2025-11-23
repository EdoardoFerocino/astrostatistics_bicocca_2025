# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 07:52:52 2025

@author: edoardo ferocino
"""
import numpy as np
from astroML.datasets import generate_mu_z
from matplotlib import pyplot as plt
from astroML.linear_model import LinearRegression
from astroML.linear_model import PolynomialRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_predict
import pandas as pd

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234) # YOU CANNOT CHANGE THIS

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0.01, 1.8)
plt.ylim(36.01, 48)

def geterror(y_true,y_pred):
    return np.sqrt( np.sum(( y_true - y_pred )**2) / len(y_true) )

def performFit(model,X,y,dy):
    m = model
    m.fit(X, y, dy)
    return m.predict(z_grid[:,np.newaxis]), m

def drawFit(label,ax = None,X=None,y=None, dy = None,errorbarlabel='Data'):
    if ax is None:
        fig, ax = plt.subplots()
    if X is None and y is None:
        X = z_sample
        y = mu_sample
        dy = dmu
    ax.errorbar(X, y, dy, fmt='.k', ecolor='gray', lw=1, alpha = 0.3,label=errorbarlabel)
    ax.plot(z_grid, mu_predict, label = label)
    ax.set_xlim(0.01, 1.8)
    ax.set_ylim(36.01, 48)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc='lower right')
    ax.set_ylabel(r'$\mu$')
    ax.set_xlabel(r'$z$')
    return ax

FitErrors = True

z_grid = np.linspace(z_sample.min(),z_sample.max(),100)
mu_predict, model = performFit(LinearRegression(),z_sample[:,np.newaxis],mu_sample,dmu)
fit_ax = drawFit('Linear fit')

n_degree = 3
mu_predict, model = performFit(PolynomialRegression(n_degree),z_sample[:,np.newaxis],mu_sample,dmu)
drawFit(f'Poly fit with {n_degree} degrees',fit_ax)

n_degree = 6
mu_predict, model = performFit(PolynomialRegression(n_degree),z_sample[:,np.newaxis],mu_sample,dmu)
drawFit(f'Poly fit with {n_degree} degrees',fit_ax)

n_degree = 15
mu_predict, model = performFit(PolynomialRegression(n_degree),z_sample[:,np.newaxis],mu_sample,dmu)
drawFit(f'Poly fit with {n_degree} degrees',fit_ax)

n_degree = 2
mu_predict, model = performFit(PolynomialRegression(n_degree),z_sample[:,np.newaxis],mu_sample,dmu)
fit_ax = drawFit(f'Poly fit with {n_degree} degrees',None)

n_degree = 4
mu_predict, model = performFit(PolynomialRegression(n_degree),z_sample[:,np.newaxis],mu_sample,dmu)
drawFit(f'Poly fit with {n_degree} degrees',fit_ax)

z_train, z_test, mu_train, mu_test, dmu_train, dmu_test = train_test_split(z_sample[:,np.newaxis], mu_sample, dmu, test_size=0.2, random_state=42)

degreerange = list(range(0, 11))
# squared = False to match the root mean square. greater_is_better=False because is a loss function.
# Note that GridSearchCV tries to maximize the score functionDefine RMSE scorer (negative for maximization in GridSearchCV)
# scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
# scorer = 'neg_mean_squared_error'
scorer = make_scorer(geterror, greater_is_better=False)
grid = GridSearchCV(PolynomialRegression(), {'degree': degreerange}, scoring=scorer, cv=5, return_train_score = True)
grid.fit(z_train, mu_train) # Drawback: I can't fit the errors
print(grid.cv_results_)

scores = (np.column_stack((-grid.cv_results_['mean_test_score'],-grid.cv_results_['mean_train_score'])))
fig, ax = plt.subplots()
ax.plot(degreerange,scores,label=['test','train'])
# ax.plot(degreerange, 0.1 * np.ones(len(degreerange)), ':k')
ax.set_ylim(0, 4)
ax.set_ylabel('Error (score)')
ax.set_xlabel('Degrees')
plt.legend()

print(f'The best is {grid.best_params_}')

mu_predict = grid.best_estimator_.predict(z_grid[:,np.newaxis]) #fitted only on training data
best_fix_ax = drawFit(f'Best poly fit with {degreerange[grid.best_index_]} degrees',None,z_train,mu_train,dmu_train,'train data')
best_fix_ax.errorbar(z_test, mu_test, dmu_test, fmt='.r', ecolor='red', lw=1, alpha = 0.5, label = 'Test data')
best_fix_ax.legend()

# test_rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(f"Test RMSE using degreerange[grid.best_index_] degrees: {test_rmse:.3f}")

mu_predict = cross_val_predict(PolynomialRegression(degreerange[grid.best_index_]), z_sample[:,np.newaxis], mu_sample, cv=5)
fig, ax = plt.subplots()
ax.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1, alpha = 0.3)
ax.scatter(z_sample[:,np.newaxis], mu_predict, label = f'CrossValidated best poly fit with {degreerange[grid.best_index_]} degrees')
ax.set_xlim(0.01, 1.8)
ax.set_ylim(36.01, 48)
ax.legend(loc='lower right')
ax.set_ylabel(r'$\mu$')
ax.set_xlabel(r'$z$')

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(mu_sample, mu_predict, edgecolors=(0, 0, 0))
ax.plot([mu_sample.min(), mu_sample.max()], [mu_sample.min(), mu_sample.max()], 'k--', lw=4)
ax.set_xlabel('Actual [x1000]',fontsize=14)
ax.set_ylabel('Predicted [x1000]',fontsize=14)
plt.show()

if FitErrors:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    # Pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linreg', LinearRegression())
    ])
    
    param_grid = {
        'poly__degree': list(range(0, 11)),
        # 'linreg__fit_intercept': [True]
    }
    sample_weights = 1/(dmu_train**2)
    grid = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=5, return_train_score = True)
    grid.fit(z_train, mu_train,linreg__sample_weight=sample_weights)
    
    print(grid.cv_results_)

    scores = (np.column_stack((-grid.cv_results_['mean_test_score'],-grid.cv_results_['mean_train_score'])))
    fig, ax = plt.subplots()
    ax.plot(degreerange,scores,label=['test','train'])
    # ax.plot(degreerange, 0.1 * np.ones(len(degreerange)), ':k')
    ax.set_ylim(0, 4)
    ax.set_ylabel('Error (score)')
    ax.set_xlabel('Degrees')
    plt.legend()

    print(f'The best is {grid.best_params_}')

    mu_predict = grid.best_estimator_.predict(z_grid[:,np.newaxis]) #fitted only on training data
    best_fix_ax = drawFit(f'Best poly fit with {degreerange[grid.best_index_]} degrees',None,z_train,mu_train,dmu_train,'train data')
    best_fix_ax.errorbar(z_test, mu_test, dmu_test, fmt='.r', ecolor='red', lw=1, alpha = 0.5, label = 'Test data')
    best_fix_ax.legend()