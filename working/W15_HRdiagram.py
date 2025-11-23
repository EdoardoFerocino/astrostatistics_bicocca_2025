# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:42:30 2025

@author: edoardo ferocino
"""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt
import urllib.request
from sklearn import preprocessing


def perform_pca(data,number_of_components = None):
    pca = PCA(number_of_components)
    projecteddata = pca.fit_transform(data)
    n_components = pca.n_components_
    evariance = pca.explained_variance_
    evarianceratio = pca.explained_variance_ratio_
    components = pca.components_
    fig = plt.figure(figsize=(7, 7))
    projecteddata = pd.DataFrame(projecteddata,columns=[f'EV{i+1}' for i in range(n_components)])
    sns.scatterplot(data=projecteddata.iloc[:,0:2], x='EV1', y='EV2', hue=labels)
    
    # Plot the results
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(121)
    ax.plot(np.arange(n_components), evarianceratio)
    ax.scatter(np.arange(n_components), evarianceratio)
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("explained variance ratio")

    ax = fig.add_subplot(122)
    ax.plot(np.arange(n_components), evarianceratio.cumsum())
    ax.scatter(np.arange(n_components), evarianceratio.cumsum())
    ax.set_xlabel("eigenvalue number")
    ax.set_ylabel("cumulative explained variance ratio")
    
    for component in components:
     print(" + ".join("%.3f x %s" % (value, name) for value, name in
     zip(component, columns)))
    plt.show()
    return pca

urllib.request.urlretrieve("https://raw.githubusercontent.com/nshaud/ml_for_astro/main/stars.csv", "stars.csv")
df_stars = pd.read_csv("stars.csv")

le = LabelEncoder()
# Assign unique integers from 0 to 6 to each star type
df_stars['Star type'] = le.fit_transform(df_stars['Star type'])
labels = le.inverse_transform(df_stars['Star type'])
class_names = le.classes_
print(class_names)

print('\nCheck quickly if any null value\n')
df_stars.info()
print('\nCheck quickly if any nan value\n')
print(df_stars.notna().count())

fig = plt.figure(figsize=(7, 7))
sns.scatterplot(data=df_stars, x='Temperature (K)', y='Luminosity(L/Lo)', hue=labels)
plt.xscale('log')
plt.yscale('log')
plt.xticks([5000, 10000, 50000])
plt.xlim(5e4, 1.5e3)
plt.show()


columns = df_stars.columns[:4]

perform_pca(df_stars[columns])


scaler = preprocessing.StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df_stars[columns]),columns = columns)
perform_pca(scaled_df)

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaled_df = pd.DataFrame(scaler.fit_transform(df_stars[columns]),columns = columns)
perform_pca(scaled_df)

