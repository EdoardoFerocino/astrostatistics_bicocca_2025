# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:44:07 2025

@author: edoardo ferocino
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.colormaps[base_cmap]
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


digits = datasets.load_digits()
y = digits.target
X = digits.data

fig, axes = plt.subplots(7,7, figsize=(10, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

np.random.seed(4321) #### This was the seed
mychoices = np.random.choice(digits.images.shape[0],49)
np.random.seed()

for i, ax in enumerate(axes.flat):
    ax.imshow((digits.images[mychoices[i]]), 
              cmap='binary')
    ax.text(0.05, 0.05, str(digits.target[mychoices[i]]),transform=ax.transAxes, color='green', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

np.random.seed(42)

embedding = Isomap(n_components=2)
X_transformed = embedding.fit_transform(X)

# Plot the result
plt.figure()
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, edgecolor='none', alpha=0.5, 
            cmap=discrete_cmap(10,'nipy_spectral'))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title("Isomap projection of the Digits dataset")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80,random_state=2)
# better to use the full data set without "dimensionality reduction".
# the accuracy reduces when using the "reducted" data set
# also you need a max_iter very high to converge
#X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, train_size=0.80)

#clf = LogisticRegression(solver='sag', max_iter=10000).fit(X_train,y_train)
# the above if using the transformed data set
clf = LogisticRegression(max_iter=2000,solver='sag').fit(X_train,y_train)
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

print('Accuracy on training set is: ', accuracy_score(y_train, y_pred_train))
print('Accuracy on test set is: ', accuracy_score(y_test, y_pred_test))

confusion_matrix(y_train, y_pred_train, labels=range(10))
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train)

confusion_matrix(y_test, y_pred_test, labels=range(10))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow((X_test[i].reshape(8, 8)), cmap='binary')
    ax.text(0.05, 0.05, str(y_pred_test[i]), transform=ax.transAxes, 
            color='green' if (y_test[i] == y_pred_test[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])