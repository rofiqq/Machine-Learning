import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs

class data():
    def seagull(ndata) :
        X, y = make_moons(n_samples = ndata, noise=0.1, random_state=0)
        idx1 = np.argwhere(y == 0).ravel()
        idx2 = np.argwhere(y == 1).ravel()
        X1 = X[idx1]
        X1[:,0] += 1.8*np.max(X1[:,0])
        X1[:,1] += 0.2*np.max(X1[:,1])
        y1 = np.linspace(0,0, len(X1))
        
        X2 = X[idx1]
        X2[:,0] -= 0.2*np.max(X2[:,0])
        X2[:,1] += 0.2*np.max(X2[:,1])
        y2 = np.linspace(1,1, len(X2))
        
        X3 = X[idx2]
        y3 = np.linspace(2,2, len(X3))
        
        X = np.vstack((X1, X2, X3))
        y = np.hstack((y1, y2, y3))
        return X, y
    
    def circle(ndata):
        X, y = make_circles(n_samples = ndata, noise=0.1, factor=0.3, random_state=1)
        idx = np.argwhere(y == 0).ravel()
        X1 = 1.8*X[idx]
        y1 = np.linspace(2,2, len(X1))
        
        X = np.vstack((X, X1))
        y = np.hstack((y, y1))
        return X, y
    
    def blob(ndata):
        return make_blobs(n_samples=ndata, n_features=2, shuffle=True)

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter = 10000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]

colors = ['r', 'g', 'b']

# Create the colormap
cm = LinearSegmentedColormap.from_list(
    'my_color', colors, N=6)

cm_bright = ListedColormap(colors)

ndata = 200
datasets = [data.seagull(ndata = ndata),
              data.circle(ndata = ndata),
              data.blob(ndata = ndata)]

i = 1
figure = plt.figure(figsize=(3*(len(classifiers) + 1), 3*len(datasets)))
for X, Y in datasets :
    X = StandardScaler().fit_transform(X)
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    shape = xx.shape
    
    grid = np.vstack((xx.ravel(), yy.ravel())).T
    
    X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=.4, random_state=42)
    
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
	
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
			   
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if i == 1 :
        ax.set_title('Input data')
    i += 1
        
    for name, clf in zip(names, classifiers) : 
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        pred = clf.predict(grid)
        Z = pred.reshape(shape)
            
        # Put the result into a color plot
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
        
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.5,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        
        if i <= 11 :
            ax.set_title(name)
        
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right', fontweight='bold') 
        i += 1
    
plt.tight_layout()
plt.show()
figure.savefig('classifier_comparison.png',dpi = 360)