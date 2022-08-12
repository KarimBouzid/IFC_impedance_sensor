# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:06:42 2020

@author: KrimFiction
"""
# %% Definitions
import time
import math
import numpy
import pandas
import collections

from matplotlib import pyplot

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import cdist
from sklearn import svm

from IPython import display
from random import sample  

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut

import os
os.environ["OMP_NUM_THREADS"] = "1"

import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torchvision.transforms as T

def evalEM(X, y, k, init):
    EM = GaussianMixture(k,init_params=init).fit(X)
    yprob = EM.predict_proba(X)
    newy = numpy.zeros((len(y),2))
    
    #Creates array for ClassData
    for t,x in enumerate(y):
        if x==0:
            newy[t,0] = 1
        else:
            newy[t,1] = 1
    
    pGroupData = yprob

    pClassGroup = numpy.dot(newy.T,pGroupData)
    
    dividend = sum(numpy.dot(newy.T,pGroupData))[numpy.newaxis,:]
    pClassGroup = pClassGroup/dividend
    
    pClassData = numpy.dot(pClassGroup,pGroupData.T)
    
    dividend = sum(numpy.dot(pClassGroup,pGroupData.T))[numpy.newaxis,:]
    pClassData = pClassData/dividend

    pClassData = pClassData.T
    
    ypredict = y*100
    for t,x in enumerate(pClassData):
        if x[0]>x[1]:
            ypredict[t] = 0
        else:
            ypredict[t] = 1
            
    if k==17:
        b1 = ax3.scatter(X[:, 0], X[:, 1],c=ypredict,marker='v',label="ypredict")
        b2 = ax3.scatter(X[:, 0], X[:, 1],c=y,marker='^',label="y")
        ax3.set_title("Example scatter for 17 clusters")
        ax3.legend(handles=[b1,b2])
    Rand = adjusted_rand_score(y, ypredict)
    Info = adjusted_mutual_info_score(y, ypredict)
    V = v_measure_score(y, ypredict)
    
    score = (Rand,Info,V)
    return score

def evalKmeans(X, y, k):
    ypredict = KMeans(n_clusters=k).fit_predict(X)
    newy = numpy.array([100]*len(y))
    for i in numpy.unique(ypredict):
        nb1 = numpy.count_nonzero(y[ypredict == i]==1)
        nb2 = numpy.count_nonzero(y[ypredict == i]==0)
        #print(i,nb1,nb2,nb1+nb2)
        if (nb1> nb2):
            newy[ypredict==i] = 1
        else:
            newy[ypredict==i] = 0

    if k==17:
        b1 = ax1.scatter(X[:, 0], X[:, 1],c=newy,marker='v',label="ypredict")
        b2 = ax1.scatter(X[:, 0], X[:, 1],c=y,marker='^',label="y")
        ax1.set_title("Example scatter for 17 clusters")
        ax1.legend(handles=[b1,b2])
    Rand = adjusted_rand_score(y, newy)
    Info = adjusted_mutual_info_score(y, newy)
    V = v_measure_score(y, newy)
    
    score = (Rand,Info,V)
    return score

def plot_clustering(X_red, labels, title, savepath):
    x_min, x_max = numpy.min(X_red, axis=0), numpy.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    pyplot.figure(figsize=(9, 6), dpi=160)
    for i in range(X_red.shape[0]):
        pyplot.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                color=pyplot.cm.nipy_spectral(labels[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title(title, size=17)
    pyplot.axis('off')
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
    pyplot.show()

# %% Data Fetching and plotting
_times = []
pyplot.close('all')
_times.append(time.time())

df = pandas.read_csv('dataSet.csv',header=None)
data = df.to_numpy()
X = data[0:,0:-1]
y = data[0:,-1]
print(X.shape)
print(y.shape)
del df

df = pandas.read_csv('FinalFantasyDataset.csv',header=None)
data = df.to_numpy()
X = data[0:,0:-1]
y = data[0:,-1]
print(X.shape)
print(y.shape)
del df

df = pandas.read_csv('matrixTime.csv',header=None)
matrixTime = df.to_numpy()
del df

_times.append(time.time())
print('Fetching Data Time: ',_times[-1] - _times[-2])

# %% Subsampling
arc = X[y==1,:]
nonarc = X[y==0,:]
X = numpy.concatenate((arc, nonarc[0::50,:]), axis=0)
y1 = y[y==1]
y2 = y[y==0]
y = numpy.concatenate((y1,y2[0::50]))
print(X.shape)

# %% K-Means
_times.append(time.time())
figK = pyplot.figure(figsize=(15,7))
ax1 = figK.add_subplot(121)
ax2 = figK.add_subplot(122)

siz = X.shape
print(siz[0],siz[1])
if siz[1]<siz[0]:
    k = numpy.arange(2,siz[1],5)
else:
    k = numpy.arange(2,siz[0],5)

#k = numpy.arange(2,30,2)

Rand = []
Info = []
V = []
for nbCluster in k:
    nbCluster = int(nbCluster)
    score = evalKmeans(X, y, nbCluster)
    Rand.append(score[0])
    Info.append(score[1])
    V.append(score[2])
    print('k:',nbCluster, 'score:',score)

pyplot.ylim([0, 1])
ax2.set_title("K-means Clustering")
ax2.set_xlabel("k-groups")
ax2.set_ylabel("Score")

a1, = ax2.plot(k,Rand,label='Rand')
a2, = ax2.plot(k,Info,label='Info')
a3, = ax2.plot(k,V,label='V')

ax2.legend(handles=[a1,a2,a3])
_times.append(time.time())
print('K-Means Clusters Time: ',_times[-1] - _times[-2])

#%% EM
_times.append(time.time())
figEM = pyplot.figure(figsize=(15,7))
ax3 = figEM.add_subplot(121)
ax4 = figEM.add_subplot(122)

Rand = []
Info = []
V = []
for nbCluster in k:
    nbCluster = int(nbCluster)
    score = evalEM(X, y, nbCluster,'kmeans')
    Rand.append(score[0])
    Info.append(score[1])
    V.append(score[2])
    print('k:',nbCluster, 'score:',score)

pyplot.ylim([0, 1])
ax4.set_title("EM Clustering")
ax4.set_xlabel("k-groups")
ax4.set_ylabel("Score")

a1, = ax4.plot(k,Rand,label='Rand')
a2, = ax4.plot(k,Info,label='Info')
a3, = ax4.plot(k,V,label='V')

ax2.legend(handles=[a1,a2,a3])
_times.append(time.time())
print('EM Clusters Time: ',_times[-1] - _times[-2])

# %% Réduction de dimensionalité
_times.append(time.time())
pca = PCA(n_components=2)
X_red = pca.fit_transform(X)
plot_clustering(X_red, y, 'Réduction de dimensionalité PCA', 'a')

mds = MDS(n_components=2, n_init=1)
X_red = mds.fit_transform(X)
plot_clustering(X_red, y, 'Réduction de dimensionalité MDS', 'a')

#tsne = TSNE(n_components=2)
#X_red = tsne.fit_transform(X)
#plot_clustering(X_red, y, 'Réduction de dimensionalité tSNE', 'a')

_times.append(time.time())
print('Dimension Reduction PCA MDS and tSNE Time: ',_times[-1] - _times[-2])
pyplot.show()

# %% K-Means réduit
_times.append(time.time())
figK = pyplot.figure(figsize=(15,7))
ax1 = figK.add_subplot(121)
ax2 = figK.add_subplot(122)

siz = X.shape
print(siz[0],siz[1])
if siz[1]<siz[0]:
    k = numpy.arange(2,siz[1],5)
else:
    k = numpy.arange(2,siz[0],5)

#k = numpy.arange(2,30,2)

Rand = []
Info = []
V = []
for nbCluster in k:
    nbCluster = int(nbCluster)
    score = evalKmeans(X_red, y, nbCluster)
    Rand.append(score[0])
    Info.append(score[1])
    V.append(score[2])
    print('k:',nbCluster, 'score:',score)

pyplot.ylim([0, 1])
ax2.set_title("K-means Clustering réduit 2dim")
ax2.set_xlabel("k-groups")
ax2.set_ylabel("Score")

a1, = ax2.plot(k,Rand,label='Rand')
a2, = ax2.plot(k,Info,label='Info')
a3, = ax2.plot(k,V,label='V')

ax2.legend(handles=[a1,a2,a3])
_times.append(time.time())
print('K-Means Clusters Time: ',_times[-1] - _times[-2])

#%% EM Réduit
_times.append(time.time())
figEM = pyplot.figure(figsize=(15,7))
ax3 = figEM.add_subplot(121)
ax4 = figEM.add_subplot(122)

siz = X.shape
print(siz[0],siz[1])
if siz[1]<siz[0]:
    k = numpy.arange(2,siz[1],5)
else:
    k = numpy.arange(2,siz[0],5)

Rand = []
Info = []
V = []
for nbCluster in k:
    nbCluster = int(nbCluster)
    score = evalEM(X_red, y, nbCluster,'kmeans')
    Rand.append(score[0])
    Info.append(score[1])
    V.append(score[2])
    print('k:',nbCluster, 'score:',score)

pyplot.ylim([0, 1])
ax4.set_title("EM Clustering Réduit 2dim")
ax4.set_xlabel("k-groups")
ax4.set_ylabel("Score")

a1, = ax4.plot(k,Rand,label='Rand')
a2, = ax4.plot(k,Info,label='Info')
a3, = ax4.plot(k,V,label='V')

ax2.legend(handles=[a1,a2,a3])
_times.append(time.time())
print('EM Clusters Time: ',_times[-1] - _times[-2])

# %% Simple classifiers 
_times.append(time.time())

# Dictionnaire pour enregistrer les erreurs selon les classifieurs
erreurs = collections.OrderedDict()

classifieurs = [QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(), GaussianNB(), NearestCentroid()]

h = .01
x_min, x_max = X_red[:,0].min() - 0.2, X_red[:,0].max() + 0.2
y_min, y_max = X_red[:,1].min() - 0.2, X_red[:,1].max() + 0.2
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),numpy.arange(y_min, y_max, h))

# On crée une figure à plusieurs sous-graphes
fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                               tight_layout=True)
t1 = time.time()
for clf,subfig in zip(classifieurs, subfigs.reshape(-1)):
    clf_name = clf.__class__.__name__
    
    X_train, X_test, y_train, y_test = train_test_split(X_red, y, train_size=0.5, test_size=0.5)
    clf.fit(X_train,y_train)

    err = 1-clf.score(X_test,y_test)
    erreurs[clf_name] = err
    
    yyy = [int(x) for x in y]
    colors = numpy.array([x for x in "bgrcmyk"])
    Y = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Y = Y.reshape(xx.shape)
    subfig.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
    subfig.scatter(X_red[:,0], X_red[:,1],cmap=pyplot.cm.Paired,color=colors[yyy].tolist())

    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())
    subfig.set_title(clf_name)


_times.append(time.time())
print('Simple Classifiers Time: ',_times[-1] - _times[-2])
pyplot.show()
df = pandas.DataFrame(erreurs, index=['Erreurs'])
display.display(df)


# %% Quadratic SVM
_times.append(time.time())

# Dictionnaire pour enregistrer les erreurs selon les classifieurs
erreurs = collections.OrderedDict()

ccc = [1, 100, 10000, 1000000]
classifieurs = [svm.SVC(C=1, max_iter=10000), svm.SVC(C=100, max_iter=10000), svm.SVC(C=10000, max_iter=100000), svm.SVC(C=1000000, max_iter=100000)]


h = .01
x_min, x_max = X_red[:,0].min() - 0.2, X_red[:,0].max() + 0.2
y_min, y_max = X_red[:,1].min() - 0.2, X_red[:,1].max() + 0.2
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),numpy.arange(y_min, y_max, h))

# On crée une figure à plusieurs sous-graphes
fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all',
                               tight_layout=True)
t1 = time.time()
for cc, clf,subfig in zip(ccc, classifieurs, subfigs.reshape(-1)):
    clf_name = clf.__class__.__name__
    
    X_train, X_test, y_train, y_test = train_test_split(X_red, y, train_size=0.5, test_size=0.5)
    clf.fit(X_train,y_train)

    err = 1-clf.score(X_test,y_test)
    erreurs[str(cc)] = err
    
    yyy = [int(x) for x in y]
    colors = numpy.array([x for x in "bgrcmyk"])
    Y = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Y = Y.reshape(xx.shape)
    subfig.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
    subfig.scatter(X_red[:,0], X_red[:,1],cmap=pyplot.cm.Paired,color=colors[yyy].tolist(), label=str(cc))

    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())
    subfig.legend()
    subfig.set_title(clf_name)


_times.append(time.time())
print('Quadratic SVM Time: ',_times[-1] - _times[-2])
pyplot.show()
df = pandas.DataFrame(erreurs, index=['Erreurs'])
display.display(df)

# %% SVM Graphique
_times.append(time.time())
C = numpy.logspace(0, 7, num=8)

scoresSVM = []
for k in C:
    err = numpy.array([])
    clf = svm.SVC(C=k, max_iter=1000000)
    rkf = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in rkf.split(X_red):
        X_train, X_test = X_red[train_index], X_red[test_index]
        R_train, R_test = y[train_index], y[test_index]
        clf.fit(X_train,R_train)
        err = numpy.append(err,100*(1-clf.score(X_test,R_test)))
    scoresSVM = numpy.append(scoresSVM,numpy.mean(err))    

# Initialisation de la figure
fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.set_title('Impact de la variable C sur l''erreur de classification')  
ax.plot(C, scoresSVM) 
ax.set_xticks((C))
ax.grid(axis='x')
ax.set_xscale('log')
ax.set_xlabel("Values of C")
ax.set_ylabel("Erreur (%)")

_times.append(time.time())
print('SVM Time: ',_times[-1] - _times[-2])

# %% K-PPV
_times.append(time.time())
n = [1,3,5,7,11,13,15,25,35,45]

scoresUniformWeights = []
scoresDistanceWeights = []
for k in n:
    err1 = numpy.array([])
    err2 = numpy.array([])
    clf1 = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
    clf2 = KNeighborsClassifier(n_neighbors = k, weights = 'distance')
    rkf = KFold(n_splits=7)
    for train_index, test_index in rkf.split(X_red):
        X_train, X_test = X[train_index], X[test_index]
        R_train, R_test = y[train_index], y[test_index]
        clf1.fit(X_train,R_train)
        clf2.fit(X_train,R_train)
        err1 = numpy.append(err1,100*(clf1.score(X_test,R_test)))
        err2 = numpy.append(err2,100*(clf2.score(X_test,R_test)))
    scoresUniformWeights = numpy.append(scoresUniformWeights,numpy.mean(err1))
    scoresDistanceWeights = numpy.append(scoresDistanceWeights,numpy.mean(err2))
    

# Initialisation de la figure
fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.set_title('Impact de la distribution et du nombre de voisins sur l''erreur de classification')  
ax.plot(n, scoresDistanceWeights, 'r--', label="Distance weights") 
ax.plot(n, scoresUniformWeights, 'b--', label="Uniform weights")  
ax.set_xticks((n))
ax.grid(axis='x')
ax.set_xlabel("Values of K")
ax.set_ylabel("Accuracy (%)")
ax.legend(['Uniform weights', 'Distance weights'])

_times.append(time.time())
print('K_PPV Time: ',_times[-1] - _times[-2])

# %% Discriminant à Noyau et descente du gradient
_times.append(time.time())
from scipy.optimize import fmin_l_bfgs_b
from sklearn.model_selection import KFold, train_test_split


# Création du tableau pour accumuler les résultats
results = {'Classifier':["DiscriminantANoyau"],
           'Range_lambda':[],
           'Range_sigma':[],
           'Best_lambda':[],
           'Best_sigma':[],
           'Error_train':[],
           'Error_test':[],
          }


class DiscriminantANoyau:
    def __init__(self, lambda_, sigma,verbose = False):
        self.lambda_ = lambda_
        self.sigma = sigma
        self.verbose = verbose
    
    def fit(self, X, y):
        normX = cdist(X,X)
        K = numpy.exp(-(normX)/self.sigma**2)
        self.K = K
        def evaluateFunc(params):
            alpha = params[0:-1] 
            w0 = params[-1]
            err = 0
            der_part_w0 = 0
            h = numpy.sum(alpha*y*K,axis=1) + w0
            der_part_alpha = -y*(numpy.sum(y*K,axis=1))+self.lambda_
            for t in range(len(y)):
                if y[t]*h[t] < 1:
                    err += 1-y[t]*h[t]
                    der_part_w0 += -y[t]
                err += self.lambda_*alpha[t]
            
            grad = der_part_alpha.tolist() + [der_part_w0]
            return err, grad
        
        M = X.shape[0]
        params = numpy.random.uniform(0,1,M+1)
        minVal = [0]*(M)+[None]
        maxVal = [None]*(M+1)
        bounds = [(x,y) for x,y in zip(minVal,maxVal)]
        
        _times.append(time.time())
        params, minval, infos = fmin_l_bfgs_b(evaluateFunc, params, bounds=bounds) #, maxiter=2,maxfun=2, factr=1e-16
        _times.append(time.time())
        print('fmin_l_bfgs_b Time: ',_times[-1] - _times[-2])
        
        
        if self.verbose:
            print("Entraînement terminé après {it} itérations et {calls} appels à evaluateFunc".format(it=infos['nit'], calls=infos['funcalls']))
            print("\tErreur minimale : {:.5f}".format(minval))
            print("\tL'algorithme a convergé" if infos['warnflag'] == 0 else "\tL'algorithme n'a PAS convergé")
            print("\tGradients des paramètres à la convergence (ou à l'épuisement des ressources) :")
            print(infos['grad'])
        
        self.alphas = params[0:-1]
        self.w0 = params[-1]
        
        self.X, self.y = X, y
        return self
        
    def predict(self, X):
        normX = cdist(X,self.X)
        K = numpy.exp(-(normX)/self.sigma**2)
        h = numpy.sum(self.alphas*self.y*K,axis=1) + self.w0
        return numpy.sign(h)

    def score(self, X, y):
        pred = self.predict(X)
        prob = y==pred
        return numpy.sum(prob)/len(y)


i = numpy.where(y<0.5)
yyyy = numpy.ones(y.shape)
yyyy[i[0]]=-1

X_train, X_test, R_train, R_test = train_test_split(X_red, yyyy, train_size=0.5, test_size=0.5)
    
range_lambda = numpy.logspace(-9,-4,num=5)
results['Range_lambda'].append(range_lambda)

range_sigma = numpy.linspace(0.3,0.9,num=6)
results['Range_sigma'].append(range_sigma)


lamb_opti, sig_opti = [],[]
rkf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in rkf.split(X):
    X_t, X_val = X_red[train_index], X_red[test_index]
    R_t, R_val = yyyy[train_index], yyyy[test_index]
    err_min = 1
    for lamb in range_lambda:
        for sig in range_sigma:
            clf = DiscriminantANoyau(lambda_=lamb, sigma=sig,verbose=False)
            clf.fit(X_t, R_t)
            err = 1-clf.score(X_val, R_val)
            if err<err_min:
                lamb_opt, sig_opt = lamb, sig
                err_min = err
    lamb_opti = numpy.append(lamb_opti,lamb_opt)
    sig_opti = numpy.append(sig_opti,sig_opt)
lamb_opt = numpy.mean(lamb_opti)
sig_opt = numpy.mean(sig_opti)

best_lambda = lamb_opt
results['Best_lambda'].append(best_lambda)

best_sigma = sig_opt
results['Best_sigma'].append(best_sigma)

clf = DiscriminantANoyau(lambda_=best_lambda, sigma=best_sigma,verbose=False)

clf.fit(X_train,R_train)

err_train = 1-clf.score(X_train,R_train)
results['Error_train'].append(err_train)

err_test = 1-clf.score(X_test,R_test)
results['Error_test'].append(err_test)
# %% 
h = .1
x_min, x_max = X_red[:,0].min() - 1, X_red[:,0].max() + 1
y_min, y_max = X_red[:,1].min() - 1, X_red[:,1].max() + 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),numpy.arange(y_min, y_max, h))

fig = pyplot.figure()
ax = fig.add_subplot(111)

yyy = [int(x) for x in y]
colors = numpy.array([x for x in "bgrcmyk"])
Y = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)

pyplot.xlim(xx.min(), xx.max())
pyplot.ylim(yy.min(), yy.max())

ax.set_title("Analyse de la performance d'une discrimination avec noyau et descente du gradient")   # TODO Q7: À modifier
ax.set_xlabel("Dimension X")
ax.set_ylabel("Dimension Y")
ax.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
#ax.scatter(X_red[:, 0], X_red[:, 1], cmap=pyplot.cm.Paired,color=colors[yyy].tolist())

# On affiche la figure
_times.append(time.time())
print('Discriminant linéaire Time: ',_times[-1] - _times[-2])
pyplot.show()

# Affichage des résultats
df = pandas.DataFrame(results)
display.display(df)