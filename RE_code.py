#!/usr/bin/env python
# coding: utf-8

# In[43]:


from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.decomposition import PCA
from numpy import genfromtxt
import numpy as np

from scipy.stats import entropy


def get_raw_scores(data):
    K_list=[4,5,6,7]
    scoreMatrix=np.zeros((len(K_list),data.shape[0]))
    for i in range(len(K_list)):
        clf = LOF(n_neighbors=K_list[i])
        clf.fit(data)
        scoreMatrix[i,:]=clf.negative_outlier_factor_
    return scoreMatrix



def regional_ensemble(data,scoreMatrix,k,v,combinationMethod='eq5'):
    # dataset: data; outlier scores from mutiple base detectors: scoreMatrix; k-NN: k, top v biggest singular values: v; The score combination methhd: combinationMethod
    # return the revised score
    msg="wrong setting for combinationMethod, please choose one of ['eq5','eq6','eq12','eq13','eq14','eq15','eq16','eq17','eq18']"
    dist, ind =NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data).kneighbors(data)
    revisedScore=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        baseScoreMatrix=scoreMatrix[:,ind[i]]
        if combinationMethod=='eq5':
            eq5=np.mean(baseScoreMatrix)
            revisedScore[i]= eq5
        elif combinationMethod=='eq6':
            eq6=np.max(baseScoreMatrix)
            revisedScore[i]= eq6

        elif combinationMethod in ['eq12','eq13','eq14','eq15','eq16','eq17','eq18']:
            pca = PCA(n_components=v)
            pca.fit(baseScoreMatrix)
            Y_projected = pca.transform(baseScoreMatrix)
            baseScoreMatrix_reconstructed = pca.inverse_transform(Y)
            g= pca.singular_values_

            if combinationMethod=='eq12':
                eq12=np.sum(baseScoreMatrix_reconstructed)
                revisedScore[i]= eq12
            elif combinationMethod=='eq13':
                eq13=np.sum((baseScoreMatrix - baseScoreMatrix_reconstructed) ** 2, axis=1).mean()
                revisedScore[i]= eq13
            elif combinationMethod=='eq14':
                eq14=np.sum(g**2)
                revisedScore[i]= eq14
            elif combinationMethod in ['eq15','eq16','eq17','eq18']:
                value,counts = np.unique(baseScoreMatrix, return_counts=True)
                counts=counts/np.sum(counts)
                if combinationMethod=='eq15':
                    eq15=entropy(counts, base=None)
                    revisedScore[i]= eq15
                elif combinationMethod=='eq16':
                    eq16=np.sum(value*counts)
                    revisedScore[i]= eq16
                elif combinationMethod=='eq17':
                    eq17=eq5+eq15
                    revisedScore[i]= eq17
                elif combinationMethod=='eq18':
                    eq18=eq5*np.log2(1+eq15)
                    revisedScore[i]= eq18
                else:
                    print(msg)
            else:
                print(msg)
        else:
            print(msg)
    return revisedScore

data=np.random.randn(20, 4)
scoreMatrix=get_raw_scores(data)
for combinationMethod in ['eq5','eq6','eq12','eq13','eq14','eq15','eq16','eq17','eq18']:
    revisedScore=regional_ensemble(data,scoreMatrix,3,2,combinationMethod='eq5')
    print("when using ",combinationMethod,", the revised scores are:" )
    print(revisedScore)


# In[ ]:




