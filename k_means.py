#from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial import distance
import matplotlib.pyplot as plt
from math import log
import random

def dis(data, i, centers, k, ncol,  mean, sd):
    # choix de la distance
    #return distance_esp(data, i, centers, k, ncol,  mean, sd)
    return distance_KLD(data, i, centers, k, ncol)
    #return distance.euclidean(data.iloc[i].drop(columns=[3]), centers[k])

def distance_esp(data, i, centers, k, ncol,  mean, sd):
    res = 0
    eucli = distance.euclidean(data.iloc[i].drop(columns=[3]), centers[k])
    for j in range(0, ncol):
        res = res + norm.cdf(eucli, loc = mean[j], scale = sd[j])
    return res

def KLD(x, y, ncol):
    x_norme = x
    sum = 0
    for i in range(0,ncol):
        sum = sum + x_norme[i]*log(x_norme[i]/y[i])
    return sum


def distance_KLD(data, i, centers, k, ncol):
    return KLD(data.iloc[i], centers[k], ncol)/2 + KLD(centers[k], data.iloc[i], ncol)/2



def compare(centers, newcenters, k, ncol):
    fini = True
    i = 0
    j = 0
    while(fini and i<k and j<ncol):
        if not centers[i][j]==newcenters[i][j] :
            fini = False
        i = i+1
        j = j+1
    return fini
    
def k_means(data, k, mean, sd):

    nrow = data.shape[0]
    ncol = data.shape[1]
    cluster = []
    somme = []
    newcenters = []
    centers=[]
    for i in range(k):
        newcenters.append([0] * ncol)
        centers.append([0] * ncol)
    for i in range(nrow):
        cluster.append(0)

    '''
    kmeans = KMeans(k, 'k-means++', 10, 300, 0.0001, 'auto', 0, None, True, None, 'auto').fit(data)
    cluster = kmeans.labels_
    centers_init = kmeans.cluster_centers_
    
    for i in range(k):
        centers.append(centers_init[i])
    '''
    medoids = random.sample(range(0,nrow), k)
    for i in range(0,k):
        centers[i] = data.iloc[medoids[i]]
    print(medoids)
    #1ère itération
    ## affectation
    for i in range(0,nrow):
        d = dis(data, i, centers, 0, ncol,  mean, sd)
        cluster[i] = 0
        for j in range(1,k):
            temp = dis(data, i, centers, j, ncol,  mean, sd)
            if temp<d:
                cluster[i] = j
                d = temp
    
    ## calcul des nouveaux centres
    for j in range(0,k):
        newcenters.append([])
        for l in range(0,ncol):
            somme.append(0)
        n = 0
        for i in range(0,nrow):
            if cluster[i]==j:
                n = n+1
                for l in range(0,ncol):
                    somme[l] = somme[l] + data.iloc[i][l]
        for l in range(0,ncol):
            newcenters[j][l] = somme[l]/n           
    fini = compare(centers, newcenters, k, ncol)
    cpt = 0
    
    
    while not fini:
        cpt = cpt + 1
        # affectation  au cluster
        for i in range(0,k):
            for j in range(0,ncol):
                centers[i][j] = newcenters[i][j]
        
        for i in range(0,nrow):
            dist = dis(data, i, newcenters, 0, ncol,  mean, sd)
            
            for j in range(1,k):
                temp = dis(data, i, newcenters, j, ncol,  mean, sd)
                if temp<dist:
                    cluster[i] = j
                    dist = temp        
        
        # calcul des nouveaux centres
        for j in range(0,k):
            for l in range(0,ncol):
                somme[l] = 0
            n = 0
            for i in range(0,nrow):
                if cluster[i]==j:
                    n = n+1
                    for l in range(0,ncol):
                        somme[l] = somme[l] + data.iloc[i][l]
                    for l in range(0,ncol):
                        newcenters[j][l] = somme[l]/n
        fini = compare(centers, newcenters, k, ncol)
        
        
    print(cpt, fini)
    #conversion en DataFrame
    cluster_dataF = pd.DataFrame(cluster, columns=['cluster'])
    return cluster_dataF

df = pd.read_csv("C:/Users/Jongmanee Tony/Desktop/iris.txt", sep=",", names=["sepal length","sepal width", "petal length", "petal width", "classe"])
data = df.drop("classe", 1)
nrow = data.shape[0]
ncol = data.shape[1]
summary = data.describe()
mean = data.mean()
sd = data.std()

cluster = k_means(data, 3, mean, sd)
df['cluster']=cluster
print(df)
    


colormap = np.array(['r', 'g', 'b'])

plt.legend()
plt.title("iris.arff")

plt.scatter(df['sepal length'], df['sepal width'], label='', c=colormap[df['cluster']])
plt.savefig('clustering_avec_ukmeans')

plt.show()
            
            
        
