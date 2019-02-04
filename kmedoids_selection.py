import numpy as np
import pandas as pd
import random
#from scipy.stats import entropy
from math import log
import matplotlib.pyplot as plt

def KLD(x, y, ncol):
    x_norme = x
    sum = 0
    for i in range(0,ncol):
        if x_norme[i]>0 and y[i]!=0:
            sum = sum + x_norme[i]*log(x_norme[i]/y[i])
    return sum

def distance_KLD(x, y, ncol):
    return KLD(x, y, ncol)/2 + KLD(y, x, ncol)/2

def compare(medoids, newmedoids, k, ncol):
    fini = True
    i = 0
    j = 0
    #print(medoids)
    #print(newmedoids)
    while(fini and i<k and j<ncol):
        if(medoids[i][j]!=newmedoids[i][j]):
            fini = False
        i = i+1
        j = j+1
    return fini

def medoids_selection(data, k, ncol):
    d = pd.DataFrame(data)
    initial = []
    for i in range(0,k):
        initial.append(0)
    nbmax_elem_A = 0.75*d.shape[0]/k

    for l in range(0,k):
        #initialisation de la recherche
        A = []
        Aindice = []
        n = d.shape[0] 
        dist_min = distance_KLD(d.loc[d.index[0]], d.loc[d.index[1]], ncol)
        indice1 = d.index[0]
        indice2 = d.index[1]
        #recherche de la paire ayant le plus de simularité
        for i in range(1,n):
            if i in d.index:
                for j in range(0,n):
                    if j in d.index and not i == j:
                        dist_new = distance_KLD(d.loc[i], d.loc[j], ncol)
                        if dist_min>dist_new:
                                indice1 = i
                                indice2 = j
                                dist_min = dist_new
        # ajout dans A et supression des données dans d
        Aindice.append(indice1)
        Aindice.append(indice2)
        A.append(d.loc[indice1])
        A.append(d.loc[indice2])
        print(indice1, indice2)
        d = d.drop([indice1, indice2])
        while len(A)<nbmax_elem_A:
            #Calcul de Am : moyenne de tous les éléments de A
            n_Am = len(A)
            Am = []
            for j in range(0,ncol):
                Am.append(0)
                for i in range(0,n_Am):
                    Am[j] = Am[j] + A[i][j]
                Am[j] = Am[j]/n_Am
                    
            #initialisation de la recherche
            n = d.shape[0] 
            indice_min = d.index[0]
            
            
            dist_min = distance_KLD(d.loc[indice_min], Am, ncol)
        
            #recherche de la donnée ayant le plus de simularité avec Am
            for i in range(d.index[0]+1,n):
                if i in d.index:
                    dist_new = distance_KLD(d.loc[i], Am, ncol)
                    if dist_min>dist_new:
                        indice_min = i
                        dist_min = dist_new
            A.append(d.loc[indice_min])
            Aindice.append(indice_min)
            
                
            d = d.drop([indice_min])
        print(Aindice)
        # choix d'un point au hasard dans les Am
        initial[l] = Aindice[random.randint(0,len(A)-1)]
    return initial
                        
def medoids(data, k):
    nrow = data.shape[0]
    ncol = data.shape[1]
    cluster = []
    for i in range(nrow):
        cluster.append(0)
    medoids_val = []
    newmedoids_val = []
    for i in range(k):
        medoids_val.append([0] * ncol)
        newmedoids_val.append([0] * ncol)
    
    
    # phase de construction
    ## selection des medoids
    medoids = medoids_selection(data, k, ncol)
    for i in range(0,k):
        medoids_val[i] = data.iloc[medoids[i]]
    print(medoids)
    ## affectation aux clusters
    for i in range(0,nrow):
        d = distance_KLD(data.iloc[i], medoids_val[0], ncol)
        cluster[i] = 0 
        for j in range(1,k):
            temp = distance_KLD(data.iloc[i], medoids_val[j], ncol)
            if temp<d:
                cluster[i] = j
                d = temp  
    
    # phase de swapping
    fini = False
    cpt = 0
    while not fini:
        for i in range(0,k):
            for j in range(0,ncol):
                newmedoids_val[i][j] = medoids_val[i][j]
                
        ## swapping
        for j in range(0,k):
            TC = 0
            for i in range(1,nrow):
                if cluster[i] == j:
                    TC = TC + distance_KLD(data.iloc[i], medoids_val[j], ncol)
                    TCS = 0
                    for l in range(1,nrow):
                        if cluster[l] == j:
                            TCS = TCS + distance_KLD(data.iloc[i], data.iloc[l], ncol)
                    if TCS<TC:
                        newmedoids_val[j] = data.iloc[i]
        ## affectation aux clusters
        for i in range(0, nrow):
            d = distance_KLD(data.iloc[i], newmedoids_val[0], ncol)
            for j in range(1,k):
                temp = distance_KLD(data.iloc[i], newmedoids_val[j], ncol)
                if temp<d:
                    cluster[i] = j
                    d = temp
        fini = compare(medoids_val, newmedoids_val, k, ncol)
        cpt= cpt +1
        print(cpt)
        print(cluster)
    cluster_dataF = pd.DataFrame(cluster, columns=['cluster'])
    return cluster_dataF
                      
df = pd.read_csv("C:/Users/Jongmanee Tony/Desktop/iris.txt", sep=",", names=["sepal length","sepal width", "petal length", "petal width", "classe"])
data = df.drop("classe", 1)
nrow = data.shape[0]
cluster = medoids(data, 3)

df['cluster']=cluster
print(df)

'''
# Robustesse/Simulation
nb = 10
cluster = []
for i in range(nb):
        cluster.append([0] * nrow)
for i in range(0,nb):
    cluster[i] = medoids(data,3)

for i in range(nb):
    print(cluster[i])
'''
# plot
colormap = np.array(['r', 'g', 'b'])

plt.legend()
plt.title("iris.arff")

plt.scatter(df['sepal length'], df['sepal width'], label='', c=colormap[df['cluster']])
plt.savefig('clustering_kmedoids_KLD ')

plt.show()

