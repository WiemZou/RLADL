import numpy as np
from random import randint
import matplotlib.pyplot as plt


with open("CTR.txt", "r") as f:
    file=f.readlines()
    

data=[]

for row in file :
    tmp=row.split(':')[-1]
    tmp=np.array(tmp.split(';'))
    tmp=[ float(t) for t in tmp ]
    data.append(np.array(tmp))
    
data=np.array(data)
def PolitiqueRandom(nb_bras):
    return randint(0,nb_bras-1)

def PolitiqueStaticBest(data): #cumule les clics et renvoie le max
    t=np.sum(data, axis=0)
    return np.argmax(t) 
    
def PolitiqueOptimale(data,ite):
    return np.argmax(data[ite]) 

def EvaluationBaselines(data):
    pi_random=[data[0][PolitiqueRandom(len(data[0]))]]
    pi_best=[data[0][PolitiqueStaticBest(data)]]
    pi_opt=[data[0][PolitiqueOptimale(data,0)]]
    nb_ite=[i for i in range(len(data))]
    for i in range(1,len(data)):
        PolitiqueRandom(10)
        pi_random.append(pi_random[i-1]+data[i][PolitiqueRandom(len(data[i]))])
        pi_best.append(pi_best[i-1]+data[i][PolitiqueStaticBest(data)])
        pi_opt.append(pi_opt[i-1]+data[i][PolitiqueOptimale(data,i)])
    plt.plot(nb_ite,pi_random)
    plt.plot(nb_ite,pi_best)
    plt.plot(nb_ite,pi_opt)    
    plt.legend(['random','best_static','optimal'])   
    plt.xlabel("Nombre d'articles")
    plt.ylabel("Gain cumulé")
    plt.title("Evaluation des politiques")
    plt.show()
    return pi_random, pi_best, pi_opt
    

#Optimal prend le meilleur en fonction de l'article donc on triche normal que ce soit meilleur
    
    
    #on veut plot gain cumulé en fonction nb ite
#connaitre nombre de bras

#max cumulé tout le temps

#on s'attends a ce que UCB cv vers static best
#LinUCB : utilise le contexte doit etre capable de depasser le static best