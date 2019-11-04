import numpy as np
import matplotlib.pyplot as plt
import math
import Random 

fichier="CTR.txt"
  

class UCB:
    def __init__(self,nb_actions):
        self.nb_actions=nb_actions
        self.mu=np.zeros(self.nb_actions)
        self.s=np.zeros(self.nb_actions)
        self.time=0

    def get_time(self):
        return self.time
    
    def get_mu(self):
        return self.mu

    def get_choice(self): #contexte à rajouter pour linUCB
        Bt=np.zeros(self.nb_actions)
        for i in range(self.nb_actions):
            Bt[i]=self.mu[i]+math.sqrt(2*math.log(self.time)/self.s[i])
        return np.argmax(np.array(Bt))
    
    def update(self, action, reward): #contexte à rajouter pour linUCB
        self.mu[action]+=reward
        self.s[action]+=1
        self.time+=1                                                                     
    
def evaluate(fichier, modele):#si on met random en classe, faire if pour ne pas initialiser les autres 10 fois
    with open(fichier, "r") as f:
        file=f.readlines()
        
    data=[]
    for row in file :
        tmp=row.split(':')[-1]
        tmp=np.array(tmp.split(';'))
        tmp=[ float(t) for t in tmp ]
        data.append(np.array(tmp))
        
    U=modele(len(data[0]))
    pi_ucb=[]
    for i in range(len(data[0])):
        U.update(i, data[U.get_time()][i])
        pi_ucb.append(U.get_mu()[i])
        
    while U.get_time()<len(data):
        act = U.get_choice()
        U.update(act,data[U.get_time()][act])
        pi_ucb.append(U.get_mu()[act])
        
        
    return pi_ucb,data

pi_ucb,data=evaluate(fichier, UCB)
nb_ite=[i for i in range(len(data))]
pi_random, pi_best, pi_opt=Random.EvaluationBaselines(data)

fig = plt.figure(1, figsize=(15, 9))
plt.plot(nb_ite,pi_random)
plt.plot(nb_ite,pi_best)
plt.plot(nb_ite,pi_opt) 
plt.plot(nb_ite,pi_ucb)   
plt.legend(['random','best_static','optimal','ucb'])   
plt.xlabel("Nombre d'articles")
plt.ylabel("Gain cumulé")
plt.title("Evaluation des politiques")
plt.show()
#SI ON A LE TEMPS EPSILON GREEDY
#on doit jouer tous les bras pour initialiser : 2 tab de 10 esperance de reward mu et un tableau de s
#        initialise chacun s sera à 1 (pour pas diviser par 0)
#lin_ucb : bat tout sauf optimal
#mettre un zoom sur les 100 premieres iterations