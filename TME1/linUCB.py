import numpy as np
import matplotlib.pyplot as plt
import math
import Random 

fichier="CTR.txt"
  

class linUCB:
    def __init__(self,nb_actions,alpha):
        self.nb_actions=nb_actions
        self.alpha=alpha
        self.A=[np.identity(self.nb_actions)]
        self.b=np.zeros(self.nb_actions*self.nb_actions).reshape(self.nb_actions,self.nb_actions)
        self.time=0

    def get_time(self):
        return self.time

    def get_choice(self,ctx): 
        val=np.zeros(self.nb_actions)
        for i in range(self.nb_actions):
            teta=np.dot(np.linalg.inv(self.A[i]),self.b[i])
            print(teta.shape,ctx.shape)
            p=np.dot(teta.T,ctx)+self.alpha*math.sqrt(np.dot(np.dot(ctx.T,np.linalg.inv(self.A[i])),ctx))    
            val[i]=p    
        return np.argmax(np.array(val))
    
    def update(self, action, reward,ctx):
        self.A[action]+=np.dot(ctx,ctx.T)
        self.b[action]+=reward*ctx
        self.time+=1
                                                               
    
def evaluate(fichier, modele,alpha):#si on met random en classe, faire if pour ne pas initialiser les autres 10 fois
    with open(fichier, "r") as f:
        file=f.readlines()
        
    data=[]
    contexte=[]
    for row in file :
        tmp1=row.split(':')[-1]
        tmp1=np.array(tmp1.split(';'))
        tmp1=[ float(t) for t in tmp1 ]
        data.append(np.array(tmp1))
        tmp2=row.split(':')[1]
        tmp2=np.array(tmp2.split(';'))
        tmp2=[ float(t) for t in tmp2 ]
        contexte.append(np.array(tmp2))
        
    U=modele(len(data[0]),alpha)
    pi_ucb=[]
        
    while U.get_time()<len(data):
        act = U.get_choice(contexte[U.get_time()])
        U.update(act,data[U.get_time()][act],contexte[U.get_time()])
        pi_ucb.append(U.get_mu()[act])
        
        
    return pi_ucb,data

alpha=1
#
#pi_ucb,data=evaluate(fichier, UCB) 
#nb_ite=[i for i in range(len(data))]
#pi_random, pi_best, pi_opt=Random.EvaluationBaselines(data)
#
#fig = plt.figure(1, figsize=(15, 9))
#plt.plot(nb_ite,pi_random)
#plt.plot(nb_ite,pi_best)
#plt.plot(nb_ite,pi_opt) 
#plt.plot(nb_ite,pi_ucb)   
#plt.legend(['random','best_static','optimal','ucb'])   
#plt.xlabel("Nombre d'articles")
#plt.ylabel("Gain cumulé")
#plt.title("Evaluation des politiques")
#plt.show()
