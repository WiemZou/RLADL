import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np


import random

#gamma=1 toutes politiques ont le meme coup meme interet quel que soit le chemin choisi alors que pour gamma inferieur à 1 on force a choisir le plus court

#env.render()

def EvalPolicy(states,P,V,gamma): #a faire
    states_P=list(P.keys())
    pi=dict()
        
    for s in states_P:
        act = np.zeros(len(P[s])) # erreur avec etats terminaux a voir  
        for a in P[s]:
            for dest in range(len(P[s][a])):
                p = P[s][a][dest][0]
                r = P[s][a][dest][2]
                s_dest = P[s][a][dest][1]
                if s_dest in states_P:
                    d=states_P.index(s_dest)
                    act[a] += p*(r + gamma*V[d])
        
        pi[states[s]] = np.argmax(act)

    return pi        
        

def compare_dict(dict1,dict2): #fonction a tester 
    for s in dict1:
        if dict1[s] != dict2[s]:
            return False
    return True       
    


def PolicyIteration(env,eps,gamma):
    states, P = env.getMDP()
    #pi et V de dimension P nombre d'états non terminaux car terminaux par défaut c'est 0 on ne peut pas en sortir
    states_P=list(P.keys())
    pi0=dict()
    for s in states_P:
        actions=list(P[s].keys())
        act=random.randint(0,len(actions)-1)
        pi0[states[s]]=actions[act]
    #initialisation de la politique aleatoire a revoir 
    
    while 1:  
        i=0
        V0=np.array([random.random() for i in range(len(states_P))])
        while 1:
            V1=np.zeros(len(states_P))
            for si in range(len(states_P)):
                s=states_P[si]
                act=pi0[states[s]]
                for dest in P[s][act]:
                    p,d,r,final=dest
                    if d not in states_P:
                        V1[si] += p*r
                    else:
                        d=states_P.index(d)
                        
                        V1[si] += p*(r+gamma*V0[d])
#            print(V1)
            i = i+1
            if np.linalg.norm(V1-V0,ord=np.inf)<= eps:
                V0 = np.copy(V1) 
                break
            V0 = np.copy(V1) 
            print(V0)
            print(i)
            
        print('break')
        pi1 = EvalPolicy(states,P,V1,gamma)
        
        if compare_dict(pi0,pi1):
            break
        pi0=dict(pi1)
    return pi1


def PolicyIteration2(env,eps,gamma):
    states, P = env.getMDP()
    #pi et V de dimension P nombre d'états non terminaux car terminaux par défaut c'est 0 on ne peut pas en sortir
    states_P=list(P.keys())
    pi0=dict()
    for s in states_P:
        actions=list(P[s].keys())
        act=random.randint(0,len(actions)-1)
        pi0[states[s]]=actions[act]
    #initialisation de la politique aleatoire a revoir 
    
    while 1:  
        i=0
        V0=np.array([random.random() for i in range(len(states_P))])
        while 1:
            V1=np.zeros(len(states_P))
            for s in states_P:
                act=pi0[states[s]]
                for dest in P[s][act]:
                    p,d,r,final=dest
                    if d not in states_P:
                        V1[states[s]] += p*r
                    else:
                        print(d)
                        d=states_P.index(d)
                        print('new d',d)
                        print(V0[d])
                        print('s',states[s])
                        print(V1[states[s]])
                        
                        V1[states[s]] += p*(r+gamma*V0[d])
            i = i+1
            if np.linalg.norm(V1-V0,ord=np.inf)<= eps:
                V0 = np.copy(V1) 
                break
            
            
        pi1 = EvalPolicy(states,P,V1)
        
        if compare_dict(pi0,pi1):
            break
        pi0=dict(pi1)
    return pi1

def ValueIteration(env,eps,gamma):
    states, P = env.getMDP()
    states_P=list(P.keys())
    V0=np.array([random.random() for i in range(len(states_P))])
    i = 0 
    while 1:
        V1=np.zeros(len(states_P))
        for s in states_P:
            act = np.zeros(len(P[s])) # erreur avec etats terminaux a voir  
            for a in P[s]:
                for dest in P[s][act]:
                        p,d,r,final=dest
                        act[a] += p*(r + gamma*V0[states[d]])
            V1[states[s]] =  np.argmax(act)
        i = i + 1 
        if np.linalg.norm(V1-V0,ord=np.inf)<= eps:
            V0 = np.copy(V1)
            break
        
    pi = EvalPolicy(states,P,V1)

    return pi      

class Agent():
    def __init__(self, env,eps,gamma,algo):
        self.env=env
        self.eps=eps
        self.gamma=gamma
        self.pi=algo(self.env,self.eps,self.gamma)

    def act(self, etat):
        obs=self.env.state2str(etat)
        return self.pi[obs]
        
if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu 
    
    etat_init=env.reset()
    obs=env.state2str(etat_init)
    states, P = env.getMDP()
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
#    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    eps = 0.1
    rsum=0
    j=0
    gamma=1
    agent = Agent(env,eps,gamma,PolicyIteration)
    while True:
        action = agent.act(states[obs])
        obs, reward, done, _ = envm.step(action)
        rsum += reward
        j += 1
        env.render()
        if done:
            print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
            break


    
        
 #politique dictionnaire ou chaque etat dis ou on doit aller
 
# V value : dictionnaire
# Fonction qui fait policy iteration prend MDP
