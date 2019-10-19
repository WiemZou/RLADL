import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import random



#gamma=1 toutes politiques ont le meme coup meme interet quel que soit le chemin choisi alors que pour gamma inferieur à 1 on force a choisir le plus court

#env.render()
class Q_learning():
    def __init__(self,gamma,alpha,eps,states,actions): #eps pour epsilon greedy
        self.gamma=gamma
        self.alpha=alpha
        self.eps=eps
        self.states=states
        self.actions=actions
        self.Q={s :{a : 0 for a in self.actions} for s in self.states}
        
    def act(self,etat):
        v=random.random()
        if v<eps:
            qmax=dict()
            for a in self.actions:
                qmax[a]=self.Q[etat][a]
            qmax = [(value, key) for key, value in qmax.items()]
            amax = max(qmax)[1]
            return amax
        else:
            return self.actions[np.random.randint(0,len(self.actions)-1)]
    
    def maj_Q(self,etat,act,obs,reward):
        
        qmax = []
        for aprime in self.actions:
            qmax.append(self.Q[obs][aprime]) 
        qmax = max(qmax)
        self.Q[etat][act] += self.alpha*[reward + self.gamma*qmax- self.Q[etat][act]]
        return


        
if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu   
    etat_init=env.reset()
    states, _ = env.getMDP()
#    states=env.states
    actions=list(env.actions.keys())  
    
    print(obs)
    print(obs)
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
#    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed()  # Initialiser le pseudo aleatoire
    eps = 0.5
    gamma=1
    alpha=0.0001
    rsum=0
    agent = Q_learning(gamma,alpha,eps,states,actions)
    i=0
    episode_count = 10000
    for i in range(episode_count):
        s0 = env.reset()
        s0=env.state2str(s0)
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            obs=env.state2str(obs)
            agent.maj_Q(s0,action,obs,reward)
            rsum += reward
            j += 1
            env.render()
            s0 = obs
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break


    
        
 #politique dictionnaire ou chaque etat dis ou on doit aller
 
# V value : dictionnaire
# Fonction qui fait policy iteration prend MDP
