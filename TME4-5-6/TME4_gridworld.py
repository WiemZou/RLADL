import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
from torch import nn

class NN(torch.nn.Module):
    def __init__(self, inSize, outSize, layers=[]): #execute layers passages en couche
        super(NN, self).__init__()
        self.layers=nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize,x))
            inSize=x
        self.layers.append(nn.Linear(inSize,outSize))
    def forward(self, x):
        x=self.layers[0](x)
        for i in range(1, len(self.layers)):
            x=torch.nn.functional.leaky_relu(x)
            x=self.layers[i](x)
        return x

class FeaturesExtractor(object):
    def __init__(self,outSize):
        super().__init__()
        self.outSize=outSize*3
    def getFeatures(self, obs):
        state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0]=np.where(obs == 2,1,state[0])
        state[1]=np.where(obs == 4,1,state[1])
        state[2]=np.where(obs == 6,1,state[2])
        return torch.from_numpy(state.reshape(1,-1),dtype=torch.float)

class Memory():
    def __init__(self,N):
        self.N=N
        self.occupe=0 #nb de lignes occupées dans la mémoire
        self.delete=0 #indice de l'élément à supprimer
        self.D=[]
    def get_random_batch(self,batchSize):
        if len(self.D)<batchSize:
            return self.D
        batch=[random.choice(self.D) for i in range(batchSize)]
        return self.D[index]
    def add_element(self,elem):
        if len(self.D)<self.N:
            self.D.append(elem)
        else:
            self.D[occupe]=elem
            self.occupe=(self.occupe+1)%self.N

class DQN(torch.nn.Module):
    def __init__(self,inSize,outSize,layers,N,gamma): 
        super(DQN, self).__init__()
        self.Q=NN(inSize,outSize,layers)
        self.Q_hat=copy.deepcopy(self.Q)#recupere paramtres avec parameters
        self.D=Memory(N)
        self.gamma=gamma
    def act(self,sequence):
        X=self.Q.forward(sequence)
        return torch.argmax(X)

    def update(self,obs, newobs,action,reward,done,batchSize): #descente de gradient et maj
        self.D.add_element((obs,action,reward,newobs))
        minibatch=self.D.get_random_batch(batchSize)
        loss=torch.nn.SmoothL1Loss()
        if done:
            y=reward
        else:
            tmp=self.Q_hat.forward(newobs)
            m=torch.max(tmp)
            y=reward+self.gamma*m
        v=loss((y-self.Q[obs][action])**2)
        loss.backward()
        self.Q_hat=copy.deepcopy(self.Q)


if __name__ == '__main__':


    env = gym.make("gridworld-v0")

    # Enregistrement de l'Agent

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(0)  # Initialiser le pseudo aleatoire
    episode_count = 10000
    reward = 0
    done = False
    rsum = 0
    eps = 0.0001
    states, P = env.getMDP()
    N=500
    batchSize=10
    i=0
    gamma=1
    feat_extract=FeaturesExtractor(len(states))
    for episode in range(episode_count):
        agent=DQN(1,len(env.actions),[10,10],N,gamma)#on rentre un état renvoie vecteur de proba d'actions
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        i+=1
        rsum = 0
        while True:
            obs=feat_extract.getFeatures(obs)
            
            action = agent.act(obs)
            newobs, reward, done, _ = envm.step(action)
            newobs=feat_extract.getFeatures(newobs)
            update=agent.update(obs, newobs,action,reward,done,batchSize)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()

#deepcopy : 
#load state dict permet de charger ce qui est renvoyé par state_dict