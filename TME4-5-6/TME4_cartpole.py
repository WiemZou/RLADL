import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
from torch import nn
import random

class NN(torch.nn.Module):
    def __init__(self, inSize, outSize, layers=[]): #execute layers passages en couche
        super(NN, self).__init__()
        self.layers=nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize,x))
            inSize=x
        self.layers.append(nn.Linear(inSize,outSize))
    def forward(self, x):
        x=x.float()
        x=self.layers[0](x)
        for i in range(1, len(self.layers)):
            x=torch.nn.functional.leaky_relu(x)
            x=self.layers[i](x)
        return x

class Memory():
    def __init__(self,N):
        self.N=N
        self.occupe=0 #nb de lignes occupees dans la memoire 
        self.D=[]
    def get_random_batch(self,batchSize):
        if len(self.D)<batchSize:
            return self.D
        batch=[random.choice(self.D) for i in range(batchSize)]
        return batch
    def add_element(self,elem):
        if len(self.D)<self.N:
            self.D.append(elem)
        else:
            self.D[self.occupe]=elem
            self.occupe=(self.occupe+1)%self.N

class DQN(torch.nn.Module):
    def __init__(self,inSize,outSize,layers,N,gamma): 
        super(DQN, self).__init__()
        self.Q=NN(inSize,outSize,layers)
        self.Q_hat=copy.deepcopy(self.Q)#recupere paramtres avec parameters
        self.D=Memory(N)
        self.gamma=gamma
        self.C=50 #temps ou on met a jour Q_hat
        self.time=0
        self.loss=torch.nn.SmoothL1Loss()

    def act(self,sequence,eps): #faire un epsilon greedy
        if random.random()<eps:
            return random.randint(0,1)
        seq=torch.tensor(sequence)
        X=self.Q.forward(seq)
        return torch.argmax(X).item()

    def update(self,tuple_add,batchSize): #descente de gradient et maj
        self.time+=1
        self.D.add_element(tuple_add)

        minibatch=self.D.get_random_batch(batchSize)

        y=[]
        l_obs=[]
        l_actions=[]
        for x in minibatch:
            obs,action,reward,newobs,done=x
            if done:
                y.append(reward)  ## batch x 1
            else:
                #with torch.no_grad():  
                newobs=torch.tensor(newobs)
                tmp=self.Q_hat.forward(newobs)
                m=torch.max(tmp)
                y.append((reward+self.gamma*m))  
            l_obs.append(obs)
            l_actions.append(action)

            Q_obs = self.Q.forward(torch.tensor(l_obs))  ## batch x actions 
            y_hat=torch.gather(Q_obs,1,torch.tensor(l_actions).view(Q_obs.size(0),-1))
        Q_loss=self.loss(torch.tensor(y),y_hat)
        Q_loss.backward()
        #Faire un optim pour maj les params, backward ne fait que calculer les gradients
        if self.time%self.C==0:     
            self.Q_hat=copy.deepcopy(self.Q)


if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 10000
    reward = 0
    done = False
    rsum = 0
    env.verbose = True
    np.random.seed(5)


    eps = 0.2 #epsilon greedy
    N=2000 #taille de la memoire
    batchSize=30
    i=0
    gamma=0.75

    agent=DQN(4,2,[32],N,gamma)#on cree un agent qui apprends, essayer avec [10,10]

    for episode in range(episode_count):

        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        i+=1
        rsum = 0
        while True:
            print(type(obs))
            action = agent.act(obs,eps)
            newobs, reward, done, _ = env.step(action)
            print(type(newobs))
            agent.update((obs,action,reward,newobs,done),batchSize)
            rsum += reward
            j += 1
            env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
