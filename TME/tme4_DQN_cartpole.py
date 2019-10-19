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
        self.delete=0 #indice de l'element a supprimer
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
        seq=torch.tensor(sequence)
        X=self.Q.forward(seq)
        return torch.argmax(X).item()

    def update(self,obs, newobs,action,reward,done,batchSize): #descente de gradient et maj
        self.D.add_element((obs,action,reward,newobs))
        minibatch=self.D.get_random_batch(batchSize)
        loss=torch.nn.SmoothL1Loss()
        obs=torch.tensor(obs)
        newobs=torch.tensor(newobs)
        if done:
            y=reward
        else:
            tmp=self.Q_hat.forward(newobs)
            m=torch.max(tmp)
            y=reward+self.gamma*m

        Q_obs = self.Q.forward(obs)
        index = torch.argmax(Q_obs)
        Q_loss=loss(y,torch.gather(Q_obs,0,index))
        Q_loss.backward()
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


    eps = 0.0001
    N=500
    batchSize=10
    i=0
    gamma=1

    print("oo")
    for episode in range(episode_count):
        agent=DQN(4,2,[10,10],N,gamma)#on rentre un etat renvoie vecteur de proba d'actions

        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        i+=1
        rsum = 0
        while True:
            action = agent.act(obs)
            newobs, reward, done, _ = env.step(action)
            agent.update(obs, newobs,action,reward,done,batchSize)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
