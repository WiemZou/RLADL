import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch

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

#batch actor critic algorithm
class A2C(torch.nn.Module):
    def __init__(self,inSize,outSize,layers_V,layers_Pi,gamma): 
        super(A2C), self).__init__()
        self.pi=sample()
        self.V=NN(inSize,outSize,layers)

    def act(self,sequence): 
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


    alpha = 0.0001
    batchSize=10
    i=0
    gamma=1
    agent=DQN(4,2,[10,10],N,gamma)#on rentre un etat renvoie vecteur de proba d'actions

    print("oo")
    for episode in range(episode_count):

        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        i+=1
        rsum = 0
        pi=dict()

        while True: ##sampling
            action=torch.from_numpy(np.random.choice(2, 1))
            pi[obs]=action
            newobs, reward, done, _ = env.step(action)
            if done:
                break
            
            
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

