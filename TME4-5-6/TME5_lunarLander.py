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


#https://github.com/floodsung/a2c_cartpo        states,actions,rewards,state,done = spl
le_pytorch/blob/master/a2c_cartpole.py

#batch actor critic algorithm
class A2C(torch.nn.Module):
    def __init__(self,state_dim,action_dim,layers_V,layers_Pi,gamma): 
        super(A2C,self).__init__()
        self.pi=sample()

        self.V=NN(state_dim,1,layers_V)
        self.V_optim = torch.optim.Adam(self.V.parameters(),lr=0.01)
        self.Pi=NN(state_dim,action_dim,layers_Pi)
        self.Pi_optim = torch.optim.Adam(self.Pi.parameters(),lr = 0.01)

        self.soft_max=torch.nn.Softmax()
        self.huber=torch.nn.SmoothL1Loss()


    def sample(self,env): 
        actions=[]
        states=[]
        rewards=[]
        state=env.reset()
        rsum=0
        done=False

        while not done:
            f=self.Pi(torch.Tensor(state))
            s=self.soft_max(f) # Return un tensor de dimension OutSize
            a = np.random.choice(outSize,1,p = s.tolist()) #np.r.choice(Nombre de possibilités,Nombre de choix à faire,probabilités)
            actions.append(int(a))
            next_state,reward,done,_ = env.step(a)
            rewards.append(reward)
            states.append(state)
            state = next_state
            rsum+=reward

            if done:
                break

        return states,actions,rewards,state,done,rsum

    def fit(self,env,spl):
        states,actions,rewards,state,done,rsum = spl

        #fitting V
        r=rsum
        for t in reversed(range(0, len(rewards))):
            self.V_optim.zero_grad()
            v_pi=self.V(states[t])
            r=r-rewards[t]
            loss=self.huber(v_pi,r)
            loss.backward()
            #torch.nn.utils.clip_grad_norm(self.V.parameters(),0.5)
            self.V_optim.step()

        
        vs = self.V(states).detach()
        r = np.zeros_like(rewards)
        rsum=0
        for i in reversed(range(0,len(rewards)):
            rsum=rsum*self.gamma+rewards[i]    

            


    def update(self,tuple_add,batchSize): #descente de gradient et maj
 

if __name__ == '__main__':


    env = gym.make('LunarLander-v2')

    # Enregistrement de l'Agent
    agent = RandomAgent(env.action_space)

    outdir = 'LunarLander-v2/results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    env._max_episode_steps = 200
    max_step=100

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()

