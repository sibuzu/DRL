import pfrl
import torch
import torch.nn as nn
import gym
import numpy as np
import os

MODEL_NAME = "Model"
VIDEO_NAME = "Video"

class Net(nn.Module):
    def __init__(self, inputN, outputN):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            torch.nn.Linear(inputN, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, outputN)
        )
        
    def forward(self, x):
        x = self.net(x)
        x = pfrl.action_value.DiscreteActionValue(x)
        return x

class RLModel():
    def __init__(self, name, rec="", gamma=0.9, epsilon=0.1, bufSize=10**6):
        # create env
        self.env = gym.make(name)
        if rec:
            self.env = gym.wrappers.Monitor(self.env, rec, force=True)
        
        # create model
        obs_size = self.env.observation_space.low.size
        n_actions = self.env.action_space.n
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Net(obs_size, n_actions)
        optimizer = torch.optim.Adam(model.parameters(), eps=1e-2)
        
        # Use epsilon-greedy for exploration
        # explorer = pfrl.explorers.ConstantEpsilonGreedy(
        #     epsilon=epsilon, random_action_func=self.env.action_space.sample)
        explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=0.5, end_epsilon=epsilon, decay_steps=200, random_action_func=self.env.action_space.sample)

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=bufSize)

        # Now create an agent that will interact with the environment.
        self.agent = pfrl.agents.DoubleDQN(
            model,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            replay_start_size=500,
            update_interval=1,
            target_update_interval=100,
            phi=lambda x: x.astype(np.float32, copy=False),
            gpu=0,
        )
        
    def train(self, episodes=300, maxEpisoldeLen=-1, show=False):
        bestScore = self.eval(episodes=10, silence=False, show=True)
        print("Average score is {}".format(bestScore))

        for i in range(1, episodes + 1):
            obs = self.env.reset()
    
            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                if show:
                    self.env.render()
                action = self.agent.act(obs)
                obs, reward, done, _ = self.env.step(action)
                R += reward
                t += 1
                reset = t == maxEpisoldeLen
                self.agent.observe(obs, reward, done, reset)
                if done or reset:
                    break
                    
            if i % 10 == 0:
                print('episode:', i + 1, 'R:', R)                
                score = self.eval(episodes=10, silence=False, show=True)
                if score > bestScore:
                    bestScore = score
                    print("Average score is {}".format(score))
                    self.save(MODEL_NAME)

            # if i % 50 == 0:
            #     print('statistics:', self.agent.get_statistics())
        print('Finished.')

    def eval(self, episodes=10, maxEpisoldeLen=-1, show=False, silence=False):
        total_R = 0
        with self.agent.eval_mode():
            for i in range(episodes):
                obs = self.env.reset()
                R = 0
                t = 0
                while True:
                    if show:
                        self.env.render()
                    action = self.agent.act(obs)
                    obs, r, done, _ = self.env.step(action)
                    R += r
                    t += 1
                    reset = t == maxEpisoldeLen
                    if done or reset:
                        break
                if not silence:
                    print('evaluation episode:', i + 1, 'R:', R)
                total_R += R
        return total_R / episodes

    def save(self, name):
        self.agent.save(name)
        print("model {} is saved".format(name))

    def load(self, name):
        fname = name + "/model.pt"
        if os.path.isfile(fname):
            self.agent.load(name)
            print("model {} is loaded".format(name))

def train(name, show=False):
    model = RLModel(name)
    model.load(MODEL_NAME)
    model.train()
    # model.save("Model")
    model.env.close()

def eval(name, show=False):
    model = RLModel(name)
    model.load(MODEL_NAME)
    model.eval(show=show)
    model.env.close()

def record(name):
    model = RLModel(name, rec="Video")
    model.load(MODEL_NAME)
    model.eval(episodes=1)
    model.env.close()
    
if __name__ == '__main__':
    name = "MountainCar-v0"
    train(name, show=False)
    # eval(name, show=True)
    # record(name)
