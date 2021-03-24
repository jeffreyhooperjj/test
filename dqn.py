from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            action = torch.argmax(self.forward(state))
            #print("action:" + str(action))
            #print("torch.max(): " + str(torch.argmax(self.forward(state))))
            





        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    #state = Variable(torch.FloatTensor(np.float32(state)))
    state = Variable(torch.FloatTensor(np.float32(state)).squeeze(1))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here
    # target = what it should be
    # model = what it is
    #print(done)
    #print("made it here")
    terminate = torch.ones_like(done) - done
    r_j = reward
    y_j = None
    #if terminated:
     #   y_j = r_j
    #else:
    #Q_next_state = target_model(next_state).detach().cpu().numpy()
    #print("Q next state: " + str(Q_next_state))
    #max_Q_next = np.array(np.amax(max_Q_next)
    #max_Q_next = np.array(Q_next_state)
    #max_Q_next = torch.autograd.Variable(torch.from_numpy(max_Q_next)).max(1)[0]
    #print("max_Q_next:" + str(max_Q_next))
    #print("other thing" + str(target_model(next_state).max(1)[0]))
    #print(Q_next_state)
    y_j = r_j + (Variable(gamma * target_model(next_state).detach().max(1)[0]) * terminate)
    #print("done: " + str(done))
    #action_indices = torch.LongTensor(action)
    #action_index = action_indices.unsqueeze(-1)
    Q_curr = model(state)
    Q_val = Q_curr.gather(1, action.unsqueeze(1)).squeeze(1)
    #print("Q val: " + str(Q_val))
    #print(Q_curr)

    #Q_values = model(state).detach().cpu().numpy()
    
    #y_j = r_j + gamma * torch.max(Q_next_state)

    #loss = (y_j - Q_val) ** 2
    #loss = nn.MSE()(y_j, Q_val)
    loss = nn.MSELoss(reduction="sum")
    loss = loss(Q_val, y_j)
    #print("loss: " +  str(loss))
    #print("other loss: " + str((Q_val - y_j).pow(2).mean()))

    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        #print("state: " + str(state))
        #print("action: " + str(action))
        #print("reward: " + str(reward))
        #print("next state: " + str(next_state))
        #print("done: " + str(done))

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        #print(random.sample(self.buffer, batch_size)[0])
        #print(self.buffer[0][0])
        #print(self.buffer[0][1])
        #x = random.randint(0, batch_size-1)
        #state = self.buffer[x][0]
        #action = self.buffer[x][1]
        #reward = self.buffer[x][2]
        #next_state = self.buffer[x][3]
        #done = self.buffer[x][3]
        state, action, reward, next_state, done = zip(* random.sample(self.buffer, batch_size))
        #print(self.buffer[0][0])
        #indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        #states, actions, rewards, dones, next_states = zip([self.buffer[idx] for idx in indices])
        #return np.array(states), np.array(actions), np.array(rewards,dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)  
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
