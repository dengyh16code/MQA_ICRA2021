import sys
import os
import random
import logging
import numpy as np
from torch.autograd import Variable
from torch.autograd import Variable


class ReplayMemory:
    """
        -- Memory storage
    """
    def __init__(self, args):

        self.memory_dir = os.path.abspath(args.memory_dir)
        self.memory_size = args.memory_size
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float16)
        self.questions = np.empty((self.memory_size,10), dtype = np.uint8)
        self.rgbs = np.empty((self.memory_size, 3, 224, 224), dtype = np.float16)
        self.rgbs_1 = np.empty((self.memory_size, 3, 224, 224), dtype = np.float16)
        self.depths = np.empty((self.memory_size, 3, 224, 224), dtype = np.float16)
        self.depths_1 = np.empty((self.memory_size, 3, 224, 224), dtype = np.float16)
        self.terminals = np.empty(self.memory_size, dtype =np.uint8) # end or not

        self.batch_size = args.batch_size
        self.current = 0


    def add(self, rgb, depth, rgb1, depth1,ques,action,reward,terminal):
        # assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.rgbs[self.current] = rgb
        self.rgbs_1[self.current] = rgb1
        self.depths[self.current] = depth
        self.depths_1[self.current] = depth1
        self.questions[self.current] = ques
        self.terminals[self.current] = terminal
        self.current = (self.current + 1) % self.memory_size



    def sample(self):
        """
            -- Sample from the memory 
        """
        indexes = []
        while len(indexes) < self.batch_size:
            index = random.randint(0, self.current - 1)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        questions = self.questions[indexes]
        rgbs = self.rgbs[indexes]
        rgbs_1 = self.rgbs_1[indexes]
        depths = self.depths[indexes]
        depths_1 = self.depths_1[indexes]

        return rgbs, depths, rgbs_1, depths_1, questions,actions, rewards, terminals



    def save(self):
        for idx, (name, array) in enumerate(
            zip(['rgbs','depths','rgbs1','depths1','questions','actions', 'rewards', 'terminals','current'],
                [self.rgbs, self.depths, self.rgbs_1,self.depths_1,self.questions,self.actions,self.rewards, self.terminals,self.current])):
            path = os.path.join(self.memory_dir, name)
            np.save(path,array)
            print("[*] save %s" %path)

    def load(self):
        for idx, (name, array) in enumerate(
            zip(['rgbs','depths','rgbs1','depths1','questions','actions', 'rewards', 'terminals','current'],
                [self.rgbs, self.depths, self.rgbs_1,self.depths_1,self.questions,self.actions,self.rewards, self.terminals,self.current])):
            path = os.path.join(self.memory_dir, (name+'.npy'))            
            array = np.load(path)
            print("[*] load %s" %path)
