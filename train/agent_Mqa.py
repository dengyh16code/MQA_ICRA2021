"""
    -- This agent is utilizing the CNN+8 U-net layer as for the Deep part (8 U-net with 8 different direction and 2 different depth)
    -- and the input state is 4-channel screen (with different history and memory)
"""
import os
import sys
import time
import random
import numpy as np
import json
from tqdm import tqdm
import torch.nn as nn
import argparse
from torch.autograd import Variable
from replay_memory import ReplayMemory
from models import act_model,get_state
sys.path.append(r'../simulation')
import enviroment


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['questionIdxToToken'] = invert_dict(vocab['questionTokenToIdx'])
        vocab['answerIdxToToken'] = invert_dict(vocab['answerTokenToIdx'])

    assert vocab['questionTokenToIdx']['<NULL>'] == 0
    assert vocab['questionTokenToIdx']['<START>'] == 1
    assert vocab['questionTokenToIdx']['<END>'] == 2
    return vocab

def invert_dict(d):
    return {v: k for k, v in d.items()}

class DQN_Agent(object):
    def __init__(self,args,environment):
        self.args = args
        self.vocab = load_vocab(self.args.vocab_json)
        self.eval_net, self.target_net = act_model(vocab=self.vocab), act_model(vocab=self.vocab)
        print(' [*] Build Deep Q-Network')

        # initialize the parameter of DQN
        self.target_q_update_step = arg.target_q_update_step
        self.epsilon = arg.epsilon
        self.discount = args.discount
        self.learning_rate = args.learning_rate
        self.max_step = args.max_step

        self.weight_dir = r'./dqn/weights'
        self.action_num = 32*32*8
        self.env = environment
        self.memory = ReplayMemory(self.args)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.eval_net.parameters()), lr= self.learning_rate)
        self.train_group_num = 7
        self.test_group_num = 3
        self.loss_func = nn.MSELoss()

    def choose_action(self,rgb_image,depth_image,ques):
        if np.random.uniform() < self.epsilon:   # greedy
            actions_value = self.eval_net.forward(rgb_image,depth_image,ques)
            action_reshape = actions_value.view(1,-1)
            action_location = torch.max(action_reshape,1)[1].data.numpy()
            action = [action_location/(32*32),action_location%(32*32)/32,action_location%(32*32)%32]
        else:   # random
            action_location = np.random.randint(0, self.action_num)
            action = [action_location/(32*32),action_location%(32*32)/32,action_location%(32*32)%32]
        return action


    def learn(self):

        """
            Learn from the memory storage every train_frequency (mini-batch loss GD)
            and update target network's weights every target_q_update_step
        """
        if self.learn_step_counter % self.target_q_update_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  #update target_net's parameters
        self.learn_step_counter += 1

        s_t, action, reward, target, s_t_plus_1, terminal = self.memory.sample()

        q_eval = torch.max(self.eval_net(s_t,target,action),1)[0]

        q_next = self.target_net(s_t_plus_1).detach()  #don't backward

        terminal = np.array(terminal) + 0.
        q_target = reward + (1. - terminal)*self.discount * q_next.max(1)[0].view(-1, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.task_total_loss += loss
        self.task_total_q += q_target.mean()
        self.update_count += 1 
        self.learn_step_counter +=1 #



    def train(self):
        """
            -- Train Model Process
        """
        self.update_count = 0
        self.learn_step_counter = 0
        task_total_reward, self.task_total_loss, self.task_total_q = 0., 0., 0.
        max_avg_act_reward = 0

        for group_num in range(self.test_group_num):  # one eposide
            for scene_num in range(10):
                rgb_image,depth_image,all_ques = self.env.new_scene(group_num = group_num,scene_num = scene_num)      #new scene
                for single_ques in all_ques:   #one task
                    task_total_reward = 0
                    self.task_total_loss = 0
                    self.task_total_q = 0
                    self.update_count = 0
                    for act_step in range(self.max_step):

                        # 1. predict
                        rgb_image_var = Variable(rgb_image.cuda())
                        depth_image_var = Variable(depth_image.cuda())
                        questions_var = Variable(single_ques.cuda())
                        action = self.choose_action(rgb_image_var,depth_image_var,questions_var)
                        # 2. act
                        # notice the action is in [0, 18*18*8-1]
                        rgb_1_image, depth_1_image, reward, terminal = self.env.act(action)
                        # 3. observe & store
                        self.memory.add(rgb_image, depth_image, rgb_1_image, depth_1_image,single_ques,action,reward,terminal)
                        # 4. learn
                        self.learn()

                        task_total_reward += reward

                        if terminal:                    
                            break

                    avg_reward = task_total_reward / act_step       # caculate the average reward after one task
                    avg_loss = self.task_total_q / self.update_count 
                    avg_q = self.update_count / self.update_count    


                if  0.8*avg_reward > max_avg_act_reward:   #avg_reward相当于在新的场景测试集上的test score
                    checkpoint = {'state': get_state(self.eval_net),
                    'optimizer': self.optimizer.state_dict()}
                    checkpoint_path = '%s/step_%d.pt' % (self.args.checkpoint_dir, self.learn_step_counter)
                    torch.save(checkpoint, checkpoint_path)
                    print('Saving checkpoint to %s' % checkpoint_path)
                    max_avg_act_reward = max(max_avg_act_reward, avg_reward)
                    print('\n [#] Up-to-now, the max action reward is %.4f \n --------------- ' %(max_avg_act_reward))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params

    parser.add_argument('-vocab_json', default='../data/vocab.json')


    #memory params
    parser.add_argument('-memory_dir',default='/memory')
    parser.add_argument('-memory_size',default=4000)   #100 scene* 40 question
    parser.add_argument('-batch_size',default=32)


    # optim params
    parser.add_argument('-learning_rate', default=1e-3, type=float)
    parser.add_argument('-target_q_update_step', default=100, type=int)
    parser.add_argument('-epsilon', default=0.2, type=float)
    parser.add_argument('-discount', default=0.6, type=float)
    parser.add_argument('-max_step', default=5, type=int)



    # bookkeeping
    parser.add_argument('-print_every', default=5, type=int)
    parser.add_argument('-eval_every', default=10, type=int)
    parser.add_argument('-save_every', default=200, type=int) #optional if you would like to save specific epochs as opposed to relying on the eval thread
    parser.add_argument('-model', default='act')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    # checkpointing
    parser.add_argument('-checkpoint_path', default=False)
    parser.add_argument('-checkpoint_dir', default='checkpoints/act/')
    parser.add_argument('-log_dir', default='logs/act/')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')
    parser.add_argument('-max_controller_actions', type=int, default=5)
    parser.add_argument('-max_actions', type=int)
    args = parser.parse_args()
    
    args.train_h5 = os.path.abspath(args.train_h5)
    args.time_id = time.strftime("%m_%d_%H:%M")

    #MAX_CONTROLLER_ACTIONS = args.max_controller_actions

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    if args.curriculum:
        assert 'lstm' in args.model_type #TODO: Finish implementing curriculum for other model types

    logging.basicConfig(filename=os.path.join(args.log_dir, "run_{}.log".format(
                                                str(datetime.now()).replace(' ', '_'))),
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        logging.info("CPU not supported")
        exit()

    if args.checkpoint_path != False:

        print('Loading checkpoint from %s' % args.checkpoint_path)
        logging.info("Loading checkpoint from {}".format(args.checkpoint_path))

        args_to_keep = ['model_type']

        checkpoint = torch.load(args.checkpoint_path, map_location={
            'cuda:0': 'cpu'
        })

        for i in args.__dict__:
            if i not in args_to_keep:
                checkpoint['args'][i] = args.__dict__[i]

        args = type('new_dict', (object, ), checkpoint['args'])

    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.identifier)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.identifier)

    print(args.__dict__)
    logging.info(args.__dict__)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)


    if args.model_type == 'nomap':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        shared_model = actPlannerBaseModel(**model_kwargs)

    elif args.model_type == 'addmap':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        shared_model = actPlannerImproveModel(**model_kwargs)
        act_checkpoint = torch.load("act_load.pt")     #load checkpoint weights
        shared_model.load_state_dict(act_checkpoint['state'])   #create model

    else:

        exit()

    shared_model.share_memory()

    if args.checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.checkpoint_path)
        logging.info("Loading params from checkpoint: {}".format(args.checkpoint_path))
        shared_model.load_state_dict(checkpoint['state'])

    if args.mode == 'eval':

        eval(0, args, shared_model)

    elif args.mode == 'train':

        if args.num_processes > 1:
            processes = []
            for rank in range(0, args.num_processes):
                # for rank in range(0, args.num_processes):
                p = mp.Process(target=train, args=(rank, args, shared_model))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        else:
            train(0, args, shared_model)


    else:
        processes = []

        # Start the eval thread
        p = mp.Process(target=eval, args=(0, args, shared_model))
        p.start()
        processes.append(p)

        # Start the training thread(s)
        for rank in range(1, args.num_processes + 1):
            # for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

