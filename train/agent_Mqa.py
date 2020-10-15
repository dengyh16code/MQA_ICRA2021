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
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
from torch.autograd import Variable
from replay_memory import ReplayMemory
from models import act_model,get_state
import logging
sys.path.append(r'../simulation')
import environment


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
        self.target_q_update_step = args.target_q_update_step
        
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_update_step = args.epsilon_update_step

        self.discount = args.discount
        self.learning_rate = args.learning_rate
        self.max_step = args.max_step
        self.learn_step_counter = args.learn_step

        self.action_num = 28*28*9
        self.env = environment
        self.memory = ReplayMemory(self.args)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.eval_net.parameters()), lr= self.learning_rate)
        self.train_group_num = 7
        self.test_group_num = 3
        self.loss_func = nn.MSELoss()

    def rgb_norm(self,rgb_np):
        rgb_mean = np.mean(rgb_np)
        rgb_std = np.std(rgb_np)
        if rgb_std != 0:         #error image
            rgb_miner = np.ones(rgb_np.shape)*rgb_mean
            rgb_np = (rgb_np - rgb_miner) / rgb_std
        rgb_tran = rgb_np.transpose((2,0,1))
        return rgb_tran

    def depth_norm(self,dep_np):
        dep_np = dep_np*65536/10000
        dep_np = np.clip(dep_np,0.0,1.2)   # the depth range: 0.0m -1.2m
        dep_mean = np.mean(dep_np)
        dep_std  = np.std(dep_np)
        if dep_std !=0:
            dep_miner = np.ones(dep_np.shape)*dep_mean
            dep_np = (dep_np - dep_miner)/dep_std
        dep_rep = np.expand_dims(dep_np,0).repeat(3,axis=0)
        return dep_rep

    def choose_action(self,rgb_image,depth_image,ques):
        if self.learn_step_counter > self.epsilon_update_step:
            epsilon = self.epsilon_end
        else:
            epsilon = self.epsilon_start-(self.learn_step_counter / float(self.epsilon_update_step)) *(self.epsilon_start-self.epsilon_end)
        if np.random.uniform() > epsilon:   # greedy
            actions_value = self.eval_net.forward(rgb_image,depth_image,ques)
            action_reshape = actions_value.view(1,-1)
            action_location = torch.max(action_reshape,1)[1].cpu().data.numpy()
        else:   # random
            action_location = np.random.randint(0, self.action_num)
        return action_location


    def learn(self):

        """
            Learn from the memory storage every train_frequency (mini-batch loss GD)
            and update target network's weights every target_q_update_step
        """
        if self.learn_step_counter % self.target_q_update_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  #update target_net's parameters
            logging.info("updtate target q")
        self.learn_step_counter += 1

        rgbs,depths, rgbs_1, depths_1,questions,actions,rewards,terminals = self.memory.sample()

        rgbs_var = Variable(torch.FloatTensor(rgbs).cuda())
        depths_var = Variable(torch.FloatTensor(depths).cuda())
        rgbs_1_var = Variable(torch.FloatTensor(rgbs_1).cuda())
        depths_1_var = Variable(torch.FloatTensor(depths_1).cuda())
        questions_var = Variable(torch.LongTensor(questions).cuda())
        actions_var = Variable(torch.LongTensor(actions).cuda())
        rewards_var = Variable(torch.FloatTensor(rewards).cuda())
        terminals_var = Variable(torch.FloatTensor(terminals).cuda())

        q_eval_matrix = self.eval_net(rgbs_var,depths_var,questions_var)
        q_eval_matrix = q_eval_matrix.view(-1,9*28*28)
        q_eval = torch.max(q_eval_matrix,1)[0]

        q_next_matrix = self.target_net(rgbs_1_var,depths_1_var,questions_var).detach()  #don't backward
        q_next_matrix = q_next_matrix.view(-1,9*28*28)
        q_next = torch.max(q_next_matrix,1)[0]

        one_var = Variable(torch.ones_like(terminals_var))

        q_target = rewards_var + (one_var- terminals_var)*self.discount * q_next
 
        loss = self.loss_func(q_eval, q_target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.task_total_loss += loss.item()
        self.task_total_q += q_target.mean()
        self.update_count += 1 



    def train(self):
        """
            -- Train Model Process
        """
        torch.cuda.set_device(args.gpus.index(args.gpus[0 % len(args.gpus)]))

        self.eval_net.train()
        self.eval_net.cuda()
        self.target_net.eval()
        self.target_net.cuda()

        self.update_count = 0
        task_total_reward, self.task_total_loss, self.task_total_q = 0., 0., 0.
        max_avg_act_reward = 0.33

        group_num_list = list(range(self.test_group_num))
        random.shuffle(group_num_list)
        scene_num_list = list(range(10))
        random.shuffle(scene_num_list)
        
        for group_num in group_num_list:  # one eposide
            for scene_num in scene_num_list:
                rgb_image_raw,depth_image_raw,all_ques,all_encode_ques = self.env.new_scene(group_num = group_num,scene_num = scene_num)      #new scene
                
                rgb_image = self.rgb_norm(rgb_image_raw)
                depth_image = self.depth_norm(depth_image_raw)

                for i in range(len(all_encode_ques)):   #one task
                    single_encode_ques = all_encode_ques[i]
                    single_ques = all_ques[i]
                    task_total_reward = 0
                    self.task_total_loss = 0
                    self.task_total_q = 0
                    self.update_count = 0
                    task_act_num =0
                    print("target:",single_ques['obj'])
                    for act_step in range(self.max_step):

                        # 1. predict
                        rgb_image_var = Variable(torch.FloatTensor(rgb_image).cuda())
                        rgb_image_var = rgb_image_var.unsqueeze(0)
                        depth_image_var = Variable(torch.FloatTensor(depth_image).cuda())
                        depth_image_var = depth_image_var.unsqueeze(0)
                        question_var = Variable(torch.LongTensor(single_encode_ques).cuda())
                        question_var = question_var.unsqueeze(0)
                        action = self.choose_action(rgb_image_var,depth_image_var,question_var)
                        # 2. act
                        # notice the action is in [0, 18*18*8-1]
                        rgb_1_image_raw, depth_1_image_raw, reward, terminal = self.env.act(action,single_ques['obj'],single_ques['type'])
                        
                        rgb_1_image = self.rgb_norm(rgb_1_image_raw)
                        depth_1_image = self.depth_norm(depth_1_image_raw)

                        # 3. observe & store
                        self.memory.add(rgb_image, depth_image, rgb_1_image, depth_1_image,single_encode_ques,action,reward,terminal)
                        # 4. learn
                        self.learn()

                        task_total_reward += reward
                        task_act_num += 1   

                        if terminal:                                            
                            break


                    avg_reward = task_total_reward / task_act_num       # caculate the average reward after one task
                    avg_loss = self.task_total_loss / self.update_count 
                    avg_q = self.task_total_q / self.update_count
                    print("avg_loss:",avg_loss)
                    print("avg_reward:",avg_reward) 
                    logging.info("avg_reward:{}".format(avg_reward))
                    logging.info("avg_loss:{}".format(avg_loss))
                    logging.info("avg_q:{}".format(avg_q))

                    if  0.9*avg_reward > max_avg_act_reward:   #avg_reward相当于在新的场景测试集上的test score
                        checkpoint = {'state': get_state(self.eval_net),
                        'optimizer': self.optimizer.state_dict()}
                        checkpoint_t = {'state': get_state(self.target_net)}

                        checkpoint_path = '%s/step_%d.pt' % (self.args.checkpoint_dir, self.learn_step_counter)
                        torch.save(checkpoint, checkpoint_path)

                        checkpoint_path_t = '%s/t_step_%d.pt' % (self.args.checkpoint_dir, self.learn_step_counter)
                        torch.save(checkpoint_t, checkpoint_path_t)


                        print('Saving checkpoint to %s' % checkpoint_path)
                        max_avg_act_reward = max(max_avg_act_reward, avg_reward)
                        print('\n [#] Up-to-now, the max action reward is %.4f \n --------------- ' %(max_avg_act_reward))
                        logging.info("max action reward:{}".format(max_avg_act_reward))
                        self.memory.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params

    parser.add_argument('-vocab_json', default='../data/vocab.json')
    parser.add_argument(
        '-mode',
        default='train',
        type=str,
        choices=['train', 'eval', 'train+eval'])
    
    parser.add_argument(
        '-reward_type',
        default='avg',
        type=str,
        choices=['avg', 'avg+local', 'local'])


    #memory params
    parser.add_argument('-memory_dir',default='../data/memory')
    parser.add_argument('-memory_size',default=1000)   #100 scene* 40 question
    parser.add_argument('-batch_size',default=16)
    parser.add_argument('-first_train',default=False,type=bool)
    parser.add_argument('-learn_step',default=0,type=int)

    # optim params
    parser.add_argument('-learning_rate', default=1e-3, type=float)
    parser.add_argument('-target_q_update_step', default=100, type=int)
    parser.add_argument('-epsilon_start', default=1,type=float)
    parser.add_argument('-epsilon_end', default=0.1,type=float)
    parser.add_argument('-reward_weight_start', default=1,type=float)
    parser.add_argument('-reward_weight_end', default=0.1,type=float)
    parser.add_argument('-epsilon_update_step', default=2000,type=float)  #memory / 2

    parser.add_argument('-discount', default=0.6, type=float)
    parser.add_argument('-max_step', default=10, type=int)



    # bookkeeping
    parser.add_argument('-print_every', default=5, type=int)
    parser.add_argument('-eval_every', default=10, type=int)
    parser.add_argument('-save_every', default=200, type=int) #optional if you would like to save specific epochs as opposed to relying on the eval thread
    parser.add_argument('-model', default='act')


    # checkpointing
    parser.add_argument('-checkpoint_name', default='step_579.pt')
    parser.add_argument('-checkpoint_dir', default='../data/checkpoints/act/')
    parser.add_argument('-log_dir', default='logs/act/')
    args = parser.parse_args()
    args.time_id = time.strftime("%m_%d_%H:%M")

    #MAX_CONTROLLER_ACTIONS = args.max_controller_actions

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

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


    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.model)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.model)

    print(args.__dict__)
    logging.info(args.__dict__)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)


    dqn_env = environment.Environment()

    model_kwargs = {'args':args,'environment':dqn_env}
    act_model =  DQN_Agent(**model_kwargs)

    if args.mode == 'train':
        if args.first_train:
            act_model.train()
        else:
            checkpoint_file = 'step_616.pt'
            args.learn_step = 616
            act_checkpoint = torch.load(checkpoint_file)
            act_model.eval_net.load_state_dict(act_checkpoint['state'])
            act_model.target_net.load_state_dict(act_checkpoint['state'])
            act_model.optimizer.load_state_dict(act_checkpoint['optimizer'])    # load check_point
            for state in act_model.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            act_model.memory.load()   #load memory    

            act_model.train()     




