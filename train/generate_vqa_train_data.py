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
import h5py
import torch.nn as nn
import torch
import argparse
from torch.autograd import Variable
from models import act_model,get_state
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

class action_Agent(object):
    def __init__(self,args,environment):
        self.args = args
        self.vocab = load_vocab(self.args.vocab_json)
        self.max_step = 5
        self.action_num = 28*28*9
        self.env = environment
        if self.args.agent == 'DQN':
            self.agent = act_model(vocab=self.vocab)
            print(' [*] Build Deep Q-Network')
        else:
            print(' [*] Build random agent')



    def rgb_process(self,rgb_np):
        rgb_tran = rgb_np.transpose((2,0,1))
        rgb_np_process = (rgb_tran/255.0).astype(np.float16)
        return rgb_np_process

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
        if self.args.agent == 'DQN':# 
            actions_value = self.agent.forward(rgb_image,depth_image,ques)
            action_reshape = actions_value.view(1,-1)
            action_location = torch.max(action_reshape,1)[1].cpu().data.numpy()
        else:   # random
            action_location = np.random.randint(0, self.action_num)
            #print("from random")

        return action_location


    def generate_data(self):
        reward_type = 'global+local'
        reward_weight = 0.5 # parameter for actions
        for group_num in range(7,8):  # one eposide
            for scene_num in range(5,6):
                h5_file_dir = os.path.abspath('../data/vqa')     
                h5_file_name = 'group-'+ '0'+str(group_num) + '-scene-'+'0'+ str(scene_num) + '.h5'
                h5_full_file = os.path.join(h5_file_dir , h5_file_name) 
                rgb_images = []
                encode_questions = []
                answers = []
                reward_e_lists = []
                reward_q_lists = []

                rgb_image_raw,depth_image_raw,all_ques,all_encode_ques = self.env.new_scene(group_num = group_num,scene_num = scene_num)      #new scene
                ques_num_list = np.random.choice(a=40, size=20, replace=False, p=None)
                for ques_index in ques_num_list:   #one task

                    single_encode_ques = all_encode_ques[ques_index]
                    single_ques = all_ques[ques_index]
                    encode_answer = self.vocab['answerTokenToIdx'][single_ques['answer']]

                    encode_questions.append(single_encode_ques)
                    answers.append(encode_answer)


                    one_task_rgb_images = []
                    one_task_reward_e = []
                    one_task_reward_q = []
                    _,rgb_initial = self.env.camera.get_camera_data()
                    rgb_initial_image = self.rgb_process(rgb_initial)
                    one_task_rgb_images.append(rgb_initial_image)
                    last_action = 0
                    close_margin = 2
 
                    for act_step in range(self.max_step):
                                                
                        depth_image_raw,rgb_image_raw = self.env.camera.get_camera_data()
                        rgb_image = self.rgb_norm(rgb_image_raw)
                        depth_image = self.depth_norm(depth_image_raw)

                        rgb_image_var = Variable(torch.FloatTensor(rgb_image))   
                        rgb_image_var = rgb_image_var.unsqueeze(0)
                        depth_image_var = Variable(torch.FloatTensor(depth_image))
                        depth_image_var = depth_image_var.unsqueeze(0)
                        question_var = Variable(torch.LongTensor(single_encode_ques))
                        question_var = question_var.unsqueeze(0)


                        action = self.choose_action(rgb_image_var,depth_image_var,question_var)
                        rgb_1_image_raw, depth_1_image_raw, reward, terminal,reward_e,reward_q= self.env.act(action,single_ques['obj'],single_ques['type'],reward_type,reward_weight)                        
                        rgb_1_image = self.rgb_process(rgb_1_image_raw)
                        one_task_rgb_images.append(rgb_1_image)
                        one_task_reward_e.append(reward_e)
                        one_task_reward_q.append(reward_q)


                        if (action >= 6272) or (abs(action-last_action)<=close_margin):
                            print("stop")
                            for  i  in range(self.max_step-act_step-1):
                                one_task_rgb_images.append(rgb_1_image)
                                one_task_reward_e.append(reward_e)
                                one_task_reward_q.append(reward_q)
                            break

                        last_action = action

                    rgb_images.append(one_task_rgb_images)
                    reward_e_lists.append(one_task_reward_e)
                    reward_q_lists.append(one_task_reward_q)
                    if len(one_task_rgb_images) !=6:
                        print("rgb error")
                    if len(one_task_reward_e) !=5:
                        print("reward_e error")
                    if len(one_task_reward_e) !=5:
                        print("reward_q error")

                f  = h5py.File(h5_full_file,'w')
                if (len(rgb_images) != 20) or (len(encode_questions) != 20) or (len(answers) != 20) or (len(reward_e_lists) != 20) or (len(reward_q_lists) != 20):
                    print("data error")
                f['images'] = rgb_images
                f['questions'] = encode_questions
                f['answers'] = answers
                f['reward_e'] = reward_e_lists
                f['reward_q'] = reward_q_lists




if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    parser = argparse.ArgumentParser()
    # data params

    parser.add_argument('-vocab_json', default='../data/vocab.json')  
    parser.add_argument(
        '-reward_type',
        default='global+local',
        type=str,
        choices=['global', 'global+local', 'local'])
    parser.add_argument(
        '-agent',
        default='DQN',
        type=str,
        choices=['random', 'DQN'])

    args = parser.parse_args()
    dqn_env = environment.Environment()
    model_kwargs = {'args':args,'environment':dqn_env}
    act_model =  action_Agent(**model_kwargs)

    if args.agent == 'DQN':
        checkpoint_file = 'best_global_local.pt'
        act_checkpoint = torch.load(checkpoint_file,map_location=torch.device('cpu'))
        act_model.agent.load_state_dict(act_checkpoint['state'])
        print("load before")

    
    act_model.generate_data()



