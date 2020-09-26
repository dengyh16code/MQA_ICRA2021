
import time
import os
import sys
import numpy.random as random
import numpy as np
import math
from collections import defaultdict
import array
import json
import cv2 as cv
from enviroment  import Environment
from generate_questions import Question

'''
generate new data 
(including a new scene and 20 questions about this scene)
easy scene: 20 objects 0
mid scene: 35 objects 1
hard scene: 50 objects.2
'''


class scene(object):
    def __init__(self, group_order='00',scene_order ='00',scene_diff=0):
        self.output_file='group-'+group_order + '-scene-'+scene_order + '.txt'
        output_file_dir = os.path.abspath('../data/test-cases/')
        self.output_preset_file = os.path.join(output_file_dir, self.output_file)

        self.ques_file_name =  'group-' + group_order + '-scene-' + scene_order + '-ques'+'.json'
        ques_file_dir = os.path.abspath('../data/ques/')
        self.ques_full_file = os.path.join(ques_file_dir, self.ques_file_name)

        self.encode_ques_file_name =  'group-' + group_order + '-scene-' + scene_order + '-encodeques'+'.json'
        encode_ques_file_dir = os.path.abspath('../data/encodeques/')
        self.encode_ques_full_file = os.path.join(encode_ques_file_dir, self.encode_ques_file_name)
        self.vocab_dir =  os.path.abspath("../data/vocab.json")
        vocab_file = open(self.vocab_dir,'r',encoding='utf-8')
        self.vocab = json.load(vocab_file)
        self.scene_diff = scene_diff


        self.all_obj =[
            'book', 'bottle', 'calclator','can','card',
            'key', 'keyboard', 'mouse', 'pen','phone',
            'clock','coin','cube','cup','gamepad',
            'pin','roller','scissors','usb','wallet'
        ]


        self.scene_dict = defaultdict(dict)
        self.object_type =[]
        self.object_file_name = []


        box_file = os.path.abspath('../data/box.txt')
        file = open(box_file, 'r')
        file_content = file.readlines()
        file.close()
        self.box_para = file_content[0].split()    
        self.workspace_limits = np.asarray([[float(self.box_para[0]), float(self.box_para[1])], [float(self.box_para[2]), float(self.box_para[3])] ])

        if scene_diff < 3 and scene_diff >=0:
            self.obj_num = scene_diff*15 + 20

        self.obj_mesh_dir=os.path.abspath('../data/mesh/')
        self.mesh_list = os.listdir(self.obj_mesh_dir)
        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.choice(a=len(self.mesh_list), size=self.obj_num, replace=False)
        #print (self.obj_mesh_ind)    #random objects
        
        self.drop_height =0.5

        self.generate_data()
        self.test_new_scene()



    def generate_data(self):
        # Add objects to robot workspace at x,y location and orientation
        # generate the position matrix
        exist_obj_list = []

        for i in range(self.obj_num):
            object_idx = self.obj_mesh_ind[i]
            curr_obj_type = self.mesh_list[object_idx][:-5]
            exist_obj_list.append(curr_obj_type)
            curr_obj_name = self.mesh_list[object_idx]

            self.scene_dict[i]['type'] = curr_obj_type
            self.scene_dict[i]['name'] = curr_obj_name


            curr_obj_ori = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            self.scene_dict[i]['oritentation'] = curr_obj_ori

            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1

            curr_object_position = [drop_x, drop_y, self.drop_height]

            self.scene_dict[i]['position'] = curr_object_position


        outfile = open(self.output_preset_file, 'w')

        for i in range(self.obj_num):    #save the scene txt
            object_name = self.scene_dict[i]['name']
            object_position = self.scene_dict[i]['position']
            object_orientation = self.scene_dict[i]['oritentation']
            outfile.write(object_name+' ')
            for i in range(3):
                outfile.write(str(object_position[i])+' ')
            for i in range(3):
                outfile.write(str(object_orientation[i]*180/np.pi)+' ')
            outfile.write('\n')

        outfile.close()

        my_ques = Question(exist_obj_list)    #save the question json
        all_ques = my_ques.createQueue()
        #vocab = my_ques.create_vocab()
        json.dump(all_ques, open(self.ques_full_file, 'w'))

        # encode the question
        encode_questions = []
        for single_ques in all_ques:
            singe_question = single_ques['question']
            questionTokens = my_ques.tokenize(
            singe_question, punctToRemove=['?'], addStartToken=False)
            encoded_question = my_ques.encode(questionTokens, self.vocab['questionTokenToIdx'])
            encoded_question.append(0)
            while(len(encoded_question)<10):
                encoded_question.append(0)
            encode_questions.append(encoded_question)
        json.dump(encode_questions, open(self.encode_ques_full_file, 'w'))




    def test_new_scene(self):
        self.simu_env = Environment(testing_file=self.output_file,obj_num=self.obj_num)
        #random_operation_times = 5
        #for i in range(random_operation_times):
            #action = [np.random.random_sample()*180,np.random.random_sample()*180,np.random.random_sample()*180,np.random.random_sample()*180]
            #self.simu_env.camera.get_camera_data()
            #self.simu_env.UR5_action(action)




group_order='00'     #'00'-'09'
scene_order ='07'    #'00'-'02'  '03'-'05'  '06'-'09'
scene_diff = 0      #     0         1           2
my_scene = scene(group_order=group_order,scene_order=scene_order,scene_diff=2)

                        
             

        

