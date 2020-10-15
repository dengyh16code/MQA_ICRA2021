import numpy as np
import math
import os, sys, json
from question_string_builder import QuestionStringBuilder

class Question():
    def __init__(self, obj_exist_list):
        self.obj_exist_list = obj_exist_list    #the object exist in the scenes
        #self.obj_dic = obj_dic

        self.obj_count = {}
        for key in self.obj_exist_list:
            self.obj_count[key] = self.obj_count.get(key,0) + 1
        print(self.obj_count)



        self.blacklist_objects =[
            'book', 'bottle', 'calclator','can','card',
            'charger','key', 'keyboard', 'mouse', 'pen',
            'phone','clock','cube','cup','gamepad',
            'eraser','screwdriver','scissors','usb','wallet'
        ]


        self.spatial_question = ['spatial_positive','spatial_negative','spatial_logical_positive','spatial_logical_negative']


        self.templates = {
        'count':
        'how many <OBJ-plural> are there in the bin?',
        'exist':
        '<AUX> there <ARTICLE> <OBJ> in the bin?',
        'spatial_relationship':
        'what is under the <OBJ>?',
        'spatial_logic':
        '<AUX> there <ARTICLE> <OBJ2> <LOGIC> the <OBJ1>?'
        } 

   
        self.q_str_builder = QuestionStringBuilder()
        
        self.question_outputJson = os.path.abspath('../data/question.json')
        '''
        self.exist_outputJson = os.path.abspath('../questions/exist.json')
        self.count_outputJson = os.path.abspath('../questions/count.json')
        self.spatial_outputJson = os.path.abspath('../questions/spatial.json')
        self.spatiallogical_outputJson = os.path.abspath('../questions/spatiallogical.json')
        '''
        self.vocab_outputJson = os.path.abspath("../data/vocab.json")



    def createQueue(self):
        self.qns_exist = self.queryExist()
        #json.dump(self.qns_exist, open(self.exist_outputJson, 'w'))

        self.qns_count = self.queryCount()
        #json.dump(self.qns_count, open(self.count_outputJson, 'w'))

#        self.qns_spatial = self.querySpatial_relationship()
        #json.dump(self.qns_spatial, open(self.spatial_outputJson, 'w'))

#        self.qns_spatial_logic = self.querySpatial_logical()
        #json.dump(self.qns_spatial_logic, open(self.spatiallogical_outputJson, 'w'))
        
        qns = self.qns_exist + self.qns_count # +self.qns_spatial +self.qns_spatial_logic
        #json.dump(qns, open(self.question_outputJson, 'w'))
        return qns



    def queryExist(self):
        qns = []
        for obj in self.blacklist_objects:
            if obj not in self.obj_exist_list:
                qns.append(self. questionObjectBuilder(
                    'exist', obj, 'no', q_type='exist_negative'))
            else: 
                qns.append(self. questionObjectBuilder(
                    'exist', obj, 'yes', q_type='exist_positive'))
        return qns

    def queryCount(self):
        qns = []
        for obj in self.blacklist_objects:
            if obj not in self.obj_exist_list:
                qns.append(self. questionObjectBuilder(
                    'count', obj, '0', q_type='count_negative'))
            else: 
                qns.append(self. questionObjectBuilder(
                    'count', obj, str(self.obj_count[obj]), q_type='count_positive'))
        return qns


    def querySpatial_relationship(self):
        qns = []
        for obj1 in self.blacklist_objects['spatial_relationship']:
            if obj1 not in self.obj_exist_list:
                qns.append(self. questionObjectBuilder(
                    'spatial_relationship', obj1, 'Nothing', q_type='spatial_negative'))
            else: 
                under_obj ='Nothing'
                under_distance = 0.1
                obj1_index = self.get_obj_index(obj1)
                answer_index = 0
                for obj2 in self.obj_exist_list:
                    if obj2 in self.blacklist_objects['spatial_under']:
                        obj2_index = self.get_obj_index(obj2)
                        for index1 in obj1_index:
                            for index2 in obj2_index:
                                pos1 = self.obj_dic[index1]['position']
                                pos2 = self.obj_dic[index2]['position']
                                if pos1[2] > pos2[2]:  #under
                                    dis = self.distance_cal(pos1,pos2)  #closer
                                    if dis < under_distance:
                                        under_distance = dis
                                        under_obj = obj2
                                        answer_index = index2
                if under_obj is 'Nothing':
                    qns.append(self. questionObjectBuilder(
                    'spatial_relationship', obj1, 'Nothing', q_type='spatial_negative'))
                else:
                    qns.append(self. questionObjectBuilder(
                    'spatial_relationship', obj1, under_obj, q_type='spatial_positive',ans_index=answer_index))
        return qns

    def querySpatial_logical(self):
        qns = []
        under_distance =0.1
        for obj1 in self.blacklist_objects['spatial_relationship']:
            for obj2 in self.blacklist_objects['spatial_logic']:
                if obj1 != obj2:
                    if obj1 not in self.obj_exist_list or obj2 not in self.obj_exist_list:
                        qns.append(self.questionObjectBuilderForlogical(
                        'spatial_logic', obj1, obj2, 'under', 'no', q_type='spatial_logical_negative'))
                    else:
                        obj1_index = self.get_obj_index(obj1)
                        obj2_index = self.get_obj_index(obj2)
                        under_distance = 0.1
                        is_under = 0
                        answer_index = 0
                        for index1 in obj1_index:
                            for index2 in obj2_index:
                                pos1 = self.obj_dic[index1]['position']
                                pos2 = self.obj_dic[index2]['position']
                                if pos1[2] > pos2[2]:  #under
                                    dis = self.distance_cal(pos1,pos2)  #closer
                                    if dis < under_distance:
                                        under_distance = dis
                                        is_under = 1
                                        answer_index = index2
                        if is_under ==1:
                            qns.append(self.questionObjectBuilderForlogical(
                            'spatial_logic', obj1, obj2, 'under', 'yes', q_type='spatial_logical_positive',ans_index= answer_index))
                        else:
                            qns.append(self.questionObjectBuilderForlogical(
                             'spatial_logic', obj1, obj2, 'under', 'no', q_type='spatial_logical_negative'))                                                   
        return qns




    def queryExist_logic(self):
        qns = []
        for obj_1 in self.blacklist_objects['exist']:
            for obj_2 in self.blacklist_objects['exist']:
                if obj_1 != obj_2:
                    if obj_1  not in self.obj_exist_list and obj_2 not in self.obj_exist_list:
                        qns.append(self.questionObjectBuilderForlogical(
                          'exist_logic', obj_1, obj_2 ,'and',  'no', q_type='exist_logic_negative'))

                        qns.append(self.questionObjectBuilderForlogical(
                          'exist_logic', obj_1, obj_2 ,'or',  'no', q_type='exist_logic_negative'))
                    
                    elif obj_1 in self.obj_exist_list or obj_2 in self.obj_exist_list:                                               
                        qns.append(self.questionObjectBuilderForlogical(
                          'exist_logic', obj_1, obj_2 ,'and',  'no', q_type='exist_logic_negative'))

                        qns.append(self.questionObjectBuilderForlogical(
                          'exist_logic', obj_1, obj_2 ,'or',  'yes', q_type='exist_logic_positive'))
                        
                    else:
                        qns.append(self.questionObjectBuilderForlogical(
                          'exist_logic', obj_1, obj_2 ,'and',  'yes', q_type='exist_logic_positive'))

                        qns.append(self.questionObjectBuilderForlogical(
                          'exist_logic', obj_1, obj_2 ,'or',  'yes', q_type='exist_logic_positive'))
        return qns


    





    def questionObjectBuilder(self, template, object_name, a_str,q_type=None,ans_index = 0):
        if q_type == None:
            q_type = template

        q_str = self.templates[template]   
        q_str = self.q_str_builder.prepareString(q_str, object_name)
        return {
            'obj':
            object_name,
            'question':
            q_str,
            'answer':
            a_str,
            'type':
            q_type,
            'ans_index':
            ans_index
        }



    def questionObjectBuilderForlogical(self,template,obj1,obj2,logic_str,a_str,q_type=None,ans_index = 0):
        if q_type == None:
            q_type = template

        q_str = self.templates[template]   
        q_str = self.q_str_builder.prepareStringForLogic(q_str,obj1,obj2,logic_str)
        return {
            'obj':
            [obj1,obj2],
            'question':
            q_str,
            'answer':
            a_str,
            'type':
            q_type,
            'ans_index':
            ans_index
        }


    def tokenize(self,seq,delim=' ',punctToRemove=None,addStartToken=True,addEndToken=True):

        if punctToRemove is not None:
            for p in punctToRemove:
                seq = str(seq).replace(p, '')

        tokens = str(seq).split(delim)
        if addStartToken:
            tokens.insert(0, '<START>')

        if addEndToken:
            tokens.append('<END>')

        return tokens


    def buildVocab(self,sequences,
               minTokenCount=1,
               delim=' ',
               punctToRemove=None,
               addSpecialTok=False):
        SPECIAL_TOKENS = {
            '<NULL>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
        }

        tokenToCount = {}
        for seq in sequences:
            seqTokens = self.tokenize(seq,delim=delim,punctToRemove=punctToRemove,addStartToken=False,addEndToken=False)
            for token in seqTokens:
                if token not in tokenToCount:
                    tokenToCount[token] = 0
                tokenToCount[token] += 1

        tokenToIdx = {}
        if addSpecialTok == True:
            for token, idx in SPECIAL_TOKENS.items():
                tokenToIdx[token] = idx
        for token, count in sorted(tokenToCount.items()):
            if count >= minTokenCount:
                tokenToIdx[token] = len(tokenToIdx)

        return tokenToIdx



    def encode(self,seqTokens, tokenToIdx, allowUnk=False):
        seqIdx = []
        for token in seqTokens:
            if token not in tokenToIdx:
                if allowUnk:
                    token = '<UNK>'
                else:
                    raise KeyError('Token "%s" not in vocab' % token)
            seqIdx.append(tokenToIdx[token])
        return seqIdx


    def decode(self,seqIdx, idxToToken, delim=None, stopAtEnd=True):
        tokens = []
        for idx in seqIdx:
            tokens.append(idxToToken[idx])
            if stopAtEnd and tokens[-1] == '<END>':
                break
        if delim is None:
            return tokens
        else:
            return delim.join(tokens)


    def create_vocab(self):
        question_file = open(self.question_outputJson,'r',encoding='utf-8')
        questions = json.load(question_file)
        answerTokenToIdx = self.buildVocab((str(q['answer']) for q in questions
                                       if q['answer'] != 'NIL'))
        questionTokenToIdx = self.buildVocab(
            (q['question'] for q in questions if q['answer'] != 'NIL'),
            punctToRemove=['?'],
            addSpecialTok=True)

        vocab = {
            'questionTokenToIdx': questionTokenToIdx,
            'answerTokenToIdx': answerTokenToIdx,
        }
        json.dump(vocab, open(self.vocab_outputJson, 'w'))
        return vocab


    