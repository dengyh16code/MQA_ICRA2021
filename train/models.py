

from __future__ import division

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import scipy.sparse as sp
import numpy as np
import os

def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state

# ----------- act -----------

class QuestionLstmEncoder(nn.Module):    #output 128
    def __init__(self,
                 token_to_idx,
                 wordvec_dim=64,
                 rnn_dim=64,
                 rnn_num_layers=2,
                 rnn_dropout=0):
        super(QuestionLstmEncoder, self).__init__()
        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx['<NULL>']
        self.START = token_to_idx['<START>']
        self.END = token_to_idx['<END>']

        self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
        self.rnn = nn.LSTM(
            wordvec_dim,
            rnn_dim,
            rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        x_cpu = x.data.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data).long()
        idx = Variable(idx, requires_grad=False)

        hs, _ = self.rnn(self.embed(x))

        idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
        H = hs.size(2)
        return hs.gather(1, idx).view(N, H)



class RGBD_Encoder(nn.Module):
    def __init__(self,
               num_classes=32, 
               checkpoint_path_local= '../data/models/resnet101.pth'
    ):
        super(RGBD_Encoder, self).__init__()

        checkpoint_path = os.path.abspath(checkpoint_path_local)
        self.num_classes = num_classes
        res_model = torchvision.models.resnet101(pretrained=False)
        #print(res_model)
        res_model.load_state_dict(torch.load(checkpoint_path))
        
        self.rgb_layer = torch.nn.Sequential(*list(res_model.children())[:-4])    #remove fc layer
        
        
        res_model1 =  torchvision.models.resnet101(pretrained=False)
        res_model1.load_state_dict(torch.load(checkpoint_path))

        self.depth_layer = torch.nn.Sequential(*list(res_model1.children())[:-4])    #remove fc layer
        

        print('Loading resnet weights from %s' % checkpoint_path)


        self.conv_layer = nn.Sequential(
            nn.Conv2d(1024,512,1),
            nn.Conv2d(512,128,1),
            nn.Conv2d(128,self.num_classes,1),
            nn.ReLU()
        )

        for param in self.rgb_layer[1].parameters():  #fix bn
            param.requires_grad = False

        for param in self.depth_layer[1].parameters():
            param.requires_grad = False

        for name,param in self.named_parameters():   
            #if name in ['bn1','bn2','bn3']:
            if 'bn1' in name or 'bn2' in name or 'bn3' in name:
                param.requires_grad = False

    def forward(self,rgb_image,depth_image):
        output1 = self.rgb_layer(rgb_image)
        output2 = self.depth_layer(depth_image)
        input_conv = torch.cat([output1, output2], 1)
        output = self.conv_layer(input_conv)

        return output
        


class act_model(torch.nn.Module):
    def __init__(self,
                vocab,
                checkpoint_path='../data/models/resnet101.pth',
                question_wordvec_dim=64,
                question_hidden_dim=64,
                question_num_layers=2,
                question_dropout=0.5,
    ):

        super(act_model, self).__init__()

        rgbd_kwargs = {'num_classes': 32,  'checkpoint_path_local':checkpoint_path}
        self.rgbd_encode_model = RGBD_Encoder(**rgbd_kwargs)

        q_rnn_kwargs = {
            'token_to_idx': vocab['questionTokenToIdx'],
            'wordvec_dim': question_wordvec_dim,
            'rnn_dim': question_hidden_dim,
            'rnn_num_layers': question_num_layers,
            'rnn_dropout': question_dropout,
        }
        self.question_encode_model = QuestionLstmEncoder(**q_rnn_kwargs)

        self.ques_tr = nn.Sequential(
            nn.Linear(question_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        pointwise_in_channels = 32 + 32   #qustion + rgbd
        self.pointwise = nn.Sequential(
            nn.Conv2d(pointwise_in_channels,32,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,16,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            
            nn.Conv2d(16,9,1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )


    def forward(self,rgb_image, depth_image,ques):
        image_embedding = self.rgbd_encode_model(rgb_image,depth_image)   #32*28*28

        question_embedding = self.question_encode_model(ques)
        question_linear = self.ques_tr(question_embedding)
        question_reshaped = question_linear.view(-1, 32, 1, 1).repeat(1, 1, 28, 28) #32*28*28

        x = torch.cat((image_embedding, question_reshaped), dim=1)  #64*28*28
        output = self.pointwise(x) #9*28*28
        return output

# ----------- vqa -----------
def build_mlp(input_dim,
              hidden_dims,
              output_dim,
              use_batchnorm=False,
              dropout=0,
              add_sigmoid=1):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True))
        D = dim
    layers.append(nn.Linear(D, output_dim))

    if add_sigmoid == 1:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)

class MultitaskCNN(nn.Module):
    def __init__(
            self,
            num_classes=191,
            pretrained=True,
            checkpoint_path='..data/models/03_13_h3d_hybrid_cnn.pt'
    ):
        super(MultitaskCNN, self).__init__()

        self.num_classes = num_classes
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d())

        self.encoder_seg = nn.Conv2d(512, self.num_classes, 1)
        self.encoder_depth = nn.Conv2d(512, 1, 1)
        self.encoder_ae = nn.Conv2d(512, 3, 1)

        self.score_pool2_seg = nn.Conv2d(16, self.num_classes, 1)
        self.score_pool3_seg = nn.Conv2d(32, self.num_classes, 1)

        self.score_pool2_depth = nn.Conv2d(16, 1, 1)
        self.score_pool3_depth = nn.Conv2d(32, 1, 1)

        self.score_pool2_ae = nn.Conv2d(16, 3, 1)
        self.score_pool3_ae = nn.Conv2d(32, 3, 1)

        self.pretrained = pretrained
        if self.pretrained == True:
            print('Loading CNN weights from %s' % checkpoint_path)
            checkpoint_path_abs = os.path.abspath(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path_abs, map_location={'cuda:0': 'cpu'})
            self.load_state_dict(checkpoint['model_state'])
            for param in self.parameters():
                param.requires_grad = False
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * (
                        m.out_channels + m.in_channels)
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):

        # assert self.training == False
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        return conv4.view(-1, 32 * 10 * 10)

class VqaLstmCnnAttentionModel(nn.Module):
    def __init__(self,
                 vocab,
                 checkpoint_path='../data/models/03_13_h3d_hybrid_cnn.pt',
                 image_feat_dim=64,
                 question_wordvec_dim=64,
                 question_hidden_dim=64,
                 question_num_layers=2,
                 question_dropout=0.5,
                 fc_use_batchnorm=False,
                 fc_dropout=0.5,
                 fc_dims=(64, )):
        super(VqaLstmCnnAttentionModel, self).__init__()

        cnn_kwargs = {'num_classes': 191, 'pretrained': True, 'checkpoint_path':checkpoint_path}
        self.cnn = MultitaskCNN(**cnn_kwargs)
        self.cnn_fc_layer = nn.Sequential(
            nn.Linear(32 * 10 * 10, 64), nn.ReLU(), nn.Dropout(p=0.5))

        q_rnn_kwargs = {
            'token_to_idx': vocab['questionTokenToIdx'],
            'wordvec_dim': question_wordvec_dim,
            'rnn_dim': question_hidden_dim,
            'rnn_num_layers': question_num_layers,
            'rnn_dropout': question_dropout,
        }
        self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)

        self.img_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        self.ques_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        classifier_kwargs = {
            'input_dim': 64,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answerTokenToIdx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
            'add_sigmoid': 0
        }
        self.classifier = build_mlp(**classifier_kwargs)

        self.att = nn.Sequential(
            nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(128, 1))
        # print('fuck!!')

    def forward(self, images, questions):

        N, T, _, _, _ = images.size()

        # bs x 5 x 3 x 224 x 224
        img_feats = self.cnn(images.contiguous().view(-1, images.size(2), images.size(3), images.size(4)))
        img_feats = self.cnn_fc_layer(img_feats)

        img_feats_tr = self.img_tr(img_feats)

        ques_feats = self.q_rnn(questions)
        ques_feats_repl = ques_feats.view(N, 1, -1).repeat(1, T, 1)
        ques_feats_repl = ques_feats_repl.view(N * T, -1)

        ques_feats_tr = self.ques_tr(ques_feats_repl)

        ques_img_feats = torch.cat([ques_feats_tr, img_feats_tr], 1)

        att_feats = self.att(ques_img_feats)
        att_probs = F.softmax(att_feats.view(N, T), dim=1)
        att_probs2 = att_probs.view(N, T, 1).repeat(1, 1, 64)

        att_img_feats = torch.mul(att_probs2, img_feats.view(N, T, 64))
        att_img_feats = torch.sum(att_img_feats, dim=1)

        mul_feats = torch.mul(ques_feats, att_img_feats)

        scores = self.classifier(mul_feats)

        return scores, att_probs

        