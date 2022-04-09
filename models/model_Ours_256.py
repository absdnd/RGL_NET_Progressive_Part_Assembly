"""
    Scene Graph to predict the pose of each part
    adjust relation using the t in last iteration
    Input:
        relation matrxi of parts,part valids, part point clouds, instance label, iter_ind, pred_part_poses:      B x P x P, B x P, B x P x N x 3, B x P x P , (1 or 2 or 3) , B x P x 7
    Output:
        R and T:                B x P x (3 + 4)
    Losses:
        Center L2 Loss, Rotation L2 Loss, Rotation Chamder-Distance Loss
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys, os
import ipdb
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from cd.chamfer import chamfer_distance
from quaternion import qrot
import ipdb
import pdb
from scipy.optimize import linear_sum_assignment
sys.path.append(os.path.join(BASE_DIR, '../../exp_GAN/models/sampling'))
import random
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        # pdb.set_trace()
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.detach().clone().requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Gated Recurrent Unit. 
        self.gru = nn.GRU(input_size, hidden_size, n_layer, bidirectional=bidirectional, dropout=0.2 if n_layer==2 else 0)

        self.init_hidden = self.initHidden()

    def forward(self, input, init_hidden):
        # [seq_len, batch_size, feature_dim]
        # pdb.set_trace()
        """
        :param input: (seq_len, batch_size, feature_dim)
        :return:
            output: (seq_len, batch, num_directions * hidden_size)
            h_n: (num_layers * num_directions, batch, hidden_size)
        """
        # pdb.set_trace()

        output, hidden = self.gru(input, init_hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(self.n_layer * self.num_directions, batch_size, self.hidden_size, requires_grad=False)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False):
        super(DecoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.n_units_hidden1 = 256
        self.n_units_hidden2 = 128
        # pdb.set_trace()
        # GRU layer being used using input_size.


        self.gru = nn.GRU(input_size, hidden_size, n_layer, bidirectional=bidirectional, dropout=0.2 if n_layer==2 else 0)

        # Locked dropout is left fixed at some value of 0.2
        self.lockdrop = LockedDropout()
        self.dropout_i = 0.2
        self.dropout_o = 0.2

        self.init_input = self.initInput()

    def forward(self, input, hidden, lens):
        """
        :param input: (1, batch, output_size)
        :param hidden: initial hidden state
        :return:
            output: (1, batch, num_directions * hidden_size)
            hidden: (num_layers * 1, batch, hidden_size)
            output_seq: (batch, 1 * output_size)
            stop_sign: (batch, 1)
        """
        input = self.lockdrop(input, self.dropout_i)
        packed_seq = pack_padded_sequence(input, lengths = lens, batch_first = True, enforce_sorted = False)
        output, hidden = self.gru(packed_seq, hidden)
        output, lens = pad_packed_sequence(output)
        # hidden : (num_layers * 1, batch, hidden_size)
        # hidden1, hidden2 = torch.split(hidden, 1, 0)
        # output_code = self.linear1(hidden.squeeze(0))
        # stop_sign = self.linear3(hidden.squeeze(0))
        # output_seq = torch.cat([output_code, output_param], dim=1)
        # output_seq = output

        return output, hidden

    def initInput(self):
        # initial_code = torch.zeros((1, 1, self.input_size - 6), requires_grad=False)
        # initial_param = torch.tensor([0.5, 0.5, 0.5, 1, 1, 1], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)
        # initial = torch.cat([initial_code, initial_param], dim=2)
        initial = torch.zeros((1, 1, self.input_size), requires_grad=False)
        return initial

# Sequential Decoder being used over here. 
class SeqDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, conf):
        super(SeqDecoder, self).__init__()
        self.n_layer = 1
        self.forward_decoder = DecoderRNN(input_size, hidden_size, self.n_layer, bidirectional = True)
        self.hidden_size = hidden_size
        self.conf = conf

    # Forward pass on this network. 
    def forward(self, decoder_input, decoder_hidden, lens):
        batch_size = len(lens)
        # pdb.set_trace()
        decoder_output, decoder_hidden = self.forward_decoder(decoder_input.float(), decoder_hidden.float(), lens)
        decoder_output = decoder_output.transpose(0,1)
        return decoder_output

    # def initHidden(self, batch_size=1):
    #     random_noise = np.random.normal(loc=0.0, scale=self.conf.noise_scale, size=[self.n_layer, batch_size, self.conf.noise_dim]).astype(np.float32)  # B x P x 16
    #     random_noise = torch.tensor(random_noise).cuda()

    #     random_noise = random_noise.repeat(2, 1, 1)
    #     init_hidden = torch.zeros(2*self.n_layer , batch_size, self.hidden_size - self.conf.noise_dim, requires_grad=False).cuda()
    #     input_hidden = torch.cat([init_hidden, random_noise], dim = -1)
    #     return input_hidden


class MLP2(nn.Module):
    def __init__(self, feat_len):
        super(MLP2, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.mlp1 = nn.Linear(1024, feat_len)
        self.bn6 = nn.BatchNorm1d(feat_len)

    """
        Input: B x N x 3 (B x P x N x 3)
        Output: B x F (B x P x F)
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]

        x = torch.relu(self.bn6(self.mlp1(x)))
        return x


class MLP3(nn.Module):
    def __init__(self, feat_len):
        super(MLP3, self).__init__()

        self.conv1 = nn.Conv1d(2*feat_len, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_len, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_len)


    """
        Input: (B x P) x P x 2F
        Output: (B x P) x P x F
    """

    def forward(self, x):
        #num_part = x.shape[1]

        x = x.permute(0, 2, 1)
        # x = self.conv1(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        return x


class MLP4(nn.Module):
    def __init__(self, feat_len):
        super(MLP4, self).__init__()

        self.conv1 = nn.Conv1d(2*feat_len, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_len, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_len)


    """
        Input: (B x P) x P x 2F
        Output: (B x P) x P x F
    """

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        return x


class MLP5(nn.Module):

    def __init__(self, feat_len):
        super(MLP5, self).__init__()

        self.mlp = nn.Linear(feat_len, 512)

        self.trans = nn.Linear(512, 3)

        self.quat = nn.Linear(512, 4)
        self.quat.bias.data.zero_()

    """
        Input: * x F    (* denotes any number of dimensions, used as B x P here)
        Output: * x 7   (* denotes any number of dimensions, used as B x P here)
    """

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))

        trans = torch.tanh(self.trans(feat))  # consider to remove torch.tanh if not using PartNet normalization

        quat_bias = feat.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()

        out = torch.cat([trans, quat], dim=-1)
        return out

class R_Predictor(nn.Module):
    def __init__(self):
        super(R_Predictor, self).__init__()
        # Multi-Layer Perceptron is being used.
        self.mlp1 = nn.Linear(128 + 128, 256)
        self.mlp2 = nn.Linear(256,512)
        self.mlp3 = nn.Linear(512,1)
        
    def forward(self, x):
        x = torch.relu(self.mlp1(x)) 
        x = torch.relu(self.mlp2(x)) 
        x = torch.sigmoid(self.mlp3(x)) 
        return x

class Pose_extractor(nn.Module):
    def __init__(self):
        super(Pose_extractor, self).__init__()
        self.mlp1 = nn.Linear(7, 256)
        self.mlp2 = nn.Linear(256,128)
        
    def forward(self, x):
        x = torch.relu(self.mlp1(x)) 
        x = torch.relu(self.mlp2(x)) 
        return x

# Network being defined here. 
class Network(nn.Module):
    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf
        self.mlp2 = MLP2(conf.feat_len)
        self.mlp3_1 = MLP3(conf.feat_len)
        self.mlp3_2 = MLP3(conf.feat_len)
        self.mlp3_3 = MLP3(conf.feat_len)
        #self.mlp3_4 = MLP3(conf.feat_len)
        #self.mlp3_5 = MLP3(conf.feat_len)

        self.mlp4_1 = MLP4(conf.feat_len) 
        self.mlp4_2 = MLP4(conf.feat_len) 
        self.mlp4_3 = MLP4(conf.feat_len) 
        #self.mlp4_4 = MLP4(conf.feat_len) 
        #self.mlp4_5 = MLP4(conf.feat_len) 

       	# Multi-Layer perceptron type 5.
        self.mlp5_1 = MLP5(conf.feat_len * 2 +  7 + conf.max_num_part)
        self.mlp5_2 = MLP5(conf.feat_len * 2 +  7 + conf.max_num_part)
        self.mlp5_3 = MLP5(conf.feat_len * 2 +  7 + conf.max_num_part)
        #self.mlp5_4 = MLP5(conf.feat_len * 2 +  7 + conf.max_num_part + 16)
        #self.mlp5_5 = MLP5(conf.feat_len * 2 +  7 + conf.max_num_part + 16)

        # The multi-layer perceptrons of different layers are being put together.

        # self.mlp5_4 = MLP5(conf.feat_len * 2 + conf.max_num_part + 7 + 16)
        # self.mlp5_5 = MLP5(conf.feat_len * 2 + conf.max_num_part + 7 + 16)

        self.relation_predictor = R_Predictor()
        self.relation_predictor_dense = R_Predictor()
        self.pose_extractor = Pose_extractor()

        self.feature_refiner = nn.Linear(2*conf.feat_len, conf.feat_len)
        self.seq2seqae_1 = SeqDecoder(2*conf.feat_len, conf.hidden_size, conf)
        self.seq2seqae_2 = SeqDecoder(2*conf.feat_len, conf.hidden_size, conf)
        self.seq2seqae_3 = SeqDecoder(2*conf.feat_len, conf.hidden_size, conf)
        self.seq2seqae_4 = SeqDecoder(2*conf.feat_len, conf.hidden_size, conf)
        self.seq2seqae_5 = SeqDecoder(2*conf.feat_len, conf.hidden_size, conf)

        # Usin random noise. 
        
    """
        Input: B x P x P, B x P, B x P x N x 3, B x P x P
        Output: B x P x (3 + 4)
    """

    def forward(self, conf, relation_matrix, part_valids, part_pcs, cur_instance_label, class_list, shape_id = 0):

        batch_size = part_pcs.shape[0]
        num_part = part_pcs.shape[1]
        lens = part_valids.sum(1)

        # pdb.set_trace()

        relation_matrix = relation_matrix.double()[:, :num_part, :num_part]

        instance_label = torch.zeros([batch_size, num_part, conf.max_num_part], requires_grad = False).cuda()
        instance_label[:, :num_part, :num_part] = cur_instance_label

        valid_matrix = copy.copy(relation_matrix)

        pred_poses = torch.zeros((batch_size, num_part, 7)).to(conf.device)
        total_pred_poses = []

        part_feats = self.mlp2(part_pcs.view(batch_size * num_part, -1, 3)).view(batch_size, num_part, -1)  # output: B x P x F
        local_feats = part_feats

        random_noise = np.random.normal(loc=0.0, scale=self.conf.noise_scale, size=[1, batch_size, conf.noise_dim]).astype(np.float32)  # B x P x noise_dim
        random_noise = torch.tensor(random_noise, requires_grad = False).cuda()

        random_noise = random_noise.repeat(2, 1, 1)
        init_hidden = torch.zeros(2, batch_size, conf.hidden_size - conf.noise_dim, requires_grad=False).cuda()
        decoder_input_hidden = torch.cat([init_hidden, random_noise], dim = -1)

        for iter_ind in range(self.conf.iter):
            if iter_ind >= 1 :
                cur_poses = copy.copy(pred_poses).double()            
                pose_feat = self.pose_extractor(cur_poses.float())

                if iter_ind % 2 == 1: 
                    for i in range(batch_size):
                        for j in range(len(class_list[i])):
                            cur_pose_feats = pose_feat[i,class_list[i][j]]
                            cur_pose_feat = cur_pose_feats.max(dim = -2)[0] 
                            pose_feat[i,class_list[i][j]]=cur_pose_feat
                            part_feats_copy = copy.copy(part_feats)
                            with torch.no_grad():
                                part_feats_copy[i,class_list[i][j]] = part_feats_copy[i, class_list[i][j]].max(dim = -2)[0]

                pose_featA = pose_feat.unsqueeze(1).repeat(1,num_part,1,1)
                pose_featB = pose_feat.unsqueeze(2).repeat(1,1,num_part,1)
                input_relation = torch.cat([pose_featA,pose_featB],dim = -1).float()
                if iter_ind % 2 == 0:
                    new_relation = self.relation_predictor_dense(input_relation.view(batch_size,-1,256)).view(batch_size,num_part,num_part)
                elif iter_ind % 2 == 1:
                    new_relation = self.relation_predictor(input_relation.view(batch_size,-1,256)).view(batch_size,num_part,num_part)
                relation_matrix = new_relation.double() * valid_matrix

            if iter_ind>=1 and iter_ind%2==1: 
                part_feat1 = part_feats_copy.unsqueeze(2).repeat(1, 1, num_part, 1) # B x P x P x F
                part_feat2 = part_feats_copy.unsqueeze(1).repeat(1, num_part, 1, 1) # B x P x P x F
            else:
                part_feat1 = part_feats.unsqueeze(2).repeat(1, 1, num_part, 1) # B x P x P x F
                part_feat2 = part_feats.unsqueeze(1).repeat(1, num_part, 1, 1) # B x P x P x F
            
            input_3 = torch.cat([part_feat1, part_feat2], dim=-1) # B x P x P x 2F

            if iter_ind == 0:
                mlp3 = self.mlp3_1
                mlp4 = self.mlp4_1
                mlp5 = self.mlp5_1
                mlp_seq = self.seq2seqae_1
            elif iter_ind == 1:
                mlp3 = self.mlp3_2
                mlp4 = self.mlp4_2
                mlp5 = self.mlp5_2
                mlp_seq = self.seq2seqae_2
            elif iter_ind == 2:
                mlp3 = self.mlp3_3
                mlp4 = self.mlp4_3
                mlp5 = self.mlp5_3
                mlp_seq = self.seq2seqae_3
            elif iter_ind == 3:
                mlp3 = self.mlp3_4
                mlp4 = self.mlp4_4
                mlp5 = self.mlp5_4
                mlp_seq = self.seq2seqae_4
            elif iter_ind == 4:
                mlp3 = self.mlp3_5
                mlp4 = self.mlp4_5
                mlp5 = self.mlp5_5
                mlp_seq = self.seq2seqae_5

            part_relation = mlp3(input_3.view(batch_size * num_part, num_part, -1)).view(batch_size, num_part,
                                     num_part, -1) # B x P x P x F

            part_message = part_relation.double() * relation_matrix.unsqueeze(3).double() # B x P x P x F
            part_message = part_message.sum(dim=2) # B x P x F
            norm = relation_matrix.sum(dim=-1) # B x P
            delta = 1e-6
            normed_part_message = part_message / (norm.unsqueeze(dim=2) + delta) # B x P x F

            input_4 = torch.cat([normed_part_message.double(), part_feats.double()], dim=-1) # B x P x 2F

            # pdb.set_trace()
            decoder_output = mlp_seq(input_4, decoder_input_hidden, lens)
            pred_num_part = decoder_output.shape[1]
            concat_tensor = torch.zeros([batch_size, num_part - pred_num_part, 2*conf.feat_len], requires_grad = False).cuda()

            decoder_output  = torch.cat([decoder_output, concat_tensor], dim = 1)
            part_feats = self.feature_refiner(decoder_output)

            output = torch.cat([part_feats, local_feats, pred_poses, instance_label], dim = -1)
            pred_poses = mlp5(output)
            total_pred_poses.append(pred_poses)

        return total_pred_poses

    """
            Input: * x N x 3, * x 3, * x 4, * x 3, * x 4,
            Output: *, *  (two lists)
    """

    def linear_assignment(self, pts, centers1, quats1, centers2, quats2):
        import random
        pts_to_select = torch.tensor(random.sample([i for i  in range(1000)],100))
        pts = pts[:,pts_to_select] 
        cur_part_cnt = pts.shape[0]
        num_point = pts.shape[1]

        with torch.no_grad():

            cur_quats1 = quats1.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers1 = centers1.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts1 = qrot(cur_quats1, pts) + cur_centers1

            cur_quats2 = quats2.unsqueeze(1).repeat(1, num_point, 1)
            cur_centers2 = centers2.unsqueeze(1).repeat(1, num_point, 1)
            cur_pts2 = qrot(cur_quats2, pts) + cur_centers2

            cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cur_part_cnt, 1, 1).view(-1, num_point, 3)
            cur_pts2 = cur_pts2.unsqueeze(0).repeat(cur_part_cnt, 1, 1, 1).view(-1, num_point, 3)
            dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
            dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cur_part_cnt, cur_part_cnt)
            # pdb.set_trace()
            rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

        return rind, cind


    def get_trans_l2_loss(self, trans1, trans2, valids):
        loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """

    def get_rot_l2_loss(self, pts, quat1, quat2, valids):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data

    """
        Input: B x P x N x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """

    # Get rot_cd_loss. 
    def get_rot_cd_loss(self, pts, quat1, quat2, valids, device):
        batch_size = pts.shape[0]
        num_point = pts.shape[2]

        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)

        loss_per_data = loss_per_data.to(device)
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data  
        
    def get_total_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part =  pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        loss_per_data = loss_per_data.view(batch_size, -1)
        
        thre = 0.01
        loss_per_data = loss_per_data.to(device)
        acc = [[0 for i in range(num_part)]for j in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_part):
                if loss_per_data[i,j] < thre and valids[i,j]:
                    acc[i][j] = 1
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
        return loss_per_data , acc

    def get_shape_cd_loss(self, pts, quat1, quat2, valids, center1, center2, device):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        pts1 = pts1.view(batch_size,num_part*num_point,3)
        pts2 = pts2.view(batch_size,num_part*num_point,3)
        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        valids = valids.unsqueeze(2).repeat(1,1,1000).view(batch_size,-1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        
        loss_per_data = loss_per_data.to(device)
        return loss_per_data

        """
            output : B
        """
    def get_sym_point(self, point, x, y, z):

        if x:
            point[0] = - point[0]
        if y:
            point[1] = - point[1]
        if z:
            point[2] = - point[2]

        return point.tolist()

    def get_possible_point_list(self, point, sym):
        sym = torch.tensor([1.0,1.0,1.0]) 
        point_list = []
        #sym = torch.tensor(sym)
        if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
        elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
        elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
        elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
        elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
        elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
        else:
            point_list.append(self.get_sym_point(point, 0, 0, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 0))
            point_list.append(self.get_sym_point(point, 0, 1, 0))
            point_list.append(self.get_sym_point(point, 0, 0, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 0))
            point_list.append(self.get_sym_point(point, 1, 0, 1))
            point_list.append(self.get_sym_point(point, 0, 1, 1))
            point_list.append(self.get_sym_point(point, 1, 1, 1))

        return point_list
    def get_min_l2_dist(self, list1, list2, center1, center2, quat1, quat2):

        list1 = torch.tensor(list1) # m x 3
        list2 = torch.tensor(list2) # n x 3
        #print(list1[0])
        #print(list2[0])
        len1 = list1.shape[0]
        len2 = list2.shape[0]
        center1 = center1.unsqueeze(0).repeat(len1, 1)
        center2 = center2.unsqueeze(0).repeat(len2, 1)
        quat1 = quat1.unsqueeze(0).repeat(len1, 1)
        quat2 = quat2.unsqueeze(0).repeat(len2, 1)
        list1 = list1.to(self.conf.device)
        list2 = list2.to(self.conf.device)
        list1 = center1 + qrot(quat1, list1)
        list2 = center2 + qrot(quat2, list2)
        mat1 = list1.unsqueeze(1).repeat(1, len2, 1)
        mat2 = list2.unsqueeze(0).repeat(len1, 1, 1)
        mat = (mat1 - mat2) * (mat1 - mat2)
        #ipdb.set_trace()
        mat = mat.sum(dim=-1)
        return mat.min()

    """    
        Contact point loss metric
        Date: 2020/5/22
        Input B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput B
    """
    def get_contact_point_loss(self, center, quat, contact_points, sym_info):

        batch_size = center.shape[0]
        num_part = center.shape[1]
        contact_point_loss = torch.zeros(batch_size)
        total_num = 0
        count = 0
        for b in range(batch_size):
            #print("Shape id is", b)
            sum_loss = 0
            for i in range(num_part):
                for j in range(num_part):
                    if contact_points[b, i, j, 0]:
                        contact_point_1 = contact_points[b, i, j, 1:]
                        contact_point_2 = contact_points[b, j, i, 1:]
                        sym1 = sym_info[b, i]
                        sym2 = sym_info[b, j]
                        point_list_1 = self.get_possible_point_list(contact_point_1, sym1)
                        point_list_2 = self.get_possible_point_list(contact_point_2, sym2)
                        dist = self.get_min_l2_dist(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :], quat[b, j, :])  # 1
                        #print(dist)
                        if dist < 0.01:
                            count += 1
                        total_num += 1
                        sum_loss += dist
            contact_point_loss[b] = sum_loss


        #print(count, total_num)
        return contact_point_loss, count, total_num
