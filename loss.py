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
import random
import pdb

def get_losses_pretrain(conf, input_part_pcs, pred_part_poses, gt_part_poses, input_part_valids):

	trans_l2_loss_per_data = get_trans_l2_loss(pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], input_part_valids)
	rot_l2_loss_per_data = get_rot_l2_loss(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids)
	rot_cd_loss_per_data = get_rot_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, conf.device)
	shape_cd_loss_per_data = get_shape_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3], conf.device)
	total_cd_loss_per_data, acc = get_total_cd_loss(input_part_pcs, pred_part_poses[:, :, 3:], gt_part_poses[:, :, 3:], input_part_valids, pred_part_poses[:, :, :3], gt_part_poses[:, :, :3])

	# total_cd_loss_per_data = total_cd_loss_per_data.detach()
	# acc = acc.detach()
	# Valid number is obtained. 
	valid_number = input_part_valids.sum(-1).float().cpu()  # B
	# Accuracy.
	acc = torch.tensor(acc)
	acc = acc.sum(-1).float()
	acc_rate = acc / valid_number

	shape_cd_loss = shape_cd_loss_per_data.mean()
	trans_l2_loss = trans_l2_loss_per_data.mean()
	rot_l2_loss = rot_l2_loss_per_data.mean()
	rot_cd_loss = rot_cd_loss_per_data.mean()
	total_cd_loss = total_cd_loss_per_data.mean()

	# cur_losses.
	cur_losses = {}
	
	cur_losses['shape_cd_loss'] = shape_cd_loss
	cur_losses['trans_l2_loss'] = trans_l2_loss
	cur_losses['total_cd_loss'] = total_cd_loss.detach()
	cur_losses['acc'] = acc.detach()
	cur_losses['valid_num'] = valid_number
	cur_losses['rot_l2_loss'] = rot_l2_loss
	cur_losses['rot_cd_loss'] = rot_cd_loss
	cur_losses['total_loss'] =  trans_l2_loss * conf.loss_weight_trans_l2 + \
								rot_l2_loss * conf.loss_weight_rot_l2 + \
								rot_cd_loss * conf.loss_weight_rot_cd + \
								shape_cd_loss * conf.loss_weight_shape_cd

	return cur_losses
	
def update_losses(total_losses, cur_losses, type_ = 'sum'):

	if(type_ == 'sum'):
		for key in total_losses.keys():
			total_losses[key] += cur_losses[key]
	elif(type_ == 'min'):
		for key in total_losses.keys():
			if key == 'acc':
				total_losses[key] = total_losses[key].max(cur_losses[key])
			else:
				total_losses[key] = total_losses[key].min(cur_losses[key])
	else:
		raise ValueError

	return total_losses


def get_rot_l2_loss(pts, quat1, quat2, valids):
	batch_size = pts.shape[0]
	num_point = pts.shape[2]

	pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
	pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

	loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

	loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
	return loss_per_data

def get_trans_l2_loss(trans1, trans2, valids):
	loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)

	loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
	return loss_per_data

# get rot_cd loss.
def get_rot_cd_loss(pts, quat1, quat2, valids, device):
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

def get_total_cd_loss(pts, quat1, quat2, valids, center1, center2):
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
	loss_per_data = loss_per_data.cuda()
	acc = [[0 for i in range(num_part)]for j in range(batch_size)]
	for i in range(batch_size):
		for j in range(num_part):
			if loss_per_data[i,j] < thre and valids[i,j]:
				acc[i][j] = 1
	loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)
	return loss_per_data , acc

def get_shape_cd_loss(pts, quat1, quat2, valids, center1, center2, device):
	max_num_part = 20
	batch_size = pts.shape[0]
	num_part = pts.shape[1]
	num_point = pts.shape[2]

	# Pts is being obtained by stacking extra_pts and pts. 
	# extra_pts = torch.zeros([batch_size, max_num_part - num_part, num_point, 3]).cuda()
	# pts = torch.cat([pts, extra_pts], dim = 1)

	# Center1, center2.
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
	loss_per_data = (num_part/max_num_part)*(torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1))
	loss_per_data = loss_per_data.to(device)
	return loss_per_data





