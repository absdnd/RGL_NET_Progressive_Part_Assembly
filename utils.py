import os
import sys
import torch
import numpy as np
import importlib
sys.path.append("../")
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from cd.chamfer import chamfer_distance
from models import model_dynamic as model_def
from quaternion import qrot
from scipy.optimize import linear_sum_assignment
import pdb
from mayavi import mlab
import matplotlib.pyplot as plt
from PIL import Image
import subprocess

# obtain cmap and plot.

mlab.options.offscreen = True

def get_cmap(n, name = 'brg'):
	return plt.cm.get_cmap(name, n)

# visualize results using vis_path.
def visualize_results_full(vis_path, step, input_part_pcs, total_pred_part_poses, gt_part_poses, input_part_valids, k  = None):

	batch_size = input_part_pcs.shape[0]
	num_point = input_part_pcs.shape[2]

	for bs in range(batch_size):

		cur_input_part_cnt = int(input_part_valids[bs].sum().item())

		cur_pred_part_poses = total_pred_part_poses[:, bs, :cur_input_part_cnt]
		cur_gt_part_poses = gt_part_poses[bs, :cur_input_part_cnt]
		cur_input_part_pcs = input_part_pcs[bs, :cur_input_part_cnt]

		N = cur_input_part_cnt
		# pdb.set_trace()

		total_pred_part_pcs = []
		# Current predicted part point clouds are being collected. 
		for pred_pose in cur_pred_part_poses:
			pred_part_pcs = qrot(pred_pose[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + pred_pose[:, :3].unsqueeze(1).repeat(1, num_point, 1)
			total_pred_part_pcs.append(pred_part_pcs)
		
		# Ground truth part point clouds.
		gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

		cur_save_paths = []             
		for suffix in ['orig', 'prediction_1', 'prediction_2', 'prediction_3',  'gt']:
			cur_save_paths.append(os.path.join(vis_path, suffix + '_step_' + str(step) + '_item_' + str(bs)  + '.png'))

		part_pcs_to_visu = cur_input_part_pcs.cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))

		cmap = get_cmap(N)
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[0])
		mlab.clf()

		part_pcs_to_visu = total_pred_part_pcs[0].cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[1])
		mlab.clf()

		part_pcs_to_visu = total_pred_part_pcs[1].cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[2])
		mlab.clf()

		part_pcs_to_visu = total_pred_part_pcs[2].cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[3])
		mlab.clf()

		part_pcs_to_visu = gt_part_pcs.cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[4])
		mlab.clf()

		image_row = [Image.open(x) for x in cur_save_paths]
		widths, heights = zip(*(i.size for i in image_row))

		total_width = sum(widths)
		max_height = max(heights)

		# Image 'collage' is being founded.
		if 'collage' not in locals():
			collage = Image.new('RGB',(total_width, 1*max_height))

		x_offset = 0
		y_offset = 0
		if k is not None:
			save_string = os.path.join(vis_path, 'collage_k_{}_step_{}.png'.format(k, step + bs))
		else:
			save_string = os.path.join(vis_path, 'collage_step_' + str(step + bs) + '.png')

		for im in image_row:
			collage.paste(im, (x_offset, y_offset))
			x_offset += im.size[0]

		process = subprocess.run(['rm'] + cur_save_paths, stdout=subprocess.PIPE)
		collage.save(save_string) 

	mlab.close(all = True)

def visualize_results(vis_path, step, input_part_pcs, pred_part_poses, gt_part_poses, input_part_valids):

	batch_size = input_part_pcs.shape[0]
	num_point = input_part_pcs.shape[2]

	for bs in range(batch_size):

		cur_input_part_cnt = int(input_part_valids[bs].sum().item())

		cur_pred_part_poses = pred_part_poses[bs, :cur_input_part_cnt]
		cur_gt_part_poses = gt_part_poses[bs, :cur_input_part_cnt]
		cur_input_part_pcs = input_part_pcs[bs, :cur_input_part_cnt]

		N = cur_input_part_cnt


		pred_part_pcs = qrot(cur_pred_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + cur_pred_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)
		gt_part_pcs = qrot(cur_gt_part_poses[:, 3:].unsqueeze(1).repeat(1, num_point, 1), cur_input_part_pcs) + cur_gt_part_poses[:, :3].unsqueeze(1).repeat(1, num_point, 1)

		# Cur_save_paths = []
		cur_save_paths = []             
		for suffix in ['orig', 'prediction', 'gt']:
			cur_save_paths.append(os.path.join(vis_path, suffix + '_step_' + str(step) + '_item_' + str(bs)  + '.png'))

		part_pcs_to_visu = cur_input_part_pcs.cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))

		cmap = get_cmap(N)
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[0])
		mlab.clf()

		part_pcs_to_visu = pred_part_pcs.cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[1])
		mlab.clf()

		part_pcs_to_visu = gt_part_pcs.cpu().detach().numpy()
		mlab.figure(size = (250,250), bgcolor = (1,1,1))
		for i in range(N):
			mlab.points3d(part_pcs_to_visu[i,:,0], part_pcs_to_visu[i,:,1], part_pcs_to_visu[i,:,2], color=cmap(i)[:-1])
		mlab.savefig(cur_save_paths[2])
		mlab.clf()

		image_row = [Image.open(x) for x in cur_save_paths]
		widths, heights = zip(*(i.size for i in image_row))

		total_width = sum(widths)
		max_height = max(heights)

		# Image 'collage' is being founded.
		if 'collage' not in locals():
			collage = Image.new('RGB',(total_width, 1*max_height))

		x_offset = 0
		y_offset = 0

		save_string = os.path.join(vis_path, 'collage_step_' + str(step + bs) + '.png')
		for im in image_row:
			collage.paste(im, (x_offset, y_offset))
			x_offset += im.size[0]

		process = subprocess.run(['rm'] + cur_save_paths, stdout=subprocess.PIPE)
		collage.save(save_string) 
	mlab.close(all = True)


def obtain_same_class_list(batch_size, num_part, input_part_valids, part_ids):

	instance_label = torch.zeros(batch_size, num_part, num_part).cuda()
	same_class_list = []
	for i in range(batch_size):
		num_class = [ 0 for i in range(160) ]
		cur_same_class_list = [[] for i in range(160)]
		
		for j in range(num_part):
			cur_class = int(part_ids[i][j])
			if j < input_part_valids[i].sum():
				cur_same_class_list[cur_class].append(j)
			if cur_class == 0: continue
			cur_instance = int(num_class[cur_class])
			instance_label[i][j][cur_instance] = 1
			num_class[int(part_ids[i][j])] += 1
		for i in range(cur_same_class_list.count([])):
			cur_same_class_list.remove([])
		same_class_list.append(cur_same_class_list)

	return same_class_list, instance_label


def linear_assignment(pts, centers1, quats1, centers2, quats2):
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
		rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

	return rind, cind

# Match Similar parts using linear_assignment.
def match_similar(conf, network, input_part_pcs, pred_part_poses, gt_part_poses, match_ids, batch):

	# Match Similar parts. 
	# pdb.set_trace()
	cur_max_num_part = input_part_pcs.shape[1]
	for ind in range(len(batch[0])):
		cur_match_ids = match_ids[ind]

		for i in range(1,10):
			need_to_match_part = []
			for j in range(cur_max_num_part):
				if cur_match_ids[j] == i:
					need_to_match_part.append(j)


			if len(need_to_match_part) == 0:break

			cur_input_pts = input_part_pcs[ind,need_to_match_part]
			cur_pred_poses = pred_part_poses[ind,need_to_match_part]
			cur_pred_centers = cur_pred_poses[:,:3]
			cur_pred_quats = cur_pred_poses[:,3:]
			cur_gt_part_poses = gt_part_poses[ind,need_to_match_part]
			cur_gt_centers = cur_gt_part_poses[:,:3]
			cur_gt_quats = cur_gt_part_poses[:,3:]
			matched_pred_ids , matched_gt_ids = network.linear_assignment(cur_input_pts, cur_pred_centers, cur_pred_quats, cur_gt_centers, cur_gt_quats)
			match_pred_part_poses = pred_part_poses[ind,need_to_match_part][matched_pred_ids]
			pred_part_poses[ind,need_to_match_part] = match_pred_part_poses
			match_gt_part_poses = gt_part_poses[ind,need_to_match_part][matched_gt_ids]
			gt_part_poses[ind,need_to_match_part] = match_gt_part_poses


	return pred_part_poses, gt_part_poses

# AverageMeter to calculate average of loss.
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n
		self.avg = self.sum / self.count

# Prinout. 
def printout(flog, strout):
	print(strout)
	flog.write(strout + '\n')

def display_loss(data_split, epoch, conf, step, num_batch, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_loss):
	
	# loss_template.
	loss_template = "{}		{}/{}	|  {}/{}	|C {:.3f}	|L2Q {:.3f}	|Q {:.3f}	|P {:.3f}	|T {:.3f}".format(data_split, \
																						epoch, conf.num_epochs, \
																						step, num_batch, \
																						total_trans_l2_loss,\
																						total_rot_l2_loss, \
																						total_rot_cd_loss, \
																						total_shape_cd_loss,\
																						total_loss
																						)
	printout(conf.flog, loss_template)


# save checkpoints. 

def save_ckpt(model, model_name, dirname, epoch, optimizer, optimizer_name):
	# filename being saved. 
	filename = f'net_{model_name}.pth'
	filename = f'{epoch}_' + filename
	torch.save(model.state_dict(), os.path.join(dirname, filename))

	filename = 'checkpt.pth'
	filename = f'{epoch}_' + filename
	checkpt = {'epoch': epoch}
	checkpt[f'opt_{optimizer_name}'] = optimizer.state_dict()
	torch.save(checkpt, os.path.join(dirname, filename))

def save_checkpoint(models, model_names, dirname, epoch=None, prepend_epoch=False, optimizers=None, optimizer_names=None):
	if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
		raise ValueError('Number of models, model names, or optimizers does not match.')

	for model, model_name in zip(models, model_names):
		filename = f'net_{model_name}.pth'
		if prepend_epoch:
			filename = f'{epoch}_' + filename
		torch.save(model.state_dict(), os.path.join(dirname, filename))

	if optimizers is not None:
		filename = 'checkpt.pth'
		if prepend_epoch:
			filename = f'{epoch}_' + filename
		checkpt = {'epoch': epoch}
		for opt, optimizer_name in zip(optimizers, optimizer_names):
			checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
		torch.save(checkpt, os.path.join(dirname, filename))

def load_checkpoint_wo_optimizer(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
	if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
		raise ValueError('Number of models, model names, or optimizers does not match.')

	for model, model_name in zip(models, model_names):
		filename = f'net_{model_name}.pth'
		if epoch is not None:
			filename = f'{epoch}_' + filename
		model.load_state_dict(torch.load(os.path.join(dirname, filename)), strict=strict)

def load_checkpoint(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
	if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
		raise ValueError('Number of models, model names, or optimizers does not match.')

	for model, model_name in zip(models, model_names):
		filename = f'net_{model_name}.pth'
		if epoch is not None:
			filename = f'{epoch}_' + filename
		model.load_state_dict(torch.load(os.path.join(dirname, filename)), strict=strict)

	start_epoch = 0
	if optimizers is not None:
		filename = os.path.join(dirname, '{}_checkpt.pth'.format(epoch))
		if os.path.exists(filename):
			checkpt = torch.load(filename)
			start_epoch = checkpt['epoch']
			for opt, optimizer_name in zip(optimizers, optimizer_names):
				opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
			print(f'resuming from checkpoint {filename}')
		else:
			response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
			if response != 'y':
				sys.exit()

	return start_epoch

def optimizer_to_device(optimizer, device):
	for state in optimizer.state.values():
		for k, v in state.items():
			if torch.is_tensor(v):
				state[k] = v.to(device)

def get_model_module(model_version):
	importlib.invalidate_caches()
	return importlib.import_module('models.' + model_version)

def collate_feats(b):
	return list(zip(*b))

def collate_feats_with_none(b):
	b = filter (lambda x:x is not None, b)
	return list(zip(*b))

def worker_init_fn(worker_id):
	""" The function is designed for pytorch multi-process dataloader.
		Note that we use the pytorch random generator to generate a base_seed.
		Please try to be consistent.
		References:
			https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
	"""
	base_seed = torch.IntTensor(1).random_().item()
	#print(worker_id, base_seed)
	np.random.seed(base_seed + worker_id)

# pc is N x 3, feat is B x 10-dim
def transform_pc_batch(pc, feat, anchor=False):
	batch_size = feat.size(0)
	num_point = pc.size(0)
	pc = pc.repeat(batch_size, 1, 1)
	center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
	shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
	quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
	if not anchor:
		pc = pc * shape
	pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
	if not anchor:
		pc = pc + center
	return pc

def get_surface_reweighting_batch(xyz, cube_num_point):
	x = xyz[:, 0]
	y = xyz[:, 1]
	z = xyz[:, 2]
	assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
	np = cube_num_point // 6
	out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
					 (y*z).unsqueeze(dim=1).repeat(1, np*2), \
					 (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
	out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
	return out


import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def get_pc_center(pc):
	return np.mean(pc, axis=0)

def get_pc_scale(pc):
	return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0))**2, axis=1)))

def get_pca_axes(pc):
	axes = PCA(n_components=3).fit(pc).components_
	return axes

def get_chamfer_distance(pc1, pc2):
	dist = cdist(pc1, pc2)
	error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
	scale = get_pc_scale(pc1) + get_pc_scale(pc2)
	return error / scale

