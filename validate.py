import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from subprocess import call
from data_dynamic import PartNetPartDataset
import utils
torch.autograd.set_detect_anomaly(True)
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import render_using_blender as render_utils
# from train_utils import visualize_results
from quaternion import qrot
# import ipdb
import pdb
from models import model_dynamic as model_def
from loss import *
from tensorboardX import SummaryWriter

# Seed everything. 
SEED = 42

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# GlobalTrainConfig.
class GlobalTrainConfig:

	lr = 1e-3
	lr_decay_by = 0.9
	lr_decay_every = 5000
	weight_decay = 1e-5
	num_epochs = 1000
	device = 'cuda:0'
	level = '3'

	loss_weight_trans_l2 = 1.0
	loss_weight_rot_l2 = 0.0
	loss_weight_rot_cd = 10.
	loss_weight_shape_cd = 1.0

	feat_len = 256
	max_num_part = 20
	epochs = 1000
	seed = 3124256514

	no_console_log = False
	no_tb_log = False
	console_log_interval = 10

	iter = 5

	category = 'Chair'
	train_data_fn =  '{}.train.npy'.format(category) 
	val_data_fn  = '{}.val.npy'.format(category)
	
	data_dir = './prepare_data'
	log_dir = './runs'
	batch_size = 2

	exp_name = f'exp-trans-l2-{loss_weight_trans_l2}-rot_l2-{loss_weight_rot_l2}-rot_cd-{loss_weight_rot_cd}-shape_cd{loss_weight_shape_cd}'
	exp_dir = os.path.join(log_dir, exp_name)

	# log_path and vis_path.
	log_path = os.path.join(exp_dir, 'logs_v')
	vis_path = os.path.join(exp_dir, 'results')
	ckpt_path = os.path.join(exp_dir, 'checkpoints')

	# train_vis_path and val_vis_path.
	train_vis_path = os.path.join(vis_path, 'train')
	val_vis_path = os.path.join(vis_path, 'val')

	vis_interval = 10
	num_batch_every_visu = 5
	num_workers = 1
	vis = True

class Fitter: 
	def __init__(self, conf):
		self.conf = conf
		self.data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'match_ids', 'pairs', 'same_class_list']

		train_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.train_data_fn, self.data_features, \
			max_num_part=conf.max_num_part, level=conf.level)
		self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
			num_workers=0, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)
		
		val_dataset = PartNetPartDataset(conf.category, conf.data_dir, conf.val_data_fn, self.data_features, \
			max_num_part=conf.max_num_part,level=conf.level)
		self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
				num_workers=0, drop_last=True, collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)

		self.train_writer = SummaryWriter(os.path.join(conf.log_path, 'train'))
		self.val_writer = SummaryWriter(os.path.join(conf.log_path, 'val'))

		self.network = model_def.Network(conf).cuda()
		self.model_names = ['network']
		self.optimizer_names = ['network_opt']

		self.network_opt = torch.optim.Adam(self.network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
		self.network_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

		utils.optimizer_to_device(self.network_opt, conf.device)


	# Forward pass through the network.
	def forward(self, batch, is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
		log_console=False, log_tb=False, tb_writer=None, lr=None):
	
		input_part_pcs = torch.cat(batch[self.data_features.index('part_pcs')], dim=0).cuda()         
		input_part_valids = torch.cat(batch[self.data_features.index('part_valids')], dim=0).cuda()    
		input_part_pairs = torch.cat(batch[self.data_features.index('pairs')], dim=0).cuda()
		
		batch_size = input_part_pcs.shape[0]
		num_part = input_part_pcs.shape[1]
		num_point = input_part_pcs.shape[2]
		
		part_ids = torch.cat(batch[self.data_features.index('part_ids')], dim=0).cuda()      			
		match_ids=batch[self.data_features.index('match_ids')]  
		gt_part_poses = torch.cat(batch[self.data_features.index('part_poses')], dim=0).cuda()  
		shape_id = batch[self.data_features.index('shape_id')]

		# pdb.set_trace()

		same_class_list, instance_label = utils.obtain_same_class_list(batch_size, num_part, input_part_valids, part_ids)
				
		# obtaining the same_class_list.
		# pdb.set_trace()

		repeat_times = 1
		for repeat_ind in range(repeat_times):
			total_pred_part_poses = self.network(conf, input_part_pairs.float(), input_part_valids.float(), input_part_pcs.float(), instance_label, same_class_list, shape_id)

			for iter_ind in range(conf.iter):
				
				pred_part_poses = total_pred_part_poses[iter_ind]
				pred_part_poses, gt_part_poses = utils.match_similar(conf, self.network, input_part_pcs, pred_part_poses, gt_part_poses, match_ids, batch)

				input_part_pcs = input_part_pcs[:, :, :1000, :]
				cur_losses = get_losses_pretrain(conf, input_part_pcs, pred_part_poses, gt_part_poses, input_part_valids)

				# pdb.set_trace()
				
				if iter_ind == 0:
					total_losses = cur_losses	
					
				elif iter_ind == conf.iter - 1:

					total_losses = update_losses(total_losses, cur_losses)
					if repeat_ind == 0:
						res_total_loss = total_losses
					else:
						res_total_loss = update_losses(res_total_loss, total_losses, type_ = 'min')

				else:
					total_losses = update_losses(total_losses, cur_losses)

		total_loss = res_total_loss['total_loss']
		total_trans_l2_loss = res_total_loss['trans_l2_loss']
		total_rot_l2_loss = res_total_loss['rot_l2_loss']
		total_rot_cd_loss = res_total_loss['rot_cd_loss']
		total_shape_cd_loss = res_total_loss['shape_cd_loss']

		# pdb.set_trace()

		data_split = 'train'
		if is_val:
			data_split = 'val' 

		# if visualize in interval. 
		if(step % conf.vis_interval == 0 and self.conf.vis):
			if(is_val):
				utils.visualize_results(self.val_vis_path, step, input_part_pcs, pred_part_poses, gt_part_poses, input_part_valids)
			else:
				utils.visualize_results(self.train_vis_path, step, input_part_pcs, pred_part_poses, gt_part_poses, input_part_valids)


		if log_console:
				# utils.display_loss(data_split = data_split, epoch = epoch, conf = conf, step = step, num_batch = num_batch, total_trans_l2_loss = total_trans_l2_loss, total_rot_l2_loss  = total_rot_l2_loss, total_rot_cd_loss = total_rot_cd_loss, total_shape_cd_loss = total_shape_cd_loss, total_loss = total_loss)

				utils.printout(conf.flog, \
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{data_split:^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{total_trans_l2_loss.item():>10.5f}   '''
                f'''{total_rot_l2_loss.item():>10.5f}   '''
                f'''{total_rot_cd_loss.item():>10.5f}  '''
                f'''{total_shape_cd_loss.item():>10.5f}  '''
                f'''{total_loss.item():>10.5f}  '''
                )
				conf.flog.flush()

		# pdb.set_trace()
		return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss

	def validation(self):

		self.network.eval()

		val_num_batch = len(self.val_dataloader)
		val_batches = enumerate(self.val_dataloader, 0)

		last_val_console_log_step = None
		start_time = time.time()

		sum_total_trans_l2_loss = utils.AverageMeter()
		sum_total_rot_l2_loss = utils.AverageMeter()
		sum_total_rot_cd_loss = utils.AverageMeter()
		sum_total_shape_cd_loss = utils.AverageMeter()

		self.val_vis_path = os.path.join(self.conf.val_vis_path, 'epoch_{}'.format(self.epoch))
		os.makedirs(self.val_vis_path, exist_ok = True)
		for val_batch_ind, batch in val_batches:

			val_fraction_done = (val_batch_ind + 1) / val_num_batch
			val_step = self.epoch * val_num_batch + val_batch_ind

			log_console = not conf.no_console_log and (last_val_console_log_step is None or \
					val_step - last_val_console_log_step >= conf.console_log_interval)
			
			if log_console:
				last_val_console_log_step = val_step

			if(len(batch) == 0): continue

			with torch.no_grad():
				total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss = self.forward(batch=batch, is_val=True,step=val_batch_ind, epoch = self.epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
						log_console=log_console, log_tb=not conf.no_tb_log, tb_writer = self.val_writer, lr= self.network_opt.param_groups[0]['lr'])

			sum_total_trans_l2_loss.update(total_trans_l2_loss)
			sum_total_rot_l2_loss.update(total_rot_l2_loss)
			sum_total_rot_cd_loss.update(total_rot_cd_loss)
			sum_total_shape_cd_loss.update(total_shape_cd_loss)

		with torch.no_grad():
			if not conf.no_tb_log and self.val_writer is not None:
				self.val_writer.add_scalar('sum_total_trans_l2_loss', sum_total_trans_l2_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('sum_total_rot_l2_loss', sum_total_rot_l2_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('sum_total_rot_cd_loss', sum_total_rot_cd_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('sum_total_shape_cd_loss', sum_total_shape_cd_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('lr', self.network_opt.param_groups[0]['lr'], self.epoch)

	def train_one_epoch(self):
		# set network in training mode.
		self.network.train()
		train_num_batch = len(self.train_dataloader)
		val_num_batch = len(self.val_dataloader)

		train_batches = enumerate(self.train_dataloader, 0)
		val_batches = enumerate(self.val_dataloader, 0)

		last_train_console_log_step =  None
		start_time = time.time()

		sum_total_trans_l2_loss = utils.AverageMeter()
		sum_total_rot_l2_loss = utils.AverageMeter()
		sum_total_rot_cd_loss = utils.AverageMeter()
		sum_total_shape_cd_loss = utils.AverageMeter()

		for train_batch_ind, batch in train_batches:
			
			train_fraction_done = (train_batch_ind + 1) / train_num_batch
			train_step = self.epoch * train_num_batch + train_batch_ind

			log_console = not conf.no_console_log and (last_train_console_log_step is None or \
					train_step - last_train_console_log_step >= conf.console_log_interval)
			if log_console:
				last_train_console_log_step = train_step

			if len(batch)==0:continue

			total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss = self.forward(batch=batch, is_val=False, step=train_batch_ind, epoch = self.epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
					log_console=log_console, log_tb=not conf.no_tb_log, tb_writer = self.train_writer, lr= self.network_opt.param_groups[0]['lr'])

			sum_total_trans_l2_loss.update(total_trans_l2_loss)
			sum_total_rot_l2_loss.update(total_rot_l2_loss)
			sum_total_rot_cd_loss.update(total_rot_cd_loss)
			sum_total_shape_cd_loss.update(total_shape_cd_loss)

			self.network_lr_scheduler.step()
			self.network_opt.zero_grad()
			total_loss.backward()
			self.network_opt.step()

		with torch.no_grad():
			if not conf.no_tb_log and self.train_writer is not None:
				self.train_writer.add_scalar('sum_total_trans_l2_loss', sum_total_trans_l2_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('sum_total_rot_l2_loss', sum_total_rot_l2_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('sum_total_rot_cd_loss', sum_total_rot_cd_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('sum_total_shape_cd_loss', sum_total_shape_cd_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('lr', self.network_opt.param_groups[0]['lr'], self.epoch)


	def fit(self):
		if not conf.no_console_log:
			header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TransL2Loss    RotL2Loss   RotCDLoss  ShapeCDLoss   TotalLoss'

		models = [self.network]
		model_names = ['network']
		optimizers = [self.network_opt]
		optimizer_names = ['network_opt']

		# Using this is important. 
		# Validate running multiple epochs.
		for self.epoch in range(conf.epochs):
			if not conf.no_console_log:
				utils.printout(conf.flog, f'validation run {conf.exp_name}')
				utils.printout(conf.flog, header)

			utils.load_checkpoint_wo_optimizer(models = models, model_names = model_names, dirname = os.path.join(conf.ckpt_path), epoch = self.epoch, optimizers = optimizers, optimizer_names = optimizer_names)
			self.validation()
			utils.printout(conf.flog, 'DONE')


if __name__ == '__main__':

	conf = GlobalTrainConfig()
	

	os.makedirs(conf.log_dir, exist_ok = True)
	os.makedirs(conf.exp_dir, exist_ok = True)
	os.makedirs(conf.log_path, exist_ok = True)
	os.makedirs(conf.vis_path, exist_ok = True)
	os.makedirs(conf.ckpt_path, exist_ok = True)
	os.makedirs(conf.train_vis_path, exist_ok = True)
	os.makedirs(conf.val_vis_path, exist_ok = True)
	
	flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
	conf.flog = flog
	
	print("conf", conf)

	fitter = Fitter(conf)
	fitter.fit()