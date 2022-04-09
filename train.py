# Importing all necessary libraries. 
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
from torch.nn.utils.rnn import pad_sequence
from quaternion import qrot
import math
import pdb
from models import model_dynamic_backup as model_def
from loss import *
from tensorboardX import SummaryWriter

# Random seed being set here. 
SEED = 42
def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

# Seed everything. 
seed_everything(SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class GlobalTrainConfig:
	lr = 1e-3
	lr_decay_by = 0.9
	lr_decay_every = 5000
	weight_decay = 1e-5
	num_epochs = 1000
	device = 'cuda:0'
	level = '3'

	# Translation and Rotation. 
	loss_weight_trans_l2 = 1.0
	loss_weight_rot_l2 = 0.0
	loss_weight_rot_cd = 10.0
	loss_weight_shape_cd = 1.0

	# Feat len is run by noise_dim = 0. 
	feat_len = 256
	noise_dim = 0
	hidden_size = 256
	max_num_part = 20
	epochs = 1000
	start_epoch = -1
	seed = 3124256514
	noise_scale = 1

	no_console_log = False
	no_tb_log = False
	console_log_interval = 10
	repeat_times = 1
	teacher_forcing = True
	bid = True

	iter = 3
	category = 'Table'
	train_data_fn =  '{}.train.npy'.format(category) 
	val_data_fn  = '{}.val.npy'.format(category)
	
	data_dir = './prepare_data'
	log_dir = './runs'
	batch_size = 16

	exp_name = f'exp-{category}-lr-decay-{lr_decay_every}-batch-{batch_size}-trans-l2-{loss_weight_trans_l2}-rot_l2-{loss_weight_rot_l2}-rot_cd-{loss_weight_rot_cd}-shape_cd-{loss_weight_shape_cd}-iter-{iter}-repeat-{repeat_times}-noise-{noise_dim}-scale-{noise_scale}'
	exp_dir = os.path.join(log_dir, exp_name)

	log_path = os.path.join(exp_dir, 'logs')
	vis_path = os.path.join(exp_dir, 'results')
	ckpt_path = os.path.join(exp_dir, 'checkpoints')

	train_vis_path = os.path.join(vis_path, 'train')
	val_vis_path = os.path.join(vis_path, 'val')

	train_vis_interval = 100
	val_vis_interval = 10

	num_batch_every_visu = 5
	EPOCH_INTERVAL = 5
	num_workers = 1
	vis = False

class Fitter: 
	def __init__(self, conf):
		self.conf = conf
		self.best_acc = 0.
		self.best_shape_cd_loss = math.inf

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
		self.train_last_writer = SummaryWriter(os.path.join(conf.log_path, 'train_last'))
		self.val_writer = SummaryWriter(os.path.join(conf.log_path, 'val'))
		self.val_last_writer = SummaryWriter(os.path.join(conf.log_path, 'val_last'))

		self.network = model_def.Network(conf).cuda()
		self.model_names = ['network']
		self.optimizer_names = ['network_opt']

		self.network_opt = torch.optim.Adam(self.network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
		self.network_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

		utils.optimizer_to_device(self.network_opt, conf.device)


	def forward(self, batch, is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
		log_console=False, log_tb=False, tb_writer=None, lr=None):
		
		input_part_pcs = pad_sequence(batch[self.data_features.index('part_pcs')], batch_first = True, padding_value = 0).cuda()
		gt_part_poses = pad_sequence(batch[self.data_features.index('part_poses')], batch_first = True, padding_value = 0).cuda()   
		
		input_part_valids = pad_sequence(batch[self.data_features.index('part_valids')], batch_first = True, padding_value = 0).cuda()
		input_part_pairs = torch.cat(batch[self.data_features.index('pairs')], dim=0).cuda()
		
		batch_size = input_part_pcs.shape[0]
		num_part = input_part_pcs.shape[1]
		num_point = input_part_pcs.shape[2]
		

		part_ids = pad_sequence(batch[self.data_features.index('part_ids')], batch_first = True, padding_value = 0).cuda()      			
		match_ids= pad_sequence(batch[self.data_features.index('match_ids')], batch_first = True, padding_value = 0).cuda()
		shape_id = batch[self.data_features.index('shape_id')]

		same_class_list, instance_label = utils.obtain_same_class_list(batch_size, num_part, input_part_valids, part_ids)
		
		# Repeat times. 
		for repeat_ind in range(self.conf.repeat_times):
			total_pred_part_poses = self.network(conf, input_part_pairs.float(), input_part_valids.float(), input_part_pcs.float(), instance_label, same_class_list, shape_id)

			for iter_ind in range(conf.iter):
				
				pred_part_poses = total_pred_part_poses[iter_ind]
				pred_part_poses, gt_part_poses = utils.match_similar(conf, self.network, input_part_pcs, pred_part_poses, gt_part_poses, match_ids, batch)

				input_part_pcs = input_part_pcs[:, :, :1000, :]
				cur_losses = get_losses_pretrain(conf, input_part_pcs, pred_part_poses, gt_part_poses, input_part_valids)

				if iter_ind == 0:
					total_losses = cur_losses	
					
				elif iter_ind == conf.iter - 1:

					total_losses = update_losses(total_losses, cur_losses)
					if repeat_ind == 0:
						res_total_loss = total_losses
					else:
						res_total_loss = update_losses(res_total_loss, total_losses, type_ = 'min')

					if repeat_ind == 0:
						last_loss = cur_losses
					else:
						last_loss = update_losses(last_loss, cur_losses, type_ = 'min')

				else:
					total_losses = update_losses(total_losses, cur_losses)

		total_loss = res_total_loss['total_loss']
		total_trans_l2_loss = res_total_loss['trans_l2_loss']
		total_rot_l2_loss = res_total_loss['rot_l2_loss']
		total_rot_cd_loss = res_total_loss['rot_cd_loss']
		total_shape_cd_loss = res_total_loss['shape_cd_loss']

		data_split = 'train'
		if is_val:
			data_split = 'val' 

		if(is_val):
			if(step % self.conf.val_vis_interval == 0 and self.conf.vis):
				utils.visualize_results(self.val_vis_path, step, input_part_pcs, pred_part_poses, gt_part_poses, input_part_valids)

		else:
			if(step % self.conf.train_vis_interval == 0 and self.conf.vis):
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


		return res_total_loss, last_loss #total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss

	# Code for running validation.
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
		sum_total_cd_loss = utils.AverageMeter()

		last_total_trans_l2_loss = utils.AverageMeter()
		last_total_rot_l2_loss = utils.AverageMeter()
		last_total_rot_cd_loss = utils.AverageMeter()
		last_total_shape_cd_loss = utils.AverageMeter()
		last_total_cd_loss = utils.AverageMeter()

		acc_list = []
		valid_part_list = []

		self.val_vis_path = os.path.join(conf.val_vis_path, 'epoch_{}'.format(self.epoch))
		os.makedirs(self.val_vis_path, exist_ok = True)

		# val_batch_ind and batch in val_batches.
		for val_batch_ind, batch in val_batches:

			val_fraction_done = (val_batch_ind + 1) / val_num_batch
			val_step = self.epoch * val_num_batch + val_batch_ind

			log_console = not conf.no_console_log and (last_val_console_log_step is None or \
					val_step - last_val_console_log_step >= conf.console_log_interval)
			
			if log_console:
				last_val_console_log_step = val_step

			if(len(batch) == 0): continue

			with torch.no_grad():
				res_total_loss, last_loss = self.forward(batch=batch, is_val=True,step=val_step, epoch = self.epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
						log_console=log_console, log_tb=not conf.no_tb_log, tb_writer = self.val_writer, lr= self.network_opt.param_groups[0]['lr'])

			sum_total_trans_l2_loss.update(res_total_loss['trans_l2_loss'])
			sum_total_rot_l2_loss.update(res_total_loss['rot_l2_loss'])
			sum_total_rot_cd_loss.update(res_total_loss['rot_cd_loss'])
			sum_total_shape_cd_loss.update(res_total_loss['shape_cd_loss'])
			sum_total_cd_loss.update(res_total_loss['total_cd_loss'])

			last_total_trans_l2_loss.update(last_loss['trans_l2_loss'])
			last_total_rot_l2_loss.update(last_loss['rot_l2_loss'])
			last_total_rot_cd_loss.update(last_loss['rot_cd_loss'])
			last_total_shape_cd_loss.update(last_loss['shape_cd_loss'])
			last_total_cd_loss.update(last_loss['total_cd_loss'])

			# Percentage of parts correctly classified in a batch.
			# valid_num = input_part_valids.sum()
			acc_list += last_loss['acc'].numpy().tolist()
			valid_part_list += last_loss['valid_num'].numpy().tolist()
			# acc = last_loss['acc'].sum()
			# valid_num = input_part_valids.sum()
			# acc_per = acc/valid_num
			# last_total_acc.update(acc/valid_num)
			# pdb.set_trace()
			# last_total_acc.update(last_loss[])

		epoch_acc = sum(acc_list)/sum(valid_part_list)
		epoch_shape_cd_loss = last_total_rot_cd_loss.avg.item()

		if epoch_acc > self.best_acc:
			self.best_epoch_acc = epoch_acc
			self.update_best_acc = True

		if epoch_shape_cd_loss < self.best_shape_cd_loss:
			self.best_shape_cd_loss = epoch_shape_cd_loss
			self.update_best_shape_cd_loss = True

		with torch.no_grad():
			if not conf.no_tb_log and self.val_writer is not None:
				self.val_writer.add_scalar('sum_total_trans_l2_loss', sum_total_trans_l2_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('sum_total_rot_l2_loss', sum_total_rot_l2_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('sum_total_rot_cd_loss', sum_total_rot_cd_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('sum_total_cd_loss', sum_total_cd_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('sum_total_shape_cd_loss', sum_total_shape_cd_loss.avg.item(), self.epoch)
				self.val_writer.add_scalar('lr', self.network_opt.param_groups[0]['lr'], self.epoch)

				self.val_last_writer.add_scalar('sum_total_trans_l2_loss', last_total_trans_l2_loss.avg.item(), self.epoch)
				self.val_last_writer.add_scalar('sum_total_rot_l2_loss', last_total_rot_l2_loss.avg.item(), self.epoch)
				self.val_last_writer.add_scalar('sum_total_rot_cd_loss', last_total_rot_cd_loss.avg.item(), self.epoch)
				self.val_last_writer.add_scalar('sum_total_cd_loss', last_total_cd_loss.avg.item(), self.epoch)
				self.val_last_writer.add_scalar('sum_total_shape_cd_loss', last_total_shape_cd_loss.avg.item(), self.epoch)
				self.val_last_writer.add_scalar('part_accuracy', sum(acc_list)/sum(valid_part_list), self.epoch)
				self.val_last_writer.add_scalar('lr', self.network_opt.param_groups[0]['lr'], self.epoch)

	def train_one_epoch(self):
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

		last_total_trans_l2_loss = utils.AverageMeter()
		last_total_rot_l2_loss = utils.AverageMeter()
		last_total_rot_cd_loss = utils.AverageMeter()
		last_total_shape_cd_loss = utils.AverageMeter()

		self.train_vis_path = os.path.join(conf.train_vis_path, 'epoch_{}'.format(self.epoch))
		os.makedirs(self.train_vis_path, exist_ok = True)

		for train_batch_ind, batch in train_batches:
			
			train_fraction_done = (train_batch_ind + 1) / train_num_batch
			train_step = self.epoch * train_num_batch + train_batch_ind

			log_console = not conf.no_console_log and (last_train_console_log_step is None or \
					train_step - last_train_console_log_step >= conf.console_log_interval)
			if log_console:
				last_train_console_log_step = train_step

			if len(batch)==0:continue

			# Res total loss and last loss in forward pass. 
			res_total_loss, last_loss = self.forward(batch=batch, is_val=False, step=train_step, epoch = self.epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
					log_console=log_console, log_tb=not conf.no_tb_log, tb_writer = self.train_writer, lr= self.network_opt.param_groups[0]['lr'])

			sum_total_trans_l2_loss.update(res_total_loss['trans_l2_loss'])
			sum_total_rot_l2_loss.update(res_total_loss['rot_l2_loss'])
			sum_total_rot_cd_loss.update(res_total_loss['rot_cd_loss'])
			sum_total_shape_cd_loss.update(res_total_loss['shape_cd_loss'])

			last_total_trans_l2_loss.update(last_loss['trans_l2_loss'])
			last_total_rot_l2_loss.update(last_loss['rot_l2_loss'])
			last_total_rot_cd_loss.update(last_loss['rot_cd_loss'])
			last_total_shape_cd_loss.update(last_loss['shape_cd_loss'])

			total_loss = res_total_loss['total_loss']
			
			self.network_lr_scheduler.step()
			self.network_opt.zero_grad()
			total_loss.backward()
			self.network_opt.step()
			del total_loss
			torch.cuda.empty_cache()
		with torch.no_grad():
			if not conf.no_tb_log and self.train_writer is not None:
				self.train_writer.add_scalar('sum_total_trans_l2_loss', sum_total_trans_l2_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('sum_total_rot_l2_loss', sum_total_rot_l2_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('sum_total_rot_cd_loss', sum_total_rot_cd_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('sum_total_shape_cd_loss', sum_total_shape_cd_loss.avg.item(), self.epoch)
				self.train_writer.add_scalar('lr', self.network_opt.param_groups[0]['lr'], self.epoch)

				self.train_last_writer.add_scalar('sum_total_trans_l2_loss', last_total_trans_l2_loss.avg.item(), self.epoch)
				self.train_last_writer.add_scalar('sum_total_rot_l2_loss', last_total_rot_l2_loss.avg.item(), self.epoch)
				self.train_last_writer.add_scalar('sum_total_rot_cd_loss', last_total_rot_cd_loss.avg.item(), self.epoch)
				self.train_last_writer.add_scalar('sum_total_shape_cd_loss', last_total_shape_cd_loss.avg.item(), self.epoch)
				self.train_last_writer.add_scalar('lr', self.network_opt.param_groups[0]['lr'], self.epoch)


	def fit(self):
		if not conf.no_console_log:
			header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TransL2Loss    RotL2Loss   RotCDLoss  ShapeCDLoss   TotalLoss'

		models = [self.network]
		model_names = ['network']
		optimizers = [self.network_opt]
		optimizer_names = ['network_opt']

		# Start epoch is being loaded. 
		if(self.conf.start_epoch > 0):
			utils.load_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.ckpt_path), epoch= self.conf.start_epoch, optimizers=optimizers, optimizer_names=optimizer_names)
			network_lr_scheduler = torch.load(os.path.join(conf.ckpt_path, '{}_network_lr_scheduler.pth'.format(self.conf.start_epoch)))
			self.network_lr_scheduler.load_state_dict(network_lr_scheduler)

		for self.epoch in range(conf.start_epoch + 1, conf.epochs):
			if not conf.no_console_log:
				utils.printout(conf.flog, f'training run {conf.exp_name}')
				utils.printout(conf.flog, header)

			self.validation()
			self.train_one_epoch()
			
			if self.update_best_acc:
				best_state = {
						'epoch': self.epoch,
						'state_dict': self.network.state_dict(),
						'acc': self.best_acc
						}
				torch.save(best_state, os.path.join(conf.ckpt_path, 'best_acc.pth'))

			if self.update_best_shape_cd_loss:
				best_state = {
						'epoch': self.epoch, 
						'state_dict': self.network.state_dict(),
						'shape_cd_loss': self.best_shape_cd_loss
						}
				torch.save(best_state, os.path.join(conf.ckpt_path, 'best_shape_cd.pth'))
			
			if self.epoch % conf.EPOCH_INTERVAL == 0:
				torch.save(self.network_lr_scheduler.state_dict(), os.path.join(conf.ckpt_path, '{}_network_lr_scheduler.pth'.format(self.epoch)))
				utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.ckpt_path), epoch= self.epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=optimizer_names)
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
