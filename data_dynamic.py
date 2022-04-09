"""
    PartNetPartDataset
"""
import os
import torch
import sys
import torch.utils.data as data
import numpy as np
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from torch.utils.data import DataLoader, random_split
import ipdb
from quaternion import qrot_np
import pdb

# PartNet shape dataset.

# def collate_fn(batch):
#     pdb.set_trace()
#     pass

# Symmetry being obtained from original pcs.
def get_sym(gt_pcs): 
    thre_for_sym = 0.025
    num_part = len(gt_pcs)
    sym = np.zeros((num_part,3))
    for j in range(3):
            for i in range(num_part):
                sym_pcs = gt_pcs[i].copy()
                sym_pcs[:,j] *= -1
                error = get_chamfer_distance(gt_pcs[i], sym_pcs)
                if error < thre_for_sym :
                    sym[i,j] = 1 
    return sym

class PartNetPartDataset(data.Dataset):

    def __init__(self, category, data_dir, data_fn, data_features, level,\
            max_num_part=20):
        # store parameters
        self.data_dir = data_dir        # a data directory inside [path/to/codebase]/data/
        self.data_fn = data_fn          # a .npy data indexing file listing all data tuples to load
        self.category = category

        # self.max_num_part.
        self.max_num_part = max_num_part
        self.max_pairs = max_num_part * (max_num_part-1) / 2
        self.level = level

        # load data.
        self.data = np.load(os.path.join(self.data_dir, data_fn))

        # data features
        self.data_features = data_features

        # load category semantics information
        self.part_sems = []
        self.part_sem2id = dict()

    def get_part_count(self):
        return len(self.part_sems)
        
    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_dir: %s, data_fn: %s, max_num_part: %d' % \
                (self.category, len(self), self.data_dir, self.data_fn, self.max_num_part)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Shape_ID
        shape_id = self.data[index]

        cur_data_fn = os.path.join(self.data_dir, 'shape_data/%s_level' % shape_id + self.level + '.npy')
        cur_data = np.load(cur_data_fn, allow_pickle=True ).item()   # assume data is stored in seperate .npz file
        
        cur_contact_data_fn = os.path.join(self.data_dir, 'contact_points/pairs_with_contact_points_%s_level' % shape_id + self.level + '.npy')
        cur_contacts = np.load(cur_contact_data_fn,allow_pickle=True)
        
        data_feats = ()
        
        for feat in self.data_features:

            if feat == 'contact_points':
                cur_num_part = cur_contacts.shape[0]
                out = np.zeros((self.max_num_part,self.max_num_part,4), dtype=np.float32)
                out[:cur_num_part,:cur_num_part,:] = cur_contacts
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'sym':
                cur_sym = cur_data['sym']
                cur_part_ids = cur_data['geo_part_ids']                 # p
                cur_num_part = cur_sym.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                out = np.zeros((self.max_num_part, cur_sym.shape[1]), dtype=np.float32)
                out[:cur_num_part] = cur_sym
                out = torch.from_numpy(out).float().unsqueeze(0)    # p x 3
                data_feats = data_feats + (out,)
                
            elif feat == 'semantic_ids':
                cur_part_ids = cur_data['part_ids']
                cur_num_part = len(cur_part_ids)
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part), dtype=np.float32)
                out[:cur_num_part] = cur_part_ids
                out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 
                data_feats = data_feats + (out,)
            
            # Part point clouds being obtained.              
            elif feat == 'part_pcs':
                out = cur_data['part_pcs']
                # pdb.set_trace()                      # p x N x 3 (p is unknown number of parts for this shape)
                # num_point = cur_pts.shape[1]
                # cur_gt_poses = cur_data['part_poses']
                # gt_part_pcs = qrot_np(np.tile(np.expand_dims(cur_gt_poses[:, 3:], 1),(1, num_point, 1)), cur_pts) + np.tile(np.expand_dims(cur_gt_poses[:, :3], 1),(1, num_point, 1))
                # sym = get_sym(gt_part_pcs)
                # cur_part_ids = cur_data['geo_part_ids']                 # p
                cur_num_part = out.shape[0]
                # print("part_num:",cur_num_part)
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                # out = np.zeros((self.max_num_part, cur_pts.shape[1], 3), dtype=np.float32)
                # out[:cur_num_part] = cur_pts
                out = torch.from_numpy(out).float() # 1 x 20 x N x 3
                data_feats = data_feats + (out,)

            elif feat == 'part_poses':
                out = cur_data['part_poses']                   # p x (3 + 4)
                cur_num_part = out.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                # out = np.zeros((self.max_num_part, 3 + 4), dtype=np.float32)
                # out[:cur_num_part] = cur_pose
                out = torch.from_numpy(out).float()    # 1 x 20 x (3 + 4)
                data_feats = data_feats + (out,)

            elif feat == 'part_valids':
                cur_pose = cur_data['part_poses']                   # p x (3 + 4)
                cur_num_part = cur_pose.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item

                out = np.ones((cur_num_part), dtype = np.float32)
                out = torch.from_numpy(out).float()
                data_feats = data_feats + (out,)
            
            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'part_ids':
                cur_part_ids = cur_data['geo_part_ids']
                cur_num_part = cur_pose.shape[0]

                if cur_num_part > self.max_num_part:
                    return None
                # out = np.zeros((self.max_num_part), dtype=np.float32)
                # out[:cur_num_part] = cur_part_ids

                out = torch.from_numpy(np.array(cur_part_ids)).float()
                data_feats = data_feats + (out,)
            
            elif feat == 'pairs':
                cur_pose = cur_data['part_poses']                   # p x (3 + 4)
                cur_num_part = cur_pose.shape[0]
                if cur_num_part > self.max_num_part:
                    return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
                cur_pose = cur_data['part_poses']                   # p x (3 + 4)
                cur_vaild_num = len(cur_pose)
                valid_pair_martix = np.ones((cur_vaild_num, cur_vaild_num))
                pair_martix = np.zeros((self.max_num_part, self.max_num_part))
                pair_martix[:cur_vaild_num,:cur_vaild_num] = valid_pair_martix
                out = torch.from_numpy(pair_martix).unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'match_ids':

                cur_part_ids = cur_data['geo_part_ids']
                cur_num_part = cur_pose.shape[0]
                if cur_num_part > self.max_num_part:
                    return None
                out = np.array(cur_part_ids, dtype=np.float32)
                # out[:cur_num_part] = cur_part_ids
                index = 1
                for i in range(1,58):
                    idx = np.where(out==i)[0]
                    idx = torch.from_numpy(idx)
                    if len(idx)==0: continue
                    elif len(idx)==1: out[idx]=0
                    else:
                        out[idx] = index
                        index += 1

                # pdb.set_trace()
                out = torch.from_numpy(out)
                data_feats = data_feats + (out,)

            elif feat == 'same_class_list':
                pass

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats
# for test:

# def getitem():
#     # shape_id = self.data[index]
#     shape_id = 11111
#     data_dir = "./data/partnet_dataset/"
#     max_num_part = 10
#     data_features = ['part_pcs','part_poses','part_valids']
#     cur_data_fn = os.path.join(data_dir, '%s.npy' % shape_id)
#     cur_data = np.load(cur_data_fn, allow_pickle=True ).item()     # assume data is stored in seperate .npz file
#     # import ipdb; ipdb.set_trace(
#     data_feats = ()
#     for feat in data_features:
#         if feat == 'part_pcs':
#             cur_pts = cur_data['part_pcs']                      # p x N x 3 (p is unknown number of parts for this shape)
#             cur_num_part = cur_pts.shape[0]
#             if cur_num_part > max_num_part:
#                 return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
#             out = np.zeros((max_num_part, cur_pts.shape[1], 3), dtype=np.float32)
#             out[:cur_num_part] = cur_pts
#             out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 x N x 3
#             data_feats = data_feats + (out,)

#         elif feat == 'part_poses':
#             cur_pose = cur_data['part_poses']                   # p x (3 + 4)
#             cur_num_part = cur_pose.shape[0]
#             if cur_num_part > max_num_part:
#                 return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
#             out = np.zeros((max_num_part, 3 + 4), dtype=np.float32)
#             out[:cur_num_part] = cur_pose
#             out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 x (3 + 4)
#             data_feats = data_feats + (out,)

#         elif feat == 'part_valids':
#             cur_pose = cur_data['part_poses']                   # p x (3 + 4)
#             cur_num_part = cur_pose.shape[0]
#             if cur_num_part > max_num_part:
#                 return None         # directly returning a None will let data loader with collate_fn=utils.collate_fn_with_none to ignore this data item
#             out = np.zeros((max_num_part), dtype=np.float32)
#             out[:cur_num_part] = 1
#             out = torch.from_numpy(out).float().unsqueeze(0)    # 1 x 20 (return 1 for the first p parts, 0 for the rest)
#             data_feats = data_feats + (out,)
        
#         elif feat == 'shape_id':
#             data_feats = data_feats + (shape_id,)

#         else:
#             raise ValueError('ERROR: unknown feat type %s!' % feat)
#     # print(data_feats.shape)
#     # import ipdb; ipdb.set_trace()

#     return data_feats


# if __name__ == "__main__":
#     dataset = PartNetPartDataset(category='Chair', data_dir="./data/partnet_dataset/", data_fn="Chair.train.npy", data_features=['part_pcs','part_poses','part_valids'], \
#             max_num_part=20)
#     # getitem()
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=False,
#                                 num_workers=1, pin_memory=True)
#     for i, data in enumerate(dataloader):
#         print(i)
#         print(data)
#         ipdb.set_trace()
