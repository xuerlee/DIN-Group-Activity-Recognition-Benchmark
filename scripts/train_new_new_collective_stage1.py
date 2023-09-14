import sys
sys.path.append(".")
from train_net import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cfg=Config('new_new_collective')

cfg.device_list="0,1"
cfg.training_stage=1
cfg.train_backbone=True

# ResNet18
cfg.backbone = 'res18'
cfg.image_size = 480, 720
cfg.out_size = 15, 23
cfg.emb_features = 512

# VGG16
# cfg.backbone = 'vgg16'
# cfg.image_size = 480, 720
# cfg.out_size = 15, 22
# cfg.emb_features = 512
# cfg.stage1_model_path = 'result/basemodel_CAD_vgg16.pth'


#cfg.image_size=480, 720
#cfg.out_size=57,87
cfg.num_boxes=13
cfg.num_actions=6
cfg.num_activities=5
cfg.num_frames=10

cfg.batch_size=2
# cfg.batch_size=32
cfg.test_batch_size=2
# cfg.test_batch_size=8
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=100

cfg.exp_note='New_new_collective_stage1'
train_net(cfg)