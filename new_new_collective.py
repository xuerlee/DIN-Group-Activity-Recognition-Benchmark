import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
from utils import out_group_black, re_organize_seq
import torchvision.models as models

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
import sys
from config import *

FRAMES_NUM = {1: 301, 2: 346, 3: 193, 4: 256, 10: 301, 12: 1083, 13: 850, 14: 722, 23: 310,
              26: 733, 28: 469, 29: 634, 30: 355, 31: 689, 33: 192, 36: 912, 41: 706, 42: 418,
              43: 409, 45: 150, 46: 173, 58: 628, 59: 898, 63: 432, 68: 944, 72: 750, 74: 398}

FRAMES_SIZE = {1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 10: (480, 720),
               12: (480, 720), 13: (480, 720), 14: (480, 720), 23: (450, 800), 26: (480, 720),
               28: (480, 720), 29: (480, 720), 30: (480, 720), 31: (480, 720),  33: (480, 720),
               36: (480, 720), 41: (480, 720), 42: (480, 720), 43: (480, 720), 45: (480, 640),
               46: (480, 640), 58: (480, 640), 59: (480, 640), 63: (720, 1280), 68: (720, 1280),
               72: (720, 1280), 74: (720, 1280)}

IDIVIDUAL_ACTIVITIES = ['standing', 'jogging', 'dancing', 'walking', 'biking', 'none']  # individual activity
# SECOND_ACTIVITIES = ['crossing', 'waiting', 'queueing', 'walking', 'talking', 'None']
GROUP_ACTIVITIES = ['dancing', 'queuing', 'jogging', 'talking', 'none']  # group activity

IDIVIDUAL_ACTIVITIES_ID = {a: i for i, a in enumerate(IDIVIDUAL_ACTIVITIES)}
# SECOND_ACTIONS_ID = {a: i for i, a in enumerate(SECOND_ACTIVITIES)}
GROUP_ACTIVITIES_ID = {a: i for i, a in enumerate(GROUP_ACTIVITIES)}

Action6to5 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
Activity5to4 = {0: 0, 1: 1, 2: 2, 3: 0, 4: 3}

def new_new_collective_read_annotations(path, sid):

    annotations = {}
    path = path + '/annotations/seq_%02d.xml' % sid

    ann_file = open(path)
    tree = ET.parse(ann_file)
    root = tree.getroot()

    group_activity = None
    actions = []
    bboxes = []

    for i in range(FRAMES_NUM[sid]):
        annotations[i] = {}
        annotations[i]['groups'] = []
        annotations[i]['persons'] = []

        for j, track in enumerate(root.findall('track')):  # one track one person
            if track.get('label') == 'group':
                for box in track.findall('box'):
                    if int(int(box.get('frame')) == i):
                        xtl = float(box.get('xtl'))
                        ytl = float(box.get('ytl'))
                        xbr = float(box.get('xbr'))
                        ybr = float(box.get('ybr'))
                        H, W = FRAMES_SIZE[sid]
                        bboxes = [ytl / H, xtl / W, ybr / H, xbr / W]

                        group_id = box.find('attribute[@name="group_id"]').text
                        group_activity = box.find('attribute[@name="group_activity"]').text

                        annotations[i]['groups'].append({
                            'frame_id': i,
                            'group_id': group_id,
                            'group_activity': group_activity,
                            'bboxes': bboxes
                        })

            if track.get('label') == 'person':
                for box in track.findall('box'):
                    if int(int(box.get('frame')) == i):
                        xtl = float(box.get('xtl'))
                        ytl = float(box.get('ytl'))
                        xbr = float(box.get('xbr'))
                        ybr = float(box.get('ybr'))
                        w = xbr - xtl
                        h = ybr - ytl
                        H, W = FRAMES_SIZE[sid]
                        bboxes = [ytl / H, xtl / W, ybr / H, xbr / W]
                        group_id = box.find('attribute[@name="group_id"]').text
                        person_id = box.find('attribute[@name="person_id"]').text
                        individual_activity = box.find('attribute[@name="individual_activity"]').text
                        # second_activity = box.find('attribute[@name="second activity"]').text

                        annotations[i]['persons'].append({
                            'frame_id': i,
                            'group_id': group_id,
                            'perons_id': person_id,
                            'individual_activity': individual_activity,
                            # 'second_activity': second_activity,
                            'bboxes': bboxes
                        })

    return annotations


def new_new_collective_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = new_new_collective_read_annotations(path, sid)
    return data

'''
data: [{ann1},{ann2}, ... ,{annseq}]
ann1: {1: {ann1-1}, 2: {ann1-2}, ... , frame: {ann1-frame}}
ann1-1: {groups: [{frame_id: 1, group_id: 1, ...}, {...}], 
        persons: [{...}, {...}]}
'''



def new_new_collective_all_frames(anns):
    return [(s, f) for s in anns for f in anns[s] if f != 0]  # remove frame0000

'''
[(seq_id, frame_id), ...]
e.g. [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),...,(1, frame)]
[(10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8)...]
'''




class NewNewCollectiveDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """

    def __init__(self, anns, frames, images_path, image_size, feature_size, num_boxes=13, num_groups=3, num_frames=10,
                 is_training=True, is_finetune=False):
        self.anns = anns
        self.frames = frames  # [(seq_id, frame_id), ...]
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_gorups = num_groups
        self.num_frames = num_frames  # maximum intervals between anns and random chosen frames

        self.is_training = is_training
        self.is_finetune = is_finetune


        # self.frames_seq = np.empty((1337, 2), dtype = np.int)
        # self.flag = 0

        # Set data position
        cfg = Config('new_new_collective')
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list
        if cfg.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        lists (1 list for 1 image or 1 sequece) of (data, kinds of labels) -> collate_fn -> stacked batch
        """
        # Save frame sequences
        # self.frames_seq[self.flag] = self.frames[index] # [0], self.frames[index][1]
        # if self.flag == 764: # 1336
        #     save_seq = self.frames_seq
        #     np.savetxt('vis/Collective/frames_seq.txt', save_seq)
        # self.flag += 1

        select_frames = self.get_frames(self.frames[index])
        sample = self.load_samples_sequence(select_frames)
        while sample[-1] == 0:
            index = torch.randint(0, len(self.frames), (1,)).item()
            select_frames = self.get_frames(self.frames[index])
            sample = self.load_samples_sequence(select_frames)
        return sample


    def get_frames(self, frame):

        sid, src_fid = frame  # self.frames[index]: one couple of (seq_id, frame_id)  e.g.: 1, 205

        if self.is_finetune:
            if self.is_training:
                if src_fid + self.num_frames - 1 < FRAMES_NUM[sid]:
                    fid = random.randint(src_fid, src_fid + self.num_frames)  # choose frame between e.g. random(205,214), not the ann id exactly. kind of shuffling
                    return [(sid, src_fid, fid)]
                else:
                    fid = random.randint(src_fid, FRAMES_NUM[sid])
                    return [(sid, src_fid, fid)]

            else:
                if src_fid + self.num_frames - 1 < FRAMES_NUM[sid]:
                    return [(sid, src_fid, fid)
                            for fid in range(src_fid, src_fid + self.num_frames)]  # each test loading 10 frames
                else:
                    list1 = [(sid, src_fid, fid)
                            for fid in range(src_fid, FRAMES_NUM[sid])]
                    list2 = [(sid, src_fid, FRAMES_NUM[sid])] * (self.num_frames - len(list1))
                    list = list1 + list2
                    return list


        else:
            # if self.is_training:
            #     sample_frames=random.sample(range(src_fid,src_fid+self.num_frames),3)
            #     return [(sid, src_fid, fid) for fid in sample_frames]
            #
            # else:
            #     sample_frames=[ src_fid, src_fid+3, src_fid+6, src_fid+1, src_fid+4, src_fid+7, src_fid+2, src_fid+5, src_fid+8 ]
            #     return [(sid, src_fid, fid) for fid in sample_frames]
            if self.is_training:
                if src_fid + self.num_frames - 1 < FRAMES_NUM[sid]:
                    return [(sid, src_fid, fid)
                            for fid in range(src_fid, src_fid + self.num_frames)]  # each test loading 10 frames
                else:
                    list1 = [(sid, src_fid, fid)
                            for fid in range(src_fid, FRAMES_NUM[sid])]
                    list2 = [(sid, src_fid, FRAMES_NUM[sid])] * (self.num_frames - len(list1))
                    list = list1 + list2
                    return list
            else:
                if src_fid + self.num_frames - 1 < FRAMES_NUM[sid]:
                    return [(sid, src_fid, fid)
                            for fid in range(src_fid, src_fid + self.num_frames)]  # each test loading 10 frames
                else:
                    list1 = [(sid, src_fid, fid)
                            for fid in range(src_fid, FRAMES_NUM[sid])]
                    list2 = [(sid, src_fid, FRAMES_NUM[sid])] * (self.num_frames - len(list1))
                    list = list1 + list2
                    return list


    def load_img_anns(self, select_frames):
        '''
        Return: images and anns for _getitem_
        '''

        OH, OW = self.feature_size

        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        for i, (sid, src_fid, fid) in enumerate(select_frames):
            img = Image.open(self.images_path + '/ActivityDataset/seq%02d/frame%04d.jpg' % (sid, fid))  # frame_fid
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)
            # H,W,3 -> 3,H,W
            image = img.transpose(2, 0, 1)
            images.append(image)

            group_boxes = []
            groups_num = []
            temp_activities = []
            temp_group_num = 0
            # print('anns:', self.anns[sid][src_fid])
            if len(self.anns[sid][src_fid]['groups']) > 0:
                for group in self.anns[sid][src_fid]['groups']:
                    temp_group_num += 1
                    gg_id = group['group_id']
                    box = group['bboxes']
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    group_boxes.append([int(gg_id), x1, y1, x2, y2])
                    temp_activities.append(GROUP_ACTIVITIES_ID[group['group_activity']])
            else:
                temp_group_num = 1
                group_boxes.append([0, 0, 0, OW, OH])  # there is no group in the image
                temp_activities.append(4)

            temp_boxes = []
            temp_actions = []
            temp_bboxes_num = []
            for i, group_box in enumerate(group_boxes):
                temp_temp_boxes = []
                temp_temp_actions = []
                temp_temp_boxes.append(group_box)
                for person in self.anns[sid][src_fid]['persons']:
                    pg_id = person['group_id']
                    if int(pg_id) == group_box[0]:
                        box = person['bboxes']
                        action = IDIVIDUAL_ACTIVITIES_ID[person['individual_activity']]
                        y1, x1, y2, x2 = box
                        w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                        temp_temp_boxes.append([int(pg_id), w1, h1, w2, h2])
                        temp_temp_actions.append(action)
                temp_boxes.append(temp_temp_boxes)  # for each image
                temp_actions.append(temp_temp_actions)
                temp_bboxes_num.append(len(temp_temp_actions))
            '''
            [
            [[group box], [person box], [person box], ... (all with same id)],
            [...],
            ...
            ]
            '''
            '''
            [
            [action1, action2,..., ],
            [...],
            ...
            ]
            '''
            '''
            [
            bbox_num1, bbox_num2, ...
            ]
            '''
            groups_num.append(temp_group_num)
            bboxes_num.append(temp_bboxes_num)

            for i, boxes_in_group in enumerate(temp_boxes):
                while len(boxes_in_group) != self.num_boxes + 1:
                    boxes_in_group.append([4, -2, -2, -2, -2])
                    temp_actions[i].append(5)

            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            activities.append(temp_activities)


        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float).reshape(-1, groups_num[i], self.num_boxes,5)  # the numbers of boxes are equal (including the group box)
        actions = np.array(actions, dtype=np.int32).reshape(-1, groups_num[i], self.num_boxes)
        # bboxes = np.array(bboxes, dtype=np.float64).reshape(-1, self.num_boxes, 4)  # the numbers of boxes are equal (including the group box)
        # actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        return images, bboxes, actions, activities, bboxes_num



    def load_samples_sequence(self, select_frames):
        """
        load samples sequence for training

        Returns:
            pytorch tensors
        """
        OH, OW = self.feature_size

        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        real_bboxes_num = []
        # groups_num = []



        for i, (sid, src_fid, fid) in enumerate(select_frames):  # single frame
            print('selected frames:', sid, src_fid, fid)

            # sequence id, ann id, frame id
            img = Image.open(self.images_path + '/ActivityDataset/seq%02d/frame%04d.jpg' % (sid, fid))  # frame_fid
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)

            # print(self.anns)
            # group_boxes = []
            # temp_activities = []
            # temp_group_num = 0
            # # print('anns:', self.anns[sid][src_fid])
            # if len(self.anns[sid][src_fid]['groups']) > 0:
            #     for group in self.anns[sid][src_fid]['groups']:
            #         temp_group_num += 1
            #         gg_id = group['group_id']
            #         box = group['bboxes']
            #         y1, x1, y2, x2 = box
            #         w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
            #         group_boxes.append([int(gg_id), w1, h1, w2, h2])
            #         temp_activities.append(GROUP_ACTIVITIES_ID[group['group_activity']])
            # else:
            #     temp_group_num = 1
            #     group_boxes.append([0, 0, 0, OW, OH])  # there is no group in the image
            #     temp_activities.append(-1)
            #
            #
            # temp_boxes = []
            # temp_actions = []
            # temp_bboxes_num = []
            # for i, group_box in enumerate(group_boxes):
            #     temp_temp_boxes = []
            #     temp_temp_actions = []
            #     temp_temp_boxes.append(group_box)
            #     for person in self.anns[sid][src_fid]['persons']:
            #         pg_id = person['group_id']
            #         if int(pg_id) == group_box[0]:
            #             box = person['bboxes']
            #             action = IDIVIDUAL_ACTIVITIES_ID[person['individual_activity']]
            #             y1, x1, y2, x2 = box
            #             w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
            #             temp_temp_boxes.append([int(pg_id), w1, h1, w2, h2])
            #             temp_temp_actions.append(action)
            #     temp_boxes.append(temp_temp_boxes)
            #     temp_actions.append(temp_temp_actions)
            #     temp_bboxes_num.append(len(temp_temp_actions))
            # '''
            # [
            # [[group box], [person box], [person box], ... (all with same id)],
            # [...],
            # ...
            # ]
            # '''
            # '''
            # [
            # [action1, action2,..., ],
            # [...],
            # ...
            # ]
            # '''
            # '''
            # [
            # bbox_num1, bbox_num2, ...
            # ]
            # '''
            # groups_num.append(temp_group_num)
            # bboxes_num.append(temp_bboxes_num)
            #
            # for i, boxes_in_group in enumerate(temp_boxes):
            #     while len(boxes_in_group) != self.num_boxes + 1:
            #         boxes_in_group.append([-2, -2, -2, -2, -2])
            #         temp_actions[i].append(-2)
            #
            # bboxes.append(temp_boxes)
            # actions.append(temp_actions)
            # activities.append(temp_activities)
            # print('bboxes:', bboxes)

            if len(self.anns[sid][src_fid]['groups']) > 0:
                for group in self.anns[sid][src_fid]['groups']:
                    gg_id = int(group['group_id'])
                    group_box = group['bboxes']
                    gy1, gx1, gy2, gx2 = group_box
                    # gw1, gh1, gw2, gh2 = gx1 * OW, gy1 * OH, gx2 * OW, gy2 * OH
                    gw1, gh1, gw2, gh2 = gx1 * img.shape[2], gy1 * img.shape[1], gx2 * img.shape[2], gy2 * img.shape[1]
                    # print('xyxy', gx1 * img.shape[2], gy1 * img.shape[1], gx2 * img.shape[2], gy2 * img.shape[1])
                    if (gw1 > gw2 or gh1 > gh2 or (gw2-gw1)*(gh2-gh2)<10) and len(self.anns[sid][src_fid]['groups']) == 1:
                        group_flag = 0
                        # activities.append(4)
                        # images.append(img)
                        # temp_boxes = []
                        # temp_actions = []
                        # for person in self.anns[sid][src_fid]['persons']:
                        #     pg_id = person['group_id']
                        #     if int(pg_id) == 0:
                        #         person_box = person['bboxes']
                        #         py1, px1, py2, px2 = person_box
                        #         pw1, ph1, pw2, ph2 = px1 * OW, py1 * OH, px2 * OW, py2 * OH
                        #         dw1, dh1, dw2, dh2 = px1 * img.shape[2], py1 * img.shape[1], px2 * img.shape[2], py2 * \
                        #                              img.shape[1]
                        #         action = IDIVIDUAL_ACTIVITIES_ID[person['individual_activity']]
                        #         temp_boxes.append([pw1, ph1, pw2, ph2])
                        #         temp_actions.append(action)
                        # real_bboxes_num.append(len(temp_boxes))
                        # while len(temp_boxes) != self.num_boxes:
                        #     temp_boxes.append([-2, -2, -2, -2])
                        #     temp_actions.append(5)
                        # bboxes_num.append(len(temp_boxes))
                        # bboxes.append(temp_boxes)
                        # actions.append(temp_actions)
                    else:
                        group_flag = 1
                        # img_paint_black = out_group_black([gw1, gh1, gw2, gh2], img)
                        # images.append(img_paint_black)
                        images.append(img)
                        imagetodraw = Image.fromarray(img.transpose(1, 2, 0))
                        draw = ImageDraw.Draw(imagetodraw)
                        activities.append(GROUP_ACTIVITIES_ID[group['group_activity']])
                        temp_boxes = []
                        temp_actions = []
                        for person in self.anns[sid][src_fid]['persons']:
                            pg_id = person['group_id']
                            if int(pg_id) == gg_id:
                                person_box = person['bboxes']
                                py1, px1, py2, px2 = person_box
                                pw1, ph1, pw2, ph2 = px1 * OW, py1 * OH, px2 * OW, py2 * OH
                                dw1, dh1, dw2, dh2 = px1 * img.shape[2], py1 * img.shape[1], px2 * img.shape[2], py2 * \
                                                     img.shape[1]
                                action = IDIVIDUAL_ACTIVITIES_ID[person['individual_activity']]
                                temp_boxes.append([pw1, ph1, pw2, ph2])
                                temp_actions.append(action)
                        #         draw.rectangle([dw1, dh1, dw2, dh2], outline="red", width=4)
                        # imagetodraw.show()
                        # plt.imshow(img_paint_black.transpose(1, 2, 0))
                        # plt.show()
                        real_bboxes_num.append(len(temp_boxes))
                        while len(temp_boxes) != self.num_boxes:
                            temp_boxes.append([-2, -2, -2, -2])
                            temp_actions.append(5)
                        bboxes_num.append(len(temp_boxes))
                        bboxes.append(temp_boxes)
                        actions.append(temp_actions)

            else:
                group_flag = 0
                # activities.append(4)
                # images.append(img)
                # temp_boxes = []
                # temp_actions = []
                # imagetodraw = Image.fromarray(img.transpose(1, 2, 0))
                # draw = ImageDraw.Draw(imagetodraw)
                # for person in self.anns[sid][src_fid]['persons']:
                #     pg_id = person['group_id']
                #     if int(pg_id) == 0:
                #         person_box = person['bboxes']
                #         py1, px1, py2, px2 = person_box
                #         pw1, ph1, pw2, ph2 = px1 * OW, py1 * OH, px2 * OW, py2 * OH
                #         dw1, dh1, dw2, dh2 = px1 * img.shape[2], py1 * img.shape[1], px2 * img.shape[2], py2 * \
                #                              img.shape[1]
                #         action = IDIVIDUAL_ACTIVITIES_ID[person['individual_activity']]
                #         temp_boxes.append([pw1, ph1, pw2, ph2])
                #         temp_actions.append(action)
                # #         draw.rectangle([dw1, dh1, dw2, dh2], outline="red", width=4)
                # # imagetodraw.show()
                # real_bboxes_num.append(len(temp_boxes))
                # while len(temp_boxes) != self.num_boxes:
                #     temp_boxes.append([-2, -2, -2, -2])
                #     temp_actions.append(5)
                # bboxes_num.append(len(temp_boxes))
                # bboxes.append(temp_boxes)
                # actions.append(temp_actions)
        '''
        bboxes:
        [
        [[person box1], [person box2],...,](image_group1)
        [[person box1], [person box2],...,](image_group2)
        ]
        actions:
        [
        [action1, action2,...,](image_group1)
        [action1, action2,...,](image_group2)
        ]
        activities:
        [
        activity1, activity2, ...
        ]
        '''
        # # images = np.stack(images)
        # activities = np.array(activities, dtype=np.int32)
        # bboxes_num = np.array(bboxes_num, dtype=np.int32)
        # real_bboxes_num = np.array(real_bboxes_num, dtype=np.int32)
        # # bboxes = np.array(bboxes, dtype=np.float).reshape(-1, groups_num[i], self.num_boxes+1,5)  # the numbers of boxes are equal (including the group box)
        # # actions = np.array(actions, dtype=np.int32).reshape(-1, groups_num[i], self.num_boxes)
        # bboxes = np.array(bboxes, dtype=np.float64).reshape(-1, self.num_boxes, 4)  # the numbers of boxes are equal (including the group box)
        # actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)
        #
        # # convert to pytorch tensor
        # # images = torch.from_numpy(images).float()
        # bboxes = torch.from_numpy(bboxes).float()
        # actions = torch.from_numpy(actions).long()
        # activities = torch.from_numpy(activities).long()
        # bboxes_num = torch.from_numpy(bboxes_num).int()
        # real_bboxes_num = torch.from_numpy(real_bboxes_num).int()


        # the dimension should be equal to each other -> batch integration
        # print('tensor:', bboxes, actions, activities, bboxes_num)
        return images, bboxes, actions, activities, bboxes_num, real_bboxes_num, group_flag

    def collate_fn(self, batch):
        '''
        stack the images and labels
        construct sequence for each group
        one image, one bboxes tensor
        '''
        B_images = []
        B_bboxes = []
        B_activities = []
        B_actions = []
        B_bboxes_num = []
        if self.is_finetune:
            if self.is_training:
                num_frames = 1
            else:
                num_frames = self.num_frames
        else:
            num_frames = self.num_frames

        for i, item in enumerate(batch):
            images, bboxes, actions, activities, bboxes_num, _, _ = item  # sequence 1
            images = np.array(images)
            bboxes = np.array(bboxes, dtype=np.float64)
            actions = np.array(actions, dtype=np.int32)
            activities = np.array(activities, dtype=np.int32)
            bboxes_num = np.array(bboxes_num, dtype=np.int32)
            images = torch.from_numpy(images).float().to(self.device)
            bboxes = torch.from_numpy(bboxes).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            activities = torch.from_numpy(activities).long().to(self.device)
            bboxes_num = torch.from_numpy(bboxes_num).int().to(self.device)

            images = re_organize_seq(images, num_frames)
            bboxes = re_organize_seq(bboxes, num_frames)
            actions = re_organize_seq(actions, num_frames)
            activities = re_organize_seq(activities, num_frames)
            bboxes_num = re_organize_seq(bboxes_num, num_frames)
            B_images.append(images)
            B_bboxes.append(bboxes)
            B_activities.append(activities)
            B_actions.append(actions)
            B_bboxes_num.append(bboxes_num)

        B_images = torch.cat(B_images)
        B_bboxes = torch.cat(B_bboxes)
        B_activities = torch.cat(B_activities)
        B_actions = torch.cat(B_actions)
        B_bboxes_num = torch.cat(B_bboxes_num)




            # for j, image in enumerate(images):
            #     B_images.append(image)
            #     B_bboxes.append(bboxes[j])
            #     B_activities.append(activities[j])
            #     B_actions.append(actions[j])
            #     B_bboxes_num.append(bboxes_num[j])

        # B_images = np.stack(B_images)
        # B_images = torch.from_numpy(B_images).float()
        #
        # B_activities = np.array(B_activities, dtype=np.int32)
        # B_bboxes_num = np.array(B_bboxes_num, dtype=np.int32)
        # B_bboxes = np.array(B_bboxes, dtype=np.float64).reshape(-1, self.num_boxes, 4)  # the numbers of boxes are equal (including the group box)
        # B_actions = np.array(B_actions, dtype=np.int32).reshape(-1, self.num_boxes)
        #
        # # convert to pytorch tensor
        # B_bboxes = torch.from_numpy(B_bboxes).float()
        # B_actions = torch.from_numpy(B_actions).long()
        # B_activities = torch.from_numpy(B_activities).long()
        # B_bboxes_num = torch.from_numpy(B_bboxes_num).int()


        return B_images, B_bboxes, B_actions, B_activities, B_bboxes_num




