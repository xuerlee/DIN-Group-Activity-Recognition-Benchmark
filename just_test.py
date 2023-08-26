import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random
import xml.etree.ElementTree as ET
import sys

FRAMES_NUM = {1: 1, 2: 347, 3: 194, 4: 257}

FRAMES_SIZE = {1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720)}

IDIVIDUAL_ACTIVITIES = ['crossing', 'waiting', 'queueing', 'walking', 'talking', 'biking']  # individual activity
# SECOND_ACTIVITIES = ['crossing', 'waiting', 'queueing', 'walking', 'talking', 'None']
GROUP_ACTIVITIES = ['waiting', 'queueing', 'walking', 'talking']  # group activity

ACTIONS_ID = {a: i for i, a in enumerate(IDIVIDUAL_ACTIVITIES)}
# SECOND_ACTIONS_ID = {a: i for i, a in enumerate(SECOND_ACTIVITIES)}
ACTIVITIES_ID = {a: i for i, a in enumerate(GROUP_ACTIVITIES)}

Action6to5 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
Activity5to4 = {0: 0, 1: 1, 2: 2, 3: 0, 4: 3}

path = '/home/jiqqi/data/new-new-collective/collective_activity_01-04'  # data path for the new_new_collective dataset
train_seqs = [1]  # video id list of train set
test_seqs = []  # video id list of test set

def new_new_collective_read_annotations(path, sid):
    annotations = {}
    path = path + '/seq%02d/annotations.xml' % sid

    ann_file = open(path)
    tree = ET.parse(ann_file)
    root = tree.getroot()

    group_activity = None
    actions = []
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
                        bboxes = [ytl / H, xtl / W, xbr / H, ybr / W]


                        group_id = box.find('attribute[@name="group id"]').text
                        group_activity = box.find('attribute[@name="group activity"]').text

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
                        bboxes = [ytl / H, xtl / W, (ytl + h) / H, (xtl + w) / W]

                        group_id = box.find('attribute[@name="group id"]').text
                        person_id = box.find('attribute[@name="person id"]').text
                        individual_activity = box.find('attribute[@name="individual activity"]').text
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
    return [(s, f) for s in anns for f in anns[s]]  # (seq_id, frame_id)

train_anns=new_new_collective_read_dataset(path, train_seqs)
train_frames=new_new_collective_all_frames(train_anns)

print(train_anns)
print(train_frames)

