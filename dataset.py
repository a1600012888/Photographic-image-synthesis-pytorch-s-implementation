#!/usr/bin/env mdlf

from cityscapesscripts.preparation.json2labelImg import createLabelImage

from cityscapesscripts.helpers.annotation import Annotation
import os
import numpy as np
import skimage.io as io
from torch.utils.data import Dataset, DataLoader
import torch
import scipy.misc as misc
import torch.nn.functional as F


class TorchDataset(Dataset):

    def __init__(self, name = 'train', resolution = 512):

        assert name in ['train', 'val', 'test']

        self.name = name
        self.super_resolution = resolution
        self.label_path = '/home/zhangtianyuan/sfzhang/gtFine_trainvaltest/gtFine/' + name + '/'
        self.img_path = '/home/zhangtianyuan/sfzhang/leftImg8bit_trainvaltest/leftImg8bit/' + name + '/'

        self.ids = []
        self.id2label = {}
        self.id2img = {}
        _r, dirs, _f = next(os.walk(self.label_path))

        print('All dirs to walk: ')
        #print(dirs)
        for dir in dirs:
            full_dir = os.path.join(self.label_path, dir)
            print(dir)
            new_ids = get_all_json(full_dir)

            for id in new_ids:
                self.id2label[id] = os.path.join(full_dir, id)
                self.id2img[id] = os.path.join(self.img_path, dir, (id[:-20] + 'leftImg8bit.png'))

            self.ids = self.ids + new_ids

        self.instance_per_epoch = len(self.ids)
        print('Datset name: {}, Instance per epoch: {}'.format(name, self.instance_per_epoch))

    def __len__(self):
        return self.instance_per_epoch

    def __getitem__(self, idx):

        id = self.ids[idx]
        label, img = self.read_label_img(id)

        sample = {
            'label': label,
            'data': img
        }

        return self.totensor_and_process(sample)

    def totensor_and_process(self, sample):

        assert sample['data'].ndim == 3 and sample['label'].ndim == 3, 'shape!!'

        assert sample['label'].shape[-1] == 4, 'label shape wrong!!'

        #resize img：
        sample['data'] = misc.imresize(sample['data'], (self.super_resolution, self.super_resolution * 2))

        # [X, Y, 3] -> [3, X, Y]
        sample['data'] = sample['data'].transpose((2, 0, 1))
        sample['label'] = sample['label'].transpose((2, 0, 1))

        sample['data'] = torch.tensor(sample['data'], dtype = torch.float)
        sample['label'] = torch.tensor(sample['label'], dtype = torch.float)

        return sample

    def test_ids(self):

        for i in range(self.instance_per_epoch):
            id = self.ids[i]

            print('id: {}, label exist: {}, img exist: {}'.format(id,
                                                                  os.path.isfile(self.id2label[id]),
                                                                  os.path.isfile(self.id2img[id])))
            print('label: {}, img: {}'.format(self.id2label[id], self.id2img[id]))
            '''
            print(self.id2label[id])
            print(self.id2img[id])
            label = read_label(self.id2label[id])

            label = np.asarray(label)

            img = io.imread(self.id2img[id])

            print(label.shape, np.array(img).shape)
            '''
    def read_label_img(self, id):
        label = read_label(self.id2label[id])

        label = np.asarray(label)

        img = np.array(io.imread(self.id2img[id]))

        return label, img


'''
def read_label(json_path):
    color_path = ''.join(json_path[:-13]) + 'color.png'

    print(color_path)

    label = io.imread(color_path)

    print(label)
'''
def read_label(json_path):
    assert os.path.isfile(json_path), 'file not exists'

    annotation = Annotation()
    annotation.fromJsonFile(json_path)
    labelImg = createLabelImage(annotation, 'color')

    return labelImg
def get_all_json(dir_path = '/home/zhangtianyuan/sfzhang/gtFine_trainvaltest/gtFine/train/aachen/'):

    os.path.isdir(dir_path)
    files = os.listdir(dir_path)

    json_files = [i for i in files if i.endswith('json') and not i.startswith('.')]

    #a = np.random.randint(0, len(json_files))
    #print(json_files[a])
    #print(json_files[1])
    #read_label(dir_path +  json_files[0][2:])

    return json_files
#get_all_json()
def test():

    ds_train = TorchDataset('train')

    print('len of dataset: {}'.format(ds_train.instance_per_epoch))

    dl_train = DataLoader(ds_train, batch_size=1)

    print('loader ready!')

    '''
    i, batch = next(enumerate(dl_train))

    print(batch['data'], batch['label'])

    print(batch['data'].size(), batch['label'].size())

    '''
    #ds_train.test_ids()

if __name__ == '__main__':
    test()
# vim: ts=4 sw=4 sts=4 expandtab
