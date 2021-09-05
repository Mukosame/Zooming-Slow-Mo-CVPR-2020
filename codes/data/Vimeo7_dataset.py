'''
Vimeo7 dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class Vimeo7Dataset(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(Vimeo7Dataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        self.half_N_frames = opt['N_frames'] // 2
        self.LR_N_frames = 1 + self.half_N_frames
        assert self.LR_N_frames > 1, 'Error: Not enough LR frames to interpolate'
        # determine the LQ frame list
        '''
        N | frames
        1 | error
        3 | 0,2
        5 | 0,2,4
        7 | 0,2,4,6
        '''
        self.LR_index_list = []
        for i in range(self.LR_N_frames):
            self.LR_index_list.append(i*2)

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        # low resolution inputs
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True
        # directly load image keys
        if opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            cache_keys = opt['cache_keys']
        else:
            cache_keys = 'Vimeo7_train_keys.pkl'
        logger.info('Using cache keys - {}.'.format(cache_keys))
        self.paths_GT = pickle.load(open('./data/{}'.format(cache_keys), 'rb'))

        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(
            osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(
            osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(
            osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()

        scale = self.opt['scale']
        # print(scale)
        N_frames = self.opt['N_frames']
        GT_size = self.opt['GT_size']
        key = self.paths_GT['keys'][index]
        name_a, name_b = key.split('_')

        center_frame_idx = random.randint(2, 6)  # 2<= index <=6

        # determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 7:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 1:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   7) or (center_frame_idx - self.half_N_frames * interval < 1):
                center_frame_idx = random.randint(2, 6)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()

        self.LQ_frames_list = []
        for i in self.LR_index_list:
            self.LQ_frames_list.append(neighbor_list[i])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))

        # get the GT image (as the center frame)
        img_GT_l = []
        for v in neighbor_list:
            if self.data_type == 'mc':
                img_GT = self._read_img_mc_BGR(
                    self.GT_root, name_a, name_b, '{}.png'.format(v))
                img_GT = img_GT.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_GT = util.read_img(
                    self.GT_env, key + '_{}'.format(v), (3, 256, 448))
            else:
                img_GT = util.read_img(None, osp.join(
                    self.GT_root, name_a, name_b, 'im{}.png'.format(v)))
            img_GT_l.append(img_GT)

       # get LQ images
        LQ_size_tuple = (3, 64, 112) if self.LR_input else (3, 256, 448)
        img_LQ_l = []
        for v in self.LQ_frames_list:
            if self.data_type == 'mc':
                img_LQ = self._read_img_mc(
                    osp.join(self.LQ_root, name_a, name_b, '/{}.png'.format(v)))
                img_LQ = img_LQ.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_LQ = util.read_img(
                    self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)
            else:
                img_LQ = util.read_img(None,
                                       osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size,
                              rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT_l = [v[rnd_h_HR:rnd_h_HR + GT_size,
                              rnd_w_HR:rnd_w_HR + GT_size, :] for v in img_GT_l]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size,
                              rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT_l = [v[rnd_h:rnd_h + GT_size,
                              rnd_w:rnd_w + GT_size, :] for v in img_GT_l]

            # augmentation - flip, rotate
            img_LQ_l = img_LQ_l + img_GT_l
            rlt = util.augment(
                img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-N_frames]
            img_GT_l = rlt[-N_frames:]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GTs = np.stack(img_GT_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTs = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img_GTs, (0, 3, 1, 2)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GTs, 'key': key}

    def __len__(self):
        return len(self.paths_GT['keys'])
