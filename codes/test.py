'''
test Zooming Slow-Mo models on arbitrary datasets
write to txt log file
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

# import utils.util as util
# import data.util as data_util
# import models.modules.Sakuya_arch as Sakuya_arch

def test_index_generation(skip, N_out, len_in):
    '''
    params: 
    skip: if skip even number; 
    N_out: number of frames of the network; 
    len_in: length of input frames

    example:
  len_in | N_out  | times | (no skip)                  |   (skip)
    5    |   3    |  4/2  | [0,1], [1,2], [2,3], [3,4] | [0,2],[2,4]
    7    |   3    |  5/3  | [0,1],[1,2][2,3]...[5,6]   | [0,2],[2,4],[4,6] 
    5    |   5    |  2/1  | [0,1,2] [2,3,4]            | [0,2,4]
    '''
    print(skip, N_out, len_in)
    # number of input frames for the network
    N_in = 1 + N_out // 2
    # input length should be enough to generate the output frames
    assert N_in <= len_in

    sele_list = []
    if skip: 
        right = N_out # init
        while (right <= len_in):
            h_list = [right-N_out+x for x in range(N_out)]
            l_list = h_list[::2]
            right += (N_out - 1)
            sele_list.append([l_list,h_list])
    else:
        right = N_out # init
        right_in = N_in
        while (right_in <= len_in):
            h_list = [right-N_out+x for x in range(N_out)]
            l_list = [right_in-N_in+x for x in range(N_in)]
            right += (N_out - 1)
            right_in += (N_in - 1)
            sele_list.append([l_list,h_list])
    # check if it covers the last image, if not, we should cover it 
    if (skip) and (right != len_in - 1):
        h_list = [len_in - N_out + x for x in range(N_out)]
        l_list = h_list[::2]
        sele_list.append([l_list,h_list])
    if (not skip) and (right_in != len_in - 1):
        right = len_in * 2 - 1;
        h_list = [right-N_out+x for x in range(N_out)]
        l_list = [len_in - N_in + x for x in range(N_in)]
        sele_list.append([l_list,h_list])        
    return sele_list

def main():
    scale = 4
    N_ot = 7 #3
    N_in = 1+ N_ot // 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #### model 
    #### TODO: change your model path here
    model_path = '../experiments/pretrained_models/xiang2020zooming.pth'
    model = Sakuya_arch.LunaTokis(64, N_ot, 8, 5, 40)

    #### dataset
    data_mode = 'Custom' #'Vid4' #'SPMC'#'Middlebury'#

    if data_mode == 'Vid4':
        test_dataset_folder = '/data/xiang/SR/Vid4/LR/*'
    if data_mode == 'SPMC':
        test_dataset_folder = '/data/xiang/SR/spmc/*'
    if data_mode == 'Custom':
        test_dataset_folder = '../test_example/*' # TODO: put your own data path here

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'replicate'
    save_imgs = False #True#
    if 'Custom' in data_mode: save_imgs = True
    ############################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')
    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    model_params = util.get_model_total_params(model)

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))

    def read_image(img_path):
        '''read one image from img_path
        Return img: HWC, BGR, [0,1], numpy
        '''
        img_GT = cv2.imread(img_path)
        img = img_GT.astype(np.float32) / 255.
        return img

    def read_seq_imgs(img_seq_path):
        '''read a sequence of images'''
        img_path_l = sorted(glob.glob(img_seq_path + '/*'))
        # img_path_l.sort(key = lambda x: int(x.split('/')[-1][5:-4]))
        img_l = [read_image(v) for v in img_path_l]
        # stack to TCHW, RGB, [0,1], torch
        imgs = np.stack(img_l, axis=0)
        imgs = imgs[:, :, :, [2, 1, 0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
        return imgs
    

    def single_forward(model, imgs_in):
        with torch.no_grad():
            # imgs_in.size(): [1,n,3,h,w]
            b,n,c,h,w = imgs_in.size()
            h_n = int(4*np.ceil(h/4))
            w_n = int(4*np.ceil(w/4))
            imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
            imgs_temp[:,:,:,0:h,0:w] = imgs_in

            model_output = model(imgs_temp)
            # model_output.size(): torch.Size([1, 3, 4h, 4w])
            model_output = model_output[:, :, :, 0:scale*h, 0:scale*w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))

    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_psnr_y_l = []
    sub_folder_name_l = []
    # total_time = []
    # for each sub-folder
    for sub_folder in sub_folder_l:
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

        if data_mode == 'SPMC':
            sub_folder = sub_folder + '/LR/'
        img_LR_l = sorted(glob.glob(sub_folder + '/*'))

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []
        if data_mode == 'SPMC':
            sub_folder_GT = osp.join(sub_folder.replace('/LR/', '/truth/'))
        else:
            sub_folder_GT = osp.join(sub_folder.replace('/LR/', '/HR/'))

        if 'Custom' not in data_mode:
            for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT,'*'))):
                img_GT_l.append(read_image(img_GT_path))

        avg_psnr, avg_psnr_sum, cal_n = 0,0,0
        avg_psnr_y, avg_psnr_sum_y = 0,0
        
        if len(img_LR_l) == len(img_GT_l):
            skip = True
        else:
            skip = False
        
        if 'Custom' in data_mode:
            select_idx_list = test_index_generation(False, N_ot, len(img_LR_l))
        else:
            select_idx_list = test_index_generation(skip, N_ot, len(img_LR_l))
        # process each image
        for select_idxs in select_idx_list:
            # get input images
            select_idx = select_idxs[0]
            gt_idx = select_idxs[1]
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            output = single_forward(model, imgs_in)

            outputs = output.data.float().cpu().squeeze(0)            

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output

                outputs = outputs / 4

            # save imgs
            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)
                output_f = outputs[idx,:,:,:].squeeze(0)

                output = util.tensor2img(output_f)
                if save_imgs:                
                    cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(name_idx+1)), output)

                if 'Custom' not in data_mode:
                    #### calculate PSNR
                    output = output / 255.

                    GT = np.copy(img_GT_l[name_idx])

                    if crop_border == 0:
                        cropped_output = output
                        cropped_GT = GT
                    else:
                        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border, :]
                    crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
                    cropped_GT_y = data_util.bgr2ycbcr(cropped_GT, only_y=True)
                    cropped_output_y = data_util.bgr2ycbcr(cropped_output, only_y=True)
                    crt_psnr_y = util.calculate_psnr(cropped_output_y * 255, cropped_GT_y * 255)
                    logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB  PSNR-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_psnr_y))
                    avg_psnr_sum += crt_psnr
                    avg_psnr_sum_y += crt_psnr_y
                    cal_n += 1

        if 'Custom' not in data_mode:
            avg_psnr = avg_psnr_sum / cal_n
            avg_psnr_y = avg_psnr_sum_y / cal_n
    
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_psnr, avg_psnr_y, cal_n))
    
            avg_psnr_l.append(avg_psnr)
            avg_psnr_y_l.append(avg_psnr_y)

    if 'Custom' not in data_mode:
        logger.info('################ Tidy Outputs ################')
        for name, psnr, psnr_y in zip(sub_folder_name_l, avg_psnr_l, avg_psnr_y_l):
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB. '
                       .format(name, psnr, psnr_y))
        logger.info('################ Final Results ################')
        logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
        logger.info('Padding mode: {}'.format(padding))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info('Flip Test: {}'.format(flip_test))
        logger.info('Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} clips. '
                    .format(
                        sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_psnr_y_l) / len(avg_psnr_y_l), len(sub_folder_l)))
        # logger.info('Total Runtime: {:.6f} s Average Runtime: {:.6f} for {} images.'
                    # .format(sum(total_time), sum(total_time)/171, 171))

if __name__ == '__main__':
    # main()
    print(test_index_generation(False, 7, 20))
