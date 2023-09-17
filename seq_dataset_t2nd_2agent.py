__author__ = 'gaozhifeng'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random

logger = logging.getLogger(__name__)

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, is_training=True, num_views=1, img_channel=3, baseline='1_NN_4_img_GCN',
                  eval_batch_size=1, n_epoch=1, args=None):
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    try:
        test_seq_length = args.eval_num_step + args.num_past
    except:
        print('No args.eval_num_step, use seq_length as test_seq_length')
        test_seq_length = seq_length
    # if dataset_name in ['circle_motion', 'sumo_sanjose-2021-11-06_20.28.57_30', 'carla_town02_20211201', 'students003']:
    input_param = {'paths': valid_data_list,
                   'image_width': img_width,
                   'minibatch_size': eval_batch_size,
                   'seq_length': test_seq_length,
                   'input_data_type': 'float32',
                   'name': dataset_name + ' iterator',
                    'num_views': num_views,
                   'img_channel': img_channel,
                   'baseline': baseline,
                   'n_epoch': n_epoch,
                   'n_cl_step': args.num_cl_step,
                   'cl_mode': args.cl_mode,}
    input_handle = DataProcess(input_param)
    if is_training:
        train_input_param = {'paths': train_data_list,
                            'image_width': img_width,
                            'minibatch_size': batch_size,
                            'seq_length': seq_length,
                            'input_data_type': 'float32',
                            'name': dataset_name + ' iterator',
                            'num_views': num_views,
                            'img_channel': img_channel,
                            'baseline': baseline,
                            'n_epoch': n_epoch,
                            'n_cl_step': args.num_cl_step,
                            'cl_mode': args.cl_mode,}
        train_input_handle = DataProcess(train_input_param)
        train_input_handle = train_input_handle.get_train_input_handle()
        train_input_handle.begin(do_shuffle=True)
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        return train_input_handle, test_input_handle
    else:
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        return test_input_handle


class InputHandle:
    def __init__(self, datas, datas_t2nd, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.num_views = input_param.get('num_views', 4)
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.datas_t2nd = datas_t2nd
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.img_channel = input_param.get('img_channel', 3)
        self.n_epoch = input_param.get('n_epoch', 1)
        self.n_cl_step = input_param.get('n_cl_step', 1)
        self.cl_mode = input_param.get('cl_mode', 1)

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True, epoch=None):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        if epoch:
            # self.current_position = int(self.total() * (epoch // self.n_cl_step) * self.n_cl_step / self.n_epoch)
            self.current_position = int(self.total() * epoch / self.n_epoch)
        else:
            self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self, epoch=None):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        # elif epoch is not None and self.current_position != 0 and \
        #         self.current_position + self.minibatch_size >= \
        #         self.total() * min(((epoch // self.n_cl_step) + 1) * self.n_cl_step / self.n_epoch, 1.):
        #     return True
        elif epoch is not None and self.current_position != 0 and \
                self.current_position + self.minibatch_size >= \
                self.total() * min((epoch + self.n_cl_step) / self.n_epoch, 1.):
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width,
             self.img_channel * self.num_views)).astype(self.input_data_type)
        input_t2nd_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width,
             1 * self.num_views)).astype(self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            # print('begin: {}, end: {}'.format(begin, end))
            data_slice = self.datas[begin:end, :, :, :]
            data_t2nd_slice = self.datas_t2nd[begin:end, :, :, :]
            # print('data_t2nd_slice.shape: ', data_t2nd_slice.shape)
            # data_t2nd_slice.shape:  (14, 128, 128, 1)
            # print(input_batch[i, :self.current_input_length, :, :, :].shape, data_slice.shape)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            input_t2nd_batch[i, :self.current_input_length, :, :, :] = data_t2nd_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        input_t2nd_batch = input_t2nd_batch.astype(self.input_data_type)
        input_batch = np.concatenate([input_batch, input_t2nd_batch], axis=-1)
        # print('input_batch.shape: ', input_batch.shape)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))

class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.image_width = input_param['image_width']

        # self.train_person = os.listdir(self.paths[0])
        # self.test_person = os.listdir(self.paths[0])

        self.input_param = input_param
        self.seq_len = input_param['seq_length']
        self.num_views = 1  # input_param.get('num_views', 4)
        self.img_channel = input_param.get('img_channel', 3)
        self.baseline = input_param.get('baseline', '1_NN_4_img_GCN')
        self.n_epoch = input_param.get('n_epoch', 1)
        self.n_cl_step = input_param.get('n_cl_step', 1)
        self.cl_mode = input_param.get('cl_mode', 1)

    def load_data(self, paths, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        folder_name_list = [0, 1]
        time_scale = 1
        old_index = 0
        indices = []
        for idx, each_folder_name in enumerate(folder_name_list):
            # print(each_folder_name)  # 0
            # 1
            path = os.path.join(paths[0], '_out_{}'.format(each_folder_name))
            t2nd_path = os.path.join(paths[0]+'_KNN_t2no', '_out_{}'.format(each_folder_name))
            print('begin load data ' + str(path) + ', ' + str(t2nd_path))

            frames_np = []
            frames_file_name = []

            t2nd_frames_np = []
            t2nd_frames_file_name = []

            c_dir_list = sorted(os.listdir(path))
            for c_dir in c_dir_list:
                c_dir_path = os.path.join(path, c_dir)
                t2nd_c_dir_path = os.path.join(t2nd_path, 't2no_'+c_dir)
                if not os.path.exists(t2nd_c_dir_path):
                    continue
                frame_im = Image.open(os.path.join(c_dir_path, ))
                frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                # print('frame_np.shape: ', frame_np.shape)  # (600, 600, 4)
                frames_np.append(frame_np[..., :-1])
                frames_file_name.append(c_dir_path)

                # try:
                # print('os.path.join(t2nd_c_dir_path,): ', os.path.join(t2nd_c_dir_path,))
                t2nd_frame_im = Image.open(os.path.join(t2nd_c_dir_path, ))
                # except:
                #     t2nd_frame_im = np.ones_like(frame_im)[..., -1]
                t2nd_frame_np = np.array(t2nd_frame_im)  # (1000, 1000) numpy array
                # print('t2nd_frame_np.max(): {}, t2nd_frame_np.min(): {}'.format(t2nd_frame_np.max(), t2nd_frame_np.min()))
                # t2nd_frame_np.max(): 10, t2nd_frame_np.min(): 1
                t2nd_frames_np.append(t2nd_frame_np)
                t2nd_frames_file_name.append(t2nd_c_dir_path)

            frames_np_array = np.asarray(frames_np) # [:200]
            data = np.zeros((frames_np_array.shape[0], self.image_width, self.image_width, self.img_channel))
            for i in range(len(frames_np_array)):
                temp = np.float32(frames_np_array[i])
                data[i, :, :, :] = cv2.resize(temp, (self.image_width, self.image_width)) / 255
            new_data = np.zeros((frames_np_array.shape[0], self.image_width, self.image_width,
                                 self.img_channel))
            for i in range(1):
                new_data = data
            each_data = new_data[::time_scale]

            t2nd_frames_np_array = np.asarray(t2nd_frames_np) # [:200]
            print('t2nd_frames_np_array.shape: ', t2nd_frames_np_array.shape)
            t2nd_data = np.zeros((t2nd_frames_np_array.shape[0], self.image_width, self.image_width, 1))
            for i in range(len(t2nd_frames_np_array)):
                temp = np.float32(t2nd_frames_np_array[i])
                t2nd_data[i, :, :, 0] = cv2.resize(temp, (self.image_width, self.image_width))
            new_data_t2nd = np.zeros((t2nd_frames_np_array.shape[0], self.image_width, self.image_width,
                                 1))
            for i in range(1):
                new_data_t2nd = t2nd_data
            each_data_t2nd = new_data_t2nd[::time_scale]

            each_indices = []
            label_1 = []
            index = old_index + len(each_data) - 1
            old_index_2 = old_index + len(each_data)
            while index >= self.seq_len - 1 + old_index:
                each_indices.append(index - self.seq_len + 1)
                index -= 1
                label_1.append(0)
            old_index = old_index_2
            data_1 = each_data
            data_1_t2nd = each_data_t2nd
            indices_1 = each_indices[::-1]
            if idx == 0:
                final_data = data_1
                final_data_t2nd = data_1_t2nd
                # first_data = data_1
                indices = indices_1
            else:
                final_data = np.concatenate([final_data, data_1], axis=-1)
                final_data_t2nd = np.concatenate([final_data_t2nd, data_1_t2nd], axis=-1)
                # indices.extend(indices_1)
        print("there are " + str(final_data.shape[0]) + " pictures with shape {}".format(final_data.shape))
        print("there are " + str(len(indices)) + " sequences")
        data = final_data

        return data, final_data_t2nd, indices

    def get_train_input_handle(self):
        train_data, train_data_t2nd, train_indices = self.load_data(self.paths, mode='train')
        print('train_indices: ', train_indices)
        return InputHandle(train_data, train_data_t2nd, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_data_t2nd, test_indices = self.load_data(self.paths, mode='test')
        print('test_indices: ', test_indices)
        return InputHandle(test_data[:200], test_data_t2nd[:200], test_indices[:200], self.input_param)

