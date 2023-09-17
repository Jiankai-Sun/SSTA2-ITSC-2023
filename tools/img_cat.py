from typing import IO
import os
import pathlib

from PIL import Image, ImageDraw
from moviepy.editor import *
import numpy as np
import gc
import cv2
import matplotlib.pyplot as plt
import pylab
import pickle, json
import moviepy.editor as mpy

def img_concat(input_dir='', target_path=''):
    bg_img_path = sorted([i for i in os.listdir(input_dir) if 'bg_' in i])[::2]
    frame_img_path = sorted([i for i in os.listdir(input_dir) if 'frame_' in i])[::2]
    diff_img_path = sorted([i for i in os.listdir(input_dir) if 'diff_' in i])[::2]
    diffT2ND_img_path = sorted([i for i in os.listdir(input_dir) if 'diffT2ND_' in i])[::2]
    mask_img_path = sorted([i for i in os.listdir(input_dir) if 'mask_' in i])[::2]
    t2no_img_path = sorted([i for i in os.listdir(input_dir) if 't2no_' in i])[::2]
    t2nd_img_path = sorted([i for i in os.listdir(input_dir) if 't2nd_' in i])[::2]
    bg_img_list = []
    for each_bg_img_name in bg_img_path:
        each_bg_img_path = os.path.join(input_dir, each_bg_img_name)
        print('Loading img from {}.'.format(each_bg_img_path))
        bg_image = cv2.imread(each_bg_img_path)
        # bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        bg_img_list.append(bg_image)
        bg_img_list.append(np.ones_like(bg_image)[:, :4])
    bg_img = np.concatenate(bg_img_list, axis=1)
    frame_img_list = []
    for each_frame_img_name in frame_img_path:
        each_frame_img_path = os.path.join(input_dir, each_frame_img_name)
        print('Loading img from {}.'.format(each_frame_img_path))
        frame_image = cv2.imread(each_frame_img_path)
        frame_img_list.append(frame_image)
        frame_img_list.append(np.ones_like(frame_image)[:, :4])
    frame_img = np.concatenate(frame_img_list, axis=1)
    diff_img_list = []
    for each_diff_img_name in diff_img_path:
        each_diff_img_path = os.path.join(input_dir, each_diff_img_name)
        print('Loading img from {}.'.format(each_diff_img_path))
        diff_image = cv2.imread(each_diff_img_path)
        # diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)
        diff_img_list.append(diff_image)
        diff_img_list.append(np.ones_like(diff_image)[:, :4])
    diff_img = np.concatenate(diff_img_list, axis=1)
    try:
        diffT2ND_img_list = []
        for each_diffT2ND_img_name in diffT2ND_img_path:
            each_diffT2ND_img_path = os.path.join(input_dir, each_diffT2ND_img_name)
            print('Loading img from {}.'.format(each_diffT2ND_img_path))
            diffT2ND_image = cv2.imread(each_diffT2ND_img_path)
            # diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)
            diffT2ND_img_list.append(diffT2ND_image)
            diffT2ND_img_list.append(np.ones_like(diffT2ND_image)[:, :4])
        diffT2ND_img = np.concatenate(diffT2ND_img_list, axis=1)
    except:
        print('No diffT2ND img found')
        diffT2ND_img = np.ones_like(frame_img)[:1]
    try:
        mask_img_list = []
        for each_mask_img_name in mask_img_path:
            each_mask_img_path = os.path.join(input_dir, each_mask_img_name)
            print('Loading img from {}.'.format(each_mask_img_path))
            mask_image = cv2.imread(each_mask_img_path)
            # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
            mask_img_list.append(mask_image)
            mask_img_list.append(np.ones_like(mask_image)[:, :4])
        mask_img = np.concatenate(mask_img_list, axis=1)
    except:
        print('No mask img found')
        mask_img = np.ones_like(frame_img)[:1]
    t2no_img_list = []
    for each_t2no_img_name in t2no_img_path:
        each_t2no_img_path = os.path.join(input_dir, each_t2no_img_name)
        print('Loading img from {}.'.format(each_t2no_img_path))
        t2no_image = cv2.imread(each_t2no_img_path)
        # t2no_image = cv2.cvtColor(t2no_image, cv2.COLOR_BGR2RGB)
        t2no_img_list.append(t2no_image)
        t2no_img_list.append(np.ones_like(t2no_image)[:, :4])
    t2no_img = np.concatenate(t2no_img_list, axis=1)
    try:
        t2nd_img_list = []
        for each_t2nd_img_name in t2nd_img_path:
            each_t2nd_img_path = os.path.join(input_dir, each_t2nd_img_name)
            print('Loading img from {}.'.format(each_t2nd_img_path))
            t2nd_image = cv2.imread(each_t2nd_img_path)
            # t2no_image = cv2.cvtColor(t2no_image, cv2.COLOR_BGR2RGB)
            t2nd_img_list.append(t2nd_image)
            t2nd_img_list.append(np.ones_like(t2nd_image)[:, :4])
        t2nd_img = np.concatenate(t2nd_img_list, axis=1)
    except:
        print('No t2nd img found')
        t2nd_img = np.ones_like(frame_img)[:1]
    frame = np.concatenate([frame_img, np.ones_like(frame_img)[:4],
                            mask_img, np.ones_like(frame_img)[:4],
                            bg_img, np.ones_like(frame_img)[:4],
                            diff_img, np.ones_like(frame_img)[:4],
                            diffT2ND_img, np.ones_like(frame_img)[:4],
                            t2no_img,  np.ones_like(frame_img)[:4], t2nd_img], axis=0)
    cv2.imwrite(os.path.join(target_path), frame)
    print('Saved to {}'.format(os.path.join(target_path)))


def img2video(input_dir='', target_path=''):
    # bg_img_path = sorted([i for i in os.listdir(input_dir) if 'bg_' in i])[::2]
    # diff_img_path = sorted([i for i in os.listdir(input_dir) if 'diff_' in i])[::2]
    # frame_img_path = sorted([i for i in os.listdir(input_dir) if 'frame_' in i])[::2]
    # mask_img_path = sorted([i for i in os.listdir(input_dir) if 'mask_' in i])[::2]
    t2no_img_path = sorted([i for i in os.listdir(input_dir) if 't2no_' in i])[::2]
    # bg_img_list = []
    # for each_bg_img_name in bg_img_path:
    #     each_bg_img_path = os.path.join(input_dir, each_bg_img_name)
    #     print('Loading img from {}.'.format(each_bg_img_path))
    #     bg_image = cv2.imread(each_bg_img_path)
    #     # bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    #     bg_img_list.append(bg_image)
    #     bg_img_list.append(np.ones_like(bg_image)[:, :4])
    # bg_img = np.concatenate(bg_img_list, axis=1)
    # diff_img_list = []
    # for each_diff_img_name in diff_img_path:
    #     each_diff_img_path = os.path.join(input_dir, each_diff_img_name)
    #     print('Loading img from {}.'.format(each_diff_img_path))
    #     diff_image = cv2.imread(each_diff_img_path)
    #     # diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)
    #     diff_img_list.append(diff_image)
    #     diff_img_list.append(np.ones_like(diff_image)[:, :4])
    # diff_img = np.concatenate(diff_img_list, axis=1)
    # frame_img_list = []
    # for each_frame_img_name in frame_img_path:
    #     each_frame_img_path = os.path.join(input_dir, each_frame_img_name)
    #     print('Loading img from {}.'.format(each_frame_img_path))
    #     frame_image = cv2.imread(each_frame_img_path)
    #     # frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
    #     frame_img_list.append(frame_image)
    #     frame_img_list.append(np.ones_like(frame_image)[:, :4])
    # frame_img = np.concatenate(frame_img_list, axis=1)
    # try:
    #     mask_img_list = []
    #     for each_mask_img_name in mask_img_path:
    #         each_mask_img_path = os.path.join(input_dir, each_mask_img_name)
    #         print('Loading img from {}.'.format(each_mask_img_path))
    #         mask_image = cv2.imread(each_mask_img_path)
    #         # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    #         mask_img_list.append(mask_image)
    #         mask_img_list.append(np.ones_like(mask_image)[:, :4])
    #     mask_img = np.concatenate(mask_img_list, axis=1)
    # except:
    #     print('No mask img found')
    #     mask_img = np.ones_like(frame_img)[:1]
    t2no_img_list = []
    for each_t2no_img_name in t2no_img_path:
        each_t2no_img_path = os.path.join(input_dir, each_t2no_img_name)
        print('Loading img from {}.'.format(each_t2no_img_path))
        t2no_image = cv2.imread(each_t2no_img_path)
        # t2no_image = cv2.cvtColor(t2no_image, cv2.COLOR_BGR2RGB)
        t2no_img_list.append(t2no_image)
        # t2no_img_list.append(np.ones_like(t2no_image)[:, :4])
    clip = mpy.ImageSequenceClip(t2no_img_list, fps=1)
    clip.write_videofile(os.path.join(target_path.replace('png', 'mp4')), fps=1)
    print('Saved to {}'.format(os.path.join(target_path.replace('png', 'mp4'))))


def img_concat_cdna(input_dir='', target_path=''):
    # t2no_img_path = sorted([i for i in os.listdir(input_dir) if 'f5_m2' in i])
    t2no_img_path = sorted([i for i in os.listdir(input_dir) if 'raw' in i])
    t2no_img_list = []
    for each_t2no_img_name in t2no_img_path:
        each_t2no_img_path = os.path.join(input_dir, each_t2no_img_name)
        print('Loading img from {}.'.format(each_t2no_img_path))
        t2no_image = cv2.imread(each_t2no_img_path)
        t2no_image = cv2.cvtColor(t2no_image, cv2.COLOR_BGR2RGB)
        t2no_img_list.append(t2no_image)
    frame = np.concatenate(t2no_img_list, axis=1)
    # clip = mpy.ImageSequenceClip(t2no_img_list, fps=1)
    # clip.write_videofile(os.path.join(target_path.replace('png', 'mp4')), fps=1)
    # print('Saved to {}'.format(os.path.join(target_path.replace('png', 'mp4'))))
    cv2.imwrite(os.path.join(target_path), frame)
    print('Saved to {}'.format(os.path.join(target_path)))


if __name__ == "__main__":
    input_root = 'carla_town02_8_view_20220303_color_KNN_t2no'
    input_path = os.path.join(os.path.join(input_root, '_out_1'))
    # img_concat(input_dir=input_path, target_path='{}.png'.format(input_root))
    # img2video(input_dir=input_path, target_path='{}.png'.format(input_root))
    # input_path = os.path.join('c:\\', 'Downloads', 'cdna', 'imgs', 'ep9')
    input_path = os.path.join('c:\\', 'Downloads', '009')
    img_concat_cdna(input_dir=input_path, target_path='{}.png'.format(input_path))