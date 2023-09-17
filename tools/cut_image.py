import numpy as np
import cv2, os

def split_img(input_img_name):
    input_img = cv2.imread(input_img_name)
    output_dir = input_img_name + '_output'
    os.makedirs(output_dir, exist_ok=True)
    img_width = 128
    orig_small_horizontal_space = 1
    orig_large_horizontal_space = 4
    orig_large_vertical_space = 4
    orig_vertical_space = 1
    splited_results = {}
    if 'vis_carla_town02_2_view_20220205_split_epoch_100_' in input_img_name:
        gt = input_img[img_width * 0:img_width * 1]
        pred = input_img[img_width * 1: img_width * 2]
        diff = input_img[img_width * 2: img_width * 3]
        msg = input_img[img_width * 3: img_width * 4]
        for i in range(10):
            gt_view_1 = gt[:, orig_large_horizontal_space * i + orig_small_horizontal_space * i +img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (i * 2+1)]
            gt_view_2 = gt[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i+1) + img_width * (i*2+1): orig_large_horizontal_space * i + orig_small_horizontal_space * (i+1) + img_width * (i * 2+2)]
            if i % 2 == 0:
                splited_results['gt_step_{}_view_1'.format(i)] = gt_view_1
                splited_results['gt_step_{}_view_2'.format(i)] = gt_view_2

            pred_view_1 = pred[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (
                                    i * 2 + 1)]
            pred_view_2 = pred[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                        i * 2 + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                        i * 2 + 2)]
            if i % 2 == 0:
                splited_results['pred_step_{}_view_1'.format(i)] = pred_view_1
                splited_results['pred_step_{}_view_2'.format(i)] = pred_view_2

            msg_view_1 = msg[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (
                                  i * 2 + 1)]
            msg_view_2 = msg[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                                  i * 2 + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                      i + 1) + img_width * (i * 2 + 2)]
            if i % 2 == 0:
                splited_results['msg_step_{}_view_1'.format(i)] = msg_view_1
                splited_results['msg_step_{}_view_2'.format(i)] = msg_view_2
    elif 'vis_carla_town02_4_view_20220205_split_epoch_100_' in input_img_name:
        gt_1 = input_img[img_width * 0:img_width * 1]
        gt_2 = input_img[img_width * 1:img_width * 2]
        pred_1 = input_img[img_width * 2: img_width * 3]
        pred_2 = input_img[img_width * 3: img_width * 4]
        diff_1 = input_img[img_width * 4: img_width * 5]
        diff_2 = input_img[img_width * 5: img_width * 6]
        msg_1 = input_img[img_width * 6: img_width * 7]
        msg_2 = input_img[img_width * 7: img_width * 8]
        for i in range(50):
            gt_view_1 = gt_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * i +img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (i * 2+1)]
            gt_view_2 = gt_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i+1) + img_width * (i*2+1): orig_large_horizontal_space * i + orig_small_horizontal_space * (i+1) + img_width * (i * 2+2)]
            gt_view_3 = gt_2[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (
                                    i * 2 + 1)]
            gt_view_4 = gt_2[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                        i * 2 + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                        i + 1) + img_width * (i * 2 + 2)]

            if i % 2 == 0:
                splited_results['gt_step_{}_view_1'.format(i)] = gt_view_1
                splited_results['gt_step_{}_view_2'.format(i)] = gt_view_2
                splited_results['gt_step_{}_view_3'.format(i)] = gt_view_3
                splited_results['gt_step_{}_view_4'.format(i)] = gt_view_4

            pred_view_1 = pred_1[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (
                                    i * 2 + 1)]
            pred_view_2 = pred_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                        i * 2 + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                        i * 2 + 2)]
            pred_view_3 = pred_2[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (
                                  i * 2 + 1)]
            pred_view_4 = pred_2[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                                  i * 2 + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                      i + 1) + img_width * (
                                                      i * 2 + 2)]
            if i % 2 == 0:
                splited_results['pred_step_{}_view_1'.format(i)] = pred_view_1
                splited_results['pred_step_{}_view_2'.format(i)] = pred_view_2
                splited_results['pred_step_{}_view_3'.format(i)] = pred_view_3
                splited_results['pred_step_{}_view_4'.format(i)] = pred_view_4

            msg_view_1 = msg_1[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (
                                  i * 2 + 1)]
            msg_view_2 = msg_1[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                                  i * 2 + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                      i + 1) + img_width * (i * 2 + 2)]
            msg_view_3 = msg_2[:,
                         orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * i * 2: orig_large_horizontal_space * i + orig_small_horizontal_space * i + img_width * (
                                 i * 2 + 1)]
            msg_view_4 = msg_2[:,
                         orig_large_horizontal_space * i + orig_small_horizontal_space * (i + 1) + img_width * (
                                 i * 2 + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                 i + 1) + img_width * (i * 2 + 2)]
            if i % 2 == 0:
                splited_results['msg_step_{}_view_1'.format(i)] = msg_view_1
                splited_results['msg_step_{}_view_2'.format(i)] = msg_view_2
                splited_results['msg_step_{}_view_3'.format(i)] = msg_view_3
                splited_results['msg_step_{}_view_4'.format(i)] = msg_view_4
    elif 'vis_carla_town02_8_view_20220205_split_epoch_100_' in input_img_name:
        gt_1 = input_img[img_width * 0:img_width * 1]
        gt_2 = input_img[img_width * 1:img_width * 2]
        pred_1 = input_img[img_width * 2: img_width * 3]
        pred_2 = input_img[img_width * 3: img_width * 4]
        diff_1 = input_img[img_width * 4: img_width * 5]
        diff_2 = input_img[img_width * 5: img_width * 6]
        msg_1 = input_img[img_width * 6: img_width * 7]
        msg_2 = input_img[img_width * 7: img_width * 8]
        num_horizontal_views = 4
        for i in range(50):
            gt_view_1 = gt_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) +img_width * i * num_horizontal_views: orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * (i * num_horizontal_views+1)]
            gt_view_2 = gt_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1)+1) + img_width * (i*num_horizontal_views+1): orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1)+1) + img_width * (i * num_horizontal_views+2)]
            gt_view_3 = gt_1[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1)+2) + img_width * (i * num_horizontal_views + 2): orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1)+2) + img_width * (
                                    i * num_horizontal_views + 3)]
            gt_view_4 = gt_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1) + 3) + img_width * (
                        i * num_horizontal_views + 3): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                        i*(num_horizontal_views-1) + 3) + img_width * (i * num_horizontal_views + 4)]

            gt_view_5 = gt_2[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * i*(num_horizontal_views-1) + img_width * i * num_horizontal_views: orig_large_horizontal_space * i + orig_small_horizontal_space * i*(num_horizontal_views-1) + img_width * (
                                    i * num_horizontal_views + 1)]
            gt_view_6 = gt_2[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1) + 1) + img_width * (
                        i * num_horizontal_views + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                        i*(num_horizontal_views-1) + 1) + img_width * (i * num_horizontal_views + 2)]
            gt_view_7 = gt_2[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1)+2) + img_width * (i * num_horizontal_views + 2): orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1)+2) + img_width * (
                                i * num_horizontal_views + 3)]
            gt_view_8 = gt_2[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i*(num_horizontal_views-1) + 3) + img_width * (
                    i * num_horizontal_views + 3): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                    i*(num_horizontal_views-1) + 3) + img_width * (i * num_horizontal_views + 4)]

            splited_results['gt_step_{}_view_1'.format(i)] = gt_view_1
            splited_results['gt_step_{}_view_2'.format(i)] = gt_view_2
            splited_results['gt_step_{}_view_3'.format(i)] = gt_view_3
            splited_results['gt_step_{}_view_4'.format(i)] = gt_view_4
            splited_results['gt_step_{}_view_5'.format(i)] = gt_view_5
            splited_results['gt_step_{}_view_6'.format(i)] = gt_view_6
            splited_results['gt_step_{}_view_7'.format(i)] = gt_view_7
            splited_results['gt_step_{}_view_8'.format(i)] = gt_view_8

            pred_view_1 = pred_1[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * i * num_horizontal_views: orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * (
                                    i * num_horizontal_views + 1)]
            pred_view_2 = pred_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 1) + img_width * (
                        i * num_horizontal_views + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                        i * (num_horizontal_views-1) + 1) + img_width * (i * num_horizontal_views + 2)]
            pred_view_3 = pred_1[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 2) + img_width * (
                                    i * num_horizontal_views + 2): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                    i * (num_horizontal_views-1) + 2) + img_width * (
                                                        i * num_horizontal_views + 3)]
            pred_view_4 = pred_1[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 3) + img_width * (
                    i * num_horizontal_views + 3): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                    i * (num_horizontal_views-1) + 3) + img_width * (i * num_horizontal_views + 4)]

            pred_view_5 = pred_2[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * i * num_horizontal_views: orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * (
                                i * num_horizontal_views + 1)]
            pred_view_6 = pred_2[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 1) + img_width * (
                    i * num_horizontal_views + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                    i * (num_horizontal_views-1) + 1) + img_width * (i * num_horizontal_views + 2)]
            pred_view_7 = pred_2[:,
                        orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 2) + img_width * (
                                    i * num_horizontal_views + 2): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                    i * (num_horizontal_views-1) + 2) + img_width * (
                                                        i * num_horizontal_views + 3)]
            pred_view_8 = pred_2[:, orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 3) + img_width * (
                    i * num_horizontal_views + 3): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                    i * (num_horizontal_views-1) + 3) + img_width * (i * num_horizontal_views + 4)]

            splited_results['pred_step_{}_view_1'.format(i)] = pred_view_1
            splited_results['pred_step_{}_view_2'.format(i)] = pred_view_2
            splited_results['pred_step_{}_view_3'.format(i)] = pred_view_3
            splited_results['pred_step_{}_view_4'.format(i)] = pred_view_4
            splited_results['pred_step_{}_view_5'.format(i)] = pred_view_5
            splited_results['pred_step_{}_view_6'.format(i)] = pred_view_6
            splited_results['pred_step_{}_view_7'.format(i)] = pred_view_7
            splited_results['pred_step_{}_view_8'.format(i)] = pred_view_8

            msg_view_1 = msg_1[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * i * num_horizontal_views: orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * (
                                  i * num_horizontal_views + 1)]
            msg_view_2 = msg_1[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 1) + img_width * (
                                  i * num_horizontal_views + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                  i * (num_horizontal_views-1) + 1) + img_width * (i * num_horizontal_views + 2)]
            msg_view_3 = msg_1[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 2) + img_width * (
                                  i * num_horizontal_views + 2): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                  i * (num_horizontal_views-1) + 2) + img_width * (
                                                      i * num_horizontal_views + 3)]
            msg_view_4 = msg_1[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 3) + img_width * (
                                  i * num_horizontal_views + 3): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                  i * (num_horizontal_views-1) + 3) + img_width * (i * num_horizontal_views + 4)]

            msg_view_5 = msg_2[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * i * num_horizontal_views: orig_large_horizontal_space * i + orig_small_horizontal_space * i * (num_horizontal_views-1) + img_width * (
                                  i * num_horizontal_views + 1)]
            msg_view_6 = msg_2[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 1) + img_width * (
                                  i * num_horizontal_views + 1): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                  i * (num_horizontal_views-1) + 1) + img_width * (i * num_horizontal_views + 2)]
            msg_view_7 = msg_2[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 2) + img_width * (
                                  i * num_horizontal_views + 2): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                  i * (num_horizontal_views-1) + 2) + img_width * (
                                                      i * num_horizontal_views + 3)]
            msg_view_8 = msg_2[:,
                          orig_large_horizontal_space * i + orig_small_horizontal_space * (i * (num_horizontal_views-1) + 3) + img_width * (
                                  i * num_horizontal_views + 3): orig_large_horizontal_space * i + orig_small_horizontal_space * (
                                  i * (num_horizontal_views-1) + 3) + img_width * (i * num_horizontal_views + 4)]

            splited_results['msg_step_{}_view_1'.format(i)] = msg_view_1
            splited_results['msg_step_{}_view_2'.format(i)] = msg_view_2
            splited_results['msg_step_{}_view_3'.format(i)] = msg_view_3
            splited_results['msg_step_{}_view_4'.format(i)] = msg_view_4
            splited_results['msg_step_{}_view_5'.format(i)] = msg_view_5
            splited_results['msg_step_{}_view_6'.format(i)] = msg_view_6
            splited_results['msg_step_{}_view_7'.format(i)] = msg_view_7
            splited_results['msg_step_{}_view_8'.format(i)] = msg_view_8

    for i, (item_name, item_value) in enumerate(splited_results.items()):
        save_path = os.path.join(output_dir, item_name+'.png')
        print('Processing {}'.format(save_path))
        cv2.imwrite(save_path, item_value)


if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    root_path = os.path.join(home_dir, 'Downloads')  # '../../InteractionSimulator/datasets/'
    # png_name = 'vis_carla_town02_8_view_20220205_split_epoch_100_.png'
    # png_name = 'vis_carla_town02_4_view_20220205_split_epoch_100_50_steps.png'
    png_name = 'vis_carla_town02_4_view_20220205_split_epoch_100_50_step.png'
    input_img_name = os.path.join(root_path, png_name)
    split_img(input_img_name)