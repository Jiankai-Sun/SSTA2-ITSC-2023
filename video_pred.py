# coding: utf-8
# https://linuxtut.com/en/fe2d3308b3ba56a80c7a/
# Predict T2NO

import numpy as np
import time
from matplotlib import pyplot as plt
import os, cv2

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
import random
from seq_dataset_t2nd_4agent import data_provider
from skimage.measure import compare_ssim
from VAE_model import VanillaVAE
import lpips
import wandb
from torchvision import models
from models import *
import gc
from memory_profiler import profile
os.environ["WANDB_API_KEY"] = "b00b2711e75723b6df804b383842ad17c46a84b0"
loss_fn_alex = lpips.LPIPS(net='alex')

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class FCN8s_1_agent_4_view_allcomm(nn.Module):
    def __init__(self, num_classes, pretrained=True, caffe=False):
        super(FCN8s_1_agent_4_view_allcomm, self).__init__()
        vgg = models.vgg16()
        self.encoder = nn.Conv2d(3*4 + 4, 3, kernel_size=1)
        self.num_classes = num_classes * 4
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        # print('features: ', features)
        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        # fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        # fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        # fc6.bias.data.copy_(classifier[0].bias.data)
        # fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7 = nn.Conv2d(512, (12*12) * self.num_classes, kernel_size=1)
        # fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        # fc7.bias.data.copy_(classifier[3].bias.data)
        # score_fr = nn.Conv2d(4096, self.num_classes, kernel_size=1)
        # score_fr.weight.data.zero_()
        # score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            # fc6, nn.ReLU(), nn.Dropout(),
            fc7, # nn.ReLU(), nn.Dropout(), score_fr
        )
        self.upscore2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=4, stride=1, bias=False)
        # self.upscore2 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, bias=False)
        # self.upscore_pool4 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, bias=False)
        # self.upscore8 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=16, stride=8, bias=False)

        # self.upscore2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, bias=False)
        # self.upscore_pool4 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, bias=False)
        # self.upscore8 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=16, stride=8, bias=False)
        # self.decoder = nn.Conv2d(self.num_classes, self.num_classes*2, kernel_size=1)
        # self.upscore2.weight.data.copy_(get_upsampling_weight(self.num_classes, self.num_classes, 4))
        # self.upscore_pool4.weight.data.copy_(get_upsampling_weight(self.num_classes, self.num_classes, 4))
        # self.upscore8.weight.data.copy_(get_upsampling_weight(self.num_classes, self.num_classes, 16))

    def forward(self, x, m_t, m_t_others, m_2, m_3):
        # print('x.shape, m_t.shape, m_t_others.shape, m_2.shape, m_1.shape: ', x.shape, m_t.shape, m_t_others.shape, m_2.shape, m_3.shape)
        # x.shape, m_t.shape, m_t_others.shape, m_2.shape, m_1.shape:  torch.Size([5, 1, 128, 128, 12]) torch.Size([5, 1, 128, 128, 1]) torch.Size([5, 1, 128, 128, 1]) torch.Size([5, 1, 128, 128, 1]) torch.Size([5, 1, 128, 128, 1])
        # x = torch.cat([x, m_t, m_t_others, m_2, m_3], -1)
        x = x[:, 0].permute(0, 3, 1, 2)  # 1 x.shape:  torch.Size([5, 5, 128, 128])
        # print(1, 'x.shape: ', x.shape)
        x = self.encoder(x) # 1 x.shape:  torch.Size([5, 3, 128, 128])
        # print(2, 'x.shape: ', x.shape)
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)
        score_fr = self.score_fr(pool5)
        # print('score_fr.shape: ', score_fr.shape)  # score_fr.shape:  torch.Size([5, 84, 5, 5])  # score_fr.shape:  torch.Size([5, 84, 11, 11])
        # print('4096 // self.num_classes * 5: ', 4096 // self.num_classes * 5)
        score_fr = score_fr.reshape(score_fr.shape[0], score_fr.shape[1] // (12*12),
                                    score_fr.shape[2] * 12, score_fr.shape[3] * 12)
        upscore2 = self.upscore2(score_fr)[:, :, 0:x_size[2], 0:x_size[3]]
        # print('upscore2.shape: ', upscore2.shape)
        #
        # score_pool4 = self.score_pool4(0.01 * pool4)
        # upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
        #                                    + upscore2)
        #
        # score_pool3 = self.score_pool3(0.0001 * pool3)
        # # print('score_pool3.shape: ', score_pool3.shape)  # score_pool3.shape:  torch.Size([5, 21, 41, 41])
        # upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
        #                          + upscore_pool4)
        #
        # print('upscore8: ', upscore8.shape)  # upscore8:  torch.Size([5, 21, 216, 216])
        upscore8 = upscore2
        # upscore8 = self.decoder(upscore8)
        return upscore8  # [:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()


# @profile
def training(bs, n_epoch, act, data_mode, args):
    # x_train, t_train, x_test, t_test, step_test = get_data(N, Nte, num_step, data_mode)
    train_input_handle, test_input_handle = data_provider(
        args.data_name, args.train_data_paths, args.valid_data_paths, args.bs, args.img_width,
        seq_length=args.num_step + args.num_past, is_training=True, num_views=args.num_views, img_channel=args.img_channel,
        baseline=args.baseline, eval_batch_size=args.vis_bs, n_epoch=n_epoch, args=args)
    if args.message_type in ['raw_data']:
        input_dim = 3  # + 3 + 3
    elif args.message_type in ['vae']:
        input_dim = 3  # + args.vae_latent_dim + args.vae_latent_dim
    else:
        input_dim = 3  # +1+1
    # h_units = [input_dim, input_dim]
    # h_units = [int(x) for x in args.num_hidden.split(',')]
    if args.mode == 'eval' and args.ckpt_dir is not None:
        model_0_path = os.path.join(args.ckpt_dir, "model_0.pt")
        model_1_path = os.path.join(args.ckpt_dir, "model_1.pt")
        model_0 = torch.load(model_0_path)
        print('Loaded model_0 from {}, model_1 from {}'.format(model_0_path, model_1_path))
    else:
        model_0 = FCN8s_1_agent_4_view_allcomm(num_classes=args.num_class)
        model_1 = None  # FCN8s_1_agent(num_classes=args.num_class)
        print('Created model_0, model_1')
    model_0 = model_0.to(args.device)
    if args.message_type in ['vae']:
        vae_path = os.path.join(args.vae_ckpt_dir, args.data_name, 'vae.pt')
        vae = torch.load(vae_path)
        vae = vae.to(args.device)
        print('Loaded VAE model_0 from {}'.format(vae_path))
    else:
        vae = None
    n_parameters = sum(p.numel() for p in model_0.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = optim.SGD([
        {'params': [param for name, param in model_0.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in model_0.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': 5e-4}
        ],
                          momentum=0.95)
    MSE = CrossEntropyLoss2d(size_average=False)

    tr_loss = []
    te_loss = []
    root_res_path = os.path.join(args.gen_frm_dir, args.data_name)
    if os.path.exists(os.path.join(root_res_path, "{}/Pred".format(act))) == False:
        os.makedirs(os.path.join(root_res_path, "{}/Pred".format(act)))

    start_time = time.time()
    print("START")
    best_eval_loss = np.inf
    for epoch in range(1, n_epoch + 1):
        if args.mode == 'train':
            model_0.train()
            sum_loss = 0
            print('Training ...')
            train_input_handle.begin(do_shuffle=True)
            num_training_batch = 0
            while (train_input_handle.no_batch_left() == False):
                ims = train_input_handle.get_batch()
                train_input_handle.next()
                x_batch = ims[..., :-args.num_views]
                gt_batch = ims[..., -args.num_views:]
                # print('x_batch.max(): {}, x_batch.min(): {}'.format(x_batch.max(), x_batch.min()))
                # print('gt_batch.max(): {}, gt_batch.min(): {}'.format(gt_batch.max(), gt_batch.min()))
                x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                # N = x_batch.size(0)
                gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))
                # print('Training x_batch.shape: {}, gt_batch.shape: {}'.format(x_batch.shape, gt_batch.shape)) # torch.Size([10, 1, 128, 128, 3]) torch.Size([10, 1, 128, 128])
                optimizer.zero_grad()
                # pred_batch, gt_batch = model_0(x_batch, gt_batch)
                # print('x_batch.shape: ', x_batch.shape)
                x_0_t, x_1_t, x_2_t, x_3_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)
                gt_batch_0, gt_batch_1, gt_batch_2, gt_batch_3 = torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
                gt_batch_0 = gt_batch_0[..., 0]
                gt_batch_1 = gt_batch_1[..., 0]
                gt_batch_2 = gt_batch_2[..., 0]
                gt_batch_3 = gt_batch_3[..., 0]
                # print('gt_batch_0.shape: {}, gt_batch_1.shape: {}'.format(gt_batch_0.shape, gt_batch_1.shape))
                # gt_batch_0.shape, gt_batch_1.shape:  torch.Size([5, 1, 128, 128]) torch.Size([5, 1, 128, 128])
                message_0 = vae.get_message(x_0_t).detach()
                message_1 = vae.get_message(x_1_t).detach()
                message_2 = vae.get_message(x_2_t).detach()
                message_3 = vae.get_message(x_3_t).detach()
                print('x_batch.shape: ', x_batch.shape, 'message_0.shape: ', message_0.shape)
                x_batch = torch.cat((x_batch, message_0, message_1, message_2, message_3), dim=-1)
                pred_batch = model_0(x_batch, message_0, message_1, message_2, message_3)

                # print('pred_batch.shape: ', pred_batch.shape)  # [5, 42, 128, 128]
                pred_batch_0, pred_batch_1, pred_batch_2, pred_batch_3 = torch.split(pred_batch, pred_batch.shape[1] // args.num_views, dim=1)
                # pred_batch_1 = model_1(x_1_t, message_1, message_0)
                # pred_batch.shape: torch.Size([10, 10, 2]), gt_batch.shape: torch.Size([10, 10, 2])
                # print('pred_batch.shape, gt_batch.shape: ', pred_batch.shape, gt_batch.shape)
                # pred_batch.shape, gt_batch.shape:  torch.Size([5, 11, 128, 128]) torch.Size([5, 1, 128, 128])
                # pred_batch.shape, gt_batch.shape:  torch.Size([5, 1, 128, 128, 11]) torch.Size([5, 1, 128, 128])
                # pred_reshape = pred_batch.reshape(-1, pred_batch.shape[-1])
                # gt_reshape = gt_batch.reshape(-1)
                # print('pred_batch_0.shape, pred_batch_1.shape,  gt_batch_0[:, 0].shape: ', pred_batch_0.shape, pred_batch_1.shape, gt_batch_0[:, 0].shape)
                # pred_batch_0.shape, pred_batch_1.shape,  gt_batch_0[:, 0].shape:  torch.Size([5, 21, 128, 128]) torch.Size([5, 21, 128, 128]) torch.Size([5, 128, 128])
                loss_0 = MSE(pred_batch_0.float(), gt_batch_0[:, 0].long()) / bs
                loss_1 = MSE(pred_batch_1.float(), gt_batch_1[:, 0].long()) / bs
                loss_2 = MSE(pred_batch_2.float(), gt_batch_2[:, 0].long()) / bs
                loss_3 = MSE(pred_batch_3.float(), gt_batch_3[:, 0].long()) / bs
                loss = loss_0 + loss_1 + loss_2 + loss_3
                loss.backward()
                optimizer.step()
                sum_loss += loss.data * bs
                num_training_batch += bs

            ave_loss = sum_loss / num_training_batch
            # tr_loss.append(ave_loss.cpu())
            train_stats = {'ave_loss': ave_loss}
            if args.USE_WANDB:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},}
                wandb.log(log_stats, step=epoch + 1)

            print('Evaluating ...')
            with torch.no_grad():
                te_sum_loss = []
                test_input_handle.begin(do_shuffle=False)
                ims = test_input_handle.get_batch()
                test_input_handle.next()
                x_batch = ims[..., :-args.num_views]
                gt_batch = ims[..., -args.num_views:]
                x_test_torch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                t_test_torch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))
                # print('Eval x_test_torch.shape: {}, t_test_torch.shape: {}'.format(x_test_torch.shape, t_test_torch.shape))  # torch.Size([10, 19, 128, 128, 6]) torch.Size([10, 19, 128, 128, 6])
                # print(x_test_torch.max(), t_test_torch.min(), t_test_torch.max())  # tensor(1., device='cuda:0') tensor(0., device='cuda:0') tensor(10., device='cuda:0')
                x_test_torch_0, x_test_torch_1, x_test_torch_2, x_test_torch_3 = torch.split(x_test_torch, x_test_torch.shape[-1] // args.num_views, dim=-1)
                message_0 = vae.get_message(x_test_torch_0).detach()
                message_1 = vae.get_message(x_test_torch_1).detach()
                message_2 = vae.get_message(x_test_torch_2).detach()
                message_3 = vae.get_message(x_test_torch_3).detach()
                x_test_torch = torch.cat((x_test_torch, message_0, message_1, message_2, message_3), dim=-1)
                y_test_torch = model_0(x_test_torch, message_0, message_1, message_2, message_3)
                y_test_torch_0, y_test_torch_1, y_test_torch_2, y_test_torch_3 = torch.split(y_test_torch, y_test_torch.shape[1] // args.num_views, dim=1)
                t_test_torch = t_test_torch[:, :y_test_torch_0.shape[1]]
                t_test_torch_0, t_test_torch_1, t_test_torch_2, t_test_torch_3 = torch.split(t_test_torch, t_test_torch.shape[-1] // args.num_views, dim=-1)
                t_test_torch_0 = t_test_torch_0[..., 0]
                t_test_torch_1 = t_test_torch_1[..., 0]
                t_test_torch_2 = t_test_torch_2[..., 0]
                t_test_torch_3 = t_test_torch_3[..., 0]
                loss_0 = MSE(y_test_torch_0.float(), t_test_torch_0[:, 0].long()) / bs
                loss_1 = MSE(y_test_torch_1.float(), t_test_torch_1[:, 0].long()) / bs
                loss_2 = MSE(y_test_torch_2.float(), t_test_torch_2[:, 0].long()) / bs
                loss_3 = MSE(y_test_torch_3.float(), t_test_torch_3[:, 0].long()) / bs
                loss = loss_0 + loss_1 + loss_2 + loss_3
                # te_sum_loss.append(loss.detach())
                te_sum_loss = loss.detach()
                # avg_te_sum_loss = torch.mean(torch.tensor(te_sum_loss))
                avg_te_sum_loss = te_sum_loss
                # te_loss.append(avg_te_sum_loss.data)
                test_stats = {'ave_loss': avg_te_sum_loss}
            if args.USE_WANDB:
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},}
                wandb.log(log_stats, step=epoch + 1)

            if epoch % 100 == 1:
                print("Ep/MaxEp     tr_loss     te_loss")

            if epoch % 10 == 0:
                print("{:4}/{}  {:10.5}   {:10.5}".format(epoch, n_epoch, ave_loss, float(loss.data)))

        if epoch % args.eval_per_step == 0:  # 20
            batch_id = 0
            res_path = os.path.join(root_res_path, str(epoch))
            os.makedirs(res_path, exist_ok=True)
            img_mse, ssim, psnr = [], [], []
            lp = []
            for i in range(args.eval_num_step):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                lp.append(0)
            if args.eval_mode != 'multi_step_eval':
                for i in range(args.num_past):
                    img_mse.append(0)
                    ssim.append(0)
                    psnr.append(0)
                    lp.append(0)
            test_input_handle.begin(do_shuffle=False)
            print('test_input_handle.current_batch_indices: ', test_input_handle.current_batch_indices)
            # while (test_input_handle.no_batch_left() == False):
            batch_id = batch_id + 1
            ims = test_input_handle.get_batch()
            test_input_handle.next()
            x_test = ims[..., :-args.num_views]
            t_test = ims[..., -args.num_views:]
            # print('Test x_test.shape: {}, t_test.shape: {}'.format(x_test.shape, t_test.shape))  # Test x_test.shape: (5, 1, 128, 128, 6), t_test.shape: (5, 1, 128, 128, 2)
            # print('batch_id: {}\n'.format(batch_id))
            with torch.no_grad():
                x_test_0, x_test_1, x_test_2, x_test_3 = torch.split(torch.from_numpy(x_test.astype(np.float32)).to(args.device), x_test.shape[-1] // args.num_views, dim=-1)
                message_0 = vae.get_message(x_test_0).detach()
                message_1 = vae.get_message(x_test_1).detach()
                message_2 = vae.get_message(x_test_2).detach()
                message_3 = vae.get_message(x_test_3).detach()
                x_test = torch.from_numpy(x_test.astype(np.float32)).to(args.device)
                x_test = torch.cat((x_test, message_0, message_1, message_2, message_3), dim=-1)
                y_test = model_0(x_test, message_0, message_1, message_2, message_3)
                y_test_0, y_test_1, y_test_2, y_test_3 = torch.split(y_test, y_test.shape[1] // args.num_views, dim=1)

                # t_test_0, t_test_1 = torch.split(t_test, t_test.shape[1] // 2, dim=1)
                t_test_0 = t_test[..., :t_test.shape[-1] // 4] # [:, 0, :, :, 0]
                t_test_1 = t_test[..., t_test.shape[-1] // 4:t_test.shape[-1] // 4*2] # [:, 0, :, :, 0]  # [..., 0]
                t_test_2 = t_test[..., t_test.shape[-1] // 4*2:t_test.shape[-1] // 4*3]  # [:, 0, :, :, 0]
                t_test_3 = t_test[..., t_test.shape[-1] // 4*3:]  # [:, 0, :, :, 0]  # [..., 0]
                # print('t_test_0.shape, t_test_1.shape: ', t_test_0.shape, t_test_1.shape)
                # t_test_0.shape, t_test_1.shape:  (5, 1, 128, 128, 1) (5, 1, 128, 128, 1)

            y_test_0 = y_test_0.detach().cpu().numpy()
            y_test_1 = y_test_1.detach().cpu().numpy()
            y_test_2 = y_test_2.detach().cpu().numpy()
            y_test_3 = y_test_3.detach().cpu().numpy()
            # save prediction examples
            if batch_id <= args.num_save_samples:
                path = os.path.join(res_path, str(batch_id))
                os.mkdir(path)
                y_test_0 = np.argmax(y_test_0, 1)[:, None, :, :, None]
                y_test_1 = np.argmax(y_test_1, 1)[:, None, :, :, None]
                y_test_2 = np.argmax(y_test_2, 1)[:, None, :, :, None]
                y_test_3 = np.argmax(y_test_3, 1)[:, None, :, :, None]
                # print('Test: y_test.shape: {}'.format(y_test.shape))
                his_length = args.num_class  # 10
                view_idx = 0
                # print('t_test_0.shape, y_test_0.shape, t_test_1.shape, y_test_1.shape: ', t_test_0.shape, y_test_0.shape, t_test_1.shape, y_test_1.shape)
                # t_test_0.shape, y_test_0.shape, t_test_1.shape, y_test_1.shape:  (5, 1, 128, 128, 1) (5, 1, 128, 128, 1) (5, 1, 128, 128, 1) (5, 1, 128, 128, 1)
                for i in range(y_test_0.shape[1]):
                    name = 'gt_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    # print('t_test.shape: ', t_test.shape)
                    img_gt = np.uint8(t_test_0[0, i, ...] * 255 / his_length)
                    print('Save to {}'.format(file_name))
                    cv2.imwrite(file_name, img_gt)
                for i in range(y_test_0.shape[1]):
                    name = 'pd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    # print('y_test.shape: ', y_test.shape)
                    img_pd = y_test_0[0, i, ...]
                    # in range (0, 1)
                    img_pd = np.uint8(img_pd * 255 / his_length)  #  * 255
                    cv2.imwrite(file_name, img_pd)
                    print('Save to {}'.format(file_name))
                view_idx = 1
                for i in range(y_test_1.shape[1]):
                    name = 'gt_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    # print('t_test.shape: ', t_test.shape)
                    img_gt = np.uint8(t_test_1[0, i, ...] * 255 / his_length)
                    cv2.imwrite(file_name, img_gt)
                    print('Save to {}'.format(file_name))
                for i in range(y_test_1.shape[1]):
                    name = 'pd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    # print('y_test.shape: ', y_test.shape)
                    img_pd = y_test_1[0, i, ...]
                    # in range (0, 1)
                    img_pd = np.uint8(img_pd * 255 / his_length)  # * 255
                    cv2.imwrite(file_name, img_pd)
                    print('Save to {}'.format(file_name))

                view_idx = 2
                for i in range(y_test_2.shape[1]):
                    name = 'gt_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    # print('t_test.shape: ', t_test.shape)
                    img_gt = np.uint8(t_test_2[0, i, ...] * 255 / his_length)
                    cv2.imwrite(file_name, img_gt)
                    print('Save to {}'.format(file_name))
                for i in range(y_test_2.shape[1]):
                    name = 'pd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    # print('y_test.shape: ', y_test.shape)
                    img_pd = y_test_2[0, i, ...]
                    # in range (0, 1)
                    img_pd = np.uint8(img_pd * 255 / his_length)  # * 255
                    cv2.imwrite(file_name, img_pd)
                    print('Save to {}'.format(file_name))

                view_idx = 3
                for i in range(y_test_3.shape[1]):
                    name = 'gt_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    # print('t_test.shape: ', t_test.shape)
                    img_gt = np.uint8(t_test_3[0, i, ...] * 255 / his_length)
                    # img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(file_name, img_gt)
                    print('Save to {}'.format(file_name))
                for i in range(y_test_3.shape[1]):
                    name = 'pd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                    file_name = os.path.join(path, name)
                    img_pd = y_test_3[0, i, ...]
                    # in range (0, 1)
                    img_pd = np.uint8(img_pd * 255 / his_length)  # * 255
                    cv2.imwrite(file_name, img_pd)
                    print('Save to {}'.format(file_name))
                print('Save to {}.'.format(res_path))

                # if args.USE_WANDB:
                    # metrics_stats = {'avg_mse': avg_mse, 'ssim': np.mean(ssim), 'psnr': np.mean(psnr), 'lpips': np.mean(lp)}
                    # log_stats = {**{f'test_{k}': v for k, v in metrics_stats.items()}, }
                    # wandb.log(log_stats, step=epoch + 1)
        gc.collect()
        if args.mode == 'train': #  and best_eval_loss > avg_te_sum_loss:
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
            # best_eval_loss = avg_te_sum_loss
            torch.save(model_0, os.path.join(root_res_path, "model_0.pt"))
            # torch.save(model_0.state_dict(), os.path.join(root_res_path, "model_0.pt"))

    print("END")

    total_time = int(time.time() - start_time)
    print("Time : {} [s]".format(total_time))


def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data_mode', type=str, default="(y_{t-1}, y_t)->y_{t+1}", help='(y_{t-1}, y_t)->y_{t+1}')
    parser.add_argument('--act', type=str, default="relu", help='relu')
    parser.add_argument('--mode', type=str, default="train", help='train / eval')
    parser.add_argument('--eval_mode', type=str, default='multi_step_eval', help='multi_step_eval / single_step_eval')
    parser.add_argument('--eval_num_step', type=int, default=1)
    parser.add_argument('--eval_per_step', type=int, default=100)
    parser.add_argument('--mask_per_step', type=int, default=1000000000)
    parser.add_argument('--log_per_epoch', type=int, default=10)
    parser.add_argument('--num_step', type=int, default=1)
    parser.add_argument('--num_past', type=int, default=0)
    parser.add_argument('--num_cl_step', type=int, default=20)
    parser.add_argument('--n_epoch', type=int, default=800, help='800')
    parser.add_argument('--bs', type=int, default=5)
    parser.add_argument('--vis_bs', type=int, default=5)

    parser.add_argument('--data_name', type=str, default="carla_town02_4_view_20221126_icra", help='SINE; circle_motion; students003, '
                       'fluid_flow_1, carla_manual_20220509, carla_town02_8_view_20220415_left_right')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 cuda:0; cpu:0 cpu:0')
    parser.add_argument('--with_comm', type=str2bool, default=False, help='whether to use communication')
    parser.add_argument('--train_data_paths', type=str, default="../../../../tools/circle_motion_30/train",
                        help='../tools/${DATASET_NAME}/train, ../../../../tools/circle_motion_30/train, sumo_sanjose-2021-11-06_20.28.57_30, carla_town02_20211201, students003')
    # carla_town02_20211201
    parser.add_argument('--valid_data_paths', type=str, default="../../../../tools/circle_motion_30/eval",
                        help='../tools/${DATASET_NAME}/eval, ../../../../tools/circle_motion_30/eval, sumo_sanjose-2021-11-06_20.28.57_30, carla_town02_20211201, students003')
    # sumo_sanjose-2021-11-06_20.28.57_30
    # RGB dataset
    parser.add_argument('--img_width', type=int, default=128, help='img width')
    parser.add_argument('--num_views', type=int, default=4, help='num views')
    parser.add_argument('--img_channel', type=int, default=3, help='img channel')
    parser.add_argument('--baseline', type=str, default='1_NN_4_img_GCN',
                        help='1_NN_1_img_no_GCN, 1_NN_4_img_no_GCN, 4_NN_4_img_GCN, 1_NN_4_img_GCN, 4_NN_4_img_no_GCN, '
                             '4_NN_4_img_FC, 4_NN_4_img_Identity')
    parser.add_argument('--gen_frm_dir', type=str, default='results/')
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--USE_WANDB', type=str2bool, default=False, help="use WANDB")
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--num_hidden', type=str, default='16,16', help='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--version', type=str, default='predrnn', help='version')
    parser.add_argument('--message_type', type=str, default='vae', help='normal, zeros, randn, raw_data, vae')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='checkpoint dir: dir/model_1.pt')
    parser.add_argument('--vae_ckpt_dir', type=str, default='results/vae_carla',
                        help='vae checkpoint dir: kessel: 1c-results/20220109-212140/circle_motion/vae.pt, '
                             '20220110-234817, chpc-gpu005: 1c-20220113-154323, wo ln: 20220117-044713')
    parser.add_argument('--cl_mode', type=str, default='full_history', help='full_history, sliding_window')
    parser.add_argument('--vae_latent_dim', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=21)
    parser.add_argument('--lr', type=float, default=0.0000001, help='0.00001, ')

    args = parser.parse_args()

    h_units = [10, 10]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.gen_frm_dir = os.path.join(args.gen_frm_dir, timestr)
    args.train_data_paths = "../../../../tools/{}".format(args.data_name)
    args.valid_data_paths = "../../../../tools/{}".format(args.data_name)
    if args.USE_WANDB:
        wandb.init(project='traffic_pred', config=args, settings=wandb.Settings(_disable_stats=True))
        wandb.run.name = '{}_{}'.format(wandb.run.name.split('-')[-1], args.data_name, )
    training(args.bs, args.n_epoch, args.act, args.data_mode, args)
