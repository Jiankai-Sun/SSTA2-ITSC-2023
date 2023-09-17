import os
# import configparser
# import sys
# import random
# import numpy
# import warnings
# import shutil
# import datetime
# import copy
#
# import torch.nn as nn
import torch

def save_cuda_mem_req(out_dir, out_filename='cuda_mem_req.pth.tar'):
    """
    :param out_dir: /path/to/best_model.pth.tar
    """
    out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)

    mem_req = {}
    mem_req['cuda_memory_allocated'] = torch.cuda.memory_allocated(device=None)
    mem_req['cuda_memory_cached'] = torch.cuda.memory_cached(device=None)

    torch.save(mem_req, out_path)
    print("SAVED CUDA MEM REQ {} to path: {}".format(mem_req, out_path))

def save_preprocessing_time(out_dir, time, out_filename='preprocess_time.pth.tar'):
    if os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)
    torch.save(time, out_path)
    print_timing(time, "PREPROCESSING")

def print_timing(timing, title=''):
    title = title.strip() + ' ' if title != '' else title
    print("{}TIMING >>> {} <<<".format(title, str(timing)))