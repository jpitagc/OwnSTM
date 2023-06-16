from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy


### My libs
from dataset.dataset import DAVIS_MO_Test
from model.model import STM
from utils.helpers import overlay_davis

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment


palette = [
    0, 0, 0,
    0.5020, 0, 0,
    0, 0.5020, 0,
    0.5020, 0.5020, 0,
    0, 0, 0.5020,
    0.5020, 0, 0.5020,
    0, 0.5020, 0.5020,
    0.5020, 0.5020, 0.5020,
    0.2510, 0, 0,
    0.7529, 0, 0,
    0.2510, 0.5020, 0,
    0.7529, 0.5020, 0,
    0.2510, 0, 0.5020,
    0.7529, 0, 0.5020,
    0.2510, 0.5020, 0.5020,
    0.7529, 0.5020, 0.5020,
    0, 0.2510, 0,
    0.5020, 0.2510, 0,
    0, 0.7529, 0,
    0.5020, 0.7529, 0,
    0, 0.2510, 0.5020,
    0.5020, 0.2510, 0.5020,
    0, 0.7529, 0.5020,
    0.5020, 0.7529, 0.5020,
    0.2510, 0.2510, 0]
palette = (np.array(palette) * 255).astype('uint8')

def Run_video(dataset,video, num_frames, num_objects,model,Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError
    F_last,M_last = dataset.load_single_image(video,0)
    F_last = F_last.unsqueeze(0)
    M_last = M_last.unsqueeze(0)
    E_last = M_last
    pred = np.zeros((num_frames,M_last.shape[3],M_last.shape[4]))
    all_Fs = [F_last]
    pred[0] = torch.argmax(E_last[0], dim=0).cpu().numpy().astype(np.uint8)
    for t in range(1,num_frames):

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        F_,M_ = dataset.load_single_image(video,t)

        F_ = F_.unsqueeze(0)
        M_ = M_.unsqueeze(0)
        all_Fs.append(F_.cpu().numpy())
        # segment
        with torch.no_grad():
            logit = model(F_[:,:,0], this_keys, this_values, torch.tensor([num_objects]))
        E = F.softmax(logit, dim=1)
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        pred[t] = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        E_last = E.unsqueeze(2)
        F_last = F_

    Fs = np.concatenate(all_Fs,axis=2)
    return pred,Fs

def demo(model,Testloader,output_mask_path,output_viz_path):
    for V in tqdm.tqdm(list(Testloader)[:]):
        num_objects, info = V
        seq_name = info['name']
        num_frames = info['num_frames']
        print(f'Running Video {seq_name}')
        pred,Fs = Run_video(Testloader, seq_name, num_frames, num_objects,model,Mem_every=5, Mem_number=None)

        print('Saving Results')
        # Save results for quantitative eval ######################
        seq_output_mask_path = os.path.join(output_mask_path,seq_name)
        if not os.path.exists(seq_output_mask_path):
            os.makedirs(seq_output_mask_path)

        for f in range(num_frames):
            img_E = Image.fromarray(pred[f].astype(np.uint8))
            img_E.putpalette(palette)
            img_E.save(os.path.join(seq_output_mask_path, '{:05d}.png'.format(f)))


        seq_output_viz_path = os.path.join(output_viz_path,seq_name)
        if not os.path.exists(seq_output_viz_path):
            os.makedirs(seq_output_viz_path)

        for f in range(num_frames):
            pF = (Fs[0,:,f].transpose(1,2,0) * 255.).astype(np.uint8)
            pE = pred[f].astype(np.uint8)
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(seq_output_viz_path, 'f{}.jpg'.format(f)))

        vid_path = os.path.join(output_viz_path, '{}.mp4'.format(seq_name))
        frame_path = os.path.join(output_viz_path, seq_name, 'f%d.jpg')
        os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))


def custom_load_single_image(video,f):
 
       
        mask_dir = os.path.join(video, 'masks')
        image_dir = os.path.join(video, 'images')
        shape = np.shape(np.array(Image.open(os.path.join(mask_dir, '00000.png')).convert("P")))

        N_frames = np.empty((1,)+shape+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+shape, dtype=np.uint8)

        img_file = os.path.join(image_dir, video, '{:05d}.jpg'.format(f))

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            mask_file = os.path.join(mask_dir, video, '{:05d}.png'.format(f))  
            N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms 


def custom_run(video, num_frames, num_objects,model,Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError
    F_last,M_last = custom_load_single_image(video,0)
    F_last = F_last.unsqueeze(0)
    M_last = M_last.unsqueeze(0)
    E_last = M_last
    pred = np.zeros((num_frames,M_last.shape[3],M_last.shape[4]))
    all_Fs = [F_last]
    pred[0] = torch.argmax(E_last[0], dim=0).cpu().numpy().astype(np.uint8)
    for t in range(1,num_frames):

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        F_,M_ = custom_load_single_image(video,t)

        F_ = F_.unsqueeze(0)
        M_ = M_.unsqueeze(0)
        all_Fs.append(F_.cpu().numpy())
        # segment
        with torch.no_grad():
            logit = model(F_[:,:,0], this_keys, this_values, torch.tensor([num_objects]))
        E = F.softmax(logit, dim=1)
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        pred[t] = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        E_last = E.unsqueeze(2)
        F_last = F_

    Fs = np.concatenate(all_Fs,axis=2)
    return pred,Fs
if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
        #parser.add_argument("-s", type=str, help="set", required=True)
        #parser.add_argument("-y", type=int, help="year", required=True)
        parser.add_argument("-D", type=str, help="path to data",default='./Kitchen_480p')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights",required=True)
        #parser.add_argument("-output_path", type=str, help="path to segmentation maps",default='./results/outputs')
        #parser.add_argument("-video_path", type=str, help="path to videos",default='./videos')
        return parser.parse_args()

    args = get_arguments()

    GPU = args.g
    #YEAR = args.y
    #SET = args.s
    #DATA_ROOT = args.D
    pth = args.p

    output_file_name = pth.rsplit("/")[-1].rsplit(".")[0]
    video = args.D
    output_viz_path = args.D
    output_mask_path = os.path.join(video , output_file_name , 'masks')
    output_viz_path = os.path.join(output_viz_path, output_file_name, 'video')

    if not os.path.exists(output_viz_path):
        os.makedirs(output_viz_path)  

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on ', video)

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    #Testloader = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    #print(f'Loaded test loader {DATA_ROOT + str(YEAR) + SET}')
    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    

    print('Loading CheckPoint Model')
    #Unncomment line when GPU available
    #model.load_state_dict(torch.load(pth))
    #model.load_state_dict(torch.load(pth,map_location=torch.device('cpu')))

    if GPU == -1: 
        model.load_state_dict(torch.load(pth))
    else: 
        model.load_state_dict(torch.load(pth,map_location=torch.device('cpu')))

    #demo(model,Testloader,output_mask_path,output_viz_path)
    num_frames = 201
    num_objects = 1
    pred,Fs = custom_run(video, num_frames, num_objects,model,Mem_every=5, Mem_number=None)
    print('Saving Results')
    # Save results for quantitative eval ######################
    seq_output_mask_path = os.path.join(output_mask_path,'results')
    if not os.path.exists(seq_output_mask_path):
        os.makedirs(seq_output_mask_path)

    for f in range(num_frames):
        img_E = Image.fromarray(pred[f].astype(np.uint8))
        img_E.putpalette(palette)
        img_E.save(os.path.join(seq_output_mask_path, '{:05d}.png'.format(f)))
