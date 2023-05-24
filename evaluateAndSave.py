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
import pandas as pd


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
    all_Ms = []
    for t in range(1,num_frames):

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        del prev_key,prev_value

        F_,M_ = dataset.load_single_image(video,t)

        F_ = F_.unsqueeze(0)
        M_ = M_.unsqueeze(0)
        all_Ms.append(M_.cpu().numpy())
        del M_
        # segment
        with torch.no_grad():
            logit = model(F_[:,:,0], this_keys, this_values, torch.tensor([num_objects]))
        E = F.softmax(logit, dim=1)
        del logit
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
            del this_keys,this_values
        pred[t] = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        E_last = E.unsqueeze(2)
        F_last = F_
    Ms = np.concatenate(all_Ms,axis=2)
    return pred,Ms

def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res

def customevaluateandsave(model,Testloader,metric,output_mask_path,output_viz_path):

    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    g_measures_by_video = ['Video','J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    df_video = pd.DataFrame(columns=g_measures_by_video)
    for V in tqdm.tqdm(list(Testloader)[:2]):
        num_objects, info = V
        seq_name = info['name'] 
        num_frames = info['num_frames']
        #print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
        
        print(f'Running {seq_name}')
        pred,Fs = Run_video(Testloader, seq_name, num_frames, num_objects,model,Mem_every=8, Mem_number=None)
        # all_res_masks = Es[0].cpu().numpy()[1:1+num_objects]

        print(f'Saving')
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

        print(f'Computing metrics')
        all_res_masks = np.zeros((num_objects,pred.shape[0],pred.shape[1],pred.shape[2]))
        for i in range(1,num_objects+1):
            all_res_masks[i-1,:,:,:] = (pred == i).astype(np.uint8)
        all_res_masks = all_res_masks[:, 1:-1, :, :]
        all_gt_masks = Fs[0][1:1+num_objects]
        all_gt_masks = all_gt_masks[:, :-1, :, :]
        j_metrics_res, f_metrics_res = evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
            if 'F' in metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)
        J, F = metrics_res['J'], metrics_res['F']
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
        df_video.loc[len(df_video)] = np.array([seq_name,final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])

    J, F = metrics_res['J'], metrics_res['F']
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    df = pd.DataFrame(columns=g_measures)
    df.loc[0] = g_res
    return df,df_video



if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
        parser.add_argument("-s", type=str, help="set", required=True)
        parser.add_argument("-y", type=int, help="year", required=True)
        parser.add_argument("-D", type=str, help="path to data",default='../data.nosync/DAVIS2017')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights",required=True)
        parser.add_argument("-metric_path", type=str, help="path to output metrics",default='./results/metrics')
        parser.add_argument("-output_path", type=str, help="path to segmentation maps",default='./results/outputs')

        return parser.parse_args()

    args = get_arguments()

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D
    pth = args.p

   
    metrics_file_name = pth.rsplit("/")[-1].rsplit(".")[0] + '_mean_metrics'
    metrics_file_name_video = pth.rsplit("/")[-1].rsplit(".")[0] + '_video_metrics'
    metrics_file_name = os.path.join(args.metric_path , metrics_file_name)
    metrics_file_name_video = os.path.join(args.metric_path , metrics_file_name_video)

    output_file_name = pth.rsplit("/")[-1].rsplit(".")[0]
    output_mask_path = args.output_path
    output_viz_path = args.output_path
    output_mask_path = os.path.join(output_mask_path , output_file_name , 'masks')
    output_viz_path = os.path.join(output_viz_path, output_file_name, 'videos')

    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
    if not os.path.exists(output_viz_path):
        os.makedirs(output_viz_path) 
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testloader = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    

    #Unncomment line when GPU available
    #model.load_state_dict(torch.load(pth))
    if GPU == -1: 
        model.load_state_dict(torch.load(pth))
    else: 
        model.load_state_dict(torch.load(pth,map_location=torch.device('cpu')))
    metric = ['J','F']
    df,video_df = customevaluateandsave(model,Testloader,metric,output_mask_path,output_viz_path)
    df.to_csv(metrics_file_name + '.csv',index = False)
    video_df.to_csv(metrics_file_name_video + '.csv',index = False)
