# read multiple images and produce a gif

import imageio
import os
import numpy as np
import cv2
import argparse


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='eval_vis')
    # whether to save the gif
    parser.add_argument('--gif', action='store_true', default=False)
    # whether to save the mp4
    parser.add_argument('--mp4', action='store_true', default=True)
    parser.add_argument('--width',type=int,default=256)
    parser.add_argument('--height',type=int,default=256)
    args = parser.parse_args()

    root_path = args.root_path
    w,h = args.width, args.height

    pred_path = os.path.join(root_path, 'pred_gs1.5_scale-cond1.0-ref1.0')
    pose_path = os.path.join(root_path, 'cond')
    gt_path = os.path.join(root_path, 'gt')
    ref_path = os.path.join(root_path, 'ref')
    # read all images in this directory
    files = os.listdir(pred_path)
    # sort the files by name
    files.sort()
    split_names = [x.split('_') for x in files]
    # two different names:
    # TiktokDance_203_006_1x1_00348jpg.png
    # TiktokDance_00337_0133png.png
    cur_pic_name = split_names[0]
    cur_idx = str(cur_pic_name[1]) + str(cur_pic_name[2]) if len(cur_pic_name) == 5 else str(cur_pic_name[1])
    cur_picures = []

    for i in range(1, len(split_names)):
        name = split_names[i]
        idx = str(name[1]) + str(name[2]) if len(name) == 5 else str(name[1])
        if idx != cur_idx: # save all these pictures as a mp4 videos
            if args.mp4:
                print(f'Saving {len(cur_picures)} pictures as a mp4 video with name {cur_idx}')
                imageio.mimsave('./eval_vis/mp4s/' + cur_idx + '.mp4', cur_picures, fps=30)
            if args.gif:
                print(f'Saving {len(cur_picures)} pictures as a gif with name {cur_idx}')
                imageio.mimsave('./eval_vis/gifs/' + cur_idx + '.gif', cur_picures, fps=30)
            cur_picures = []
            cur_idx = idx
        else:
            pred_img = imageio.imread(os.path.join(pred_path, files[i]))
            cond_img = imageio.imread(os.path.join(pose_path, files[i]))
            gt_img = imageio.imread(os.path.join(gt_path, files[i]))
            ref_img = imageio.imread(os.path.join(ref_path, files[i]))
            # resize these images into w,h using bilinear interpolation
            pred_img = cv2.resize(pred_img, (w,h), interpolation=cv2.INTER_LINEAR)
            cond_img = cv2.resize(cond_img, (w,h), interpolation=cv2.INTER_LINEAR)
            gt_img = cv2.resize(gt_img, (w,h), interpolation=cv2.INTER_LINEAR)
            ref_img = cv2.resize(ref_img, (w,h), interpolation=cv2.INTER_LINEAR)
            # concatenate these images into one img
            up_img = np.concatenate((pred_img, cond_img), axis=1)
            down_img = np.concatenate((gt_img, ref_img), axis=1)
            img = np.concatenate((up_img, down_img), axis=0)
            cur_picures.append(img)
            
                    
