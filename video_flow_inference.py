# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Sequence
import time
import cv2
import torch.cuda
from numpy import ndarray
from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow


try:
    import imageio
except ImportError:
    imageio = None




def inference_video(video_file:str):
    config_file = r'./configs/liteflownet/liteflownet_8x1_500k_flyingthings3d_subset_384x768.py'
    checkpoint_file = r'./checkpoints/liteflownet_8x1_500k_flyingthings3d_subset_384x768.pth'
    # build the model from a config file and a checkpoint file
    print(video_file)
    print(video_file)
    model = init_model(config_file, checkpoint_file, device='cuda:0'if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(video_file)

    assert cap.isOpened(), f'Failed to load video file {video_file}'

    # get video info
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    imgs = []
    while (cap.isOpened()):
        # Get frames
        flag, img = cap.read()
        if not flag:
            break
        imgs.append(img)

    frame_list = []

    for i in range(len(imgs) - 1):
        img1 = imgs[i]
        img2 = imgs[i + 1]
        # estimate flow
        result = inference_model(model, img1, img2)
        flow_map = visualize_flow(result, None)
        # visualize_flow return flow map with RGB order
        flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
        #concat_frame_flow = np.concatenate((flow_map, imgs[i]), axis=1)
        #frame_list.append(concat_frame_flow)
        frame_list.append(flow_map)

    size = (frame_list[0].shape[1], frame_list[0].shape[0])
    cap.release()
    out_file = video_file.replace('.mp4','_.mp4')
    create_video(frame_list, out_file, fourcc, fps, size)
  
    return out_file

def create_video(frames: Sequence[ndarray], out: str, fourcc: int, fps: int,
                 size: tuple) -> None:
    """Create a video to save the optical flow.

    Args:
        frames (list, tuple): Image frames.
        out (str): The output file to save visualized flow map.
        fourcc (int): Code of codec used to compress the frames.
        fps (int):      Framerate of the created video stream.
        size (tuple): Size of the video frames.
    """
    # init video writer
    video_writer = cv2.VideoWriter(out, fourcc, fps, size, True)

    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':

    start_time = time.time()
    config_file = r'./configs/liteflownet/liteflownet_8x1_500k_flyingthings3d_subset_384x768.py'
    checkpoint_file = r'./checkpoints/liteflownet_8x1_500k_flyingthings3d_subset_384x768.pth'

    #输入视频地址
    video_file = r'./video/example/0012--1438.mp4'

    #输出视频地址
    out_file = r'./video/example_flow/0012--1438.mp4'

    inference_video(video_file)

    end_time =time.time()
    run_time = end_time - start_time
    print(run_time)