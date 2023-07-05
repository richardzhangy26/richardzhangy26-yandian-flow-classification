import gradio as gr
import os
from video_flow_inference import inference_video
from test import infer
from test import extract_frames
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from create_dataset import UCF101Dataset
from lrcn_model import ConvLstm
from utils_action_recognition import save_setting_info, plot_label_distribution, \
    plot_images_with_predicted_labels, create_folder_dir_if_needed, load_all_dataset_to_RAM, split_data, \
    test_model
import os
import cv2

frames_dir = r'./data/test'
parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size (default:32)')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate (default:5e-4')
parser.add_argument('--num_workers', default=4, type=int,
                    help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
# 将数据集直接加载到RAM，以加快计算速度。通常在类的数量较少时使用(默认值:False)
parser.add_argument('--load_all_data_to_RAM', default=False, type=bool,
                    help='load dataset directly to the RAM, for faster computation. usually use when the num of class '
                         'is small (default:False')
# Conv FC输出的dim维数(默认值:512)
parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
# 处于LSTM隐藏状态的特征数量(默认值:256)
parser.add_argument('--hidden_size', default=256, type=int,
                    help="The number of features in the LSTM hidden state (default:256)")
# LSTM重复层的数量(默认值:2)
parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
# 将LSTM设置为双向(默认值:True)
parser.add_argument('--bidirectional', default=False, type=bool, help='set the LSTM to be bidirectional (default:True)')
# 打开一个新文件夹来保存运行信息，如果为false，信息将保存在项目目录中，如果为debug，信息将保存在debug文件夹中(默认值:True)
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project '
                         'dir, if debug the info would be saved in debug folder(default:True)')

# 加载checkpoint并继续使用它进行训练
parser.add_argument('--load_checkpoint', default=True, type=bool,
                    help='Loading a checkpoint and continue training with it')
# checkpoint路径
parser.add_argument('--checkpoint_path',
                    default=r'./checkpoint/best_epoch_198.pth.tar',
                    type=str, help='Optional path to checkpoint model')
# checkpoint保存间隔
parser.add_argument('--checkpoint_interval', default=5, type=int, help='Interval between saving model checkpoints')
# 验证测试的间隔（默认值：5）
parser.add_argument('--val_check_interval', default=5, type=int, help='Interval between running validation test')
# 保存结果的位置  os.getcwd() 方法用于返回当前工作目录
parser.add_argument('--local_dir', default=os.getcwd(), help='The local directory of the project, setting where to '
                                                             'save the results of the run')

parser.add_argument('--ucf_list_dir', default='./data',
                    type=str, help='path to find the UCF101 list, splitting the data to train and test')                                                            
# 类别数
parser.add_argument('--number_of_classes', default=6, type=int, help='The number of classes we would train on')



# from label.test import infer
def video_identity(video):
    out_video = inference_video(video)
    video_name1 = out_video.split('/')[-1]
    video_name2 = os.path.splitext(video_name1)[0]
    video_frames_dir = os.path.join(frames_dir, video_name2)
    extract_frames(out_video, video_frames_dir)
    result = infer(parser)
    return out_video,{'0012':result[0], '0221':result[1], '1012':result[2], '1102':result[3],'1122':result[4],'1221':result[5]}

demo = gr.Interface(video_identity,
                    gr.Video(),
                    ["playable_video","label"],
                    examples=[
                        os.path.join(os.path.abspath(''),
                                     "video/example/0012_1438.mp4"), os.path.join(os.path.abspath(''),
                                     "video/example/0012_1600.mp4"), os.path.join(os.path.abspath(''),
                                     "video/example/0012_2944.mp4")],
                    cache_examples=False,
                    theme="freddyaboulton/dracula_revamped",
                    description='''
                                0012  水平向左，垂直向上，逆时针，强度无明显变化
                                
                                0221 水平向左，无垂直眼震，无轴向眼震，由强变弱
                                
                                1012 水平向右，垂直向上，逆时针，强度无明显变化
                                
                                1102 水平向右，垂直向下，顺时针，强度无明显变化
                                
                                1122  水平向右，垂直向下，无轴向眼震，强度无明显变化
                                
                                1221  水平向右，无垂直眼震，无轴向眼震，由强变弱
                                ''')

if __name__ == "__main__":
    demo.launch(share=True)
