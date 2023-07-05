
import gradio as gr
import os 
import gradio as gr
import os
# from video_flow_inference import inference_video
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
import time
from sava_video import save_video
time_stamps = []
# a = [0,1,2,3,4,5]
i = 0
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



def play_video(_):

    time_stamps.append(time.time())

def pause_video(_):
    time_stamps.append(time.time())
    print(f"pause time_stamps:{time_stamps}")
def record_time_start(_):
    # 当按钮被按下时，记录当前时间
    # 如果记录了两次时间，计算并返回时间差
    if len(time_stamps) >= 2:
        time_diff = time_stamps[-1] - time_stamps[-2]
        zhen = int(time_diff*10)
        return f"起始帧: {zhen} "
    else:
        return "请再按一次play"
def record_time_end(_):
    # 当按钮被按下时，记录当前时间
    # 如果记录了两次时间，计算并返回时间差
    if len(time_stamps) >= 2:
        time_diff1 = time_stamps[-1] - time_stamps[-2]
        time_diff2 = time_stamps[1] - time_stamps[0]
        zhen = int((time_diff1+time_diff2)*10)
        return f"结束帧: {zhen} "
    else:
        return "请再按一次pause"

def video_identity1(video):
    outputvideo = save_video(video)
    return  video,outputvideo
def radio_content(level,vertical,axial,intensity,record_start,record_end):
    return  f"水平方向：{level}\n垂直方向：{vertical}\n轴向：{axial}\n眼震强度变化：{intensity}\n起始帧：{record_start}\n结束帧：{record_end}\n"
def clean_output(out):
   time_stamps=[]
   return "","",""
def record_content(records,level,vertical,axial,intensity,record_start,record_end):
    global i
    records["水平方向"][i] = level
    records["垂直方向"][i] = vertical
    records["轴向"][i] = axial
    records["眼震强度变化"][i] = intensity
    records["起始帧"][i] = record_start
    records["结束帧"][i] = record_end
    i = i + 1
    return records
def inference_video(video):
    return video
# from label.test import infer
def video_identity(video):
    out_video = inference_video(video)
    video_name1 = out_video.split('/')[-1]
    video_name2 = os.path.splitext(video_name1)[0]
    video_frames_dir = os.path.join(frames_dir, video_name2)
    extract_frames(out_video, video_frames_dir)
    result,label = infer(parser)
    return out_video,{'0012':result[0], '0221':result[1], '1012':result[2], '1102':result[3],'1122':result[4],'1221':result[5]},f"真实标签为{label[0]},预测标签为{label[1]}"

with gr.Blocks(theme="freddyaboulton/dracula_revamped",title="BPPV智能辅助诊断系统") as demo:
    with gr.Tab("智能辅助标注"):
      video = gr.Video(label="眼震视频",source="upload",interactive=True,visible=True)
      with gr.Group():
          with gr.Row():
              with gr.Column():
                  level = gr.Radio(["左(0)","右(1)","无明显水平眼震(2)","其他特殊类型眼震(3)","干扰(4)"],label="水平方向")
              with gr.Column():
                  vertical = gr.Radio(["上(0)","下(1)","无明显垂直眼震(2)","其他特殊类型眼震(3)","干扰(4)"],label="垂直方向")
          with gr.Row():
              with gr.Column():
                  axial = gr.Radio(["顺时针(0)","逆时针(1)","无明显轴向眼震(2)","其他特殊类型眼震(3)","干扰(4)"],label="轴向")
              with gr.Column():
                  intensity = gr.Radio(["上(0)","下(1)","无明显垂直眼震(2)","其他特殊类型眼震(3)","干扰(4)"],label="眼震强度变化")
      with gr.Group():
          with gr.Row():
              with gr.Column():
                  record_start = gr.Textbox(lines=1,placeholder="起始帧")
                  record_start_button = gr.Button(value="开始标记")
              with gr.Column():
                  record_end = gr.Textbox(lines=1,placeholder="结束帧")
                  record_end_button = gr.Button(value="结束标记")
      i = 0
      # video = gr.Video(label="眼震视频",source="upload",interactive=True,visible=True)

      record_start_button.click(fn=record_time_start,outputs=record_start)
      record_end_button.click(fn=record_time_end,outputs=record_end)
      output_video = gr.Video(label="眼震视频_输出",source="upload",interactive=True,visible=True)

      record_button = gr.Button(value="记录")
      record_button.click(fn=video_identity1,inputs=video,outputs=[video,output_video])
      output_video.play(fn=play_video)
      output_video.pause(fn=pause_video)
      submit_btn = gr.Button(value="提交")
      clean_btn = gr.Button(value="清空")
      # submit_btn.click(fn=radio_content,inputs=[level,vertical,axial,intensity,record_start,record_end],outputs=out)
      record = gr.Dataframe(
          headers=["水平方向", "垂直方向", "轴向","眼震强度变化","起始帧","结束帧"],
          datatype=["str", "str", "str","str","str","str"],
          row_count=3,
          col_count=(6, "fixed"),
      )
      save_btn = gr.Button(value="保存为csv")
      submit_btn.click(fn=record_content,inputs=[record,level,vertical,axial,intensity,record_start,record_end],outputs=record)
    with gr.Tab("类型智能诊断"):
      gr.Markdown(
      """
      # 标签类别说明
      0012  水平向左，垂直向上，逆时针，强度无明显变化
      
      0221 水平向左，无垂直眼震，无轴向眼震，由强变弱
      
      1012 水平向右，垂直向上，逆时针，强度无明显变化
      
      1102 水平向右，垂直向下，顺时针，强度无明显变化    
      
      1122  水平向右，垂直向下，无轴向眼震，强度无明显变化
      
      1221  水平向右，无垂直眼震，无轴向眼震，由强变弱
      """)
      with gr.Row():
        with gr.Column(scale=2):
          input_video = gr.Video(label="眼震视频",source="upload",interactive=True,visible=True)
          output_video = gr.Video(label="光流视频",source="upload",interactive=True,visible=True)
        with gr.Column(scale=2):
          button = gr.Button(value="开始计算")
          label = gr.Label(label="根据光流计算各眼震类别概率值")
      with gr.Column():
          text = gr.Textbox(value="输出眼震标签值和预测值")
          gr.Examples(
                examples=[
                      os.path.join(os.path.abspath(''),
                                    "video/example/0012_1438.mp4"), os.path.join(os.path.abspath(''),
                                    "video/example/0012_1600.mp4"), os.path.join(os.path.abspath(''),
                                    "video/example/0012_2944.mp4")],
                inputs = input_video,
                outputs=[output_video,label],
                fn = video_identity,
                cache_examples=False
          )
          
      button.click(video_identity,inputs=[input_video],outputs=[output_video,label,text])

if __name__ == "__main__":
    gr.themes.Base(primary_hue="red")
    demo.launch(share=True)