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
# 类别数
parser.add_argument('--number_of_classes', default=6, type=int, help='The number of classes we would train on')


def infer(parser):
    # ====== set the run settings ======
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data_names, val_data_names, test_data_names = split_data(args.ucf_list_dir)
    dataset_order = ['train', 'val', 'test']
    datasets = {dataset_order[index]: UCF101Dataset(args.ucf_list_dir, x, mode=dataset_order[index])
                for index, x in enumerate([train_data_names, val_data_names, test_data_names])}

    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True)
                   for x in ['train', 'val', 'test']}
    # ======= if args.load_all_data_to_RAM True load dataset directly to the RAM (for faster computation) ======
    if args.load_all_data_to_RAM:
        dataloaders = load_all_dataset_to_RAM(dataloaders, dataset_order, args.batch_size)
    # plot_label_distribution(dataloaders, folder_dir, args.load_all_data_to_RAM, label_decoder_dict)
    num_class = args.number_of_classes
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    # ====== setting optimizer and criterion parameters ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    # ====== start training the model ======
    for epoch in range(args.epochs):
        test_loss, test_acc, predicted_labels, images, true_labels, index,output = test_model(model, dataloaders['test'],
                                                                                       device, criterion,
                                                                                       mode='save_prediction_label_list')
        dict = {0: '0012', 1: '0221', 2: '1012', 3: '1102', 4: '1122', 5: '1221'}
        for m, n in zip(true_labels, predicted_labels):
            for x, y in zip(m, n):
                x = int(x)
                y = int(y)
                mlabel = dict[x]
                nlabel = dict[y]

                return output.reshape(-1).tolist(),[mlabel,nlabel]


def extract_frames(video_path, frames_dir):
    # 创建目标文件夹
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    chouzhen = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # 抽帧  两帧取一

        if chouzhen % 2 == 0:
            frame_path = os.path.join(frames_dir, 'frame{:06d}.jpg'.format(frame_count))
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        chouzhen += 1

    # 释放VideoCapture对象
    video_capture.release()


if __name__ == '__main__':
    # 视频地址  随意改
    video_dir = r'D:\pycharmProject\yandian_flow\label\video\1012_103.mp4'



    # 勿动
    frames_dir = r'.\data\test'
    parser.add_argument('--ucf_list_dir', default='./data',
                        type=str, help='path to find the UCF101 list, splitting the data to train and test')
    video_name1 = video_dir.split('\\')[-1]
    video_name2 = os.path.splitext(video_name1)[0]
    video_frames_dir = os.path.join(frames_dir, video_name2)
    print(video_frames_dir)
    extract_frames(video_dir, video_frames_dir)

    infer()
