"""train TransRAC model """
from platform import node
import os
## if your data is .mp4 form, please use RepCountA_raw_Loader.py (slowly)
from dataset.RepCountA_raw_Loader import MyData
## if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
# from dataset.RepCountA_Loader import MyData
# you can use 'tools.video2npz.py' to transform .mp4 to .npz
from models.TransRAC import TransferModel
from training.train_looping import train_loop

# CUDA environment
N_GPU = 1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# # # we pick out the fixed frames from raw video file, and we store them as .npz file
# # # we currently support 64 or 128 frames
# data root path
root_path = os.getcwd() + '/data/'

train_video_dir = 'train'
valid_video_dir = 'valid'

# please make sure the pretrained model path is correct
checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

# TransRAC trained model checkpoint, we will upload soon.
lastckpt = './pretrained/transrac_ckpt_pytorch_171.pt'

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]

NUM_EPOCHS = 300
LR = 8e-6
BATCH_SIZE = 32

my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)
train_label_dir = f'train.csv'
valid_label_dir = f'valid.csv'
train_dataset = MyData(root_path, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
valid_dataset = MyData(root_path, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)
train_loop(NUM_EPOCHS, my_model, train_dataset, valid_dataset, train=True, valid=True,
        batch_size=BATCH_SIZE, lr=LR, saveckpt=True, ckpt_name='ours', log_dir='ours', device_ids=device_ids,
        lastckpt=lastckpt, mae_error=False, use_wandb=False, fold_index=0)