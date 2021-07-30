import os
import config
import numpy as np
import torch.optim as optim
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils_pkg.utils import *
import time
import copy
import radar_config as rrc
import metric
import LR.loss_function as ll
from PIL import Image

def nor(frames):
    new_frames = frames.astype(np.float32)/80.0
    return new_frames

def de_nor(frames):
    new_frames = copy.deepcopy(frames)
    new_frames *= 80.0
    new_frames[new_frames < 0] = 0
    new_frames[new_frames>80]=80
    new_frames = new_frames.astype(np.uint8)
    return new_frames

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args=config.configs.args
batch_size = args.batch_size

class refining_network(nn.Module):
    def __init__(self):
        super(refining_network, self).__init__()

        self.conv1=nn.Conv2d(2,16,kernel_size=(9,9),stride=1,padding=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5,5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=(5,5), stride=1, padding=2)

    # def sobel(self,x):
    #     #x:batch_size*2*700*900
    #     kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    #     kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
    #
    #     kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    #     kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()
    #
    #     weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    #     weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
    #
    #     grad_x = F.conv2d(x, weight_x,padding=1)
    #     grad_y = F.conv2d(x, weight_y,padding=1)
    #     gradient = torch.abs(grad_x) + torch.abs(grad_y)
    #     return gradient

    def forward(self,input):
        #input:batch_size*2*700*900
        output=self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)

        # gradient=self.sobel(output)

        return output

train_path='/home/ices/HX/rainfall_final/pred_new/train_location2/'
test_path='/home/ices/HX/rainfall_final/pred_new/location2/'
train_num=4432
test_num=1956

def sample_train(batch_size):
    imgs = []
    for batch_idx in range(batch_size):
        sample_index = random.randint(1, train_num-1)
        batch_imgs = []

        x_aligned=train_path+str(sample_index)+'/x_aligned.npy'
        x_aligned=np.load(x_aligned)

        x_pred=train_path+str(sample_index)+'/pred.npy'
        x_pred =np.load(x_pred)

        true=train_path+str(sample_index)+'/true.png'
        true = cv2.imread(true, 0)*1.0
        true[true > 80] = 0
        true[true < 15] = 0

        batch_imgs.append(x_aligned)
        batch_imgs.append(x_pred)
        batch_imgs.append(true)


        imgs.append(np.array(batch_imgs))
    imgs = np.array(imgs)
    return imgs

def sample_test(batch_size,index):
    imgs = []
    for batch_idx in range(batch_size):
        sample_index = index+batch_idx
        batch_imgs = []

        x_aligned = test_path + str(sample_index) + '/x_aligned.npy'
        x_aligned = np.load(x_aligned)

        x_pred = test_path + str(sample_index) + '/pred.npy'
        x_pred = np.load(x_pred)

        true = test_path + str(sample_index) + '/true.png'
        true = cv2.imread(true, 0) * 1.0
        true[true > 80] = 0
        true[true < 15] = 0

        batch_imgs.append(x_aligned)
        batch_imgs.append(x_pred)
        batch_imgs.append(true)
        imgs.append(np.array(batch_imgs))
    imgs = np.array(imgs)
    return imgs

def mypred(pred,index):
    #pred:4*700*900
    path='/home/ices/HX/rainfall_final/pred_new/location2/'

    for i in range(pred.shape[0]):
        if (not os.path.exists(path + str(index + i))):os.mkdir(path + str(index + i))
        tmp1 = pred[i] * 1
        tmp1 = tmp1 * 80
        tmp1 = np.array(tmp1)
        tmp1 = Image.fromarray(tmp1)
        np.save(path + str(index + i) + '/refining2_7.npy', tmp1)

    return 0

def wrapper_test(model):
    # test_num=batch_size*(int(test_all.shape[0]/batch_size))
    # test_num=8
    print('test num:',test_num)
    index = 0
    model.eval()
    count = 0

    final_mse = 0
    final_vmse_20 = 0
    final_vmse_30 = 0
    csi_20 = 0
    csi_30 = 0
    hss_20 = 0
    hss_30 = 0

    while index<test_num:
        if(index%100==0):print(index)
        dat= sample_test(batch_size, index)#8*10*101*101*1
        dat = nor(dat)
        ims=torch.Tensor(dat).cuda()
        # ims = ims.unsqueeze(-1)  # 4*3*700*900*1
        img_gen= model(ims[:,:-1])
        img_gen=img_gen.detach().cpu().numpy()#4*1*700*900

        mypred(img_gen[:, -1] * 1, index)

        final_mse += np.mean(np.square(img_gen[:, -1] - dat[:, -1]))
        final_vmse_20 += metric.valid_mse(img_gen[:, -1] * 80, dat[:, -1] * 80, 20)
        final_vmse_30 += metric.valid_mse(img_gen[:, -1] * 80, dat[:, -1] * 80, 30)
        csi_20 += metric.csi(img_gen[:, -1] * 80, dat[:, -1] * 80, 20)
        csi_30 += metric.csi(img_gen[:, -1] * 80, dat[:, -1] * 80, 30)
        hss_20 += metric.hss(img_gen[:, -1] * 80, dat[:, -1] * 80, 20)
        hss_30 += metric.hss(img_gen[:, -1] * 80, dat[:, -1] * 80, 30)

        count = count + 1
        index = index + batch_size

    print('final_mse', final_mse / count)
    print('final_vmse_20', final_vmse_20 / count)
    print('final_vmse_30', final_vmse_30 / count)
    print('csi_20', csi_20 / count)
    print('csi_30', csi_30 / count)
    print('hss_20', hss_20 / count)
    print('hss_30', hss_30 / count)

if __name__=='__main__':
    model = torch.load('/home/ices/HX/rainfall_final/model_new/refining/refining_new2/refining7.pkl')
    print(model)
    wrapper_test(model)