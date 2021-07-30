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

def wrapper_test(model):
    # test_num=batch_size*(int(test_all.shape[0]/batch_size))
    # test_num=8
    print('test num:',test_num)
    index = 0
    model.eval()
    count = 0

    final_mse = 0
    final_vmse = 0
    final_mae = 0
    csi_30 = 0
    csi_35 = 0
    csi_40 = 0
    final_ssim = 0

    while index<test_num:
        dat= sample_test(batch_size, index)#8*10*101*101*1
        dat = nor(dat)
        ims=torch.Tensor(dat).cuda()
        # ims = ims.unsqueeze(-1)  # 4*3*700*900*1
        img_gen= model(ims[:,:-1])
        img_gen=img_gen.detach().cpu().numpy()#4*1*700*900

        # mse = np.mean(np.square(img_gen[:, -1] - dat[:, -1,:,:,-1]))
        mse = metric.mse(img_gen[:, -1] * 80, dat[:, -1] * 80)
        vmse = metric.valid_mse(img_gen[:, -1] * 80, dat[:, -1] * 80)
        mae = metric.mae(img_gen[:, -1] * 80, dat[:, -1] * 80)

        final_mse += mse
        final_vmse += vmse
        # final_mae += mae
        csi_30 += metric.csi(img_gen[:, -1] * 80, dat[:, -1] * 80, 30)
        # csi_35 += metric.csi(img_gen[:, -1] * 80, dat[:, -1] * 80, 35)
        # csi_40 += metric.csi(img_gen[:, -1] * 80, dat[:, -1] * 80, 40)
        # final_ssim += metric.myssim(img_gen[:, -1] * 80, dat[:, -1] * 80)

        count = count + 1
        index = index + batch_size

    print('final_mse', final_mse / count)
    print('final_vmse', final_vmse / count)
    print('final_mae', final_mae / count)
    print('csi_30', csi_30 / count)
    print('csi_35', csi_35 / count)
    print('csi_40', csi_40 / count)
    print('ssim', final_ssim / count)
    return final_mse / count

def wrapper_train():
    test_min_mse = 10000
    ##nn config
    model=refining_network()
    model.cuda()

    MSE_criterion = nn.MSELoss(size_average=True)
    MAE_criterion = nn.L1Loss(size_average=True)
    SSIM_criterion = ll.SSIM()
    csi_criterion=ll.CSI_Loss()



    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    print('begin to train,lr=',args.lr)

    test_epoch=1
    train_time1=time.time()
    for itr in range(1, 2*args.max_iterations + 1):
        model.train()
        ims = sample_train(batch_size=batch_size)#4*3*700*900
        ims = nor(ims)# ims/80
        ims=torch.Tensor(ims).cuda()
        # ims=ims.unsqueeze(-1)#4*3*700*900*1
        optimizer.zero_grad()
        next_frames= model(ims[:,:-1])#4*1*700*900
        # loss = 0.4*MSE_criterion(next_frames[:,-1], ims[:, -1])+0.3*MAE_criterion(next_frames[:,-1], ims[:, -1])+0.3*SSIM_criterion(next_frames[:,-1], ims[:, -1])
        # print(MSE_criterion(next_frames[:, -1], ims[:, -1]),csi_criterion(next_frames[:, -1], ims[:, -1]))
        loss1=MSE_criterion(next_frames[:, -1], ims[:, -1])#0.01
        loss2=csi_criterion(next_frames[:, -1], ims[:, -1])#0.08
        loss3=SSIM_criterion(next_frames[:, -1], ims[:, -1])#0.75
        loss = loss1+0.1*loss2+0.001*loss3

        if(itr%500==0):print(loss1,loss2,loss3,loss)

        loss.backward()
        optimizer.step()

        # train_time2 = time.time()
        # print('train time:', train_time2 - train_time1)

        test_time1=time.time()
        if itr % args.test_interval == 0:
            train_time2=time.time()
            print('train time:',train_time2-train_time1)
            print('test epoch:',test_epoch)
            test_mse= wrapper_test(model)
            test_time2 = time.time()
            print('test time:', test_time2 - test_time1)
            print('test mse is:',str(test_mse))

            if(test_epoch<30):
                test_min_mse=test_mse
                save_path = '/home/ices/HX/rainfall_final/model_new/refining/refining_new1/' + 'refining' + str(test_epoch) + '.pkl'
                torch.save(model, save_path)

            test_epoch += 1
            train_time1=time.time()

if __name__=='__main__':
    wrapper_train()