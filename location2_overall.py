import os
import config
import numpy as np
import torch.optim as optim
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import radar_config as rrc
import metric
from PIL import Image

def nor(frames):
    new_frames = frames.astype(np.float32)/80.0
    return new_frames

def de_nor(frames):
    new_frames = copy.deepcopy(frames)
    new_frames *= 80.0
    new_frames[new_frames>80]=80
    new_frames = new_frames.astype(np.uint8)
    return new_frames

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args=config.configs.args
batch_size = 2

rh='2500'
# train_all=np.load(rrc.path + rh + '_18_19.npy', allow_pickle=True)
# test_all=np.load(rrc.path + rh + '_20.npy', allow_pickle=True)
train_all,test_all=config.get_train_test()
print(train_all.shape,test_all.shape)
# rh='2500'
# year='2018'
# train_all=np.load(rrc.path + rh + '_' + year + '_train.npy', allow_pickle=True)
# test_all=np.load(rrc.path + rh + '_' + year + '_test.npy', allow_pickle=True)
# print(train_all.shape,test_all.shape)

class FLloss(nn.Module):
    def __init__(self):
        super(FLloss, self).__init__()

    def forward(self,pred,true):
        tmp = pred - true
        tmp = torch.square(tmp)
        # print(torch.max(true))
        coe = 0.1 + 1 / (1 + torch.exp(-true*80 / 15))
        tmp = coe * tmp
        tmp = torch.mean(tmp)
        return tmp

class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H  # (hidden, output)

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).cuda())
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).cuda())

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1,
                               output_padding=output_padding),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

class encoder_E(nn.Module):
    def __init__(self, nc=2, nf=32):
        super(encoder_E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2 * nf, stride=2)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3

class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2 * nf, nf, stride=2)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3, 3), stride=2, padding=1,
                                       output_padding=1)  # (nc) x 64 x 64

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3

class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1)  # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2

class decoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)  # (64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1)  # (32) x 32 x 32

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2

class Seq2SeqConvGRU(nn.Module):
    def __init__(self, configs):
        super(Seq2SeqConvGRU, self).__init__()
        self.encoder_E = encoder_E()
        self.encoder_Er = encoder_specific()
        self.convcell=ConvLSTM(input_shape=(175,225), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3, kernel_size=(3, 3))
        self.decoder_Dr = decoder_specific()
        self.decoder_D = decoder_D()

    def forward(self, frames):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        for i in range(10):
            input=frames[:,i]
            input = self.encoder_E(input)
            input_conv = self.encoder_Er(input)
            hidden2, output2 = self.convcell(input_conv, i == 0)

        decoded_Dr = self.decoder_Dr(output2[-1])
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))
        return out_conv

def sample_train(batch_size):
    imgs = []
    for batch_idx in range(batch_size):
        sample_index = random.randint(1, train_all.shape[0]-1)
        batch_imgs = []
        for t in range(11):#1,2,3,4,5,6,7,8,9,10|20
            img_path = train_all[sample_index][t]
            # img = cv2.imread(img_path, 0)[:, :, np.newaxis]
            img=np.load(img_path)[:, :, np.newaxis]
            img[img > 80] = 0
            img[img < 15] = 0
            batch_imgs.append(img)
        imgs.append(np.array(batch_imgs))
    imgs = np.array(imgs)
    return imgs

def sample_test(batch_size,index):
    imgs = []
    for batch_idx in range(batch_size):
        sample_index = index+batch_idx
        batch_imgs = []
        for t in range(11):
            img_path = test_all[sample_index][t]
            img = cv2.imread(img_path, 0)[:, :, np.newaxis]
            img[img > 80] = 0
            img[img < 15] = 0
            batch_imgs.append(img)
        imgs.append(np.array(batch_imgs))
    imgs = np.array(imgs)
    return imgs

def mask_func(x):
    #x:4*10*700*900*1
    y1=x*1.0
    y1[y1<25]=0
    y1[y1>=25]=1

    # y2 = x * 1.0
    # y2[y2 < 40] = 0
    # y2[y2 >= 40] = 1

    # y=np.stack((x,y1,y2),axis=4)
    y = np.stack((x, y1), axis=4)
    y=y.squeeze(-1)
    return y

def mypred(pred,true,x10,index):
    #pred:4*700*900
    path='/home/ices/HX/rainfall_final/pred_new/location2/'

    for i in range(pred.shape[0]):
        if (not os.path.exists(path + str(index + i))):os.mkdir(path + str(index + i))

        tmp1=pred[i]*1
        tmp1=de_nor(tmp1)
        tmp1 = np.array(tmp1, dtype='uint8')
        tmp1 = Image.fromarray(tmp1)
        # tmp1.save(p1 + str(index + i) + '/true_' + str(j + 1) + '.png')
        tmp1.save(path+str(index + i)+'/pred.png')

        tmp1 = pred[i] * 1
        tmp1 = tmp1 * 80
        tmp1 = np.array(tmp1)
        # tmp1 = Image.fromarray(tmp1)
        np.save(path + str(index + i) + '/pred.npy', tmp1)

        tmp2 = true[i] * 1
        tmp2 = de_nor(tmp2)
        tmp2 = np.array(tmp2, dtype='uint8')
        tmp2= Image.fromarray(tmp2)
        # tmp1.save(p1 + str(index + i) + '/true_' + str(j + 1) + '.png')
        tmp2.save(path + str(index + i) + '/true.png')

        tmp3 = x10[i] * 1
        tmp3 = de_nor(tmp3)
        tmp3 = np.array(tmp3, dtype='uint8')
        tmp3 = Image.fromarray(tmp3)
        # tmp1.save(p1 + str(index + i) + '/true_' + str(j + 1) + '.png')
        tmp3.save(path + str(index + i) + '/x10.png')

    return 0

def wrapper_test(model):
    test_num=batch_size*(int(test_all.shape[0]/batch_size))
    index = 0
    model.eval()

    final_mse = 0
    final_vmse_20 = 0
    final_vmse_30 = 0
    csi_20 = 0
    csi_30 = 0
    hss_20 = 0
    hss_30 = 0

    count = 0

    while index<test_num:
        dat= sample_test(batch_size, index)#8*10*101*101*1
        dat = nor(dat)
        ims=mask_func(dat[:,:-1])
        ims=torch.Tensor(ims).cuda()
        img_gen= model(ims)
        img_gen=img_gen.detach().cpu().numpy()#4*1*700*900

        # mypred(img_gen[:, -1] * 1, dat[:, -1, :, :, -1] * 1, dat[:, -2, :, :, -1] * 1, index)

        final_mse += np.mean(np.square(img_gen[:, -1] - dat[:, -1, :, :, -1]))
        final_vmse_20 += metric.valid_mse(img_gen[:, -1] * 80, dat[:, -1, :, :, -1] * 80, 20)
        final_vmse_30 += metric.valid_mse(img_gen[:, -1] * 80, dat[:, -1, :, :, -1] * 80, 30)
        csi_20 += metric.csi(img_gen[:, -1] * 80, dat[:, -1, :, :, -1] * 80, 20)
        csi_30 += metric.csi(img_gen[:, -1] * 80, dat[:, -1, :, :, -1] * 80, 30)
        hss_20 += metric.hss(img_gen[:, -1] * 80, dat[:, -1, :, :, -1] * 80, 20)
        hss_30 += metric.hss(img_gen[:, -1] * 80, dat[:, -1, :, :, -1] * 80, 30)

        count = count + 1
        index=index+batch_size

    print('final_mse', final_mse / count)
    print('final_vmse_20', final_vmse_20 / count)
    print('final_vmse_30', final_vmse_30 / count)
    print('csi_20', csi_20 / count)
    print('csi_30', csi_30 / count)
    print('hss_20', hss_20 / count)
    print('hss_30', hss_30 / count)
    return final_mse/count

if __name__=='__main__':
    # wrapper_train()
    model = torch.load('/home/ices/HX/rainfall_final/model_new/lr/lr30.pkl')
    print(model)
    # wrapper_test(model)