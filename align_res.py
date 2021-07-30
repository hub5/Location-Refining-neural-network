import cv2
import numpy as np
from PIL import Image
import metric
import os

def myunsqueeze(x):
    tmp=[x]
    return np.asarray(tmp)

path = '/home/ices/HX/rainfall_final/pred_new/location2/'
all=os.listdir(path)
all.sort()

final_mse = 0
final_vmse_20 = 0
final_vmse_30 = 0
csi_20 = 0
csi_30 = 0
hss_20 = 0
hss_30 = 0

for i in range(len(all)):
    # if(i==4):break
    if(i%100==0):print(i)
    pp=path+str(i)

    # print(pp)

    # s1 = cv2.imread(pp+'/pred.png', 0)
    s1=np.load(pp+'/x_aligned.npy')
    # print(np.max(s1),np.min(s1))
    true=cv2.imread(pp+'/true.png', 0)
    # s1[s1 > 80] = 0
    # s1[s1 < 15] = 0
    true[true>80]=0
    true[true < 15] = 0

    s1=myunsqueeze(s1)
    true=myunsqueeze(true)

    img_gen=s1*1.0
    dat=true*1.0

    final_mse += metric.mse(img_gen, dat)
    final_vmse_20 += metric.valid_mse(img_gen, dat,20)
    final_vmse_30 += metric.valid_mse(img_gen, dat, 30)
    csi_20 += metric.csi(img_gen, dat, 20)
    csi_30 += metric.csi(img_gen, dat, 30)
    hss_20 += metric.hss(img_gen, dat, 20)
    hss_30 += metric.hss(img_gen, dat, 30)

count=len(all)
print('final_mse', final_mse / count)
print('final_vmse_20', final_vmse_20 / count)
print('final_vmse_30', final_vmse_30 / count)
print('csi_20', csi_20 / count)
print('csi_30', csi_30 / count)
print('hss_20', hss_20 / count)
print('hss_30', hss_30 / count)