import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

def ecc(im1,im2):
    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    im1_tmp=im1.astype(np.float32)
    im2_tmp = im2.astype(np.float32)
    # im1_tmp[im1_tmp<15]=0
    # im1_tmp[im1_tmp >= 15] = 1
    # im2_tmp[im2_tmp < 15] = 0
    # im2_tmp[im2_tmp >= 15] = 1
    # print(np.sum(im1_tmp),np.sum(im2_tmp))


    (cc, warp_matrix) = cv2.findTransformECC(im1_tmp, im2_tmp, warp_matrix, warp_mode, criteria,inputMask=None,gaussFiltSize=1)
    # print(warp_matrix)


    # Use warpAffine for Translation, Euclidean and Affine
    im1_aligned = cv2.warpAffine(im1, warp_matrix, (sz[1], sz[0]))

    # Show final results
    # cv2.imshow("Image 1", im1)
    # cv2.imshow("Image 2", im2)
    # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(0)
    return im1_aligned

def main():
    path = '/home/ices/HX/rainfall_final/pred_new/train_location2/'
    all=os.listdir(path)
    all.sort()

    for i in range(len(all)):
        if(i%100==0):print(i)
        # print(i)
        pp=path+str(i)

        s1 = cv2.imread(pp+'/x10.png', 0)
        # s2 = cv2.imread(pp+'/pred.png', 0)
        s2=np.load(pp+'/pred.npy')

        s1[s1 > 80] = 0
        s1[s1 < 15] = 0
        # s2[s2 > 80] = 0
        # s2[s2 < 10] = 0

        if(np.max(s1)==0 or np.max(s2)==0):
            print(i)
            cc=s1
        else:cc = ecc(s1*1.0, s2*1.0)


        # cc = np.array(cc, dtype='uint8')
        # cc = Image.fromarray(cc)
        # cc.save(pp+'/x_aligned.png')
        cc=np.asarray(cc)
        np.save(pp+'/x_aligned.npy',cc)


if __name__=='__main__':
    main()