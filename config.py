import argparse
import numpy as np

class configs:
    parser = argparse.ArgumentParser(description='PyTorch satellite prediction model - Seq2SeqConvGRU')
    parser.add_argument('--is_training', type=int, default=1)

    # data
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--total_length', type=int, default=11)
    parser.add_argument('--img_width', type=int, default=900)
    parser.add_argument('--img_height', type=int, default=700)
    parser.add_argument('--img_channel', type=int, default=1)

    # model
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--layer_norm', type=int, default=1)

    # optimization
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=500*100)
    parser.add_argument('--display_interval', type=int, default=500)
    parser.add_argument('--test_interval', type=int, default=500)

    args = parser.parse_args()

def get_train_test():
    path='/home/ices/HX/dataset/radar_all/2500/'
    train1=np.load(path+'2015.npy', allow_pickle=True)
    train2 = np.load(path + '2016.npy', allow_pickle=True)
    train3 = np.load(path + '2017.npy', allow_pickle=True)

    train=np.concatenate((train1,train2,train3))
    test=np.load(path + '2018.npy', allow_pickle=True)

    return train,test
