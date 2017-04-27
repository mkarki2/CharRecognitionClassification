import cv2
import h5py
from parameters_file import tic, toc, initialize, natural_sort
import numpy as np
import gc
from os import listdir
import random
import pickle

import constants_file as CONST
import theano.sandbox.cuda

import time

start_time = time.time()
gpu = "gpu2"
theano.sandbox.cuda.use(gpu)

#Parameters for running the code. Parameters for Stage 2 are in init_file

epochs = 20
params = initialize()             # OPTIONS FOR TRAINING, TESTING
# params.test_flag = 1                 # 0: TRAIN 1: TEST
# params.load_from_file = 1             # Load Saved Data from File (?)
epoch_ctr=0 # Set to 1 mostly right now
params.finetune_var.retrain = False

if params.test_flag == 0 and params.finetune_var.retrain == False:
    params.finetune_var.model_name = params.finetune_var.model_name + gpu

file_ctr=0
def predictor(parameters):
    params=parameters
    # X_all,Y_all,norm=load_saved_data(params.train_data_folder,train=-2)
    # data = divide_data(X_all, Y_all, params.num_train, params.num_val)
    # X_, Y_ = data[0]

    if params.load_from_file == 1:
        X, Y = load_saved_data(params.test_data_folder,train=0,num_imgs=0)
    else:
        from Stage1 import CoreSample
        X, Y= CoreSample([], # to use default model
                         params.image_names,
                         params.save,
                         params.test_data_folder,#folder
                         params.test_flag,
                         [])#filename if updating normalization parameters
        X=Normalize(X)

    # file = h5py.File('test1.h5', 'r')
    #
    # X = file['/my_data/X'][:]
    # Y = file['/my_data/Y'][:]
    # X=Normalize(X)
    # X=np.clip(X,0,1)

    from Stage2 import predict
    output = predict(X,filename=params.finetune_var.model_test)
    if Y !=[]:
        from sklearn.metrics import accuracy_score
        result = accuracy_score((Y/255).astype(int), output)*100

    # mse = ((Y * 255 - output * 255) ** 2).mean(axis=None)

    # print('Prediction Completed with a MSE of :' + str(mse))
    if params.reconstruct_image == 1:
          # save_images((X[:,0]), name='train')
          # save_images((Y/255).astype(np.int), name='train')
          # save_images(output, name='out_' + params.finetune_var.model_name + '_acc_' + str(int(result)))
          save_images(output, name=params.ctr,params=params)

    return

def save_images(Y, name,params):
    # Y_channel = X[:, 0]
    num_samples = int(Y.shape[0] / (CONST.NUM_PIXELS))
    # Y_channel = Y_channel.reshape(num_samples, 224, 224, 1)
    img_out = ((Y.reshape(num_samples,CONST.IMAGE_DIM[0], CONST.IMAGE_DIM[1]))*255)


    # channel_3 = np.ones((Y.shape[0:3])) * .5
    # channel_2 = np.ones((Y.shape[0:3])) * .5
    #
    # channel_3=np.expand_dims(channel_3, axis=3).astype(np.float32)
    # channel_2=np.expand_dims(channel_2, axis=3).astype(np.float32)
    #)
    # Y = np.concatenate((Y,channel_2, channel_3), axis=3).astype(np.float32)
    # CrCb_out = Y.reshape(num_samples, 224, 224, 3)

    # Y = np.concatenate((Y_channel, CrCb_out), axis=3).astype(np.float32)
    # Y=np.clip(Y, 0, 1)
    for i in range(num_samples):
        # cv2.imshow('disp',Y[i, :, :, :])
        # if name != 'train':
        #     gray = cv2.cvtColor(Y[i, :, :, :], cv2.COLOR_BGR2GRAY)
        #     cv2.imwrite(params.output_folder + "gray_"+ str(i) + ".png", gray * 255)
        # img=cv2.cvtColor(Y[i, :, :, :], cv2.COLOR_YCR_CB2BGR)
        # if name == 'train':
        #   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        import os
        if not os.path.exists(params.output_folder):
            os.makedirs(params.output_folder)
        cv2.imwrite(params.output_folder + str(name+i)+".tif", img_out[i,:,:])
    print(str(name) + ": Images saved at: " + params.output_folder)
    return

def load_saved_data(folder, train,num_imgs):
    if train == 1:
        tic()
        data_list = listdir(folder)
        X_all = []
        Y_all = []
        # random.shuffle(data_list)
        ctr=epoch_ctr%len(data_list)

        file_name=data_list[ctr]
        train_file = h5py.File(folder + file_name, 'r')
        X = train_file['/my_data/X'][:]
        Y = train_file['/my_data/Y'][:]
        start = random.randint(0, int(X.shape[0]/CONST.NUM_PIXELS) - num_imgs)
        if X_all == []:
            X_all = X[start*CONST.NUM_PIXELS:(start+num_imgs)*CONST.NUM_PIXELS,:]
            Y_all = Y[start*CONST.NUM_PIXELS:(start+num_imgs)*CONST.NUM_PIXELS]
        else:
            Y_all = np.concatenate((Y_all, Y), axis=0)
        #Uncomment Line below
        X_all=Normalize(X_all)
        # Y_all=np.expand_dims(Y_all,axis=1)
    elif train == 0:
        tic()
        data_list = listdir(folder)
        X_all = []
        Y_all = []
        # random.shuffle(data_list)
        # for i in range(len(data_list)):
        file_name = data_list[0]#file_ctr]
        test_file = h5py.File(folder + file_name, 'r')
        X = test_file['/my_data/X'][:]
        Y = test_file['/my_data/Y'][:]
        if X_all == []:
            X_all = X
            Y_all = Y
        else:
            X_all = np.concatenate((X_all, X), axis=0)
            Y_all = np.concatenate((Y_all, Y), axis=0)
        X_all = Normalize(X_all)

    elif train == 2:
        tic()
        data_list = listdir(folder)
        data_list.sort()
        X_all = []
        Y_all = []
        # for i in range(len(data_list)):
        # random.shuffle(data_list)
        file_name =data_list[file_ctr]
        test_file = h5py.File(folder + file_name, 'r')
        X = test_file['/my_data/X'][:]
        if X_all == []:
            X_all = X
        else:
            X_all = np.concatenate((X_all, X), axis=0)
        X_all = Normalize(X_all)

    X = X_all



    Y = (Y_all)

    # tic()
    # f = h5py.File('/home/exx/PycharmProjects/SAR/Norm_'+file_name, 'w')
    # grp = f.create_group("my_data")
    # grp.create_dataset('X', data=X, compression="gzip")
    # grp.create_dataset('Y', data=Y, compression="gzip")
    # f.close()
    # toc('Data File saved to disk.')
    toc("Data loaded and Normalized from " + folder+file_name)
    # import os
    # os.remove(folder+file_name)
    return X, Y

#Normalize data from a specified minmax folder
def Normalize(x):
    norm = pickle.load(open(params.minmax_file, 'rb'))
    # Normalization
    max_x = norm[1]
    min_x = norm[0]
    d = (max_x - min_x)

    indices = [i for i, j in enumerate(d) if j == 0]
    d[indices] = 1

    d = 1 / d
    d = d[0:CONST.FEATURES]
    x = (x - min_x[0:CONST.FEATURES])
    for j in range(int(x.shape[1] / 2), 1, -1):
        if x.shape[1] % j == 0:
            factor = j
            break
    for i in range(0, x.shape[1], factor):
        x[:, i:i + factor] = x[:, i:i + factor] * d[i:i + factor]
    return x

#Train a model
def train():
    network = []

    best_mse = np.inf

    print_vars(params, 0, log_file=1)
    X_val, Y_val = load_saved_data(params.val_data_folder, train=1, num_imgs=params.num_val)

    from Stage2 import test_DBN

    for i in range(epochs):

        global epoch_ctr
        epoch_ctr = i

        X_train, Y_train = load_saved_data(params.train_data_folder, train=1, num_imgs=params.num_train)
        # from sklearn.utils import shuffle
        # X, Y = shuffle(X, Y, )

        print('Outer Epoch: ' + str(i))



        network, mse = test_DBN(best_mse,
                                network,
                                [(X_train, (Y_train/255).astype(int)), (X_val, (Y_val/255).astype(int))],  #, (X_test, Y_test)],
                                params)
        if (mse < best_mse and i > 2):
            best_mse = mse

        # network =[]
        check_gpu()
        gc.collect()
        params.finetune_var.retrain = True
        params.finetune_var.lr = max(0.01, params.finetune_var.lr * 0.95)
        print('New Finetune LR: ' + str(params.finetune_var.lr))

    print_vars(params, mse, log_file=0)
    return

#Divide the total data into Train, Val etc.
def divide_data(X, Y, num_train, num_val):
    num_train_images = num_train
    total_train = num_train_images * CONST.NUM_PIXELS
    # total_val = total_train + int(num_val) * 224 * 224  # int((20 - num_train_images) / 2) * 224 * 224
    # total_test = total_val + int(num_val) * 224 * 224

    X_train = X[0:total_train, :]
    Y_train = Y[0:total_train]

    X_val = X[total_train:, :]
    Y_val = Y[total_train:]

    return X_train, Y_train, X_val, Y_val #, X_val, Y_val

#Check free bytes in the GPU
def check_gpu():
    import theano.sandbox.cuda.basic_ops as sbcuda
    import theano.tensor as T
    T.config.floatX = 'float32'
    GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
    freeGPUMemInGBs = GPUFreeMemoryInBytes / 1024. / 1024 / 1024
    print("Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs))

#Print parameters of the current training model
def print_vars(opts, mse, log_file):
    line1 = ('Finetune Variables: lr: %f \tepochs: %i\nPretrain Variables: lr: %f \tepochs: %i \tk: %i\n' % (
    opts.finetune_var.lr, opts.finetune_var.epochs, opts.pretrain_var.lr, opts.pretrain_var.epochs,
    opts.pretrain_var.k))
    line2 = (
    'L2_reg: %f  \tbatch size: %i \t output classes: %i\n' % (opts.L2_reg, opts.batch_size, opts.num_output))
    line3 = ('Layer Sizes: ' + str(opts.layer_sizes))
    line4 = ('Dropout: ' + str(opts.dropout))
    print(line1, line2, line3, line4)
    # print('GPU USED:',gpu)
    if log_file == 1:
        with open('/home/exx/PycharmProjects/CharRecognitionClassification/logs/' + opts.finetune_var.model_name + '.txt', 'a') as out:
            out.write(line1)
            out.write(line2)
            out.write(line3)
            out.write(line4)

            # out.write('GPU USED: '+gpu)
            # out.write(Note+'\n')
            if mse != 0:
                out.write('Final Test Error: %i\n' % mse)
        print('Log File with variables for this Model: ' + '/logs/' + opts.finetune_var.model_name + '.txt')
    return

if __name__ == "__main__":

    if params.test_flag==1:
        predictor(params)
    elif params.test_flag==-1:
        pass
    else:
        train()#load "data" [maps +images] from file

    print("The Code ran for --- %f Hours --- " % ((time.time() - start_time)/3600))