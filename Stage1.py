import theano
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import constants_file as CONST
import cv2
import os
import h5py
from parameters_file import tic, toc, natural_sort
import pickle
import gc

#Defining the VGG 16 Model Architecture
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))  # 1
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))  # 3
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))  # 6
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))  # 8
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))  # 11
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))  # 13
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))  # 15
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 18
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 20
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 22
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 25
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 27
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))  # 29
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

#Loading Images
def LoadImages(X):
    tic()
    Images=[]
    Labels=[]
    for img_name in X:
        if img_name.endswith(".tif"):
            img = cv2.imread(CONST.PATH+'IMAGES/' + img_name,-1)
            label = cv2.imread(CONST.PATH+'LABELS2/' + img_name,-1)

            img= np.expand_dims(img, axis=0)
            label= np.expand_dims(label, axis=0)

            if Images == []:
                Images=img
                Labels=label

            else:
                Images=np.concatenate((Images, img), axis=0).astype(np.float32)
                Labels = np.concatenate((Labels, label), axis=0).astype(np.float32)
               #APPEND
        else:
            print('Wrong Image Type Found! Matrix is saved as all Zeros!')

    toc("Images Loaded.")
    return Images/255, Labels

def LoadImages_Test(X,path):
    tic()
    Images=[]
    for img_name in X:
        if img_name.endswith(".tif"):
            img = cv2.imread(path +'/'+ img_name,-1)

            img= np.expand_dims(img, axis=0)

            if Images == []:
                Images=img

            else:
                Images=np.concatenate((Images, img), axis=0).astype(np.float32)
               #APPEND
        else:
            print('Wrong Image Type Found! Matrix is saved as all Zeros!')

    toc("Images Loaded.")
    return Images/255

#Generates Maps, Concatenates them to a Core [Single Array]
def GenerateCore(model, Images):
    # from keras import backend as K

    tic()
    num_samples = len(Images)

    layers_extract = CONST.LAYER_NUMS
    all_hc = np.zeros((num_samples, CONST.NUM_PIXELS, CONST.FEATURES))

    layers = [model.layers[li].output for li in layers_extract]
    get_feature = theano.function([model.layers[0].input], layers,
                                  allow_input_downcast=False)


    def extract_hypercolumn(instance):
        # fc6_output = get_fc6([instance])
        # fc7_output = get_fc7([instance, 0])
        feature_maps = get_feature(instance)
        hypercolumns = np.zeros((CONST.NUM_PIXELS, CONST.FEATURES))
        # fc6_map= np.reshape(fc6_output, (64,64))
        # fc7_map= np.reshape(fc7_output, (64,64))
        # hypercolumns[:, 1473] = np.reshape(sp.misc.imresize(fc6_map, size=(224, 224), mode="F", interp='bilinear'), (50176))
        # hypercolumns[:, 1474] = np.reshape(sp.misc.imresize(fc7_map, size=(224, 224), mode="F", interp='bilinear'), (50176))

        original= instance[:,0, :, :] + .407
        hypercolumns[:, 0] = np.reshape(original, (CONST.NUM_PIXELS))

        ctr = 1
        for convmap in feature_maps:
            for fmap in convmap[0]:
                upscaled = sp.misc.imresize(fmap, size=(CONST.IMAGE_DIM),
                                            mode="F", interp='bilinear')

                hypercolumns[:, ctr] = np.reshape(upscaled, (CONST.NUM_PIXELS))
                ctr += 1
        return np.asarray(hypercolumns)

    print("Starting Loop")
    counter = 0
    for i in range(len(Images)):

        Y_Channel = Images[i, :, :]

        R = Y_Channel - .407
        G = Y_Channel - .458
        B = Y_Channel - .485
        Y_Image = np.stack((R, G, B), axis=0)
        Y_Image = np.expand_dims(Y_Image, axis=0).astype(np.float32)
        hc = extract_hypercolumn(Y_Image)
        hc = np.expand_dims(hc, axis=0)

        all_hc[counter] = hc

        counter += 1
        if not counter % np.ceil(num_samples/10):
            print(counter)
        if not counter % num_samples:
            break

    toc("Hypercolumns Extracted.")
    return all_hc

def LoadPretrainedModel(filename, weights):
    tic()
    if weights == 0:  # 1: load from weights file, 0: load from pickle file
        model = pickle.load(open('kerasmodel', 'rb'))
    else:
        model = VGG_16(filename)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

    toc("Model Loaded. Compiled.")
    # pickle.dump(model,open ('kerasmodel','wb'))
    return model

#Saves to a .h5 file using groups 'X' and 'Y'
def SaveData(X, Y, output_filename):
    tic()
    f = h5py.File(output_filename, 'w')
    grp = f.create_group("my_data")
    grp.create_dataset('X', data=X, compression="gzip")
    grp.create_dataset('Y', data=Y, compression="gzip")
    f.close()
    toc('Data File saved to disk.')
    return

#Creates Targets from Labels, For each pixel
def CreateTargets(Labels):
    num_samples = len(Labels)
    targets = np.reshape(Labels, (num_samples, CONST.NUM_PIXELS))
    return targets

def find_minmax(x):
    min_x = np.min(x, axis=0)
    max_x = np.max(x, axis=0)
    min_x[0]=0
    max_x[0]=1
    return min_x, max_x

# Using the model, image_names etc. this method calls other methods
# and saves 20 images worth of data at a time.
# Also, manipulates (reshape/resizes) the data to fit input to each method
def CoreSample(model, image_names, save, output_filename, test_flag, minmax_filename):
    if model==[]:
            model = LoadPretrainedModel(CONST.PRETRAINED_MODEL, weights=1)
    if test_flag==0:
        images,labels= LoadImages(image_names)
    else:
        images =LoadImages_Test(image_names,output_filename) #output_filename : path of images during testing
    ctr=0 #Start of Image Number

    #This loop is here to limit saving a maximum of 20 images on a file. Bigger than that has memory issues.
    for i in range(int(np.ceil((images.shape[0])/CONST.MAXIMAGES_PER_FILE))):

        last=min(len(images),ctr+CONST.MAXIMAGES_PER_FILE)
        maps = GenerateCore(model, images[ctr:last, :, :])
        maps = maps.reshape(len(maps) * CONST.NUM_PIXELS, CONST.FEATURES)

        if test_flag==0:
            tic()
            targets = CreateTargets(labels[ctr:last,:,:])
            targets = targets.reshape(len(maps))
            UpdateNorm(find_minmax(maps), minmax_filename)
            toc('Targets Created. MinMax updated .')
        ctr=ctr+CONST.MAXIMAGES_PER_FILE

        #Saving normal images before training (or even testing if it has labels
        if save == 1:
            SaveData(maps.astype(np.float32), targets.astype(np.float32), output_filename + '_' + str(ctr) + '.h5')
        #Saving test images without labels
        elif save==2:
            print('Implementation Removed from this file!')
        else:
            print('Skipped saving.')
            return maps,[]

    return

#Maintains a pickle file with normalization parameters and updates when necessary as new data arrives
def UpdateNorm(norm, fname):

    if os.path.isfile(fname):
        file = open(fname, "rb")
        global_minmax= pickle.load(file)
        file.close()
    else:
        global_minmax = [norm[0], norm[1]]
    global_min = global_minmax[0]
    global_max = global_minmax[1]
    for k in range(len(global_min)):
        if norm[0][k]<global_min[k]:
            global_min[k] = norm[0][k]
        if norm[1][k]>global_max[k]:
            global_max[k] = norm[1][k]
    pickle.dump(global_minmax, open(fname,'wb'))
    return

if __name__ == '__main__':


    # image_list = natural_sort(os.listdir(CONST.PATH+'/IMAGES/'))
    image_list = (os.listdir(CONST.PATH+'/IMAGES/'))

    import theano.sandbox.cuda
    theano.sandbox.cuda.use("gpu1")
    model = LoadPretrainedModel(CONST.PRETRAINED_MODEL, weights=1)


    save_val=1 #1-Saving with Labels 2- No labels (Implementation removed)
    start=0
    stop =1000
    images_num = len(image_list)
    ctr=start
    X=[]
    Y=[]
    while start < images_num:
        if images_num < stop:
            stop = images_num
        fname=CONST.PATH+'Data/Class_Data/'+str(ctr)

        if not os.path.isfile(fname) and ctr<images_num:
            CoreSample(model,
                   image_names =image_list[start:stop],
                   save=save_val,
                   output_filename=fname,
                   test_flag=0,
                   minmax_filename='minmax_char_class.pkl')
            print(fname+' used!')
            gc.collect()
        start=stop
        stop=start+1000
        ctr=ctr+1000
        print(str(ctr)+' files processed!')
