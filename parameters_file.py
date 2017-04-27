import time
import constants_file as CONST
def initialize():

    class pretrain(object):
        def __init__(self):
            self.lr     = 0.01
            self.epochs = 3
            self.k      = 1

    class finetune(object):
        def __init__(self):
            self.lr     = .7
            self.epochs = 10000
            self.model_name = 'class_chars_03'
            self.retrain=False
            self.model_test = 'class_chars_03gpu1'

    class Options(object):
        def __init__(self):
            self.num_samples =   500    #12
            self.num_train   =   960    #8
            self.num_val     =   240 #2
            self.save        =   1     #1= yes, 0 - no
            self.train_data_folder = '/data/MK/Data Bangla/Sample/Sample_Train/Data/Class_Data/Train/'#'/data/NIR_batches/'
            self.val_data_folder    ='/data/MK/Data Bangla/Sample/Sample_Train/Data/Class_Data/Val/'

            self.pretrain_var=pretrain()
            self.finetune_var=finetune()

            # self.num_test_samples = 5
            # self.test_output_filename    ='NIR_testnorm.h5'
            self.test_data_folder          = '/data/MK/Data Bangla/Sample/Sample_Train/Data/Class_Data/Test/'

            self.reconstruct_image = 1 #1= yes, 0 - no
            self.output_folder     = '/data/MK/Data Bangla/Sample/Sample_Train/Data/Class_Data/Output/'

            self.L1_reg=0.000
            self.L2_reg=0.0001

            self.batch_size=CONST.NUM_PIXELS*self.num_val
            self.layer_sizes=[100,40,20]
            self.num_output=2
            self.classification = True
            self.dropout = 0
            self.minmax_file ='minmax_char_class.pkl'
            self.test_flag=-1
            self.load_from_file=1
            self.image_names=[] # For prediction
            self.ctr=1000
    params=Options()
    return params


def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(display=""):

    if 'startTime_for_tictoc' in globals():
        print (display +" Elapsed time is "+ str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

import re
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)
