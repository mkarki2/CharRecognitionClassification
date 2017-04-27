IMAGE_DIM=[32,32]#[360,480]#
NUM_PIXELS = IMAGE_DIM[0] *IMAGE_DIM[1]
PATH ='/data/MK/Data Bangla/Sample/Sample_Train/'
PRETRAINED_MODEL='/home/exx/vgg16_weights.h5' # Pretrained Model Weights File
LAYER_NUMS=[3, 8, 15, 22, 29] # Layers to extract Maps from
FEATURES =1473 # Experimentally calculated the total features that comes out for the above layers
MAXIMAGES_PER_FILE = 1000