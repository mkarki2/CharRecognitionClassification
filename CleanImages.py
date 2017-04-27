from CSF_Predict import predictor,natural_sort,tic,toc
from parameters_file import initialize
import os
import gc

type=['Numeral','Char']
noise_type=['Resized', 'awgn','contrast','motion']
params=initialize()
params.test_flag = 1
params.load_from_file = 0
params.save=0

for t in type:
    if t=='Numeral':
        classes=10
        tvt=['Train','Test']
    else:
        classes = 50
        tvt=['Train','Test','Validation']
        tic()
    for nt in noise_type:
        if nt != 'Resized':
            continue
        for dt in tvt:
            params.ctr=1000
            for i in range(classes):
                path=t +'/'+nt+'/'+dt+'/'+str(i)+'/'
                params.output_folder  = '/data/MK/Data Bangla/Noisy/Clean_' +path
                params.test_data_folder = '/data/MK/Data Bangla/Noisy/Noisy_'+ path
                image_list = natural_sort(os.listdir(params.test_data_folder))
                images_num = len(image_list)

                start=0
                stop =100
                ctr=start

                while start < images_num:
                    if images_num < stop:
                        stop = images_num

                    if  ctr<images_num:
                        params.image_names =image_list[start:stop]
                        predictor(params)

                        gc.collect()
                    start=stop
                    stop=start+100
                    ctr=ctr+100
                    params.ctr=params.ctr+ctr
                    print(str(ctr)+' files processed!')

        toc('Done with '+nt+' : '+ t)