import glob
import random
from pathlib import Path
import numpy as np
import cv2
from keras.utils import Sequence
from parse_data import Parse_helper
class TrainImageGenerator(Sequence):
    def __init__(self, file_dirs, batch_size=8, label_size = 6):
        self.image_paths = list()
        self.group_paths = file_dirs
        for dir in file_dirs:
            self.image_paths.append(list(Path(dir+'image/').glob("*.bmp")))
            
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.label_size = label_size
        #self.image_size = image_size

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self,idx):
        batch_size = self.batch_size
        label_size = self.label_size
        x = np.zeros((batch_size, 240, 320, 3), dtype=np.uint8)
        y1 = np.zeros((batch_size,1), dtype=np.float64)
        y2 = np.zeros((batch_size,1), dtype=np.float64)
        y3 = np.zeros((batch_size,1), dtype=np.float64)
        y4 = np.zeros((batch_size,1), dtype=np.float64)

        sample_id = 0

        while True:
            
            id_idx = random.randint(0,len(self.image_paths)-1)
            image_path = random.choice(self.image_paths[id_idx])
            # print ('image_path:',image_path)
            # print ('self.group_paths[id_idx]:',self.group_paths[id_idx])
            parse = Parse_helper(self.group_paths[0],image_path)

            pair_data = parse.read_pair()
            image = pair_data['image']
            x[sample_id] = image

            y1[sample_id]=pair_data['pose']['r']
            y2[sample_id]=pair_data['pose']['theta']
            y3[sample_id]=pair_data['pose']['phi']
            y4[sample_id]=pair_data['pose']['yaw']
           
            h, w, _ = image.shape
           
            sample_id = sample_id + 1
            print ('sample_id',sample_id)
            if sample_id == batch_size:
                return x, {'r': y1, 'theta': y2, 'phi': y3, 'yaw': y4} 
            '''
            clip the image
            '''
            # if h >= image_size and w >= image_size:
                # h, w, _ = image.shape
                # i = np.random.randint(h - image_size + 1)
                # j = np.random.randint(w - image_size + 1)
                # clean_patch = image[i:i + image_size, j:j + image_size]
                # x[sample_id] = ''
                # y[sample_id] = ''

                # sample_id += 1

                # if sample_id == batch_size:
                    # return x, y
                    



class ValGenerator(Sequence):
    def __init__(self, val_dir):
        self.image_paths = list(Path(val_dir+'image/').glob("*.*"))
        self.image_num = len(self.image_paths)
        self.val_dir = val_dir
        self.data = []
        for image_path in self.image_paths:
            parse = Parse_helper(val_dir,image_path)
            pair_data = parse.read_pair()
            x = pair_data['image']

            y = [[],[],[],[]]
            y[0]=np.expand_dims(pair_data['pose']['r'],axis=0)
            y[1]=np.expand_dims(pair_data['pose']['theta'],axis=0)
            y[2]=np.expand_dims(pair_data['pose']['phi'],axis=0)
            y[3]=np.expand_dims(pair_data['pose']['yaw'],axis=0)
              
            #y_label = {'r': y[0], 'theta': y[1], 'phi': y[2], 'yaw': y[3]}
            self.data.append([x,y])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
       # print (self.data[idx])
        batch_size = self.__len__()
        x = np.zeros((batch_size, 240, 320, 3), dtype=np.uint8)
        y1 = np.zeros((batch_size,1), dtype=np.float64)
        y2 = np.zeros((batch_size,1), dtype=np.float64)
        y3 = np.zeros((batch_size,1), dtype=np.float64)
        y4 = np.zeros((batch_size,1), dtype=np.float64)
        sample_id = 0
        for data in self.data: 
            x[sample_id] = data[0]

            y1[sample_id]= data[1][0]
            y2[sample_id]= data[1][1]
            y3[sample_id]= data[1][2]
            y4[sample_id]= data[1][3]
            sample_id = sample_id + 1
        return x,{'r': y1, 'theta': y2, 'phi': y3, 'yaw': y4}