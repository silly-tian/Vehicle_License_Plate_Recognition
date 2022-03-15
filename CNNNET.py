import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau


data_path = r'C:\Users\15098\Desktop\cnn_char_train'
character_folders = os.listdir(data_path)
label = 0
LABEL_temp = {}
if(os.path.exists('./train_data.list')):
    os.remove('./train_data.list')
if(os.path.exists('./test_data.list')):
    os.remove('./test_data.list')
for character_folder in character_folders:
    with open('./train_data.list', 'a') as f_train:
        with open('./test_data.list', 'a') as f_test:
            if character_folder == '.DS_Store' or character_folder == '.ipynb_checkpoints' or character_folder == 'data23617':
                continue
            #print(character_folder + " " + str(label))
            LABEL_temp[str(label)] = character_folder     #存储一下标签的对应关系
            character_imgs = os.listdir(os.path.join(data_path, character_folder))
            for i in range(len(character_imgs)):
                if i%10 == 0:
                    f_test.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
                else:
                    f_train.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
    label = label + 1
print('图像列表已生成')