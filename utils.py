import os
import glob

import cv2
import cvlib as cv
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator

class DataHandle:
    def __init__(self):
        self.db = []
        self.img = None
        self.X_postion = (0, 0)
        self.Y_postion = (0, 0)
        
        self.emotion = {
            'angry': 0, 
            'embarrassed': 1, 
            'happy': 2,
            'neutral': 3, 
            'sad': 4
            }
        self.new_file_path = ''
    
    def _save_crop_img(self):
        try:
            img = self.img.copy()
            roi = img[
                self.Y_position[0]:self.Y_position[1],
                self.X_position[0]:self.X_position[1],
                ]
            img = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            self.img = img
            return True
        except:
            return False
    
    def _detect_face(self, img_path):
        try:
            self.img = cv2.imread(img_path)
            faces, _ = cv.detect_face(self.img, enable_gpu=False)
            self.X_position = faces[0][0], faces[0][2]
            self.Y_position = faces[0][1], faces[0][3]
            return True
        except:
            os.remove(img_path)
            return False
        
    def _random_name(self):
        return ''.join(list(map(chr, np.random.randint(low=97, high=122, size=15))))

    def data_augmentation(self, img):
        def random_noise(x):
            x = x + np.random.normal(size=x.shape) * np.random.uniform(1, 5)
            x = x - x.min()
            x = x / x.max()
            
            return x * 255.0
        
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.07,
            height_shift_range=0.07,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            preprocessing_function=random_noise
        )
        
        i = 0
        for _ in datagen.flow(img, batch_size=16, save_to_dir='./augmentation', save_prefix=os.path.basename(self.new_file_path).split('.')[0], save_format='jpg'):
            i += 1
            if i > 5:
                break
            
        augmentation_img_list = glob.glob(os.path.join('./augmentation', '*.jpg'))
        for item in augmentation_img_list:
            item_name = os.path.basename(item)
            new_item_path = f'./dataset/{item_name}'
            self.db.append({
                    'path': new_item_path,
                    'label': item_name.split('_')[0]
                })
            os.rename(item, new_item_path)
            
    
    def work(self, img_path):
        (_, emotion, _) = img_path.split('_')[1].split('/')
        if self._detect_face(img_path) and self._save_crop_img():
            self.new_file_path = f'./dataset/{self.emotion[emotion]}_{self._random_name()}.jpg'
            self.db.append({
                'path': self.new_file_path,
                'label': self.emotion[emotion]
            })
            cv2.imwrite(self.new_file_path, self.img)

if __name__ == '__main__':
    def main():
        dbHandle = DataHandle()
        folder_list = glob.glob(os.path.join('./pre_dataset', '*/'))
        for folder in folder_list:
            img_list = glob.glob(os.path.join(folder, '*.jpg'))
            for img_path in img_list:
                dbHandle.work(img_path)
                x = dbHandle.img
                x = x.reshape(1, 96, 96, 1)
                dbHandle.data_augmentation(x)
        pd.DataFrame(dbHandle.db).to_csv('dataset.csv', index=False)
    main()