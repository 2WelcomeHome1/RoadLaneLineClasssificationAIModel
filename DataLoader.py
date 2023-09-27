import cv2
import os
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.compat.v1.config.experimental.list_physical_devices("GPU") else "OFF" }')

class LoadData:
    def __init__(self, path) -> None:
        self.le = LabelEncoder()
        self.path = path
        self.num_classes = len(os.listdir(self.path))
        self.input_size = (256,256)
        pass
    
    def get_inputsize(self):
        return self.input_size

    def rectify_image(self, imagepath, input_size):
        image = cv2.cvtColor(cv2.imread(imagepath, 1) ,cv2.COLOR_BGR2RGB)
        return cv2.resize(image, input_size)
        
    def load_data(self):
        data, labels = [], []
        for file in os.listdir(self.path):
            for file_2 in os.listdir(str(self.path+file)):
                image = self.rectify_image(str(str(self.path+file) + '/' + file_2), self.input_size)
                data.append(image)
                labels.append(file)
        return data, labels
    
    def split_data(self, data, labels, test_size):
        labels = self.le.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 42, 
                                                            test_size = test_size, stratify = labels)
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        return X_train, X_test, y_train, y_test, self.num_classes

