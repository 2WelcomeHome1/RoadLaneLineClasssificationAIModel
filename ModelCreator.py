import math
import numpy as np
from keras import initializers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense,  Dropout, BatchNormalization, GlobalAveragePooling2D


class SequentialModel(Sequential):
    def __init__(self, num_classes, shape):
        super().__init__()
        self.initializer = initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512), seed=42)
        self.num_classes = num_classes
        self.shape = shape
        pass
    
    def model(self):
        self.add(Conv2D(filters=32, kernel_size=5, activation='swish', input_shape=self.shape))
        self.add(Conv2D(filters=32, kernel_size=5, activation='swish'))

        self.add(MaxPool2D(pool_size=(2, 2)))
        self.add(BatchNormalization (momentum=0.9, epsilon=1e-5))
        self.add(Dropout(rate=0.05))

        self.add(Conv2D(filters=64, kernel_size=5, activation='swish'))
        self.add(Conv2D(filters=64, kernel_size=5, activation='swish'))

        self.add(MaxPool2D(pool_size=(2, 2)))
        self.add(BatchNormalization (momentum=0.9, epsilon=1e-5))
        self.add(Dropout(rate=0.1))

        self.add(Conv2D(filters=128, kernel_size=3, activation='swish'))
        self.add(Conv2D(filters=128, kernel_size=3, activation='swish'))

        self.add(MaxPool2D(pool_size=(2, 2)))
        self.add(BatchNormalization (momentum=0.9, epsilon=1e-5))
        self.add(Dropout(rate=0.15))

        self.add(Conv2D(filters=256, kernel_size=3, activation='swish'))
        self.add(Conv2D(filters=256, kernel_size=3, activation='swish'))

        self.add(MaxPool2D(pool_size=(2, 2)))
        self.add(BatchNormalization (momentum=0.9, epsilon=1e-5))
        self.add(Dropout(rate=0.15))

        self.add(Conv2D(filters=512, kernel_size=1,activation='swish'))

        self.add(GlobalAveragePooling2D()) 
        self.add(Dense(1024, activation='swish'))
        self.add(Dense(self.num_classes,kernel_initializer=self.initializer, bias_initializer=self.initializer, activation='softmax')) 

        self.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics=['accuracy'])

        return self
