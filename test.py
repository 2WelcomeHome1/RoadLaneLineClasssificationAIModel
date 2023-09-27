import os
import numpy as np
from DataLoader import *
from ModelCreator import *
from sklearn.metrics import f1_score

class TestModel:
    def __init__(self, path, modelpath, num_classes, shape) -> None:
        self.path = path
        self.input_size = LoadData(self.path).get_inputsize()
        self.model = SequentialModel(num_classes, shape).model()
        self.model.load_weights(modelpath)
        pass
    
    def rectify_image(self, imagepath, input_size):
        return LoadData(self.path).rectify_image(imagepath, input_size)

    def run(self):
        all_classes, pred_classes = [], []
        for i in os.listdir(self.path):
            for z in os.listdir(self.path+str(i)):
                image =  np.array(self.rectify_image(self.path + str(i) + '/' + str(z), self.input_size))
                img_batch = np.expand_dims(image, 0) 
                test_pred = self.model.predict(img_batch)
                prediction = np.argmax(test_pred, axis=1)
                all_classes.append(int(i))
                pred_classes.append(prediction[0])
        print (f1_score(all_classes, pred_classes, average='weighted'))



if __name__ == "__main__":
    TestModel('./Data/', './CNN.h5/', num_classes=6, shape=(256,256,3)).run()    

    
       