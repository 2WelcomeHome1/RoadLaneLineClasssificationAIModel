from DataLoader import *
from ModelCreator import *

class TrainModel:
    def __init__(self, path) -> None:
        self.Loader = LoadData(path)
        pass

    def train(self, model, X_train, X_test, y_train, y_test):
        history = model.fit(np.array(X_train), np.array(y_train), batch_size=5, 
                                epochs=20,validation_data=(np.array(X_test), np.array(y_test)))
        model.save_weights('./CNN.h5')
    
    def run(self):
        data, labels = self.Loader.load_data()
        X_train, X_test, y_train, y_test, num_classes = self.Loader.split_data(data, labels, 0.2)
        model = SequentialModel(num_classes, np.array(X_train).shape[1:]).model()
        model.summary()
        self.train (model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    TrainModel('./Data/').run()    
    

        