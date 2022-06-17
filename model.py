import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

class MODEL:
    def __init__(self):
        self.paths = glob.glob('.\\lable\\*\\*.jpg')
        self.paths = np.random.permutation(self.paths)
        self.independent_variables = np.array([plt.imread(self.paths[i]) for i in range(len(self.paths))])
        self.dependent_variables = np.array([self.paths[i].split('\\')[2] for i in range(len(self.paths))])
        self.imgSize = (200, 200)
        self.shape = (self.independent_variables.shape[0],self.independent_variables.shape[1],self.independent_variables.shape[2], 1)
    
    def create_model(self):
        print(self.independent_variables.shape, self.dependent_variables.shape)
        self.independent_variables = self.independent_variables.reshape(self.shape)
        self.dependent_variables = pd.get_dummies(self.dependent_variables)
        print(self.independent_variables.shape, self.dependent_variables.shape)
        

        X = tf.keras.layers.Input(self.independent_variables.shape[1:])

        H1 = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='swish')(X)
        H1 = tf.keras.layers.MaxPool2D()(H1)
        
        H1 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='swish')(H1)
        H1 = tf.keras.layers.MaxPool2D()(H1)

        H1 = tf.keras.layers.Flatten()(H1)
        H1 = tf.keras.layers.Dense(64, activation='swish')(H1)
        H1 = tf.keras.layers.Dense(32, activation='swish')(H1)

        H2 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same', activation='swish')(X)
        H2 = tf.keras.layers.MaxPool2D()(H2)

        H2 = tf.keras.layers.Conv2D(32, kernel_size=5, activation='swish')(H2)
        H2 = tf.keras.layers.MaxPool2D()(H2)

        H2 = tf.keras.layers.Flatten()(H2)
        H2 = tf.keras.layers.Dense(64, activation='swish')(H2)
        H2 = tf.keras.layers.Dense(32, activation='swish')(H2)

        H3 = tf.keras.layers.Conv2D(16, kernel_size=7, padding='same', activation='swish')(X)
        H3 = tf.keras.layers.MaxPool2D()(H3)

        H3 = tf.keras.layers.Conv2D(32, kernel_size=7, activation='swish')(H3)
        H3 = tf.keras.layers.MaxPool2D()(H3)

        H3 = tf.keras.layers.Flatten()(H3)
        H3 = tf.keras.layers.Dense(64, activation='swish')(H3)
        H3 = tf.keras.layers.Dense(32, activation='swish')(H3)

        Y1 = tf.keras.layers.Dense(self.dependent_variables.shape[1], activation='sigmoid')(H1)
        Y2 = tf.keras.layers.Dense(self.dependent_variables.shape[1], activation='sigmoid')(H2)
        Y3 = tf.keras.layers.Dense(self.dependent_variables.shape[1], activation='sigmoid')(H3)
        
        model = [tf.keras.models.Model(X, Y1), tf.keras.models.Model(X, Y2), tf.keras.models.Model(X, Y3)]
        for m in model:
            m.compile(loss='categorical_crossentropy', metrics='accuracy')
        print('model created')
        return model


    def train_model(self):
        model = self.create_model()
        cnt = 0
        for m in model:
            m.fit(self.independent_variables, self.dependent_variables, epochs=10)
            m.save(f'.\\model\\model_{cnt}.h5')
            cnt += 1
        print('model saved')
    

    def load_model(self):
        model = glob.glob('.\\model\\*.h5')
        print(model)
        model = [tf.keras.models.load_model(m) for m in model]
        print('model loaded')
        return model


    def test_model(self):
        model = self.load_model()
        test_lable = glob.glob('.\\test_lable\\*\\*.jpg')

        column = glob.glob('.\\test_lable\\*')
        column = [column[i].split('\\')[-1] for i in range(len(column))]


        independent = np.array([plt.imread(test_lable[i]) for i in range(len(test_lable))])
        dependent = np.array([test_lable[i].split('\\')[2] for i in range(len(test_lable))])
        pred = tf.add_n([model[0].predict(independent), model[1].predict(independent), model[2].predict(independent)])
        df = pd.DataFrame(pred, columns=column)
        print(dependent)
        print(df)
    
        

obj = MODEL()
while True:
    querry = input('Enter your querry: ')
    if querry == 'create':
        obj.create_model()
    if querry == 'train':
        obj.train_model()
    if querry == 'test':
        obj.test_model()
    if querry == 'load':
        obj.load_model()
    if querry == 'exit':
        break
print('process exterminated')