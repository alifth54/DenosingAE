"""
Title: Denoising Auto Encoder
Author: Ali Fathi
alifathi8008@gmail.com
"""

import keras as ks
import keras.layers as ksl 
from keras.layers.convolutional import Conv2D as cv
from keras.layers.convolutional import Conv2DTranspose as cvt
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard

(x,_),(_,_) = mnist.load_data()
xnoisy =x+0.5* np.random.normal(size=x.shape)
xnoisy = np.clip(xnoisy, 0., 1.)

layer0 = ksl.Input(shape=[28,28,1])
layer1 = cv(32,kernel_size=3,strides=2,padding='same',
            activation='relu',input_shape=[28,28,1])(layer0)
layer2 = cv(32,kernel_size=3,strides=2,padding='same',
            activation='relu')(layer1)
layer3 = cvt(32,kernel_size=3,strides=2,padding='same',
             activation='relu')(layer2)
layer4 = cvt(1,kernel_size=3,strides=2,padding='same',
             activation='sigmoid')(layer3)

model = ks.models.Model(layer0,layer4)
model.compile(loss='binary_crossentropy',optimizer=Adam(),
              metrics=['accuracy'])

clbk = [TensorBoard('./logs')]
model.fit(xnoisy,x,epochs=10,batch_size=128,callbacks=clbk)

model.save('DenosingAE.h5')

a = np.expand_dims(xnoisy,axis=3)
r=model.predict(a[1])

plt.imshow(r[:,:,3,0],cmap='gray')
plt.show()
plt.imshow(xnoisy[1],cmap='gray')