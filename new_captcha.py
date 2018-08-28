import numpy as np
import random
from PIL import Image
import h5py

model_path = './imgs/c.h5'
model_weights_path = './imgs/cw.h5'

CHARS = '123456789'
width, height, n_len, n_class = 60, 20, 4, len(CHARS)
h5_file = './imgs/dataset.h5'
pos = 0

def gen(batch_size=64):
    X = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
    y = [np.zeros( (batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    f = h5py.File(h5_file, 'r')
    global pos
    while True:
        X = f['x'][pos:pos+batch_size,:]
        y = f['y'][pos:pos+batch_size,:]
        y = [y[:,n_class * i : n_class * i + n_class] for i in range(n_len)]
        # print(pos)
        pos += batch_size
        # print(pos)
        # for i in range(batch_size):
        #     random_str = ''.join([random.choice(CHARS) for j in range(n_len)])
        #     X[i] = gg.generate_image(random_str)
        #     for j, ch in enumerate(random_str):
        #         y[j][i, :] = 0
        #         y[j][i, CHARS.find(ch)] = 1
        yield X, y

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([CHARS[x] for x in y])

from keras.models import Input, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense

def get_model():
    input_tensor = Input( (height, width, 1) )
    x = input_tensor
    #for i in range(2):
    x = Convolution2D(32, (3, 3), activation='relu')(x)
    x = Convolution2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D( (2,2))(x)
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D( (2,2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )
    return model
    

def train_model(model):
    
    model.fit_generator(
        gen(), samples_per_epoch=3000, nb_epoch=2,
        nb_worker=2, pickle_safe=True,
        validation_data=gen(), nb_val_samples=500
    )

    model.save(model_path)
    model.save_weights(model_weights_path)
    
def test_model():
    model = get_model()
    model.load_weights(model_weights_path)
    img = Image.open('/home/imaxct/Documents/learn/imgs/test2.png')
    img = np.asarray(img, dtype=np.uint8)
    img = img.reshape( (-1, height, width, 1) )
    y_pre = model.predict(img)
    print(decode(y_pre))


if __name__ == '__main__':
    # train_model()
    # X, y = next(gen(1))
    # print(decode(y))
    test_model()
