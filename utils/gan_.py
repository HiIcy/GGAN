# coding:utf-8
import math
import random

import numpy as np
from PIL import Image
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Activation,InputLayer,Conv2DTranspose,Dense, BatchNormalization, Reshape, UpSampling2D, Conv2D, MaxPooling2D, Flatten
from keras import layers
import os
from keras.optimizers import Adam,SGD
from keras import utils

H=28
W=28
C=1
batch = 64

def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(Activation('tanh'))

    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    # model.add(Dense(128 * 7 * 7))
    # model.add(BatchNormalization())
    # model.add(Activation('tanh'))

    # Reshape层用来将输入shape转换为特定的shape，将含有128*7*7个元素的向量转化为7×7×128张量
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    # 2维上采样层，即将数据的行和列分别重复2次
    # model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(128,(4,4),(2,2),padding="same"))

    model.add(Conv2D(64, (5, 5), padding='same', activation='tanh'))
    model.add(Conv2DTranspose(64,(4,4),(2,2),padding="same"))

    # model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))

    return model


# 定义个判别模型
def discriminator_model():
    model = Sequential()

    model.add(Conv2D(64,(5,5),input_shape=(H,W,C)))
    model.add(Activation('tanh'))
    # 为空域信号施加最大值池化，pool_size取（2，2）代表使图片在两个维度上均变为原长的一半
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(5,5),activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # #Flatten层把多维输入一维化，常用在从卷积层到全连接层的过渡
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable=False
    model.add(d)
    return model


# 生成拼接的图片（即将一个batch所有生成图片放到一个图片中）：
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0],
                      width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0],j*shape[1]:(j+1)*shape[1]] = \
                                                 img[:,:,0]
    return image

class Dse(utils.Sequence):
    def __init__(self, images_root, shuffle=True,train=True):
        self.images_root = images_root
        self.pair = list(zip(*self.init()))
        random.shuffle(self.pair)
        self.shuffle = shuffle
        self.train=train

    def init(self):
        imgs = []
        labels = []
        dirs = os.listdir(self.images_root)
        dirs.sort()
        for i, dir in enumerate(dirs):
            cdir = os.path.join(self.images_root, dir)
            for img in os.listdir(cdir):
                imgs.append(os.path.join(dir, img))
                labels.append(i)
        return imgs, labels

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.pair)

    def __len__(self):
        return len(self.pair) // batch

    def __getitem__(self, item):
        pairs = self.pair[item * batch:(item + 1) * batch]
        images, labels = list(zip(*pairs))
        images = [os.path.join(self.images_root, im) for im in images]
        # images = [read_image(impath,self.train) for impath in images]

        return np.array(images, np.float), np.array(labels,dtype=np.uint8)

def load_data():
    path=r'/data/soft/javad/COCO/mnist.npz'
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


# 训练
def train(BATCH_SIZE):
    # 下载的地址为：https://s3.amazonaws.com/img-datasets/mnist.npz
    (X_train, y_train), (X_test, y_test) = load_data()
    print(type(X_train),X_train.shape,X_test.shape)
    print(type(y_train),y_train.shape)

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g,d)
    d_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)

    g.compile(loss='binary_crossentropy',optimizer='SGD')
    d_on_g.compile(loss='binary_crossentropy',optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    for epoch in range(30):
        print("epoch is",epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise,verbose=0)
            print(np.shape(generated_images))
            if index % 100 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "./GAN"+str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch,generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X,y)
            print("batch %d d_loss : %f" % (index, d_loss))

            # 随机生成的噪声服从均匀分布
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))

            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise,[1]*BATCH_SIZE)

            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))

            if index % 100 == 0:
                g.save_weights('generator',True)
                d.save_weights('discriminator',True)


def generate(BATCH_SIZE,nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images,verbose=1)
        index = np.arange(0,BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20,1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,)+generated_images.shape[1:3],dtype=np.float32)
        nice_images = nice_images[:,:,:,None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=0)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "./GAN/generated_image.png")
train(batch)

