import argparse
import os
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from mpl_toolkits.axes_grid1 import ImageGrid

K.set_image_dim_ordering('tf')
print(K.image_data_format())

## required for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend
from sklearn.model_selection import train_test_split
from ops import *

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


## required for efficient GPU use


class CNN():
    def __init__(self, datapath, batch_size=32, epochs=100,
                 droprate=0.5, img_row=64, img_col=64,
                 num_features=32, num_class=2,
                 lr=0.01, c_dim=1, output_name="training",
                 checkpoint_dir='checkpoint', CV_num=1, model_type='simple', testpath='test',dense_unit = 256,
                 valid_size=0.2, psfpath="CNN", decay=0.0, c_ratio=[0.001, 0.002]):
        self.data = np.load(datapath)
        self.psf_pl = np.load(psfpath)
        self.test_data = np.load(testpath)
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_rows = img_row
        self.img_cols = img_col
        self.lr = lr
        self.num_class = num_class
        self.num_features = num_features
        self.output_name = output_name
        self.c_dim = c_dim
        self.CV_num = CV_num
        self.valid_size = valid_size
        self.decay = decay
        self.c_ratio = c_ratio
        self.model_type = model_type
        self.droprate = droprate
        self.dense_unit = dense_unit
        print('data shape: {}'.format(self.data.shape))

        self.run_model()

        sys.exit()

    def local_normal(self, data):
        new_imgs_list = []
        for imgs in data:
            local_min = np.min(imgs)
            new_imgs = (imgs - local_min) / np.max(imgs - local_min)
            new_imgs_list.append(new_imgs)
        return np.array(new_imgs_list).reshape(-1, 64, 64)

    def data_preprocess(self, data):

        ## inject planet for train_data
        injected_samples = np.zeros([len(data), 64, 64])
        planet_loc_maps = np.zeros([len(data)*2, 64, 64])
        for i in range(len(data)):
            new_img, loc_map = self.inject_planet(data[i].reshape(64, 64), self.psf_pl, c_ratio=self.c_ratio,no_blend=True)
            injected_samples[i] += new_img
            planet_loc_maps[i] += loc_map

        normalised_injected = self.local_normal(injected_samples)
        nor_data = self.local_normal(data)

        dataset = np.zeros([int(len(data) * 2), 64, 64])

        ## Here we normalised each images into [0,1]
        dataset[:len(data)] += normalised_injected
        dataset[len(data):] += nor_data

        label = np.zeros((len(dataset)))
        label[:len(data)] += 1
        print("label size =", label.shape)
        print("data size=", dataset.shape)
        print("number of positive examples", np.sum(label))

        return dataset.reshape(-1, 64, 64, 1), label,planet_loc_maps

    def inject_planet(self,data, psf_library, c_ratio=[0.01, 0.1], x_bound=[4, 61], y_bound=[4, 61], no_blend=False):
        """Inject planet into random location within a frame
        data: single image
        psf_library: collection of libarary (7x7)
        c_ratio: the contrast ratio between max(speckle) and max(psf)*, currently accepting a range
        x_bound: boundary of x position of the injected psf, must be within [0,64-7]
        y_bound: boundary of y position of the injected psf, must be within [0,64-7]
        no_blend: optional flag, used to control whether two psfs can blend into each other or not, default option allows blending.

        """

        image = data.copy()
        pl_num = np.random.randint(1, high=4)
        pos_label = np.zeros([64, 64])
        used_xy = np.array([])
        c_prior = np.linspace(c_ratio[0], c_ratio[1], 100)
        if x_bound[0] < 4 or x_bound[0] > 61:
            raise Exception("current method only injects whole psf")
        if y_bound[0] < 4 or y_bound[0] > 61:
            raise Exception("current method only injects whole psf")

        for num in range(pl_num):
            while True:
                np.random.shuffle(c_prior)
                psf_idx = np.random.randint(0, high=psf_library.shape[0])
                Nx = np.random.randint(x_bound[0], high=x_bound[1])
                Ny = np.random.randint(y_bound[0], high=y_bound[1])
                if len(used_xy) == 0:
                    pass
                else:
                    if no_blend:
                        if np.any(dist([Nx, Ny], used_xy) < 3):
                            pass
                    else:
                        if np.any(np.array([Nx, Ny]) == used_xy):
                            pass
                if dist([Nx, Ny], (32.5, 32.5)) < 4:
                    pass
                else:
                    planet_psf = psf_library[psf_idx]
                    brightness_f = c_prior[0] * np.max(image) / np.max(planet_psf)
                    image[Ny - 4:Ny + 3, Nx - 4:Nx + 3] += planet_psf * brightness_f
                    used_xy = np.append(used_xy, [Nx, Ny]).reshape(-1, 2)
                    pos_label[Ny - 4:Ny + 3, Nx - 4:Nx + 3] = 1
                    break
        return image, pos_label
    def model_fn(self, model_type):

        # prepare callbacks
        if not os.path.exists(os.path.join(self.checkpoint_dir, 'ckt')):
            os.makedirs(os.path.join(self.checkpoint_dir, 'ckt'))

        filter_pixel = 3

        # input image dimensions
        input_shape = (self.img_rows, self.img_cols, self.c_dim)

        # Start Neural Network
        self.model = Sequential()

        if model_type == 'vgg':
            # convolution 1st layer
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(BatchNormalization(axis=3))
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(BatchNormalization(axis=3))
            # model.add(LeakyReLU())
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))

            # convolution 2nd layer
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))
            #         #
            # convolution 3rd layer
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))
            # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))
            # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))
            # model.add(LeakyReLU())

            # Fully connected 1st layer
            self.model.add(Flatten())  # 7
            self.model.add(Dense(self.dense_unit, use_bias=False, activation='relu'))  # 13
            self.model.add(Dropout(self.droprate))  # 15

            # Fully connected final layer
            self.model.add(Dense(2))  # 8
            self.model.add(Activation('sigmoid'))  # 9
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr, decay=self.decay),
                               metrics=['accuracy'])

        elif model_type == 'res_net':
            # convolution 1st layer
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=(64, 64, 1), dim_ordering="tf", kernel_initializer='random_uniform'))  # 0
            self.model.add(BatchNormalization(axis=3))
            # model.add(LeakyReLU())
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

            # convolution 2nd layer
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  border_mode="same", dim_ordering="tf", kernel_initializer='random_uniform'))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  border_mode="same", dim_ordering="tf", kernel_initializer='random_uniform'))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
            #
            # Fully connected 1st layer
            self.model.add(Flatten())  # 7
            self.model.add(Dense(256, use_bias=False, kernel_initializer='random_uniform', activation='relu'))  # 13
            # self.model.add(LeakyReLU()) #14
            self.model.add(Dropout(self.droprate))  # 15

            # Fully connected final layer
            # self.model.add(Dense(128,use_bias=False, kernel_initializer='random_uniform', activation='relu'))  # 8
            self.model.add(Dense(2, use_bias=False, kernel_initializer='random_uniform', activation='softmax'))
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr, decay=self.decay),
                               metrics=['accuracy'])

        elif model_type == 'vgg_triple':
            # convolution 1st layer
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(BatchNormalization(axis=3))
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(BatchNormalization(axis=3))
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))

            # convolution 2nd layer
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))
            #         #
            # convolution 3rd layer
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))
            # model.add(LeakyReLU())

            # Fully connected 1st layer
            self.model.add(Flatten())  # 7
            self.model.add(Dense(self.dense_unit, use_bias=True, activation='relu'))  # 13
            self.model.add(Dropout(self.droprate))  # 15

            # Fully connected final layer
            self.model.add(Dense(2))  # 8
            self.model.add(Activation('sigmoid'))  # 9
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr, decay=self.decay),
                               metrics=['accuracy'])
        elif model_type == 'vgg_deep':
            # convolution 1st layer
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='linear',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(keras.layers.LeakyReLU(alpha=0.3))
            self.model.add(BatchNormalization(axis=3))
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='linear',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(keras.layers.LeakyReLU(alpha=0.3))
            self.model.add(BatchNormalization(axis=3))
            # model.add(LeakyReLU())
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))

            # convolution 2nd layer
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='linear',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(keras.layers.LeakyReLU(alpha=0.3))
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='linear',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(keras.layers.LeakyReLU(alpha=0.3))
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))
            #         #
            # convolution 3rd layer
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='linear',
                                  padding="same", dim_ordering="tf"))
            self.model.add(keras.layers.LeakyReLU(alpha=0.3))
            # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='linear',
                                  padding="same", dim_ordering="tf"))
            self.model.add(keras.layers.LeakyReLU(alpha=0.3))
            # 1
            self.model.add(BatchNormalization(axis=3))
            # self.model.add(keras.layers.GlobalAveragePooling2D(data_format=None))
            self.model.add(keras.layers.GlobalMaxPooling2D(data_format=None))

            # Fully connected 1st layer
            # Fully connected final layer
            self.model.add(Dense(2))  # 8
            self.model.add(Activation('sigmoid'))  # 9
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr, decay=self.decay),
                               metrics=['accuracy'])
        self.model.summary()
        # sys.exit()

    def return_heatmap(self, model, org_img, normalise=True):
        test_img = model.output[:, 1]
        if self.model_type == 'simple' or 'simple_alt':
            last_conv_layer = model.get_layer('conv2d_6')
        else:
            last_conv_layer = model.get_layer('conv2d_6')
        grads = K.gradients(test_img, last_conv_layer.output)[0]

        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        message = K.print_tensor(pooled_grads, message='pool_grad = ')
        iterate = K.function([model.input, K.learning_phase()],
                             [message, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([org_img.reshape(-1, 64, 64, 1), 0])
        for i in range(conv_layer_output_value.shape[2]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        if normalise:
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
        return heatmap

    def plot_heatmap(self, heatmap, diff, index, cv_num):

        fig = plt.figure(figsize=(16, 8))

        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         )

        # Add data to image grid
        im = grid[0].imshow(heatmap)
        im = grid[1].imshow(diff)

        plt.savefig(os.path.join(self.checkpoint_dir, 'heatmap_{}_{}'.format(index, cv_num)), bbox_inches='tight')
        np.savetxt(os.path.join(self.checkpoint_dir, 'heatmap_{}_{}.txt'.format(index, cv_num)), heatmap)



    def run_model(self):

        ##Early Stopping

        if not os.path.exists(os.path.join(self.checkpoint_dir, 'history')):
            os.makedirs(os.path.join(self.checkpoint_dir, 'history'))
        cvscore = []
        test_score = []

        train_data, train_label,_ = self.data_preprocess(self.data)
        test, test_label,loc_map = self.data_preprocess(self.test_data)
        test_label = keras.utils.to_categorical(test_label, num_classes=2, dtype='float32')

        index = list(range(len(train_data)))
        np.random.shuffle(index)
        train_data_shu, train_label_shu = train_data[index], train_label[index]


        # data, test, data_label,test_label = train_test_split(self.data, label, test_size=self.testsize, shuffle=True,random_state=42)
        print("test_size:{}".format(test.shape))
        print("test_label:{}".format(np.sum(test_label)))
        np.save(os.path.join(self.checkpoint_dir, 'test_data'), test)
        np.save(os.path.join(self.checkpoint_dir, 'test_loc_map'), loc_map)
        np.savetxt(os.path.join(self.checkpoint_dir, 'test_label.txt'), test_label)
        for i in range(self.CV_num):
            csv_logger = keras.callbacks.CSVLogger(
                os.path.join(self.checkpoint_dir, 'history/training_{}.log'.format(i)))
            self.callbacks = [
                ModelCheckpoint(os.path.join(self.checkpoint_dir, 'ckt/checkpt_{}.h5'.format(i)),
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min',
                                verbose=1), csv_logger
            ]

            X_train, X_valid, y_train, y_valid = train_test_split(train_data_shu, train_label_shu,
                                                                  test_size=self.valid_size,
                                                                  shuffle=True)
            y_train = keras.utils.to_categorical(y_train, num_classes=2, dtype='float32')
            y_valid = keras.utils.to_categorical(y_valid, num_classes=2, dtype='float32')

            self.model_fn(self.model_type)
            history = self.model.fit(X_train.reshape(-1, 64, 64, 1), y_train,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     verbose=1,
                                     validation_data=(X_valid.reshape(-1, 64, 64, 1), y_valid), shuffle=True,
                                     callbacks=self.callbacks)
            score = self.model.evaluate(X_valid.reshape(-1, 64, 64, 1), y_valid, verbose=0)
            cvscore.append(score[1])
            keras.backend.clear_session()
            model = load_model(os.path.join(self.checkpoint_dir, 'ckt/checkpt_{}.h5'.format(i)))
            pred = model.evaluate(test.reshape(-1, 64, 64, 1), test_label)
            test_score.append(pred[1])
            pred = model.predict(test.reshape(-1, 64, 64, 1))
            over_confidence = np.where(pred[:, 1] == 1.)[0]
            np.savetxt(os.path.join(self.checkpoint_dir, 'over_confidence_inst_{}.txt'.format(i)), over_confidence)
            pos_index = np.where(pred[:, 1] > 0.9)[0]
            # for k in range(5):
            #     heatmap = self.return_heatmap(model, test[pos_index[k]])
            #     diff = test[pos_index[k]] - test[pos_index[k] + int(len(test) / 2)]
            #     self.plot_heatmap(heatmap, diff.reshape(64, 64), k, i)

            keras.backend.clear_session()

        final_score = np.array([np.mean(cvscore), np.std(cvscore)])
        final_test_score = np.array([np.mean(test_score), np.std(test_score)])
        np.savetxt(os.path.join(self.checkpoint_dir, 'CV_result.txt'), final_score)
        np.savetxt(os.path.join(self.checkpoint_dir, 'CV_history.txt'), np.array(cvscore))
        np.savetxt(os.path.join(self.checkpoint_dir, 'test_history.txt'), np.array(test_score))
        np.savetxt(os.path.join(self.checkpoint_dir, 'test_result.txt'), final_test_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='datapath')
    parser.add_argument('--checkpt', type=str, default='checkpt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--droprate', type=float, default=0.35)
    parser.add_argument('--num_features', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--output_name', type=str, default='training')
    parser.add_argument('--cv_num', type=int, default=5)
    parser.add_argument('--dense_unit', type=int, default=512)
    parser.add_argument('--model_type', type=str, default='vgg')
    parser.add_argument('--testpath', type=str, default='test')
    parser.add_argument('--psfpath', type=str, default='psf')
    parser.add_argument('--c_ratio', nargs='+', type=float)

    args = parser.parse_args()
    DLmodel = CNN(datapath=args.datapath, batch_size=args.batch_size, epochs=args.epochs, droprate=args.droprate,
                  num_features=args.num_features, lr=args.lr, output_name=args.output_name,
                  checkpoint_dir=args.checkpt, CV_num=args.cv_num,
                  dense_unit=args.dense_unit, model_type=args.model_type, testpath=args.testpath,
                  psfpath=args.psfpath, decay=args.decay, c_ratio=args.c_ratio)
