import argparse
import os
import sys
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from sklearn.model_selection import train_test_split


K.set_image_dim_ordering('tf')
print(K.image_data_format())

## required for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


class CNN():
    def __init__(self, datapath,psf_path,test_path, batch_size=32, epochs=100,
                 droprate=0.5, img_row=64, img_col=64,
                 num_features=32, num_class=2,
                 lr=0.0001, c_dim=1,
                 checkpoint_dir='checkpoint', CV_num=1, dense_unit=128, model_type='simple',
                 valid_size=0.2, SNR=10,decay = 0.0):
        self.data = np.load(datapath)
        self.psf_pl = np.load(psf_path)
        self.test_data = np.load(test_path)
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.droprate = droprate
        self.img_rows = img_row
        self.img_cols = img_col
        self.lr = lr
        self.num_class = num_class
        self.num_features = num_features
        self.c_dim = c_dim
        self.CV_num = CV_num
        self.dense_unit = dense_unit
        self.model_type = model_type
        self.valid_size = valid_size
        self.SNR = SNR
        self.decay = decay
        print('data shape: {}'.format(self.data.shape))

        self.run_model()

    def data_preprocess(self, data, SNR):
        """Data preprocess stage. Here each negative example is duplicated to produce a postive example by injecting a planet psf onto it.
        The injection method is a separate method and can be changed at any time.
        Labels for both training and test data is also created here. """
        ## inject planet for train_data
        injected_samples = np.zeros([len(data), self.img_rows, self.img_cols])

        for i in range(len(data)):
            new_img, Nx, Ny = self.SNR_injection(data[i].reshape(self.img_rows, self.img_cols), self.psf_pl, SNR=SNR)
            injected_samples[i] += new_img
        normalised_injected = self.local_normal(injected_samples)
        nor_data = self.local_normal(data)

        dataset = np.zeros([int(len(data) * 2), self.img_rows, self.img_cols])
        dataset[:len(data)] += normalised_injected
        dataset[len(data):] += nor_data

        label = np.zeros((len(dataset)))
        label[:len(data)] += 1
        print("label size =", label.shape)
        print("train data size=", dataset.shape)
        print("label sum=", np.sum(label))

        return dataset.reshape(-1, self.img_rows, self.img_cols, self.c_dim), label
    def SNR_injection(self, data, tinyPSF, SNR=20, verbose=False, num_pixel=4):
        """Planet injection method. """
        pl_PSF = tinyPSF
        pad_length = int(pl_PSF.shape[0] / 2)
        pad_data = np.pad(data, ((pad_length, pad_length), (pad_length, pad_length)), 'constant',
                          constant_values=(100000))
        width = int(num_pixel / 2)
        while True:

            Nx = np.random.randint(0, high=self.img_rows)
            Ny = np.random.randint(0, high=self.img_rows)
            aperture = pad_data[Ny + 19 - width:Ny + 19 + width, Nx + 19 - width:Nx + 19 + width]
            aperture = aperture[aperture < 100000]
            noise_std = np.std(aperture)
            FWHM_contri = np.sum(pl_PSF[19 - width:19 + width, 19 - width:19 + width])
            pl_brightness = (noise_std * SNR * len(aperture.flatten()) / FWHM_contri)

            if np.max(data) > np.max(pl_PSF * pl_brightness):
                break
            else:
                pass

        pad_data[Ny:Ny + pad_length * 2, Nx:Nx + pad_length * 2] += pl_PSF * pl_brightness
        if verbose:
            print("planet_PSF_signal=", np.sum(pl_PSF * pl_brightness))
            print("planet_PSF_FWHMsignal=",
                  np.sum(pl_PSF[19 - width:19 + width, 19 - width:19 + width] * pl_brightness))
            print("Peak planet signal=", np.max(pl_PSF[19 - width:19 + width, 19 - width:19 + width] * pl_brightness))
            print("Peak speckle signal=", np.max(data))
            print("noise std=", noise_std)
            plt.imshow(pad_data[pad_length:pad_length + self.img_rows, pad_length:pad_length + self.img_rows])
        return pad_data[pad_length:pad_length + self.img_rows, pad_length:pad_length + self.img_rows], Nx, Ny

    def local_normal(self, data):
        """Perform image by image normalisation. Maximum and Minimum value of each image is extracted and used to create an normalised image between [0,1]"""
        new_imgs_list = []
        for imgs in data:
            local_min = np.min(imgs)
            new_imgs = (imgs - local_min) / np.max(imgs - local_min)
            new_imgs_list.append(new_imgs)
        return np.array(new_imgs_list).reshape(-1, self.img_rows, self.img_cols)

    def return_heatmap(self, model, org_img,normalise = True):
        """CAM implementation here. An activation heatmap is produced for every test images. """
        test_img = model.output[:, 1]
        if self.model_type == 'simple':
            last_conv_layer = model.get_layer('conv2d_3')
        else:
            last_conv_layer = model.get_layer('conv2d_6')
        grads = K.gradients(test_img, last_conv_layer.output)[0]

        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        message = K.print_tensor(pooled_grads, message='pool_grad = ')
        iterate = K.function([model.input],
                             [message, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([org_img.reshape(-1, self.img_rows, self.img_cols, self.c_dim)])
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        if normalise:
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
        return heatmap

    def plot_heatmap(self, heatmap,diff, index, cv_num):
        """Plotting function to show the heatmap and planet location of the corresponding test image. """
        fig = plt.figure(figsize=(16, 8))

        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         )

        # Add data to image grid
        im = grid[0].imshow(heatmap)
        im = grid[1].imshow(diff)

        plt.savefig(os.path.join(self.checkpoint_dir, 'heatmap_{}_{}'.format(index, cv_num)),bbox_inches='tight')

    def model_fn(self, model_type):
        """Architetures for different models are presented here """
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
                       input_shape=input_shape, dim_ordering="tf"))  # 0
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
            # self.model.add(LeakyReLU()) #14
            self.model.add(Dropout(self.droprate))  # 15

            # Fully connected final layer
            self.model.add(Dense(2))  # 8
            self.model.add(Activation('sigmoid'))  # 9
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr,decay=self.decay),
                               metrics=['accuracy'])

        elif model_type == 'simple':
            # convolution 1st layer
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=input_shape, dim_ordering="tf", kernel_initializer='random_uniform'))  # 0
            self.model.add(BatchNormalization(axis=3))
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
            self.model.add(Dense(2, use_bias=False, kernel_initializer='random_uniform', activation='sigmoid'))
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr,decay=self.decay),
                               metrics=['accuracy'])

        self.model.summary()

    def run_model(self):

        """This is where the whole model is ran."""

        if not os.path.exists(os.path.join(self.checkpoint_dir, 'history')):
            os.makedirs(os.path.join(self.checkpoint_dir, 'history'))
        cvscore = []
        test_score = []
        ## Data Preprocess stage ##
        train_data, train_label = self.data_preprocess(self.data, SNR=self.SNR)
        test, test_label = self.data_preprocess(self.test_data, SNR=self.SNR)
        index = list(range(len(train_data)))
        np.random.shuffle(index)
        train_data_shu, train_label_shu = train_data[index],train_label[index]

        test_label = keras.utils.to_categorical(test_label, num_classes=2, dtype='float32')

        np.save(os.path.join(self.checkpoint_dir, 'test_data'), test)
        np.savetxt(os.path.join(self.checkpoint_dir, 'test_label.txt'), test_label)
        ## Cross validation ##
        for i in range(self.CV_num):
            ## prepare call_backs, csv_logger for progress monitoring. ##
            csv_logger = keras.callbacks.CSVLogger(
                os.path.join(self.checkpoint_dir, 'history/training_{}.log'.format(i)))
            self.callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    mode='min',
                    verbose=1),
                ModelCheckpoint(os.path.join(self.checkpoint_dir, 'ckt/checkpt_{}.h5'.format(i)),
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min',
                                verbose=1), csv_logger
            ]

            X_train, X_valid, y_train, y_valid = train_test_split(train_data_shu, train_label_shu, test_size=self.valid_size,
                                                                  shuffle=True)
            y_train = keras.utils.to_categorical(y_train, num_classes=2, dtype='float32')
            y_valid = keras.utils.to_categorical(y_valid, num_classes=2, dtype='float32')

            self.model_fn(self.model_type)
            ## Training Phase ##
            history = self.model.fit(X_train.reshape(-1, self.img_rows, self.img_cols, self.c_dim), y_train,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     verbose=0,
                                     validation_data=(X_valid.reshape(-1, self.img_rows, self.img_cols, self.c_dim), y_valid), shuffle=True,
                                     callbacks=self.callbacks)
            score = self.model.evaluate(X_valid.reshape(-1, self.img_rows, self.img_cols, self.c_dim), y_valid, verbose=0)
            cvscore.append(score[1])

            ## Test Phase ##
            keras.backend.clear_session()
            model = load_model(os.path.join(self.checkpoint_dir, 'ckt/checkpt_{}.h5'.format(i)))
            pred = model.evaluate(test.reshape(-1, self.img_rows, self.img_cols, self.c_dim), test_label)
            test_score.append(pred[1])
            pred = model.predict(test.reshape(-1,self.img_rows, self.img_cols, self.c_dim))
            pos_index = np.where(pred[:, 1] > 0.9)[0]
            for k in range(5):
                heatmap = self.return_heatmap(model, test[pos_index[k]])
                diff = test[pos_index[k]] - test[pos_index[k] + int(len(test) / 2)]
                self.plot_heatmap(heatmap,diff.reshape(self.img_rows,self.img_cols), k, i)

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
    parser.add_argument('--testpath', type=str, default='test')
    parser.add_argument('--psf_path', type=str, default='psf')
    parser.add_argument('--checkpt', type=str, default='checkpt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--droprate', type=float, default=0.35)
    parser.add_argument('--num_features', type=int, default=8)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--cv_num', type=int, default=3)
    parser.add_argument('--dense_unit', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='simple')
    parser.add_argument('--SNR', type=float, default=10)


    args = parser.parse_args()
    DLmodel = CNN(datapath=args.datapath, batch_size=args.batch_size, epochs=args.epochs, droprate=args.droprate,
                  num_features=args.num_features, num_class=args.num_class, lr=args.lr,
                   checkpoint_dir=args.checkpt, CV_num=args.cv_num,
                  dense_unit=args.dense_unit, model_type=args.model_type, test_path=args.testpath, SNR=args.SNR,
                  psf_path=args.psf_path,decay=args.decay)
