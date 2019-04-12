# Direct_Imaging_with_CNN
A public repository to store codes used in the paper "Pushing the Limits of Exoplanet Discovery via Direct Imaging with Deep Learning"

## Data
Data can be downloaded at: https://osf.io/ez7pk/


## Paper
The link to the paper will be available soon. 

## Running the GAN
To train the GAN, please type the following command in the terminal:

	$python train-dcgan.py --dataset LINK/TO/DATAFILES --epoch 35 --batch_size 50 —loss_ftn dcgan

Once you finished training the GAN, you can inspect its quality by asking it to generate images unseen by the network. 

	$python complete.py --imgSize 64 --dataset LINK/TO/UNSEENDATA --batch_size 1 --nIter 2000 --train_size 75

batch size specifies the number of input images you pass through to the GAN at each time
nIter specifies the number of restoration iterations GAN must go through before outputing the final image.
train_size specifies the number of input images
