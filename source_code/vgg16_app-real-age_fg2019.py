#------------------------------------------------------------------------
# This source code was used to generate the results (of the proposed model,
# case 3) of the paper "On the effect of age perception biases for real age
# regression", accepted in the 14th IEEE International Conference on Automatic
# Face and Gesture Recognition (FG 2019).
#
# When using it, please cite the following reference:
#
# @inproceedings{jacques:FG2019,
# author={Julio C. S. Jacques Junior and Cagri Ozcinar and Marina Marjanovic
#         and Xavier Baro and Gholamreza Anbarjafari and Sergio Escalera},
# booktitle={IEEE International Conference on Automatic Face and Gesture
#            Recognition (FG)},
# title={On the effect of age perception biases for real age regression},
# year={2019},
# }
#
# For intructions and additional details (e.g., about the adopted dataset,
# pre-processing procedures, input/output format, etc), please check:
# https://github.com/juliojj/app-real-age
#
# Author: Julio C. S. Jacques Junior | juliojj at gmail dot com
# Version: 1.0
# Date: 19 of Feb 2019.
#------------------------------------------------------------------------



#----------------------
# loading libraries
#--------------
import sys
import os.path
import numpy as np
import h5py
import pickle
# keras modules:
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, load_model, Model
from keras.layers import  Input, Activation, Flatten, Conv2D
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate
from keras.utils import plot_model
# to generate and save a plot of the loss over the epochs:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#-----------------



#----------------------------------------------------
# loading the age data (preprocessed data)
def load_data_set(data_dir, set):
    # loading samples (images: cropped_faces)
    with h5py.File(os.path.join(data_dir,set+'_samples.h5'), 'r') as hf:
        data_set = hf[set+'_samples'][:]
    # loading age labels (real)
    with h5py.File(os.path.join(data_dir,set+'_real_labels.h5'), 'r') as hf:
        data_real_labels = hf[set+'_real_labels'][:]
    # loading age labels (app)
    with h5py.File(os.path.join(data_dir,set+'_app_labels.h5'), 'r') as hf:
        data_app_labels = hf[set+'_app_labels'][:]

    return data_set, data_real_labels, data_app_labels



#----------------------------------------------------
# loading the extra features / attributes (preprocessed data)
def load_data_extra_set(data_dir, set):
    # i.e., gender, race, makeup and facial expression
    with h5py.File(os.path.join(data_dir,set+'_all_extra_labels.h5'), 'r') as hf:
        all_extra_labels = hf[set+'_all_extra_labels'][:]

    return all_extra_labels



#----------------------------------------------------
# evaluate the results (MAE)
def evaluate(gt, p, n_factor):
    # loading ground truth and predictions
    gt = gt.astype(np.float64)
    predictions = p.astype(np.float64)
    # un-normalizing ground truth and predictions
    gt = gt*float(n_factor)
    predictions = predictions*float(n_factor)

    error = []
    for i in range(0,len(gt)):
        error.append(abs(gt[i]-predictions[i][0]))
    mae = np.array(error).mean()
    print "Mean Absolute Error:", mae
    print "--"
    return mae



#----------------------------------------------------
# saving the train history for further analysis
def save_train_history(output_dir, model_filename, train_history):
    plt.plot(train_history.history['predict_app_mean_absolute_error'])
    plt.plot(train_history.history['predict_real_mean_absolute_error'])
    plt.plot(train_history.history['val_predict_app_mean_absolute_error'])
    plt.plot(train_history.history['val_predict_real_mean_absolute_error'])
    plt.title('model accuracy')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train_app_mae', 'train_real_mae','val_app_mae', 'val_real_mae'], loc='upper left')
    # saving the plot as an image:
    plt.savefig(os.path.join(output_dir,model_filename+'_acc.png'))
    # saving the train history for further analysis as a pkl file:
    pkl_file = os.path.join(output_dir,model_filename+'_train_history.pkl')
    with open(pkl_file, 'wb') as handle:
        pickle.dump(train_history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)




#----------------------------------------------------
# Defining and training the network with new features (attributes)
# (check Figures 2 and 3 of the paper, for a complete illustration of the model)
def train_extra_feat(train_set, train_app_labels, train_real_labels, valid_set, valid_app_labels, valid_real_labels, train_extra_labels, valid_extra_labels, output_dir, stage_training, new_filename_st1, new_filename_st2, learning_rate, batch_size, epochs):
    # input image shape (width, height, channels)
    input_1 = Input(shape=(224, 224, 3))
    # getting the size of extra features from the input file (pre-processed data)
    # (according to the paper, the size = "13")
    input_2 = Input(shape=(train_extra_labels.shape[1],))

    # 1rst stage training:
    if(stage_training=='1'):
        # loading the VGG16 model, pretrained on Imagenet
        vgg16_model = VGG16(input_tensor=input_1, weights='imagenet', include_top=True)

        # getting the output of the last conv layer
        last_convMaxPool = vgg16_model.get_layer('block5_pool').output
        # including on top of it a new conv layer to reduce dimensionality:
        new_conv_layer = Conv2D(512, (7, 7), activation='relu',name='block6_conv1')(last_convMaxPool)
        # flatten the output
        flatten_new_conv_layer = Flatten()(new_conv_layer)

        # creating a hidde layer to be included on top of input_2 (attributes)
        h_layer = Dense(10,name='hidden_layer')(input_2)

        # merging both inputs
        merged = concatenate([flatten_new_conv_layer, h_layer],name='concatenate_1')

        # including a FC on top of concatenated layer
        fc2 = Dense(256,name='fc2')(merged)
        # including a prediction layer on top of it (responsible to estimate APPARENT age)
        out1 = Dense(1,activation='sigmoid',name='predict_app')(fc2)

        # including a FC on top of "h_layer" (to reduce dimensionality)
        h_layer2 = Dense(5,name='hidden_layer_2')(h_layer)

        # concatenate the output of predicted (apparent) age with h_layer2 (representing the attributes)
        merged2 = concatenate([h_layer2, out1],name='concatenate_2')
        # including a FC on top of it
        out2 = Dense(6,name='fc3')(merged2)
        # including a prediction layer on top of it (responsible to estimate the REAL age)
        out2 = Dense(1,activation='sigmoid',name='predict_real')(out2)

        # building the model
        model = Model(inputs=[input_1,input_2],outputs=[out1, out2])
        print model.summary()

        # FREEZING the weights (all layers before "block5_pool" layer)
        count = 0
        for layer in model.layers:
            if(count<19):
                layer.trainable = False
                count +=1
            print layer, layer.trainable
        # saving an image of the new model
        plot_model(model, to_file='proposed_model.png')
        #
        #--------
        # creating the compiler (with same weights for both losses, and default values for the optimizer)
        model.compile(Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False),loss='mean_squared_error',metrics=['mae'], loss_weights=[1., 1.])
        # creating a callback to save the best model
        filepath = os.path.join(output_dir,'best_models',new_filename_st1+'.hdf5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_predict_real_mean_absolute_error', verbose=1, save_best_only=True, mode='min')
        # stop training if loss do not improve after N epocs (set N to 100)
        early_stopping_monitor = EarlyStopping(patience=100)
        callbacks_list = [checkpoint, early_stopping_monitor]
        #-----------------------
        # print train set size (for debug purpose)
        print "----"
        print "train set (images) shape", train_set.shape
        print "train set (attributes) shape", train_extra_labels.shape
        print "----"

        # training
        train_history = model.fit([train_set,train_extra_labels], [train_app_labels, train_real_labels], validation_data=([valid_set,valid_extra_labels],[valid_app_labels, valid_real_labels]), batch_size=batch_size, callbacks=callbacks_list, epochs=epochs, shuffle=True, verbose=2)
        # saving train history
        save_train_history(output_dir, new_filename_st1, train_history)


    elif(stage_training=='2'):
        print "Train STAGE 2. LOADING THE BEST MODEL TRAINED ON STAGE 1:"
        print new_filename_st1+'.hdf5'
        print "----"
        # loading the best model saved during stage 1
        model = load_model(os.path.join(output_dir,'best_models',new_filename_st1+'.hdf5'))
        # allowing fine-tunning the whole network
        for layer in model.layers:
            layer.trainable = True
            print layer, layer.trainable

        # creating the compiler (with same weights for both losses, and default values for the optimizer)
        model.compile(Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False),loss='mean_squared_error',metrics=['mae'], loss_weights=[1., 1.])
        #-----------------------
        # creating the callback to save the best model
        filepath = os.path.join(output_dir,'best_models',new_filename_st2+'.hdf5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_predict_real_mean_absolute_error', verbose=1, save_best_only=True, mode='min')
        # stop training if loss do not improve after N epocs (set N to 100)
        early_stopping_monitor = EarlyStopping(patience=100)
        callbacks_list = [checkpoint, early_stopping_monitor]
        #-----------------------
        # training
        train_history = model.fit([train_set,train_extra_labels], [train_app_labels, train_real_labels], validation_data=([valid_set,valid_extra_labels],[valid_app_labels, valid_real_labels]), batch_size=batch_size, callbacks=callbacks_list, epochs=epochs, shuffle=True, verbose=2)
        # saving train history
        save_train_history(output_dir, new_filename_st2, train_history)

    return model



#----------------------------------------------------
# making predictions
def predict(model, test_set, output_dir, model_filename):
    print "\n--------"
    print "Making predictions on the test set..."
    predictions = model.predict(test_set, batch_size=10, verbose=0)
    # saving the output in a file for further analysis
    np.save(os.path.join(output_dir,model_filename+'_predictions.npy'), predictions)
    print "\nPredictions saved (npy) at ", os.path.join(output_dir)
    print model_filename+'_predictions.npy'

    return predictions


#----------------------------------------------------
def main(data_h5, output_dir, train_model, stage_training, new_filename_st1, new_filename_st2, learning_rate, batch_size, epochs):

    # performing the training
    if(train_model==True):
        # loading preprocessed data
        train_set, train_real_labels, train_app_labels = load_data_set(data_h5,'train')
        valid_set, valid_real_labels, valid_app_labels = load_data_set(data_h5,'valid')

        # loading the extra features/attributes (gender, age, etc)
        train_all_extra_labels = load_data_extra_set(data_h5,'train')
        valid_all_extra_labels = load_data_extra_set(data_h5,'valid')

        #----------------------------
        print "proposed model (case 3)"
        #----------------------------
        # proposed model: 2 inputs, 2 outputs
        model = train_extra_feat(train_set, train_app_labels, train_real_labels, valid_set, valid_app_labels, valid_real_labels, train_all_extra_labels, valid_all_extra_labels, output_dir, stage_training, new_filename_st1, new_filename_st2 ,learning_rate, batch_size, epochs)
        #----------------------------

    #----------------------------------------------------
    # loading pretrained model (according to each stage of training 1|2)
    else:
        print "\n--------"
        print "Loading pre-trained model (and predicting ages)"
        print ">> Warning - model generated from stage:", stage_training
        print ">> Results shown in the paper are based on 2 stages training"
        print "--------\n"
        if(stage_training=='1'):
            model = load_model(os.path.join(output_dir,'best_models',new_filename_st1+'.hdf5'))
        else:
            model = load_model(os.path.join(output_dir,'best_models',new_filename_st2+'.hdf5'))
    #----------------------------------------------------



    #-----------------------------------------
    # making predictions on the test set (according to each stage of training 1|2)
    #
    # loadint the test set
    test_set, test_real_labels, test_app_labels = load_data_set(data_h5,'test')
    test_extra_labels = load_data_extra_set(data_h5,'test')
    # predict
    if(stage_training=='1'):
        predictions = predict(model, [test_set,test_extra_labels], output_dir, new_filename_st1)
    else:
        predictions = predict(model, [test_set,test_extra_labels], output_dir, new_filename_st2)



    #---------------------------------------------------------
    # Evaluating the results (Mean Absolute Error).
    #
    # loading predicted values (according to each stage of training 1|2):
    if(stage_training=='1'):
        predictions = np.load(os.path.join(output_dir,new_filename_st1+'_predictions.npy'))
    else:
        predictions = np.load(os.path.join(output_dir,new_filename_st2+'_predictions.npy'))

    # as the proposed model returns two outputs:
    app_predictions = predictions[0,:,:]
    real_predictions = predictions[1,:,:]
    print "------"
    # Age values are normalized [0,...,1] for training. Thus, they are un-normalized during evaluation using
    # a normalization factor = 100, adopted for data preprocessing. Such factor is based on the max age found on the dataset.
    # a2a
    print "Apparent age estimation error:"
    mae_app = evaluate(test_app_labels,app_predictions,100)
    # a2r
    print "Real age estimation error:"
    mae_real = evaluate(test_real_labels,real_predictions,100)
    #---------------------------------------------------------


def print_init_msg(data_dir, data_h5, output_dir, model_filename, train_model, stage_training, learning_rate, batch_size, epochs, learning_rate_stage1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir,'best_models')):
        os.makedirs(os.path.join(output_dir,'best_models'))
    print "-------"
    print "data_dir: ", data_dir
    print "data_h5: ", data_h5
    print "output_dir: ", output_dir
    print "train_model: ", train_model
    print "stage_training: ", stage_training
    if(stage_training=='2'):
        print "learning_rate (stage 1): ", learning_rate_stage1
    if(stage_training=='3'):
        print "learning_rate (stage 1 and 2): ", learning_rate_stage1
    print "learning_rate: ", learning_rate
    print "batch_size: ", batch_size
    print "epochs: ", epochs
    # generating the final filename (used to save train history, model, predictions)
    if(stage_training=='1'):
        new_filename_st1 = model_filename+'_stage_'+stage_training+'_st1-lr_'+str(learning_rate)
        new_filename_st2 = 'null'
        print "Final filename (st1) = ", new_filename_st1+'(.hdf5,.npy,.pkl,.png)'
    else:
        if(stage_training=='2'):
            new_filename_st2 = model_filename+'_stage_'+stage_training+'_st1-lr_'+str(learning_rate_stage1)+'_st2-lr_'+str(learning_rate)
            new_filename_st1 = model_filename+'_stage_'+str(1)+'_st1-lr_'+str(learning_rate_stage1)
            print "Final (current) filename (st2) = ", new_filename_st2+'(.hdf5,.npy,.pkl,.png)'
            print "Previous stage filename (st1) = ", new_filename_st1+'(.hdf5,.npy,.pkl,.png)'
        else:
            print "WARNING: lr of stage 1 and 2 must be the same in this case (to-do: optimize code here)"
            new_filename_st2 = model_filename+'_stage_'+stage_training+'_st1-2-lr_'+str(learning_rate_stage1)+'_st3-lr_'+str(learning_rate)
            new_filename_st1 = model_filename+'_stage_'+str(2)+'_st1-lr_'+str(learning_rate_stage1)+'_st2-lr_'+str(learning_rate_stage1)
            print "Final (current) filename (st3) = ", new_filename_st2+'(.hdf5,.npy,.pkl,.png)'
            print "Previous stage filename (st2) = ", new_filename_st1+'(.hdf5,.npy,.pkl,.png)'
    print "-------"

    return new_filename_st1, new_filename_st2


#----------------------------------------------------
if __name__== "__main__":
    #
    # e.g., run it as:
    # for stage 1: python vgg16_app-real-age_fg2019.py ../data/ True 1 1e-4 32 3000 1e-4
    # and then, for stage 2: python vgg16_app-real-age_fg2019.py ../data/ True 2 1e-4 32 1500 1e-4
    #
    # global variables:
    data_dir = sys.argv[1] # ex.: ../dataset/appa-real-release/
    # train the model (True) and predict or load pretrained model (False) and predict
    if(sys.argv[2]=='True'):
        train_model = True
    else:
        train_model = False

    stage_training = sys.argv[3] # 1=fine-tune last layers, 2=fine-tune all layers (stage 2 must be run after running option 1)
    learning_rate = float(sys.argv[4]) # 1e-3, 1e-4, 1e-5 (using adam)
    batch_size = int(sys.argv[5]) # default is 32
    epochs = int(sys.argv[6]) # ex.: 300, 500, 1000

    # if fine-tunning all layers (second stage training)
    learning_rate_stage1 = float(sys.argv[7]) # learning rate used at stage 1 (must be informed at stage 2 or same value for both if it is stage 1)

    data_h5 = os.path.join(data_dir,'data_h5') # load the (pre-processed) h5 data from
    output_dir = os.path.join(data_dir,'output/') # output predictions and some additional information (train history, plots, etc)
    model_filename = 'vgg16_app-real-age_fg2019' # filename sufix
    #---------------------------------


    new_filename_st1, new_filename_st2 = print_init_msg(data_dir, data_h5, output_dir, model_filename, train_model, stage_training, learning_rate, batch_size, epochs, learning_rate_stage1)
    main(data_h5, output_dir, train_model, stage_training, new_filename_st1, new_filename_st2, learning_rate, batch_size, epochs)
