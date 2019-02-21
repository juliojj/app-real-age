# Short description
This source code was used to generate the results of the paper 
<b>"On the effect of age perception biases for real age regression"</b>, accepted in the 
14th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2019). 
<center><img src="http://158.109.8.102/AppaRealAgeFG19/fg-2019.jpg"></center>

# Citation
In case you use this code, please cite the reference paper (<a href="https://arxiv.org/abs/1902.07653">arXiv link</a>) as:

@inproceedings{jacques:FG2019,<br>
 author={Julio C. S. Jacques Junior and Cagri Ozcinar and Marina Marjanovic
         and Xavier Baro and Gholamreza Anbarjafari and Sergio Escalera},<br>
 booktitle={IEEE International Conference on Automatic Face and Gesture
            Recognition (FG)},<br>
 title={On the effect of age perception biases for real age regression},<br>
 year={2019},<br>
 }<br>
 

# Tested on
- Linux Ubuntu 16.04.2 LTS
- NVIDIA Driver Version: 390.30 - GeForce GTX 1080
- Cuda = 9.0, CuDNN = 7.0.5.15
- Keras = 2.1.6, tensorflow = 1.8.0, python = 2.7<br>
>a docker image with required libraries is provided next

# Intructions
<b>Step 1)</b> Download the preprocessed data (<a href="http://158.109.8.102/AppaRealAgeFG19/train.zip">train</a> / <a href="http://158.109.8.102/AppaRealAgeFG19/valid.zip">valid</a> / <a href="http://158.109.8.102/AppaRealAgeFG19/test.zip">test</a>). <br>
- Create an auxiliary directory in your home, for instance, "data/data_h5"
- Uncompress each downloaded set, and move all files to "data/data_h5"

<b>Step 2)</b> Lets assume you have already downloaded the data and source code (from Github), and that you have the following strucure in your home directory:

- /home/your_username/source_code/
- /home/your_username/data/data_h5/

where, in "source_code" you have the python files, and within "data" you have the "data_h5" directory (with all .h5 files inside)

<b>Step 3)</b> Now, you can run the code inside a docker, with all required libraries installed, as described next (it requires GPU and "nvidia-docker" installed). You can optionally run the code without docker and with CPU. However, different library versions might conflict.

  - pull the docker: <b>docker pull juliojj/keras-tf-py2-gpu</b>
  - run the docker and map your local directory with data and source as:<br>
<b>nvidia-docker run -it --rm -v /home/your_username/source_code/:/root/app-real-age/source_code -v /home/your_username/data/:/root/app-real-age/data juliojj/keras-tf-py2-gpu</b>

<b>Step 4)</b> Running the code (training and predicting). Inside the docker, go to the directory you have the python source code (i.e., /root/app-real-age/source_code/) and run:
 
- Stage 1 (training): <b>python vgg16_app-real-age_fg2019.py ../data/ True 1 1e-4 32 3000 1e-4</b>
- Stage 2 (training): <b>python vgg16_app-real-age_fg2019.py ../data/ True 2 1e-4 32 1500 1e-4</b>

After training, you can optionally run the code (without training) to make predictions as:<br>
<b>python vgg16_app-real-age_fg2019.py ../data/ False 2 1e-4 32 1500 1e-4</b>

><b>Note:</b> results reported in the paper were generated after the 2 stages training (using the above parameters). You have to run stage 1 and then stage 2 to reproduce the results.

><b>Important note:</b> during training, the model might suffer from "vanishing gradients" due to initialization procedures of the new layers. If you observe the network is not learning during the first epochs, restart training. Another option can be to reduce batch size.

Parameters are defined as: [data_patch, train_model (bool), stage_num, lr (current), batch_size, epochs, lr (stage 1)]

>The <b>pre-trained model</b> (stage 2), used to generate the results reported in the paper can be downloaded <a href="http://158.109.8.102/AppaRealAgeFG19/vgg16_app-real-age_fg2019_stage_2_st1-lr_0.0001_st2-lr_0.0001.hdf5">here</a>. You can copy it into the 'best_models' directory and make predictions on the test set without the need of doing the training.
