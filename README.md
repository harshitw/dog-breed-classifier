# DOG BREED CLASSIFIER

# Udacity Project 2

A classifier that predicts the dog breed given the image of a dog. However if you give an image of a human then it will predict the closest looking breed of dog to that image. 

I have created my own neural network architecture to do so. However using Transfer Learning from pretrained models like VGG16 and ResNet-50 the accuracy obtained was significantly better. 


# Getting started

1. cd ~/Desktop

I have used AWS for using GPU for training my model. If you are using same, then follow the steps 2 - 4.  

2. ssh -i yourkeyname.pem ubuntu@ip-address

3. jupyter notebook --generate-config 

4. sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip = '*'/g" ~/.jupyter/jupyter_notebook_config.py

5. git clone https://github.com/udacity/dog-project.git 

6. cd dog-project

Now the next part is to download the compressed dataset and unzip it. For this use "wget" to download and unzip filename. 

7. download dog images dataset => wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip => unzip dogImages.zip

8. download human images dataset => wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip => unzip lfw.zip

9. download vgg16 bottleneck features => wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz => unzip DogVGG16Data.npz 

10. codna create -n dogs python=3.5

11. source activate dogs

12. pip install -r requirements/requirements-gpu.txt

13. Switching keras backend to tensorflow. KERAS_BACKEND=tensorflow python -c "from keras import backend"

Finally to run the notebook on your laptop , type the command given below. 
14. jupyter notebook dog_app.ipynb


