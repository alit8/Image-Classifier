# Image Classifier
A complete pipeline for image classification with Keras. The models consist of pretrained CNN models on imagenet(Xception, InceptionResNetV2, ResNet50, and other Keras applications) and dense classifiers. In training process, the CNN models are first frozen and only the dense part is trained. After that there is a fine-tuning step to also train an arbitrary number of layers of CNN models.

I used this image classifier piple to train an image classifier on planar brain images.

## How to use

- Each data split must contain subdirectories with the name of each class and each subdirectory contains images of that class
- Set the path to train and validation split in `train.py`
- Configure parameters in `train.py`
- Start training: `python train.py`
