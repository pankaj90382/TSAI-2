# Session 2 - Transfer Learning with Custom Classes on Mobile-Net v2 Architecture (Using Pretrained Imagenet Model)

## Objective
Train Mobilenetv2 on custom classes by using the transfer learning. I have used the trained weights of Imagenet classifier from [Pytorch](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/).

### Dataset Info

A custom dataset will be used to train model, which consists of:
- Flying Birds
- Large Quadcopters
- Small Quadcopters
- Winged Drone

### Image Augmentation

I have used the [albumentations](https://albumentations.readthedocs.io/en/latest/api/augmentations.html) library for augmentation.

- **Resize**:
	- Downscale the images to be able to train for lower dimensions first.
- **RandomBrightnessContrast** & **HueSaturationValue**:
	- Used to reduce the dependency on image colours for prediction.
- **ShiftScaleRotate**:
	- Translate, scale and rotate images.
- **'GridDistortion'**:
	- Grid analysis estimates the distortion from a checkerboard or thin line grid.
- **RandomRotate90**:
	- Images were randomly rotated within 90 degrees.

### Sample Images
<img src="Save_Model/Sample.jpg">

### Model Architecture
Last Layer of Mobilenetv2 will be replaced to get 4 classes. The architecture is taken from the Imagenet which is having 1000 classes. While training the model freezing all the layers except the last one.  

### Loss Function
CrossEntropyLoss - Used Cross Entropy Loss to get the desired results

### Analysis

#### Learning Rate

<img src="Save_Model/Learning_Rate_Curve.jpg">

#### Batch wise Training Loss
<img src="Save_Model/Batch_Train_Val_Loss_Curve.jpg">

#### Accuracy

<img src="Save_Model/Accuracy_Curve.jpg">

#### Validation Curve
<img src="Save_Model/Validation_Curve.jpg">

## Results

### Miss Classified Images
The Miss classified images with their gradcam results.
<img src="Save_Model/Mis-classified Images.jpg">
<figure>
    <figcaption>Grad-Cam Incorrect Images</figcaption>
    <img src="Save_Model/gradcam_Incorrect.png">
</figure>

### Correctly Classified Images
The Correctly classified images with their gradcam results.
<img src="Save_Model/Corr-classified Images.jpg">
<figure>
    <figcaption>Grad-Cam Correct Images</figcaption>
    <img src="Save_Model/gradcam_Correct.png">
</figure>

## Appendix
 - Great thanks to the [blog.](https://www.analyticsvidhya.com/blog/2019/10/how-to-master-transfer-learning-using-pytorch/)
 - [Pytorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 
 - [Pytorch Neural Transfer Style](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
