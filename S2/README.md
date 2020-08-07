# Session 2 - Transfer Learning with Custom Classes on Mobile-Net v2 Architecture (Using Pretrained Imagenet Model)

## Objective
Train Mobilenetv2 on custom classes by using the transfer learning. I have used the trained weights of Imagenet classifier from [Pytorch](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/).

### Dataset Info

A custom dataset will be used to train model, which consists of:
- Flying Birds
- Large Quadcopters
- Small Quadcopters
- Winged Drone

## Resizing Strategy

### SPP net Strategy Pretrained Mobilenet:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sobti/TSAI/blob/master/Drone%20Prediction/MobilenetV2_Model.ipynb)

 ![Sppnet](https://user-images.githubusercontent.com/42212648/89426205-857e6080-d757-11ea-8510-3147acea6a78.png)
 
Usually deep neural network requires fixed size input images. This is obtained mostly by transformation (resize and center crop) strategy. with transformation , there is possible chances that we loose important infromation from the images. Sppnet overcome the requiremnet to trtanform the images to arbitary size and hence neural net equip with SPP-net can run with images having varying sizes.Pyramid pooling is also robust to object deformations. With these advantages, SPP-net should in general improve all CNN-based image classification methods.

####  implenmenting SPP net:

- Sppnet is mostly implement before FC layer.

      model1 = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
      
      class sppmob(nn.Module):
        def __init__(self):
           super(sppmob, self).__init__()
           count=1 
           for child in model1.children(): 
            if count==1:  
             self.conv1=child
             count=count+1 
             break;

           self.lin=nn.Linear(in_features=26880,out_features=4,bias=True)    

        def forward(self, x):
           x=self.conv1(x)
           spp=spatial_pyramid_pool(x,1,[int(x.size()[2]),int(x.size()[3])],[4,2,1])
           x=self.lin(spp)
           return x
           
  in_features=26880 -> fetaures out from Sppnet for each images

- It requires custom Dataloader - Collate_Fn needs to be customised to create data loader of variable size images as dataloader stacks similiar size images.

         def my_collate(batch):
           data={}
           data['total_drones'] = [(item['total_drones']) for item in batch]
           data['total_drones'] = (torch.Tensor(data['total_drones'][0])).unsqueeze(dim=0)
           data['labels'] = [item['labels'] for item in batch]
           data['labels'] = torch.LongTensor(data['labels'])
           return data
 
 - As mobilenet goes very deep such that image of 224 * 224 squeezes to 7 * 7 .BE alert in feeding smaller dimension images as maxpooling in Sppnet can cause issue.
 
 - Sometimes without having control on images can cause out of memory issue as image going can be having higher dimensions.

#### :point_right: Issues and Exploration:

- I have to make batch size = 1 for variable size images ( Needs to explore more on it) 

- facing Cuda out of memory as few images are very large .Advisable to keep images in certain range.

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
	
:+1: Planning to use simple image augmentation strategy. There are some errors while implementing the variable image size strategy (SPPnet).

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
    <figcaption><h3>Grad-Cam Incorrect Images</h3></figcaption>
    <img src="Save_Model/gradcam_Incorrect.png">
</figure>

### Correctly Classified Images
The Correctly classified images with their gradcam results.
<img src="Save_Model/Corr-classified Images.jpg">
<figure>
    <figcaption><h3>Grad-Cam Correct Images</h3></figcaption>
    <img src="Save_Model/gradcam_Correct.png">
</figure>

## Appendix
 - Great thanks to the [blog.](https://www.analyticsvidhya.com/blog/2019/10/how-to-master-transfer-learning-using-pytorch/)
 - [Pytorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 
 - [Pytorch Neural Transfer Style](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
 - [SPPnet Paper](https://arxiv.org/abs/1406.4729)
 - Implementation of [Sppnet](https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py) Paper
 
