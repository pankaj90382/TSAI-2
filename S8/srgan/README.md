# Image Super Resolution

[![Website](https://img.shields.io/badge/Website-green.svg)](http://face-operations.s3-website-us-east-1.amazonaws.com/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S8/srgan/SRGAN.ipynb)

The objective here is to train SR-GAN (Super Resolution- Generative Adversial Network) on the flying objects dataset to provide Super resolution of the flying object images. The model is then deployed on AWS.

## Model Hyperparameters

- Model : SRGAN
- Batch Size : 64
- Upscale Factor : 4
- Epochs : 30
- Loss function : Content Loss + Adversial Loss
- Optimizer : Adam

## Dataset Details
There are four different classes of images in the dataset:

<table>
<thead>
  <tr>
    <th>Image Class</th>
    <th>No of Images</th>
    <th>Mean</th>
    <th>Std. Dev</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Flying Birds<br></td>
    <td>7781</td>
    <td rowspan="4">[0.3749, 0.4123, 0.4352]</td>
    <td rowspan="4">[0.3326, 0.3393, 0.3740]</td>
  </tr>
  <tr>
    <td>Large QuadCopters</td>
    <td>3609</td>
  </tr>
  <tr>
    <td>Small QuadCopters</td>
    <td>3957</td>
  </tr>
  <tr>
    <td>Winged Drones</td>
    <td>3163</td>
  </tr>
</tbody>
</table>

## Model Architecture

GAN based model includes deep residual network for Generator network which is able to recover photo-realistic textures from heavily downsampled images. 

Model uses perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes the solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, the content loss compares deep features extracted from Super-resolved and High Resolution images with a pre-trained VGG network.

## Results

### Model results
Below are the results from SR-GAN. It consists of HR restored image (restored by Bicubic interpolation of the Low resolution image), actual input HR image (centercropped) and the Super Resolution result from the SR-GAN model.

![](Save_Model/sr_results.png)

### Generator/Discriminator Loss Trend versus Epoch

Shared below is the Generator/Discriminator Loss trend during training.

![](Save_Model/loss_vs_epoch.JPG)

### PSNR and SSIM trend

Shared below are the Peak Signal to Noise Ratio and Structural Similarity Index trend during training.

![](Save_Model/psnr_ssim.JPG)

## Animation
![](Save_Model/srgan.gif)

### Upscale Factor of 4
We trained the model from scratch, keeping the upscale factor 4, and the output is presented here. Column 1 represents the results from Upsampling Bicubic Interpolation, column 2 represents the ground truth image and the third column is the super resolution image.

![](Save_Model/epoch_30_index_1.png)
![](Save_Model/epoch_30_index_2.png)
![](Save_Model/epoch_30_index_3.png)

## Super Resolution GAN Refrences
- [SRGAN Pytorch Implementation](https://github.com/leftthomas/SRGAN)

