import math
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from PIL import Image

# Provide valid crop size. Returns multiple of upscale_factor
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# Provides set of Low resolution images, High resolution images restored by Bicubic interpolation and the actual input HR image (centercropped.)
def image_transform_LR_HR_Restored(HR_image, crop_size, upscale_factor):
    LR_scale = Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    HR_scale = Resize(crop_size, interpolation=Image.BICUBIC)

    HR_image = CenterCrop(crop_size)(HR_image)
    LR_image = LR_scale(HR_image)
    HR_restored_image = HR_scale(LR_image)
    return ToTensor()(LR_image), ToTensor()(HR_restored_image), ToTensor()(HR_image)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


def load_image(image):
    """Apply transformations to an input image."""
    try:
        transformations = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)


def upscale(image, model_path):
    w, h = image.size
    UPSCALE_FACTOR=4
    print('Getting Crop size')
    crop_size = calculate_valid_crop_size(min(w, h), UPSCALE_FACTOR)
    print('Getting all HR and LR images')
    val_lr, val_hr_restored, val_hr = image_transform_LR_HR_Restored(image, crop_size, UPSCALE_FACTOR)
    print('Converting and Squeezing Image')
    val_lr = val_lr.to(torch.device("cpu")).unsqueeze(0)
    print('Generator Loading')
    # Upscale is 4
    model = Generator(UPSCALE_FACTOR).eval()
    model.load_state_dict(torch.load(model_path))
    print('Generator Loading Done')
    output = model(val_lr)
    print('Get final Image')
    return ToPILImage()(output[0]), ToPILImage() (val_hr), ToPILImage() (val_hr_restored)
