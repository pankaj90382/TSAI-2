try:
    import unzip_requirements
except ImportError:
    pass

import io
import json
import random
import numpy as np
import base64
import boto3
import torch
from PIL import Image
from requests_toolbelt.multipart import decoder
import torchvision
import torchvision.transforms as transforms

MODEL_PATH = 'red_car_gan_generator.traced.pt'

print("Loading Model")
model = torch.jit.load(MODEL_PATH)

def fetch_input_image(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final picture that will be used by the model
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Picture obtained')
    
    return picture.content


def decode(self, z):
    """
    Maps the given latent codes
    onto the image space.
    :param z: (Tensor) [B x D]
    :return: (Tensor) [B x C x H x W]
    """
    result = self.decoder_input(z)
    result = result.view(-1, 512, 4, 4)
    result = self.decoder(result)
    result = self.final_layer(result)
    return result


def generate():
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        fixed_noise = torch.randn(64,100,device='cpu')
        samples = decode(model, fixed_noise)
        print('Image Generated')
        generated_image = torchvision.utils.make_grid(samples.cpu(), normalize=True).permute(1,2,0).detach().numpy()*255
        generated_image = Image.fromarray(generated_image.astype(np.uint8), mode='RGB')
        return generated_image

def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.570838093757629, 0.479552984237671, 0.491760671138763), (0.279659748077393, 0.309973508119583, 0.311098515987396))])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return torchvision.utils.make_grid(model(tensor)[0][0], normalize=True).permute(1,2,0).detach().numpy()

def reconstruct(event, context):
    """Reconstruct the Input Image."""
    try:
        # Get image from the request
        picture = fetch_input_image(event)
        print('getting prediction')
        prediction = get_prediction(image_bytes=picture)
        print('Transforming Image')        
        output = Image.fromarray((prediction * 255).astype(np.uint8))

        # Convert output to bytes
        print('Prepare buffer for recon image')
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG")
        output_bytes = base64.b64encode(buffer.getvalue())

        print('Genrating Sample Image')
        generated_image = generate()
        print('preparing buffer for Generating Image')
        buffer2 = io.BytesIO()
        generated_image.save(buffer2, format="JPEG")
        generated_image_bytes = base64.b64encode(buffer2.getvalue())

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data1': output_bytes.decode('ascii'),
            'data': generated_image_bytes.decode('ascii')})
        }
    except Exception as e:
        print(repr(e))
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'error': repr(e)})
        }