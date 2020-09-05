try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import io
import json
import base64
import boto3
from requests_toolbelt.multipart import decoder

from classes import class_labels

# Define dat file
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'pankaj90382-dnn'
RECOG_PATH = os.environ['RECOG_PATH'] if 'RECOG_PATH' in os.environ else 'traced_face_recog.pt'
# MODEL_PATH = 'shape_predictor_5_face_landmarks.dat'

print('Downloading Model')

s3 = boto3.client('s3')


try:
    if os.path.isfile(RECOG_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=RECOG_PATH)
        print("Creating Model Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(size=(160,160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()



def rface(event, context):
    """Align the Input Image."""
    try:
        # Get image from the request
        print('Getting Content')
        if 'Content-Type' in event['headers']:
            content_type_header = event['headers']['Content-Type']
        else:
            content_type_header = event['headers']['content-type']
        print("Content Loaded.")
        #print(event['body'])  # printing the actual hex image on lambda console
        body = base64.b64decode(event["body"])
        print("BODY LOADED")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print(prediction)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'file': filename.replace('"',''),'predicted':prediction, 'predicted-class':class_labels[prediction]})
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
