try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import base64
import boto3
from PIL import Image
from requests_toolbelt.multipart import decoder

from captioning import caption_image

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'pankaj90382-dnn'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'Caption_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
WORDMAP_PATH = 'WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'

print('Downloading Model')

s3 = boto3.client('s3')

def loading_from_s3(PATH):
    try:
        if os.path.isfile(PATH) != True:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=PATH)
            print(f"Creating {PATH} Bytestream")
            bytestream = io.BytesIO(obj['Body'].read())
            print("Loading :- ", PATH)
            # model = torch.jit.load(bytestream)
            print(f"{PATH} Loaded...")
            return bytestream
    except Exception as e:
        print(repr(e))
        raise(e)

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


def Imcaption(event, context):
    """Caption input image."""
    try:
        # Get image from the request
        picture = fetch_input_image(event)
        image = Image.open(io.BytesIO(picture))

        print("loading Model")
        model = loading_from_s3(MODEL_PATH)
        print("Loaded Model")

        # Get Caption
        print('Getting caption')
        output = caption_image(image, model, WORDMAP_PATH)
        print('Caption:', output)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data': output})
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

