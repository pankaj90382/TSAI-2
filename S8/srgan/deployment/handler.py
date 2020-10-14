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

from super_resolution import upscale


MODEL_PATH = 'netG.pt'


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
    
    return Image.open(io.BytesIO(picture.content))


def srgan(event, context):
    """Super Resolution"""
    try:
        # Get image from the request
        print('fetching image from buffer')
        image = fetch_input_image(event)
        print('fetched image successfully')
        # Upscale the image
        sr, val_hr, val_hr_restored = upscale(image, MODEL_PATH)

        # Convert output to bytes
        print("Loading output to buffer")
        buffer_hr_restored = io.BytesIO()
        val_hr_restored.save(buffer_hr_restored, format="JPEG")
        hr_restored_bytes = base64.b64encode(buffer_hr_restored.getvalue())

        buffer_hr = io.BytesIO()
        val_hr.save(buffer_hr, format="JPEG")
        hr_bytes = base64.b64encode(buffer_hr.getvalue())

        buffer_sr = io.BytesIO()
        sr.save(buffer_sr, format="JPEG")
        sr_bytes = base64.b64encode(buffer_sr.getvalue())

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'hr_restored': hr_restored_bytes.decode('ascii'),
                            'hr': hr_bytes.decode('ascii'), 'sr': sr_bytes.decode('ascii')})
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
