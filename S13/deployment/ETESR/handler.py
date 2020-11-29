try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import base64
import boto3
import sys
from requests_toolbelt.multipart import decoder

from model import srtotext


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'pankaj90382-dnn'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'ETESR.pt'

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
        
def fetch_input_audio(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final audio that will be used by the model
    audio = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Audio obtained')
    
    return audio.content


def ETESR(event, context):
    """Speech to Text."""
        # Get image from the request
    try:
        audio_file = '/tmp/temp.wav'
        audio = fetch_input_audio(event)
        print('Size:-', sys.getsizeof(audio))
        print(audio)

        with open(audio_file, 'wb') as f:
            f.write(audio)
        print('File Written')
        print('Path Exsits:- ',os.path.exists(audio_file))
        print(os.listdir('/tmp/'))

        print("loading Model")
        model = loading_from_s3(MODEL_PATH)
        print("Loaded Model")

        # Get Caption
        print('Getting caption')
        output = srtotext(audio_file, model)
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
