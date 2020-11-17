try:
    import unzip_requirements
except ImportError:
    pass

import io
import json
import base64
import boto3
import torch
import os

print('torch version:',torch.__version__)

from requests_toolbelt.multipart import decoder

from attention import get_text_translate_function

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'pankaj90382-dnn'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'translation-encoder-decoder-de-en.pt'
METADATA_PATH = os.environ['METADATA_PATH'] if 'METADATA_PATH' in os.environ else 'de-to-en-meta.dill.pkl'

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


def fetch_inputs(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final input that will be used by the model
    input_text = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Input obtained')
    
    return input_text.content.decode('utf-8')


def translate(event, context):
    """Style the content image."""
    try:
        input_text = fetch_inputs(event)
        print(input_text)

        print("loading Model and Meta Datafile")
        model = loading_from_s3(MODEL_PATH)
        metadata = loading_from_s3(METADATA_PATH)
        print("Loaded Model and Meta Data File")
        # Output Sentiment
        output = get_text_translate_function(model, metadata, input_text)
        print(output)

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