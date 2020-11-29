try:
    import unzip_requirements
except ImportError:
    pass

import os
import sys
import io
import json
import base64
from requests_toolbelt.multipart import decoder

from model import sptotex


MODEL_PATH = 'STT1.pt'


print('Downloading Model')

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
    print(type(audio.content))
    print('Audio obtained')
    
    return audio.content


def STT(event, context):
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

        # Get Caption
        print('Getting caption')
        output = sptotex(audio_file, MODEL_PATH)
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
