from PIL import Image
from models.style_image import stylize
from flask import Flask, jsonify, request, redirect, flash, url_for, render_template
import io
import base64

MODEL_PATH = 'models/neural.model'

app = Flask(__name__)
app.secret_key = "secret key"

def style(image_bytes, style_idx=1):
    """Style the content image."""

    # Get image from the request
    image = Image.open(io.BytesIO(image_bytes))

    # Style
    output = stylize(image, style_idx, model=MODEL_PATH)
    output.save('output.jpg')        
    # Convert output to bytes
    buffer = io.BytesIO()
    output.save(buffer, format="JPEG")
    output_bytes = base64.b64encode(buffer.getvalue())
    return output_bytes.decode('ascii')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        style_image = request.form['style_image']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            img_bytes = file.read()
            picture = style(image_bytes=img_bytes,style_idx=int(style_image))
            return render_template('index.html', image=picture)


if __name__ == "__main__":
    app.run(host='0.0.0.0')	
