import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, session
from PIL import Image

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR/'input'
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR/'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


app = Flask(__name__)
app.secret_key = 'debug'  # TODO: replace this with a configuration variable


@app.route('/')
def index():
    if 'id' not in session:
        session['id'] = os.urandom(8).hex()
        session['state'] = 'input'
    return render_template('index.html')

ALLOWED_EXTENSIONS = ['gif', 'jpg', 'jpeg', 'png']


@app.route('/upload_image', methods=['POST'])
def upload_image():
    redir = redirect('/')
    if 'file' not in request.files:
        flash('File upload was empty.')
        return redir
    input_file = request.files['file']
    if input_file.filename == '':
        flash('Filename was empty.')
        return redir
    extension = input_file.filename.rpartition('.')[2]
    if extension not in ALLOWED_EXTENSIONS:
        flash('Allowed image types: %s.' % ', '.join(ALLOWED_EXTENSIONS))
        return redir
    filename = '%s.%s' % (os.urandom(8).hex(), extension)
    input_file.save(str(INPUT_DIR/filename))
    session['input_image'] = filename
    flash('Image uploaded.')
    return redir


@app.route('/render')
def render():
    redir = redirect('/')
    if 'image' not in session:
        flash('You didn\'t upload an image first.')
        return redir
    try:
        image = Image.open(str(INPUT_DIR/session['input_image']))
    except OSError:
        flash('Error reading your previously uploaded image. Try again?')
        return redir
    output_filename = session['input_image'].rpartition('.')[0] + '.jpg'
    image = image.resize((256, 256), 1)
    image = image.save(str(OUTPUT_DIR/output_filename), quality=85)
    session['output_image'] = output_filename
    flash('Rendered image.')
    return redir
