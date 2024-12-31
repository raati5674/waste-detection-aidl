from flask import Flask, flash, request, redirect, send_file,url_for,send_from_directory
import os
import logging
import pathlib

import torch
from PIL import Image

from model.vits import ViT

from torchvision import transforms

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'jpg'}
FILE_NAME="sample.jpg"

MODEL=None

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FILE_NAME']=FILE_NAME

with app.app_context():
    # First load into memory the variables that we will need to predict
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / "checkpoint/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path)

    MODEL = ViT(checkpoint['img_size'],
            checkpoint['patch_size'],
            checkpoint['num_hiddens'],
            checkpoint['mlp_num_hiddens'],
            checkpoint['num_heads'],
            checkpoint['num_blks'],
            checkpoint['emb_dropout'],
            checkpoint['blk_dropout'])
    
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def __process_file__(request):
    if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file_name =app.config['FILE_NAME']
        path_file=os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        if os.path.exists(path_file):
            os.remove(path_file)
        file.save(path_file)
        
        img=Image.open(path_file)

        x=transforms.functional.pil_to_tensor(img)

        predict=MODEL(x.view((1,1,28,28)).float())
        distribution=torch.nn.Softmax(dim=-1)(predict)
        distribution=distribution.detach().cpu().numpy()
        predict=torch.argmax(torch.nn.Softmax(dim=-1)(predict))
        predict=predict.cpu().item()
        
        return {
            "distribution":distribution.tolist(),
            "predict":predict
        }
        
    else:
        flash('file not allowed')
        return redirect(request.url)
    
@app.route('/')
def hello():
	return "Hello World!"


@app.route('/upload', methods=['GET'])
def upload_file():    
    return '''
    <!doctype html>
    <title></title>
    <h1>Classification of FashionMNIST from ViT</h1>
    <form method="post" enctype="multipart/form-data" id="multipart">
      <input type="file" name="file"/>
      <input type="submit" value="predict" formaction ="predict" />
    </form>
    '''
@app.route('/predict', methods=['POST'])
def show_file():    
    preditc=__process_file__(request)    
    return preditc


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000,debug=True)
 