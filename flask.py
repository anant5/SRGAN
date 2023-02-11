# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

!pip install flask-ngrok
!pip install tensorlayer

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
from tensorlayer.models import Model
import os
import numpy as np
# from IPython.display import HTML
from flask_ngrok import run_with_ngrok
from flask import Flask,render_template, request, redirect
import time
from threading import Thread

def get_G(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nin = Input(input_shape)
    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
        nn = BatchNorm2d(gamma_init=g_init)(nn)
        nn = Elementwise(tf.add)([n, nn])
        n = nn

    n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=g_init)(n)
    n = Elementwise(tf.add)([n, temp])
    # B residual blacks end

    n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)
    G = Model(inputs=nin, outputs=nn)
    return G

lr_img_path = '/content/drive/My Drive/data/valid_LR'
save_dir = "/content/drive/My Drive/data/results"
checkpoint_dir = "/content/drive/My Drive/data/models"
hr_img_path = '/content/drive/My Drive/data/valid_HR'
imids = 0

valid_hr_img_list = sorted(tl.files.load_file_list(path=hr_img_path, regx='.*.png', printable=False))
valid_lr_img_list = sorted(tl.files.load_file_list(path=lr_img_path, regx='.*.png', printable=False))
valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=lr_img_path, n_threads=32)
valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=hr_img_path, n_threads=32)
G1 = get_G([1, None, None, 3])
G1.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
G1.eval()

def evaluate():
    global imids,G1
    imid = imids[1:]
    imid = int(imid) - 1
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
   
    valid_lr_img = (valid_lr_img / 127.5) - 1      

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G1(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  
    print("[*] save images")
    tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
    tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))
    tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

from flask import redirect, url_for
app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
@app.route("/")
def home():
    # return render_template("test.html")    
     return """
     <style>
     .loader {
      border: 16px solid #f3f3f3;
      border-radius: 50%;
      border-top: 16px solid blue;
      border-right: 16px solid green;
      border-bottom: 16px solid red;
      border-left: 16px solid yellow;
      width: 120px;
      height: 120px;
      -webkit-animation: spin 2s linear infinite;
      animation: spin 2s linear infinite;
     }

     @-webkit-keyframes spin {
      0% { -webkit-transform: rotate(0deg); }
      100% { -webkit-transform: rotate(360deg); }
     }

     @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
     }
     </style>
     
     <center><h1>Welcome to Super-Resolution GAN App</h1>
     </center><center><h3>Please click on any image to generate its super-resolution</h3></center> 
     <iframe sandbox="allow-scripts" src="https://drive.google.com/embeddedfolderview?id=1pnVqN61GQ_V2nvQzrquin9dcWF8d7vJ6#grid" style="width:100%; height:450px; border:0;"></iframe>
     <br>
     <br>
     <center>
     <form method="POST">
     <input type ="text" name ="text">
     <input type="submit">
     </form>
     """



@app.route('/', methods=['POST'])
# def disp():
  # return "<h1>Processing</h1>",redirect('/output')

# @app.route('/output')
def my_form_post():
    # thr = Thread(target=disp)
    # thr.start()
    text = request.form['text']
    processed_text = text.upper()
    global imids
    imids = processed_text
    evaluate()
    time.sleep(10)
    # return """<iframe src="https://drive.google.com/embeddedfolderview?id=12vf69Wgjw8P9ja7dEZWIwhkHV-5uTgHS#grid" style="width:100%; height:450px; border:0;"></iframe>"""
    return """
    <center>
    <img src="https://drive.google.com/uc?id=1VIJDE6Wm2SeQqnQ94om6cGkGckp3UutH" alt="Generated image" style="max-width: 100%;max-height: 97%;">
    <br>
    <h3>Generated Image</h3>
    <br><br>
    <img src="https://drive.google.com/uc?id=1cefOmlu7K92QiYJeL8ncUw1vXqW_SiTa" alt="Low-Resolution image">
    <br>
    <h3>Low Resolution Image(Ground Truth)</h3>
    <br><br>
    <img src="https://drive.google.com/uc?id=1-3eB5o_f-5DEkmFGvnEPRhESJJJ6E2ii" alt="High-Resolution image" style="max-width: 100%;max-height: 97%;">
    <br>
    <h3>High Resolution Image(Ground Truth)</h3>
    <br><br>
    </center>
    """



app.run()

