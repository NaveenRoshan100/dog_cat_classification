from flask import Flask, render_template,request
import cv2
import numpy as n
import torch
import torch.nn as nn
from utils.model import model_arc

app=Flask(__name__)

@app.route('/')
def home():
          return render_template('home.html')

@app.route('/result',methods=['post'])
def results():
        x=request.files['user_images']
        img=n.frombuffer(x.read(),n.uint8)
        img=cv2.imdecode(img,cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (156, 156))        # Resize if needed
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32)
           
        model = model_arc(inp_siz=3)  # Instantiate the model class
        model.load_state_dict(torch.load(r'C:\Users\navee\Desktop\flask_apps\dog_cat_classification\models\model_weights.pth', map_location='cpu'))  
        model.eval()
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0) 
        with torch.no_grad():
                output = model(img)
        print(torch.sigmoid(output).item())
        if torch.sigmoid(output).item() <0.5:
                return render_template('submit.html',output='cat')
        else:
                return render_template('submit.html',output='dog')
app.run(debug=True)
