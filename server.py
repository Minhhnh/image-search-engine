from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from flask import request
from flask import render_template
from torch_utils_50 import find_matches, CLIPModel, get_one_image_embeddings
import pandas as pd
import torch
import pickle
import io
import os
import cloudinary
import cloudinary.uploader

df_file = "images.csv"
pkl_file = "31kimage_embeded_50.pickle"

CLOUD_NAME = "thefour123"
API_KEY = "717175813831459"
API_SECRET = "LgOqSL4uAr21kZIInC9Q8l7D0VI"

# utility functions


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)

# init
app = Flask(__name__)
# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = './flickr30k_images/flickr30k_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['POST'])
@cross_origin(origin='*')
def predict():
    image_embeddings = CPU_Unpickler(open(pkl_file, "rb")).load()
    # print(image_embeddings.shape)
    valid_df = pd.read_csv(df_file)
    # print(valid_df)
    query = request.get_data().decode('utf-8')
    print(query)
    matches = find_matches(model, image_embeddings, query=query,
                           image_filenames=valid_df['image'].values, n=9)
    links = []
    for match in matches:
        uploaded_image = cloudinary.uploader.upload(
            "./flickr30k_images/flickr30k_images/" + match, public_id="flickr30k/" + match)
        links.append(uploaded_image.get('url'))
    data = {'links': links}
    return jsonify(data)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load("best_resnet50.pt", map_location=device))
    model.eval()
    app.run(port='3000', debug=True)
