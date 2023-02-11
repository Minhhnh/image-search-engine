from re import I
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import render_template
from torch_utils_50 import find_matches, CLIPModel, get_one_image_embeddings
import pandas as pd
import torch
import pickle
import io
import os

df_file = "image.csv"
pkl_file = "embeded_resnet50.pickle"

# utility functions
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


# init
app = Flask(__name__)
# Apply Flask CORS
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

UPLOAD_FOLDER = "./flickr30k_images/flickr30k_images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
@cross_origin(origin="*")
def mainpage():
    return render_template("index.html")


@app.route("/", methods=["POST"])
@cross_origin(origin="*")
def predict():
    image_embeddings = CPU_Unpickler(open(pkl_file, "rb")).load()
    print("Embedding shape: ", image_embeddings.shape)
    valid_df = pd.read_csv(df_file)
    print("DF shape: ", valid_df.shape)
    query = request.form["query"]
    print(query)
    matches = find_matches(
        model,
        image_embeddings,
        query=query,
        image_filenames=valid_df["image"].values,
        n=9,
    )
    return render_template("index.html", matches=matches)


@app.route("/upload", methods=["GET"])
@cross_origin(origin="*")
def uploadpage():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
@cross_origin(origin="*")
def upload():
    # get imagefile
    imagefile = request.files["imagefile"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], imagefile.filename)
    imagefile.save(path)
    # create new dataframe with 1 value
    upload_df = pd.DataFrame([imagefile.filename])
    upload_df.columns = ["image"]
    # embedding image
    one_image_embedding = get_one_image_embeddings(upload_df, model)
    print(one_image_embedding.shape)
    # concat exist embedded_images with embedded_image
    image_embeddings = CPU_Unpickler(open(pkl_file, "rb")).load()
    print(image_embeddings.shape)
    new_image_embeddings = torch.cat((image_embeddings, one_image_embedding), 0)
    print(new_image_embeddings.shape)
    # save to file pickle
    pickle.dump(
        new_image_embeddings, open(pkl_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL
    )
    # concat exist dataframe with new dataframe
    valid_df = pd.read_csv(df_file)
    valid_df = pd.concat([valid_df, upload_df], ignore_index=True)
    print(valid_df)
    valid_df.to_csv(df_file, index=False)
    return render_template("upload.html")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load("resnet50_best.pt", map_location=device))
    model.eval()
    app.run(port="3000", debug=True)
