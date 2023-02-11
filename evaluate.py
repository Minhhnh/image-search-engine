import cv2
import numpy as np
import albumentations as A
import timm
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from tqdm import tqdm
import pickle
import io
import pandas as pd

class CFG:
    debug = False
    image_path = "flickr30k_images/flickr30k_images"
    captions_path = ""
    batch_size = 32
    num_workers = 1
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model_vn = "vinai/phobert-base"
    text_embedding = 768
    text_tokenizer_vn = "vinai/phobert-base"
    max_length = 200

    pretrained = True  # for both image encoder and text encoder
    trainable = True  # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

df_file = "word_segmented.csv"
pkl_file = "31kimage_embeded_50.pickle"

# utility functions
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

annotator = VnCoreNLP("VnCoreNLP-master/VnCoreNLP-1.1.1.jar",
                      annotators="wseg", max_heap_size='-Xmx500m')
tokenizer = AutoTokenizer.from_pretrained(
    CFG.text_tokenizer_vn, use_fast=False)
print(annotator)
specialchars = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}',
                '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                '9', '']
# # text = "Một con chó nâu và một con chó trắng chạy qua một cánh đồng"


def word_segmentation_one(query):
    for c in specialchars:
        if c in query:
            query = query.replace(c, '')
    word_segmented_text = np.squeeze(annotator.tokenize(query))
    try:
        query = ' '.join(map(str, word_segmented_text))
    except:
        print('one word: ', word_segmented_text)
        return True, word_segmented_text
    return False, query


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, transforms):
        self.image_filenames = image_filenames
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        print(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        return item

    def __len__(self):
        return len(self.image_filenames)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model_vn, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        self.model = AutoModel.from_pretrained(CFG.text_encoder_model_vn)

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def build_loaders(dataframe, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def get_one_image_embeddings(valid_df, model):
    valid_loader = build_loaders(valid_df, mode="valid")
    model.eval()

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    only_one_word, query = word_segmentation_one(query)
    print(query)
    if only_one_word:
      encoded_query = tokenizer([str(query)])
    else:
      encoded_query = tokenizer([query])

    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
        print(text_embeddings.shape)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    print(indices)
    matches = [image_filenames[idx] for idx in indices]
    return matches

def compute_top_k_accuracy(df, k):
    hits = 0
    image_paths = df['image'][::5].values
    num_batches = int(np.ceil(len(image_paths) / CFG.batch_size))
    for idx in tqdm(range(num_batches)):
        start_idx = idx * CFG.batch_size
        end_idx = start_idx + CFG.batch_size
        current_image_paths = image_paths[start_idx:end_idx]
        # print(current_image_paths)
        query = df['caption'][idx*5]
        for c in specialchars:
                if c in query:
                    query = query.replace(c, ' ')
        # print(query)
        result = find_matches(model, image_embeddings, query, df['image'].values, n=k)
        hits += sum(
            [
                image_path in matches
                for (image_path, matches) in list(zip(current_image_paths, result))
            ]
        )
        print(hits)
    return hits / len(image_paths)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load("best_resnet50.pt", map_location=device))
    model.eval()
    print(model)
    image_embeddings = CPU_Unpickler(open(pkl_file, "rb")).load()
    # valid_df = pd.read_csv(df_file)
    # acc = compute_top_k_accuracy(valid_df, k=100)
    # print(acc)