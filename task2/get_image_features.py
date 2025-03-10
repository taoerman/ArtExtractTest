import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms

OBJECTS_PATH = "opendata/objects.csv"
PUBLISHED_IMAGES_PATH = "opendata/published_images.csv"
EFFICIENT_NET_MODEL = "efficientnet-b7"
SENTENCE_BERT_MODEL = "all-mpnet-base-v2"
BASE_URL = "https://api.nga.gov/iiif/{}/full/224,224/0/default.jpg"
BATCH_SIZE = 16
OUTPUT_DIR = "features"

objects = pd.read_csv(OBJECTS_PATH, low_memory=False)
published_images = pd.read_csv(PUBLISHED_IMAGES_PATH)

data = pd.merge(
    objects,
    published_images,
    how="inner",
    left_on="objectid",
    right_on="depictstmsobjectid",
)
data = data[(data["classification"] == "Painting") & (data["isvirtual"] == 0)]
data["id"] = data["iiifurl"].map(lambda x: x.split("/")[-1])
columns = ["id", "title"]
data = data[columns].dropna(subset=columns).drop_duplicates()
data = data.reset_index(drop=True)
data.to_csv(os.path.join(OUTPUT_DIR, "data.csv"), index=False)

efficient_net = EfficientNet.from_pretrained(EFFICIENT_NET_MODEL)
sentence_bert = SentenceTransformer(SENTENCE_BERT_MODEL)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def fetch_image(url, transform):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image, transform(image)


id_list = []
image_list = []
image_features = []
title_list = []
title_features = []
for i, row in data.iterrows():
    url = BASE_URL.format(row["id"])
    id_list.append(row["id"])
    image, transformed_image = fetch_image(url, transform)
    image.save(os.path.join(OUTPUT_DIR, "images", f"{row['id']}.jpg"))
    image_list.append(transformed_image)
    title_list.append(row["title"])
    if (i + 1) % BATCH_SIZE == 0 or i == len(data) - 1:
        with torch.no_grad():
            image_feature = efficient_net.extract_features(torch.stack(image_list))
            image_feature = efficient_net._avg_pooling(image_feature).squeeze((2, 3))
            title_feature = sentence_bert.encode(title_list)
        image_features.append(image_feature)
        title_features.append(title_feature)
        image_list.clear()
        title_list.clear()
        print(f"Finished {i + 1} rows", flush=True)
image_features = torch.cat(image_features, dim=0).numpy()
title_features = np.vstack(title_features)

np.save(os.path.join(OUTPUT_DIR, "ids.npy"), id_list)
np.save(os.path.join(OUTPUT_DIR, "image_features.npy"), image_features)
np.save(os.path.join(OUTPUT_DIR, "title_features.npy"), title_features)
