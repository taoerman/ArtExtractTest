import torch
import torch.nn.functional as F
from dataset import DatasetManager
from model import ArtClassifier

IMAGE_DIR = "wikiart"
STYLE_CLASS_PATH = "ArtGAN/style_class.txt"
STYLE_TRAIN_PATH = "ArtGAN/style_train.csv"
GENRE_CLASS_PATH = "ArtGAN/genre_class.txt"
GENRE_TRAIN_PATH = "ArtGAN/genre_train.csv"
ARTIST_CLASS_PATH = "ArtGAN/artist_class.txt"
ARTIST_TRAIN_PATH = "ArtGAN/artist_train.csv"
MODEL_OUTPUT_NAME = "model.pth"

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10


style_training_loader, style_class_mapping = DatasetManager.get_dataset_loader(
    STYLE_TRAIN_PATH, STYLE_CLASS_PATH, IMAGE_DIR, BATCH_SIZE
)
genre_training_loader, genre_class_mapping = DatasetManager.get_dataset_loader(
    GENRE_TRAIN_PATH, GENRE_CLASS_PATH, IMAGE_DIR, BATCH_SIZE
)
artist_training_loader, artist_class_mapping = DatasetManager.get_dataset_loader(
    ARTIST_TRAIN_PATH, ARTIST_CLASS_PATH, IMAGE_DIR, BATCH_SIZE
)

model = ArtClassifier(
    num_styles=len(style_class_mapping),
    num_genres=len(genre_class_mapping),
    num_artists=len(artist_class_mapping),
)


def train_one_epoch(data_loader, train_type, optimizer):
    weights = data_loader.dataset.weights
    num_batches = 0
    for batch in data_loader:
        images = batch[0]
        labels = batch[1]
        preds = model(images, train_type, output_probabilities=False)
        loss = F.cross_entropy(preds, labels, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batches += 1
        if num_batches % 10 == 0:
            print(f"Batch {num_batches}, loss = {loss.item()}", flush=True)


for layer in model.children():
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
for epoch in range(NUM_EPOCHS):
    print(f"Start epoch {epoch + 1}", flush=True)
    print(f"Start style training", flush=True)
    train_one_epoch(style_training_loader, "style", optimizer)
    print(f"Start genre training", flush=True)
    train_one_epoch(genre_training_loader, "genre", optimizer)
    print(f"Start artist training", flush=True)
    train_one_epoch(artist_training_loader, "artist", optimizer)
    torch.save(model, MODEL_OUTPUT_NAME)
