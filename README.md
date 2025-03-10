# ArtExtract-test

## Task 1
I wrote a simple CNN + GRU for this task, trained it on ArtGAN training data, and evaluated with val data.
- ```model.py```: the CNN + RNN model.
  - This is a relatively small model due to resource constraints (I don't have GPU). For simplicity all three tasks (style, genre, artist) share the same CNN + RNN, and has separate FC layers for classification.
  - Some ideas to improve:
    - Use pre-trained models like EfficientNet / ResNet50. I've tried this, but even just the inference would take too much time on CPU.
    - Genre and artist may benefit from RNN that can detect brushstrokes or relationships between patches, and style classification relies more on global patterns. We could design separate RNNs or ViT for the three tasks given this difference, but it's good to still share CNN for initial feature extraction.
- ```dataset.py```: for loading training data.
  - Because the ArtGAN data has imbalanced classes, especially for style classification, the data loader maintains a weight vector for the classes used in training. The weight of a class is inversely proportional to the occurrences of the class in training data.
- ```train.py```: for training the model.
- ```output.log```: the training logs, showing losses along the training.
- ```evaluate.ipynb```: a demo for model evaluation
  - While accuracy is reported, it's not a good metric given the imbalanced classes. AUC and confusion matrix are better metrics.
  - The model performance is limited, probably due to the simple model architecture.

## Task 2
I collected paintings and titles from the National Gallery of Art open data, used the latest pre-trained models to extract features, specifically EfficientNet-b7 for images and SentenceBERT all-mpnet-base-v2 for titles. Then I calculated cosine similarity between image features.

Because there's no ground-truth labels of similar painting pairs, I took two evaluation approaches:
- Manually examine clusters of images.
- Use similarity between titles as proxy, based on the assumption that similar paintings may have similar titles.

Files breakdown:
- ```get_image_features.py```: read metadata, download images, and get embeddings for painting images and titles.
  - A total of 3,926 paintings are read, which is the result from inner join between ```objects``` and ```published_images``` tables with ```classification="Painting"```.
- ```evaluate.ipynb```: a demo for evaluation
  - Visualizing t-SNE for image and title embeddings.
  - Example retrieval task by image and title embeddinggs.
  - Siamese network to improve image embeddings using positive/negative pairs obtained from title embeddings.
