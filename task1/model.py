import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtClassifier(nn.Module):
    def __init__(self, num_styles, num_genres, num_artists):
        super(ArtClassifier, self).__init__()
        self.cnn_dim = 64
        self.rnn_dim = 32

        # CNN backbone to extract features
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # RNN on top of CNN features
        self.rnn = nn.GRU(
            self.cnn_dim, self.rnn_dim, batch_first=True, bidirectional=True
        )

        # FC layers for classification
        self.style_fc = nn.Linear(self.rnn_dim * 2, num_styles)
        self.genre_fc = nn.Linear(self.rnn_dim * 2, num_genres)
        self.artist_fc = nn.Linear(self.rnn_dim * 2, num_artists)

    def forward(self, x, return_type="", output_probabilities=True):
        cnn_features = self.pool(F.relu(self.conv1(x)))  # (batch, 16, 64, 64)
        cnn_features = self.pool(
            F.relu(self.conv2(cnn_features))
        )  # (batch, 32, 16, 16)
        cnn_features = self.pool(F.relu(self.conv3(cnn_features)))  # (batch, 64, 4, 4)
        cnn_features = torch.flatten(cnn_features, start_dim=2)  # (batch, 64, 16)
        cnn_features = cnn_features.permute(0, 2, 1)  # (batch, 16, 64)

        rnn_out, _ = self.rnn(cnn_features)  # (batch, 16, rnn_dim * 2)
        rnn_out_forward = rnn_out[:, -1, : self.rnn_dim]  # Forward RNN output
        rnn_out_backward = rnn_out[:, 0, self.rnn_dim :]  # Backward RNN output
        rnn_out = torch.cat(
            (rnn_out_forward, rnn_out_backward), dim=1
        )  # (batch, rnn_dim * 2)

        if return_type == "style":
            return (
                F.softmax(self.style_fc(rnn_out), dim=1)
                if output_probabilities
                else self.style_fc(rnn_out)
            )
        if return_type == "genre":
            return (
                F.softmax(self.genre_fc(rnn_out), dim=1)
                if output_probabilities
                else self.genre_fc(rnn_out)
            )
        if return_type == "artist":
            return (
                F.softmax(self.artist_fc(rnn_out), dim=1)
                if output_probabilities
                else self.artist_fc(rnn_out)
            )

        style_pred = (
            F.softmax(self.style_fc(rnn_out), dim=1)
            if output_probabilities
            else self.style_fc(rnn_out)
        )
        genre_pred = (
            F.softmax(self.genre_fc(rnn_out), dim=1)
            if output_probabilities
            else self.genre_fc(rnn_out)
        )
        artist_pred = (
            F.softmax(self.artist_fc(rnn_out), dim=1)
            if output_probabilities
            else self.artist_fc(rnn_out)
        )

        return style_pred, genre_pred, artist_pred
