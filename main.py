import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
from tqdm import tqdm


BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "data")

# just to simply the code to put the default parameters here.
# in formal codes, it should be in the function default settings.
categorical_sizes = [24, 7, 4, 7, 10, 20, 100]
embedding = 7
embedding_sizes = [embedding for _ in range(len(categorical_sizes))]
embedding_dim = sum(embedding_sizes)  # 60  #21
product_layer_dim = 25
hidden_dim = 25  # sum(embedding_sizes) 20
hidden_dim2 = 25
num_heads = 4  # 3
batch_size = 16
epochs = 200
learning_rate = 1e-5
dropout_rate = 0.5
use_user_id = False


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads=num_heads, bias=True, dropout=0.):
        super().__init__()
        """
        dim_self = embedding
        dim_ref = optional, default = embedding
        """
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)  # layer_num * layer num
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias) # layer_num * 2 * layer num
        self.project = nn.Linear(dim_self, dim_self)  # layer_num * layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c = x.shape # batch, num_length, embedding, if batch_first = True
        _, d = x.shape # void,  num_length, embedding, if batch_first = True

        queries = self.to_queries(x).reshape(b, self.num_heads, c // self.num_heads)
        # b 2 h dh --> expand to 4D [batch, (clip_length + pre_length), 2, num_heads, embedding // num_heads]
        keys_values = self.to_keys_values(x).reshape(b, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        attention = attention.softmax(dim=2)

        out = torch.matmul(attention, values).reshape(b, c)
        out = self.project(out)
        return out, attention


class CtrPredictionModel(nn.Module):
    def __init__(self, categories_sizes=categorical_sizes, embedding_sizes=embedding_sizes, hidden_dim=hidden_dim,
                 product_layer_dim=product_layer_dim):
        super(CtrPredictionModel, self).__init__()        
        self.categories_sizes = categories_sizes
        self.embedding_sizes = embedding_sizes
        self.hidden_dim = hidden_dim

        self.fc0 = nn.Linear(embedding_sizes, embedding_sizes)
        self.batch_norm0_0 = nn.BatchNorm1d(embedding_sizes)
        self.first_order_weight = nn.Parameter(torch.randn((product_layer_dim, 1)), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(product_layer_dim), requires_grad=True)
        self.second_order_weight = nn.Parameter(torch.randn((product_layer_dim, self.embedding_sizes)),
                                                requires_grad=True)

        # case 1
        self.batch_norm0 = nn.BatchNorm1d(product_layer_dim)
        self.fc1 = nn.Linear(product_layer_dim, hidden_dim)

        # case 2
        # self.fc1 = nn.Linear(self.embedding_sizes, hidden_dim)
        # self.attn = MultiHeadAttention(hidden_dim, hidden_dim, num_heads)

        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.res = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        concat_row = x.unsqueeze(1)  # b, 1, input_size
        first_order = torch.matmul(self.first_order_weight, concat_row)  # e.g.20, 1  X  b, 1, 8 -> # b, 20, 8
        first_order = torch.sum(first_order, dim=2)  # b, 20
        temp = torch.matmul(concat_row.transpose(1, 2), concat_row)  # b, 8, 1  X b, 1, 8 --> b, 8, 8 
        temp = temp.squeeze(-1)  # incaseof embedding layer is used. embeddings to be squeezed.
        second_order = torch.matmul(self.second_order_weight, temp)  # 20, 8  X b, 8, 8  --> b, 20, 8
        second_order = torch.sum(second_order, dim=2)    # b, 20
        product_layer = first_order + second_order + self.bias  # b, 20
        x = product_layer

        x = self.batch_norm0(x)
        x = self.fc1(x)  # linear 1
        x = self.relu(x)        
        x = self.dropout(x)

        x = self.batch_norm(x)
        x = self.res(x)  # linear 2
        x = self.relu(x)
        x = self.dropout(x)

        x = self.batch_norm2(x)
        x = self.fc2(x)  # output layer
        x = self.sigmoid(x)

        return x


def read_files(paths) -> np.array:
    """
    batch reading files.
    """
    data = pd.read_csv(paths)

    return data.to_numpy()


def random_sample(X_train, y_target):

    positive_indices = np.where(y_target == 1)[0]
    negative_indices = np.where(y_target == 0)[0]
    positive_size = len(positive_indices)

    oversample_ratio = len(positive_indices) / len(negative_indices)

    # initial random state
    oversampled_indices = np.random.choice(negative_indices, size=int(0.5 * positive_size / oversample_ratio), replace=False)
    undersampled_indices = np.random.choice(positive_indices, size=int(0.5 * positive_size / oversample_ratio), replace=True)

    balanced_indices = np.concatenate([oversampled_indices, undersampled_indices])
    np.random.shuffle(balanced_indices)

    X_train_s = X_train[balanced_indices]
    y_target_s = y_target[balanced_indices]

    X_train_tensor = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_target_s, dtype=torch.float32)

    return X_train_tensor, y_train_tensor

def main():
    X_train = read_files(os.path.join(INPUT_DIR, "x_train.csv"))
    y_target = read_files(os.path.join(INPUT_DIR, "y_target.csv"))

    X_train = X_train[:, 1:]  # remove record id
    y_target = y_target[:, 1:]  # remove record id
    y_target = np.ravel(y_target)

    X_train_tensor, y_train_tensor = random_sample(X_train, y_target)

    # setting for NN model
    embedding_sizes = X_train_tensor.shape[1]
    model = CtrPredictionModel(categories_sizes=categorical_sizes,
                               embedding_sizes=embedding_sizes,
                               hidden_dim=hidden_dim)
    criterion = nnf.binary_cross_entropy_with_logits
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adjust the learning rate as needed
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0)  #  compare with SGD optimizer

    for epoch in tqdm(range(epochs)):

        model.train()
        for i in range(0, X_train_tensor.size(0), batch_size):
            # Get mini-batch
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 1))  # Assuming y_train_tensor is a column vector

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # prepare for next loop re-randomize index
        X_train_tensor, y_train_tensor = random_sample(X_train, y_target)

        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(INPUT_DIR, f"train_model_{epoch:05d}.pt"),
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
