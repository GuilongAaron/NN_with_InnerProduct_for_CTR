import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

categorical_sizes = [24, 7, 4, 7, 10, 20, 100]
embedding_size = 7
embedding_sizes = [embedding_size for _ in range(len(categorical_sizes))]
embedding_dim = sum(embedding_sizes)  # 60  #21
product_layer_dim = 25
hidden_dim = 25  # sum(embedding_sizes) 20
hidden_dim2 = 25
# num_heads = 4  # 3
batch_size = 16
epochs = 2_000
learning_rate = 1e-5 # do not change
dropout_rate = 0.5
use_user_id = False


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):

        super().__init__()
        """
        dim_self = embedding_size
        dim_ref = optional, default = embedding_size
        """
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)  # layer_num * layer num
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias) # layer_num * 2 * layer num
        self.project = nn.Linear(dim_self, dim_self)  # layer_num * layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        b, c = x.shape # batch, num_length, embedding size, if batch_first = True
        _, d = x.shape # void,  num_length, embedding_size, if batch_first = True

        queries = self.to_queries(x).reshape(b, self.num_heads, c // self.num_heads)
        # b 2 h dh --> expand to 4D [batch, (clip_length + pre_length), 2, num_heads, embedding_size // num_heads]
        keys_values = self.to_keys_values(x).reshape(b, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]

        attention = torch.bmm(queries, keys.transpose(1, 2)) * self.scale

        attention = attention.softmax(dim=2)

        out = torch.matmul(attention, values).reshape(b, c)
        out = self.project(out)
        return out, attention


class CtrPredictionModel(nn.Module):
    def __init__(self, num_categories_list=categorical_sizes, embedding_sizes=embedding_sizes, hidden_dim=hidden_dim):
        super(CtrPredictionModel, self).__init__()        
        self.num_categories = num_categories_list
        self.embedding_sizes = embedding_sizes
        self.hidden_dim = hidden_dim

        self.fc0 = nn.Linear(embedding_sizes, embedding_sizes)
        self.batch_norm0_0 = nn.BatchNorm1d(embedding_sizes)
        self.first_order_weight = nn.Parameter(torch.randn((product_layer_dim, 1)), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(product_layer_dim), requires_grad=True)
        self.second_order_weight = nn.Parameter(torch.randn((product_layer_dim, self.embedding_sizes)), requires_grad=True)

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
        # x = self.fc0(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.batch_norm0_0(x)
        concat_row = x.unsqueeze(1)  # b, 1, 8
        first_order = torch.matmul(self.first_order_weight, concat_row)  # 20, 1  X  b, 1, 8 -> # b, 20, 8
        first_order = torch.sum(first_order, dim = 2)  # b, 20 
        temp = torch.matmul(concat_row.transpose(1, 2), concat_row)  # b, 8, 1  X b, 1, 8 --> b, 8, 8 
        temp = temp.squeeze(-1) 
        second_order = torch.matmul(self.second_order_weight, temp)  # 20, 8  X b, 8, 8  --> b, 20, 8
        second_order = torch.sum(second_order, dim = 2)    # b, 20    
        product_layer = first_order + second_order + self.bias  # b, 20
        x = product_layer

        x = self.batch_norm0(x)
        x = self.fc1(x)  # linear
        x = self.relu(x)        
        x = self.dropout(x)

        resnet = self.batch_norm(x)        
        x = self.res(resnet)  # linear
        x = self.relu(x)
        x = self.dropout(x)

        x = self.batch_norm2(x)
        # x = x + resnet          # resnet addition
        x = self.fc2(x)  # linear 
        x = self.sigmoid(x)

        return x

def categorizing(total):

    # categorical_columns = ["browser", "os"]
    # frequency / counts
    category = "user_id"
    temp = data_recat_freq(total, [f"{category}"], 1)
    total = encode_time_f(total, temp, [f"{category}"])
    category = "campaign_id"
    temp = data_recat_freq(total, [f"{category}"], total[f"{category}"].nunique())
    total = encode_time_f(total, temp, [f"{category}"])
    category = "main_category"
    temp = data_recat_freq(total, [f"{category}"], total[f"{category}"].nunique())
    total = encode_time_f(total, temp, [f"{category}"])
    category = "sub_category"
    temp = data_recat_freq(total, [f"{category}"], total["main_category"].nunique())
    total = encode_time_f(total, temp, [f"{category}"])
    category = "time"
    temp = data_recat_freq(total, [f"{category}"], 5)
    total = encode_time_f(total, temp, [f"{category}"])
    category = "browser"
    temp = data_recat_freq(total, [f"{category}"], 4)
    total = encode_time_f(total, temp, [f"{category}"])
    category = "os"
    temp = data_recat_freq(total, [f"{category}"], 4)
    total = encode_time_f(total, temp, [f"{category}"])

    # CTR, rate
    category = "dayweek"
    temp = data_recat_rate(total, [f"{category}"], 3)  
    total = encode_time(total, temp, [f"{category}"])
    category = "user_id" # category changed to _recat    
    temp = data_recat_rate(total, [category], 3)
    total = encode_time(total, temp, [f"{category}"])
    category = "campaign_id" # category changed to _recat
    temp = data_recat_rate(total, [category], total[f"{category}"].nunique())
    total = encode_time(total, temp, [f"{category}"])
    category = "main_category" # category changed to _recat
    temp = data_recat_rate(total, [f"{category}"], total[f"{category}"].nunique())  
    total = encode_time(total, temp, [f"{category}"])
    category = "sub_category"
    temp = data_recat_rate(total, [f"{category}"], total["main_category"].nunique())  
    total = encode_time(total, temp, [f"{category}"])
    category = "time"
    temp = data_recat_rate(total, [f"{category}"], 5)  
    total = encode_time(total, temp, [f"{category}"])
    # categorical_columns = ["browser", "os"]
    category = "browser"
    temp = data_recat_rate(total, [f"{category}"], 4)  
    total = encode_time(total, temp, [f"{category}"])
    category = "os"
    temp = data_recat_rate(total, [f"{category}"], 3)  
    total = encode_time(total, temp, [f"{category}"])

    return total


def read_files(paths):
    """
    batch reading files.
    """
    data = pd.DataFrame()
    if paths:
        for file in paths:
            df = pd.read_csv(file)
            data = pd.concat([data, df], ignore_index=True)
        data["datetime"] = pd.to_datetime(data["datetime"])

    return data


def data_process(data_p, cate):
    """
    process and drop irrelevant columns
    """
    data_p = data_p.drop("device", axis=1)
    data_p = pd.merge(data_p, cate, left_on="article_id", right_on="article_id", how="left")

    data_p = data_p.drop("article_id", axis=1)
    data_p["time"] = data_p["datetime"].apply(lambda x: x.hour)
    data_p["dayweek"] = data_p["datetime"].dt.day_name()
    data_p["date"] = data_p["datetime"].dt.date
    data_p = data_p.drop("datetime", axis=1)

    return data_p


def data_recat_freq(total, categorical_columns, n_clusters):
    # distribution is very different, use frequency
    frequency_tables = []
    for column in categorical_columns:
        frequency = total[column].value_counts()
        frequency = frequency.reset_index()
        frequency = frequency.rename(columns={"index": f"{column}", f"{column}": f"{column}_counts"})
        frequency_tables.append(frequency)
    
    encode_tables = []
    for table, category in zip(frequency_tables, categorical_columns):
        time_dict = table.iloc[:, 1]
        time_dict = time_dict.values.reshape(-1, 1)
        time_dict = np.nan_to_num(time_dict, nan = 0)
        if len(time_dict) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            time_dict = kmeans.fit_predict(time_dict)
        table[f"{category}_recat_freq"] = time_dict
        encode_tables.append(table)

    return encode_tables

def data_recat_rate(data_p, categorical_columns, n_clusters):
    # distribution is almost same, then use rate
    pivot_dfs = []
    for column in categorical_columns:
        percentage_df = data_p.groupby(['click', column]).size().reset_index(name='count')
        percentage_df = percentage_df.groupby([column, 'click']).agg({"count": "sum"})
        percentage_df = percentage_df.groupby(level=0, group_keys=False).apply(
            lambda x: 100 * x / float(x.sum() + 0.001)).reset_index()
        pivot_df = percentage_df.pivot(index=column, columns="click", values='count')
        pivot_dfs.append(pivot_df)

    encode_tables = []
    for time_table, category in zip(pivot_dfs, categorical_columns):
        time_dict = time_table[1]
        time_dict = time_dict.values.reshape(-1, 1)
        time_dict = np.nan_to_num(time_dict, nan = 0)
        if len(time_dict) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            time_dict = kmeans.fit_predict(time_dict)
        time_table[f"{category}_recat"] = time_dict
        encode_tables.append(time_table)

    return encode_tables


def encode_time(data_p, encode_tables, categorical_columns):

    for time_table, category in zip(encode_tables, categorical_columns):
        data_p = pd.merge(data_p, time_table[f"{category}_recat"], left_on=category, right_on=category, how="left")
        # data_p = data_p.drop(f"{category}", axis=1)

    return data_p


def encode_time_f(data_p, encode_tables, categorical_columns):
    for time_table, category in zip(encode_tables, categorical_columns):
        data_p = pd.merge(data_p, time_table[[f"{category}", f"{category}_recat_freq"]], left_on=category, right_on=category, how="left")
        # data_p = data_p.drop(f"{category}", axis=1)

    return data_p


def user_encode(data_p, frequency_tables):

    data_p = pd.merge(data_p, frequency_tables[0][["user_id", "user_id_is_elite"]], left_on="user_id", right_on="user_id", how="left")
    data_p = data_p.drop("user_id", axis=1)

    return data_p

    # to be removed before submission
    

# forming statistical data
def find_freq(data_p, categorical_columns):
    # categorical_columns = ["dayweek", "user_id", "campaign_id", "main_category", "sub_category", "time", "browser", "os"]
    frequency_tables = []
    columns_to_log = ["user_id", "os", "browser"]
    columns_to_reciprocal = []  # 
    columns_to_linear_reciprocal= []  # 
    # columns_to_add_reciprocal = ["os", "browser"]
    for column in categorical_columns:   

        if column == "dayweek":
            freq = data_p.groupby(['dayweek', 'date']).size().reset_index(name='count')
            count = freq.groupby("dayweek").size().reset_index(name = f"{column}_count")
            frequency = data_p.groupby("dayweek").size().reset_index(name=f"{column}_sum")
            frequency = pd.merge(frequency, count, left_on = "dayweek", right_on="dayweek", how = "left")
            frequency[f"{column}_counts"] = frequency[f"{column}_sum"] / frequency[f"{column}_count"]
            frequency = frequency.drop(f"{column}_sum", axis = 1)
            frequency = frequency.drop(f"{column}_count", axis = 1)
            frequency_tables.append(frequency)
        else:
            frequency = data_p[column].value_counts()
            frequency = frequency.reset_index()
            frequency = frequency.rename(columns={"index": f"{column}", f"{column}": f"{column}_counts"})
            if column in columns_to_log:
                frequency[f"{column}_counts"] = np.log(frequency[f"{column}_counts"])
            elif column in columns_to_linear_reciprocal:
                frequency[f"{column}_counts"] = -frequency[f"{column}_counts"] - 1 / frequency[f"{column}_counts"]
            elif column in columns_to_reciprocal:
                frequency[f"{column}_counts"] = -np.log(frequency[f"{column}_counts"])
            # if column in columns_to_add_reciprocal:
            #     frequency[f"{column}_counts_recip"] = 1 / frequency[f"{column}_counts"]
            frequency_tables.append(frequency)
    return frequency_tables


def find_freq_recip(data_p, categorical_columns):
    # categorical_columns = ["browser", "os"]
    frequency_tables = []
    for column in categorical_columns:  
        frequency = data_p[column].value_counts()
        frequency = frequency.reset_index()
        frequency = frequency.rename(columns={"index": f"{column}", f"{column}": f"{column}_counts"})
        frequency[f"{column}_counts_recip"] = 1 / frequency[f"{column}_counts"]
        frequency_tables.append(frequency)

    return frequency_tables

def find_ctr(data_p, categorical_columns):
    
    pivot_dfs = []
    columns_to_square = []
    columns_to_recip = []
    columns_to_log = ["user_id", "browser", "os"]  # move left 
    # columns_to_log = []  # move left 
    # columns_to_square = ["dayweek", "ti`me"]  # move right
    columns_to_sqrt = []  # move left
    columns_to_cubic = []  # move left
    columns_to_exp = []
    # columns_to_recip = ["browser", "os"]  # recp expo

    for column in categorical_columns:
        percentage_df = data_p.groupby(['click', column]).size().reset_index(name=f"{column}_rate")
        percentage_df = percentage_df.groupby([column, 'click']).agg({f"{column}_rate": "sum"})
        percentage_df = percentage_df.groupby(level=0, group_keys=False).apply(
            lambda x: 100 * x / float(x.sum() + 0.001)).reset_index()
    #         pivot_df = percentage_df.pivot(index=column, columns="click", values='count')
    #         pivot_df = pivot_df.rename(columns = {"0": f"{column}_f", "1":f"{column}_rate"})
    #         pivot_dfs.append(pivot_df)
        if column in columns_to_log:
            percentage_df[f"{column}_rate"] = np.log(percentage_df[f"{column}_rate"])
        elif column in columns_to_square:
            percentage_df[f"{column}_rate"] = np.square(percentage_df[f"{column}_rate"])
        elif column in columns_to_sqrt:
            percentage_df[f"{column}_rate"] = np.sqrt(percentage_df[f"{column}_rate"])
        elif column in columns_to_exp:
            percentage_df[f"{column}_rate"] = np.exp(percentage_df[f"{column}_rate"])
        elif column in columns_to_recip:
            percentage_df[f"{column}_rate"] = np.exp(1 / (percentage_df[f"{column}_rate"]))
    #     print(f"{column} is going to have error?")
        pivot_dfs.append(percentage_df[percentage_df["click"] == 1])
        
    return pivot_dfs


def main():


    return 0

if __name__ == '__main__':
    sys.exit(main())
