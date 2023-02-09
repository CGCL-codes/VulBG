from http.client import ImproperConnectionState
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
import sklearn
import gc
import pandas as pd
import random
import os
import sys

seed = 202203
random.seed(seed)
os.environ['PYHTONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model = model.to(device)

def embed_one_batch(codes, cpu=False, max_length = 512):
    inputs = tokenizer(
        codes, padding=True, truncation=True, return_tensors="pt", max_length=max_length
    )
    if cpu:
        inputs.to("cpu")
    else:
        inputs.to(device)
    return model(**inputs)

def mycollate_fn_for_embed(data):
    files = []
    codes = []
    label_data = []
    for i in data:
        label_data.append(i["vul"])
        codes.append(i["code"])
        files.append(i["file"])
    label_data = torch.tensor(label_data)
    length = len(files)
    return codes, label_data, files, length


def embed_all(data, all_dataloader, batch_size):
    all_embeds = {}
    torch.cuda.empty_cache()
    gc.collect()
    for i, codes_with_labels in enumerate(all_dataloader):    
        print(i)
        codes, labels, files, length = codes_with_labels
        output = embed_one_batch(codes, False)[1]
        gc.collect()
        output = output.cpu().detach().numpy()
        for j in range(length):
            all_embeds[files[j]] = output[j]
    return all_embeds
            


if __name__ == "__main__":
    dataset = sys.argv[1]
    max_length = sys.argv[2]

    if dataset == "reveal":
        path = "/root/data/VulBG/dataset/devign_dataset.pkl"
    elif dataset == "devign":
        path = "/root/data/VulBG/dataset/reveal_dataset.pkl"
    else:
        path = dataset

    dataset = pickle.load(open(path, "rb"))
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn=mycollate_fn_for_embed)

    embeds = embed_all(dataset, dataloader, batch_size)

    for i in dataset:
        i["codebert"] = embeds[i["file"]]
        
    f = open(path, "wb")
    pickle.dump(dataset, f)
    f.close()

