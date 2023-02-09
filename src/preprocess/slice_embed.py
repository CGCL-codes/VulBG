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

def embed_one_batch(codes, cpu=False, max_length = 256):
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
    for i in data:
        codes.append(i["s"])
        files.append(i["file"])
    length = len(files)
    return codes, files, length


def embed_all(data, all_dataloader, batch_size):
    all_embeds = {}
    for item in data:
        all_embeds[item["file"]] = []
    torch.cuda.empty_cache()
    gc.collect()
    for i, codes in enumerate(all_dataloader):    
        print(i)
        codes, files, length = codes
        output = embed_one_batch(codes, False)[1]
        gc.collect()
        output = output.cpu().detach().numpy()
        for j in range(length):
            all_embeds[files[j]].append(output[j])
    return all_embeds
            


if __name__ == "__main__":
    dataset = sys.argv[1]
    # max_length = sys.argv[2]

    if dataset == "reveal":
        path = "/root/data/VulBG/dataset/devign_dataset.pkl"
    elif dataset == "devign":
        path = "/root/data/VulBG/dataset/reveal_dataset.pkl"
    else:
        path = dataset

    dataset = pickle.load(open(path, "rb"))

    slices_dataset = []
    for i in dataset:
        for s in i["slices"][:20]:
            slices_dataset.append({"file": i["file"], "s": s})

    batch_size = 12
    dataloader = DataLoader(slices_dataset, batch_size = batch_size, shuffle = False, collate_fn=mycollate_fn_for_embed)

    print(len(slices_dataset))
    embeds = embed_all(dataset, dataloader, batch_size)

    for i in dataset:
        i["slices_vec"] = embeds[i["file"]]
        
    f = open(path, "wb")
    pickle.dump(dataset, f)
    f.close()

