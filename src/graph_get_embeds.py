import pickle
import random 
import os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
import sys

def derandom():
    seed = 202203
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def do_embed(edges_fname, output_fname, have_weight = False):
    assert(os.path.exists(edges_fname))
    cmdline = "/root/data/ase2022/src/snap/examples/node2vec/node2vec -i:%s -o:%s -e:2"
    if have_weight:
        cmdline += "-w"

    ret = os.system(cmdline % (edges_fname, output_fname))
    print("node2vec returns %d" % ret)
    return ret


def load_graph(n, final_data, embed_fname):
    node_vecs = {}
    f = open(embed_fname, "r").read()
    f = f.split("\n")
    for line in f[1:]:
        if not line:
            continue
        line = line.split(" ")
        node_id = int(line[0])
        vec = [float(i) for i in line[1:]]
        node_vecs[node_id] = np.array(vec)#, dtype="float32")
        
    func_idx_begin = n
    for func in final_data:
        if func_idx_begin not in node_vecs:
            func["graph_vec"] = np.zeros(128)
            print(func_idx_begin)
            print("zero!")
        else:
            func["graph_vec"] = node_vecs[func_idx_begin]
        func_idx_begin += 1


if __name__ == "__main__":
    args = sys.argv[1:]
    dataset_path, edge_file, have_weight, n_cluster = args

    n_cluster = int(n_cluster)
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    ret = do_embed(edge_file, "tmp_embed", have_weight)

    if ret:
        print("Error occured in node embedding!")
        exit(0)

    load_graph(n_cluster, dataset, "tmp_embed")

    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    
    


    
