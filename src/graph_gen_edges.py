import pickle
import random 
import os
import numpy as np
import sys
import torch
from sklearn.cluster import MiniBatchKMeans

# 聚类中心数
N_CLUSTERS=1140
final_data = None

def derandom():
    seed = 202203
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_kmeans(n, inputs, output):
    kmeans = MiniBatchKMeans(n_clusters=n, random_state=0, verbose=1, max_iter=300, batch_size=3000)
    for i in range(0, len(inputs), 3000):
        kmeans.partial_fit(inputs[i:i+3000])
    with open(output, "wb") as ff:
        pickle.dump(kmeans, ff)
    return kmeans

def generate_edges(n, final_data, cluster_model, have_weight = False):
    edges = []
    func_idx_begin = n
    for func in final_data:
        tmp_vec = [np.array(i) * 1000 for i in func["slices_vec"]]
        tmp_labs = cluster_model.predict(tmp_vec)

        for i in range(len(tmp_vec)):
            lab = tmp_labs[i]
            d = np.linalg.norm(cluster_model.cluster_centers_[lab] - (tmp_vec[i]))
            if have_weight:
                edges.append([func_idx_begin, lab, 10000/(d+1)])
            else:
                edges.append([func_idx_begin, lab])
        func_idx_begin += 1
    return edges

def save_edges(edges, fname, have_weight = False):
    f = open(fname, "w")
    for i in edges:
        if have_weight:
            f.write(str(i[0]) + " " + str(i[1]) +" " + str(i[2]) +"\n")
        else:
            f.write(str(i[0]) + " " + str(i[1]) + "\n")
            
    f.close()
    print(f"Parameters: cluster centers:{N_CLUSTERS}, have weight:{have_weight}")
    print(f"Edges saved to {fname}.")


if __name__ == "__main__":
    print(f"Usage {__file__} dataset.pkl use_weight output_name [nclusters]")

    dataset = sys.argv[1]
    dataset = pickle.load(open(dataset, "rb"))

    use_weight = sys.argv[2]
    if int(use_weight) != 0:
        use_weight = True
    else:
        use_weight = False

    output_name = sys.argv[3]

    if len(sys.argv) >= 5:
        N_CLUSTERS = int(sys.argv[4])

    inputs = []
    for i in dataset:
        for slice_vec in i["slices_vec"]:
            inputs.append(np.array(slice_vec) * 1000)

    kmeans = train_kmeans(N_CLUSTERS, inputs, output_name+"_kmeans.pkl")
    edges = generate_edges(N_CLUSTERS, dataset, kmeans, use_weight)
    save_edges(edges, output_name+".edges", use_weight)
    print("Done")



    
