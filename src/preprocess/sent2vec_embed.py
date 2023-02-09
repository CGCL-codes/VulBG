import pickle
import numpy as np
import torch

def load_embedding(final_data, sent2vec_data):
    idx = 0
    for data in final_data:
        vec_sent2vec = []
        for s in sent2vec_data[idx]:
            ret = np.zeros_like(s[0])
            for sent in s:
                ret += sent
            ret /= len(s)
            vec_sent2vec.append(ret)
        data['slices_vec'] = torch.tensor(vec_sent2vec, dtype = torch.float32)
    
if __name__ == "__main__":
    final_data = pickle.load(open("/root/data/vulbg/final_data_ffmpeg_qemu.pkl", "rb"))
    sent2vec_data = pickle.load(open("/root/data/vulbg/s2v_embs_fq_768.pkl", "rb"))
    load_embedding(final_data, sent2vec_data)
    print(final_data[0]['slices_vec'])
    print(final_data[0]['slices_vec'].shape)
    with open("/root/data/vulbg/final_data_ffmpeg_qemu_s2v_768.pkl", "wb") as f:
        pickle.dump(final_data, f)
