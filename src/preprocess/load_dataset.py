import pickle
import os
import json



def load_one_slice(fname):
    with open(fname) as f:
        slice_json = json.load(f)
        if slice_json:
            slices = slice_json[0].split("$$$$$$$$")
            return slices


def load_slices(final_data, slice_path):
    for item in final_data:
        slice_name = slice_path + '/' + item["file"] + ".json"
        try:
            slices = load_one_slice(slice_name)
            item["slices"] = slices

        except FileNotFoundError:
            continue

def load_src(final_data, src_path, is_vul = 0):
    for file in os.listdir(src_path):
        if file[-2:] != ".c":
            continue
        with open(src_path + "/" + file) as f:
            content = f.read()
            item = {"file": file, "code": content, "vul": is_vul, "label": is_vul, "slices":[]}
            final_data.append(item)


if __name__ == "__main__":
    vul_path = "/root/data/0-src/vul_new/"
    novul_path = "/root/data/0-src/novul_new/"

    vul_slice_path = "/root/data/0-src/vul_slice/"
    novul_slice_path = "/root/data/0-src/novul_slice/"

    final_data1 = []
    load_src(final_data1, vul_path, 1)
    load_slices(final_data1, vul_slice_path)

    final_data0 = []
    load_src(final_data0, novul_path, 0)
    load_slices(final_data0, novul_slice_path)

    _final_data = final_data0 + final_data1
    print(len(_final_data))
    final_data = []
    
    for i in _final_data:
        if i["slices"]:
            final_data.append(i)

    print(len(final_data))

    with open("./msr_final_data.pkl", "wb") as f:
        pickle.dump(final_data, f)








