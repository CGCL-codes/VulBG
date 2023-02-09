
# collate_fn returns baseline_input, label, bg_input, length of current batch
def gen_collate_fn(baseline_column):
    def collate_fn(data):
        baseline_input = []
        label_data = []
        bg_input = []
        for i in data:
            baseline_input.append(i[baseline_column])
            label_data.append(i["vul"])
            bg_input.append(i["graph_vec"])
        label_data = torch.tensor(label_data)
        bg_input = np.array(bg_input)
        bg_input = torch.tensor(bg_input, dtype=torch.float32)
        baseline_input = torch.tensor(baseline_input)
        data_length = len(label_data)
        return baseline_input, label_data, bg_input,data_length
    return collate_fn
