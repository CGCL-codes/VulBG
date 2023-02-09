import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
def get_accuracy(labels, prediction):    
    cm = confusion_matrix(labels, prediction)
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    def linear_assignment(cost_matrix):    
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]    
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy 

def get_accuracy_with_new_label(labels, prediction):    
    cm = confusion_matrix(labels, prediction)
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    def linear_assignment(cost_matrix):    
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]    
    accuracy = np.trace(cm2) / np.sum(cm2)  # 对角线/总的

    label_dict_in = {}
    label_dict_out = {}
    label_change = {}
    for i,label, in enumerate(unique_labels(labels, prediction)):
        label_dict_in[label] = i
        label_dict_out[i] = label
    for i,ind, in enumerate(js):
        label_change[i] = ind
    labels_new = [label_dict_out[label_change[label_dict_in[i]]] for i in labels]
    return accuracy , labels_new

def get_scores(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    precision, recall, f_score, _ = sklearn.metrics.precision_recall_fscore_support(labels, predictions,average='binary', pos_label=1)
    return f_score, precision, recall, accuracy

def get_MCM_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    precision, recall, f_score, true_sum, MCM = sklearn.metrics.precision_recall_fscore_support(labels, predictions,average='binary', pos_label=1)
    return f_score, precision, recall, accuracy
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0] 
    tp = MCM[:, 1, 1] 
    fpr_array = fp / (fp + tn)
    fnr_array = fn / (tp + fn)
    f1_array = 2 * tp / (2 * tp + fp + fn)
    sum_array = fn + tp
    M_fpr = fpr_array.mean()
    M_fnr = fnr_array.mean()
    M_f1 = f1_array.mean()
    W_fpr = (fpr_array * sum_array).sum() / sum( sum_array )
    W_fnr = (fnr_array * sum_array).sum() / sum( sum_array )
    W_f1 = (f1_array * sum_array).sum() / sum( sum_array )
    return {
        "M_fpr": M_fpr,
        "M_fnr": M_fnr,
        "M_f1" : M_f1,
        "W_fpr": W_fpr,
        "W_fnr": W_fnr,
        "W_f1" : W_f1,
        "ACC"  : accuracy
    }

def get_MCM_score_with_new_label(labels, predictions):
    accuracy , labels_new = get_accuracy_with_new_label(labels, predictions)
    precision, recall, f_score, true_sum, MCM = precision_recall_fscore_support(labels_new, predictions,average='macro')
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0] 
    tp = MCM[:, 1, 1] 
    fpr_array = fp / (fp + tn)
    fnr_array = fn / (tp + fn)
    f1_array = 2 * tp / (2 * tp + fp + fn)
    sum_array = fn + tp
    M_fpr = fpr_array.mean()
    M_fnr = fnr_array.mean()
    M_f1 = f1_array.mean()
    W_fpr = (fpr_array * sum_array).sum() / sum( sum_array )
    W_fnr = (fnr_array * sum_array).sum() / sum( sum_array )
    W_f1 = (f1_array * sum_array).sum() / sum( sum_array )
    return {
        "M_fpr": M_fpr,
        "M_fnr": M_fnr,
        "M_f1" : M_f1,
        "W_fpr": W_fpr,
        "W_fnr": W_fnr,
        "W_f1" : W_f1,
        "ACC"  : accuracy
    }

