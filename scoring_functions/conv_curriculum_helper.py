from IPython import embed
import pandas as pd
import random
import numpy as np

def load_dataset(dataset_path):
    dataset = []
    query_neg_count = 0
    with open(dataset_path,'r') as f:
        for line in f:
            query = line.split("\t")[1:-1]
            doc = line.split("\t")[-1][0:-2]
            rel = line.split("\t")[0]
            if rel == '1':
                query_neg_count = 0
            else:
                query_neg_count+=1
            if query_neg_count <= 1:
                dataset.append((rel, query, doc))
    return dataset

def save_curriculum(curriculum, output_path):
    with open(output_path, 'w') as f:
        for label in curriculum:
            f.write(str(label)+"\n")


def ap(y_true, y_pred, rel_threshold=0):
    def _to_list(x):
        if isinstance(x, list):
            return x
        return [x]
    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = [a for a in zip(y_true, y_pred)]
    import random
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / ( j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s