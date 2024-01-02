import json
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='',ckpt=True):
    tries = 15
    error = None
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if ckpt:
                torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)


def load_config(opt, file_path):
    with open(file_path, "r") as f:
        opt.__dict__ = json.load(f)

def cosine_similarity_matrix(a,b):
    if 'numpy' in str(type(a)):
        # return cosine_similarity(a,b)

        return np.matmul(a, np.matrix.transpose(b))
    else:
        return a @ b.t()





