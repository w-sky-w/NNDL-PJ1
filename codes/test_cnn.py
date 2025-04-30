import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

conv_params = [
    (1, 8, 3, 1, 0, 1e-4),
    (8, 16, 3, 1, 0, 1e-4)
]

fc_params = [
    (16*24*24, 128, 1e-4),
    (128, 10, 1e-4)
]

cnn_model = nn.models.Model_CNN(conv_params=conv_params, fc_params=fc_params, act_func='ReLU')
cnn_model.load_model(r'.\saved_models\best_model.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs[:1000]
test_labs = test_labs[:1000]

test_imgs = test_imgs / test_imgs.max()

test_imgs = test_imgs.reshape(-1, 1, 28, 28)

logits = cnn_model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))