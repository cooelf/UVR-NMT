import numpy as np

image_embedding_file = "features_resnet50/train-resnet50-avgpool.npy"
embeding_weights = np.load(image_embedding_file)