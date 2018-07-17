
from tensorflow.python.keras.models import Model, load_model
import os

base_path = '/cs/labs/peleg/asaph/playground/avse/'
model_path = os.path.join(base_path, 'cache/models/classifier2/dics.h5py')
test_set = os.path.join(base_path, 'cache/preprocessed/obama-libri-lips-test/data.npz')
train_set = os.path.join(base_path, 'cache/preprocessed/obama-libri-lips-train/data.npz')

model = load_model(model_path)

