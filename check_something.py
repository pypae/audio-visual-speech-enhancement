
from tensorflow.python.keras.models import Model, load_model
import os
import numpy as np

base_path = '/cs/labs/peleg/asaph/playground/avse/'
model_path = os.path.join(base_path, 'cache/models/classifier6/disc.h5py')
test_set_path = os.path.join(base_path, 'cache/preprocessed/obama-libri-lips-test/data.npz')
train_set_path = os.path.join(base_path, 'cache/preprocessed/obama-libri-lips-train/data.npz')

model = load_model(model_path)

test_set = np.load(test_set_path)
train_set = np.load(train_set_path)

print 'loading'

mixed_test = np.swapaxes(test_set['mixed_spectrograms'][:10], 1, 2)[:,:1000,:]
clean_test = np.swapaxes(test_set['source_spectrograms'][:10], 1, 2)[:,:1000,:]
mixed_train = np.swapaxes(train_set['mixed_spectrograms'][:10], 1, 2)[:,:1000,:]
clean_train = np.swapaxes(train_set['source_spectrograms'][:10], 1, 2)[:,:1000,:]

mixed_test = mixed_test.reshape(100,100,80)
clean_test = clean_test.reshape(100,100,80)
mixed_train = mixed_train.reshape(100,100,80)
clean_train = clean_train.reshape(100,100,80)

print('shape', mixed_train.shape)

# pred_mixed_test = model.predict(mixed_test)
# pred_clean_test = model.predict(clean_test)
# pred_mixed_train = model.predict(mixed_train)
# pred_clean_train = model.predict(clean_train)


# print(pred_mixed_test)
# print(pred_clean_test)
# print(pred_mixed_train)
# print(pred_clean_train)

loss_mixed_test = model.evaluate(mixed_test, np.zeros([mixed_test.shape[0], 1]))
loss_clean_test = model.evaluate(clean_test, np.ones([mixed_test.shape[0], 1]))
loss_mixed_train = model.evaluate(mixed_train, np.zeros([mixed_test.shape[0], 1]))
loss_clean_train = model.evaluate(clean_train, np.ones([mixed_test.shape[0], 1]))

print(model.metrics_names)
print('mixed test:', loss_mixed_test)
print('clean test:', loss_clean_test)
print('mixed train:', loss_mixed_train)
print('clean train:', loss_clean_train)

