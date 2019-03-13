import os

import pickle
import matplotlib.pyplot as plt


directory = os.path.join(os.path.dirname(__file__), 'raw')
filename_base = 'server_power_Data_Size_20190312191446_4188_models_final.pkl'

with open(os.path.join(directory, filename_base), 'rb') as fid:
    data = pickle.load(fid)


plt.figure()
plt.plot(data['Y_odq_predict'])
plt.plot(data['Y_reservoir_predict'])
plt.plot(data['Y_test'])
plt.legend(('ODQ', 'Reservoir', 'Test'))

plt.figure()
plt.plot(data['history_odq'].history['loss'])
plt.plot(data['history_reservoir'].history['loss'])
plt.legend(('ODQ', 'Reservoir'))

plt.figure()
plt.plot(data['history_odq'].history['val_loss'])
plt.plot(data['history_reservoir'].history['val_loss'])
plt.legend(('ODQ', 'Reservoir'))

plt.show()
print('')