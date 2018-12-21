import os
import glob

data_path = 'data2'

for folder in os.listdir(data_path):
    for file in glob.glob(os.path.join(data_path, folder, '[0-9]*')):
        os.remove(file)
