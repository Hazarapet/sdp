import json
import numpy as np

train_mean = train_mean.tolist()
with open('mean.json', 'w') as outfile:
    json.dump(train_mean, outfile)

