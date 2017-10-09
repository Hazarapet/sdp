import json

with open('mean.json', 'r') as outfile:
    mean = json.load(outfile)
    print mean