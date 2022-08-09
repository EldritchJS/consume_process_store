import json
import requests

with open('./ukr_clusters.json', 'r') as f:
    data = json.load(f)
    # data = {'69': ['/path/image/one', '/path/image/two']}
    r = requests.put('http://127.0.0.1:8080/results', json=data)
    print(r)
