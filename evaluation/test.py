import requests

fp = open('data/qa/hyps.json')
line = fp.readline()
url = 'http://127.0.0.1:5000/qa'
r = requests.post(url, data = line.encode('utf8'))
print(r.text)