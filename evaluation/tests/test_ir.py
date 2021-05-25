from pathlib import Path
import requests
import json


url = "http://127.0.0.1:5000/ir"


root = Path("tests/data/ir")

for filename in root.glob("*"):
    print(filename)
    with open(filename) as fp:
        s = fp.read()
        data = json.dumps(json.loads(s), ensure_ascii=False).encode('utf8')
    r = requests.post(url, data=data)
    print(r.text)
    print(r.elapsed)
