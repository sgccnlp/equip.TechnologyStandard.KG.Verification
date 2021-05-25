from pathlib import Path
import requests
import json


url = "http://127.0.0.1:5000/qa"


root = Path("tests/data/qa")

for filename in root.glob("qa_test_extr_raw.json"):
    print(filename)
    with open(filename) as fp:
        s = fp.read()
        data = json.dumps(json.loads(s), ensure_ascii=False).encode('utf8')
    r = requests.post(url, data=data)
    print(r.text)
    print(r.elapsed)
