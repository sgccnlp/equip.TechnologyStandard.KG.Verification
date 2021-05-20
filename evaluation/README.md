## 环境配置

python3.6

安装rouge

```shell 
pip install rouge
```


## 使用手册

启动服务

```shell script
python3 app.py
```

采用post方法调用。具体接口参见`data/data/qa/hyp.json`和`data/data/ie/hyps.json`.
每行为1个请求json。具体例子如下：

```python
import requests

url = "http://127.0.0.1:5000/ie"
fp = open('data/data/qa/hyp.json')
line = fp.readline()
r = requests.post(url, data=line.encode('utf8'))
print(r.text)
```

## 最后

欢迎大家一起捉虫