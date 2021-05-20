# 电网设备技术标准精准检索与智能问答评测脚本说明
## 环境配置
python3.6

安装 rouge

```shell 
    pip install rouge
```


## 使用手册

启动服务 bash start.sh
``` bash
  python3 app.py
```

采用post方法调用。具体接口参见

[检索接口](data/ir/hyps.json)

[问答接口](data/qa/hyps.json)

每行为1个请求json。

具体示例如下：

检索赛道评价
```python
import requests

url = "http://127.0.0.1:5000/ir"
fp = open('data/ir/hyps.json')
line = fp.readline()
r = requests.post(url, data=line.encode('utf8'))
print(r.text)
```
问答赛道评价
```python
import requests

url = "http://127.0.0.1:5000/qa"
fp = open('data/qa/hyps.json')
line = fp.readline()
r = requests.post(url, data=line.encode('utf8'))
print(r.text)
```

## 备注
有疑问请联系开发者　或直接在 issue里提问

@宋博川 [songbochuan@geiri.sgcc.com.cn](songbochuan@geiri.sgcc.com.cn)

@张强 [zhangqiang1@geiri.sgcc.com.cn](zhangqiang1@geiri.sgcc.com.cn)

欢迎大家一起捉虫