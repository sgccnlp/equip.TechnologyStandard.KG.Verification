#   共性问题：

**1.检索结果中的空格、缩进、中英文括号(()和（）)、中英文句号（。和.）等与原条款内容有所不同，是否会影响得分？当答案涉及多个段落，“换行/回车”是否参与计算？**

```
标点符号、空格、换行符不在考察范围，不影响得分。
```

**2.参赛队伍是否需要提交可运行的代码？**
```
评测中提交可运行的容器即可，验证之后可删除。
```

**3.智能问答赛道中的“数字类”答案是否需要带单位，当单位遗漏时如何评分？如果需要带单位，单位需要做归一化吗，还是直接原文中的单位即可？**
```
如果问题不带有单位，最终答案须包含单位。如果问题带单位，答案可以选择填或不填单位，但一旦选择填写且填错，则会得0分。建议在答案中采用原文中的单位形式。
```

**4.任务中不同维度评价指标的权重在最终评分的任务书中没有说明，是否可公布？评分脚本是否可以公布？**
```
承办方在评测之前公布评测脚本，并将权重设置可变参量，权重的具体设置将在评测现场公布。
```

**5.表格、图片、公式是否在考察范围内？**
```
问答赛道可能涉及对表格及其内容，检索赛道仅涉及表格标题，无需反馈其内容，表格号及表格标题作为条款摘要输出；图片、公式不在评测范围。
```

**6.在问答赛道中，参赛队伍是否需要自行判断问题的答案类型？**
```
智能问答赛道中“答案类型”将连同题目一同给出。
```

**7.本次评测工作所涉及样本资料如何获取？**
```
建议各参与单位根据“关于开展电网设备技术标准精准检索与智能问答验证工作的通知（附件2：技术验证相关技术标准文档清单）”自行获取。
```

**8.运行时间等性能是否作为考察要求，还是说只看评测的准确率等指标？**
```
本次评测不考察性能指标。参与单位的模型运行时间在本次评测中也会进行统计和展示，但不计入评分指标。
```

**9.主办方提供的最大资源是多少？**
```
应参与单位要求，进一步增加各单位计算资源配置为：64G内存、2张RTX6000显卡、1T存储、16核CPU。
```

**10.检索赛道的答案中是否需包含条款的标号和标题？什么是条款？对于条款摘要的抽取准确度在评测体系中占多少权重？**
```
需包含条款号，建议包含标题，评测中仅根据条款号进行评分，条款标题仅用于展示。摘要的抽取和匹配在本次评测中仅占较小比重，权重在现场评价时公布。
```

**11.检索结果根据什么规则进行排序，优先级如何约束？**
```answer
考察检索算法给出的答案列表中第一个正确答案（能够与任一参考答案匹配，即为正确答案）的排序位置（在算法所提供答案列表中的位置），不考察正确答案之间的相对顺序。具体见“电网设备技术标准精准检索与智能问答评测任务书”。
```

**12.问答赛道中，当抽取类的答案中缺少一个或少数关键词时如何评分？**
```answer
做出如下调整，以降低难度。对于抽取类答案，首先判断专家预先指定的关键词（短语、短句）中有几个被涵盖，以给出“分数折扣”。如抽取类题目的参考答案为String0，专家指定其中的a、b、c和d作为关键词。假设算法给出的答案为String1，而String1仅涵盖a、c和d三个关键词，因此分数折扣为3/4，最后得分为3/4 * Rouge（String1, String2），其中Rouge的计算方法具体见评测任务书。类似的，若String1涵盖a、b、c和d三个关键词，“分数折扣”为4/4；若String1涵盖零个关键词，“分数折扣”为0/4，以此类推。判断类、统计类答案的核心判据部分也是按照同样的方法进行评分。
```

**13.当标准中表格出现跨页，例如“表1（续）”，如果检索目标为“表1（续）”中对应内容，返回条款摘要应为“表1”还是“表1（续）”？**
```answer
统一用“表1”作为返回条款号
```


**14.条款的定义**
```answer
条款是文档内有标号的细分层次，由高到低、由概括到具体，如正文的1、1.1、1.1.1、1.1.1.1和附录的A、A.1、A.1.1等。一般条款含条款号、标题和内容，但部分条款也存在无标题的情况。关于“条款”的定义详见（GBT 1.1-2020 标准化工作导则 第1部分：标准化文件的结构和起草规则 第7.3条）。精准检索目的是找到最（尽量）完整反映检索词语义的条款中最（尽量）低层级条款。

```

**15.Q/GDW 1103-2012 750kV系统用油浸式变压器技术规范》与《Q/GDW 1103-2015 750kV主变压器技术规范》考察哪个？**
```answer
考察《Q/GDW 1103-2015 750kV主变压器技术规范》，不考察《Q/GDW 1103-2012 750kV系统用油浸式变压器技术规范》，详见附件《技术验证相关技术标准文档清单.docx》。
```
# 各家反馈答复详见
http://verification.rises.tech/notify

#   容器部署规则
为避免厂商比赛时镜像重名，导致覆盖，要求各厂商镜像命名格式如下：
```text
    equipment-编号-镜像名
```
    编号由组织方给定
    镜像名自行命名
厂商容器名格式如下：
```buildoutcfg
    equipment-编号-容器名
```

#   评测脚本
评测脚本工程在 /evaluation内
详见 [评测脚本说明](evaluation/README.md)
