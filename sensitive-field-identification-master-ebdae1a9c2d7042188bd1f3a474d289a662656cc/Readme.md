# 项目简介

项目主要目的是自动化的判断数据仓库接入的数据表中各个字段的 **敏感** 情况。

目前数据平台的解决方案大致如下：

1. 申请开通数据接入平台到业务系统数据库的相关权限。
2. 申请接入人员填写接入申请表，并配置相关字段信息 (包括敏感信息)。
3. 数据仓库人员审核接入申请表，并制定存储策略。
4. 接入数据，对敏感信息进行加密，落库到数据仓库。

目前方案中，在业务人员填写接入申请表时，系统根据字段名称及其注释等相关信息，利用关键词匹配等方法对疑似敏感信息的字段进行了提示，但基于关键词匹配的方案的 **准确率** 和 **覆盖率** 都相对 **较低**。

我们的方案：
对于入库到数据仓库中的数据进行自动敏感信息识别，辅助数据加密策略实施。根据数据的元信息 (例如:表名，表注释，字段名，字段注释等) 和值信息 (即字段存储的数据值)，利用 Wide & Deep 网络构建识别模型。提取传统特征构建 Wide 网络，针对文本特征，利用 Char Embedding + CNN 构建 Deep 网络，模型测试数据的 F1-Score 为 95%+ 
