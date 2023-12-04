# 敏感字段识别 API

## 接口定义

敏感字段识别 API 采用 HTTP POST 请求形式进行调用，请求体为 JSON 格式字符串，请求体所有字段均为必须 (值可以为空)。

### JSON 请求定义

一个 JSON 请求的示例

```json
{
    "database_uri": "hbase-cds-nn01.pekdc1.jdfin.local:2181,hbase-cds-nn02.pekdc1.jdfin.local:2181,hbase-cds-nn03.pekdc1.jdfin.local:2181",
    "database_port": "2181",
    "database_name": "",
    "database_comment": "",
    "database_type": "hbase",
    "database_is_shard": "0",
    "table_name": "cds_pledge_acct_convert",
    "table_comment": "质押账户转换结果表",
    "table_theme": "风控",
    "column_name": "d:cust_id",
    "column_comment": "申请人身份证号",
    "column_type": "STRING",
    "column_is_primary_key": "0",
    "column_allow_null": "1",
    "column_value": ["441424199804296078", "441424199804296078", "441424199804296078"]
}
```

其中字段说明如下：

| 字段                  | 说明               | 备注         |
| --------------------- | ------------------ | ------------ |
| database_uri          | 数据库 URI         |              |
| database_port         | 数据库接口         |              |
| database_name         | 数据库名称         |              |
| database_type         | 数据库类型         |              |
| database_is_shard     | 数据库是否分库分表 | 0: 否，1: 是 |
| table_name            | 表名称             |              |
| table_comment         | 表注释             |              |
| table_theme           | 表主题             |              |
| column_name           | 字段名称           |              |
| column_comment        | 字段注释           |              |
| column_type           | 字段类型           |              |
| column_is_primary_key | 字段是否是主键     | 0: 否，1: 是 |
| column_allow_null     | 字段是否允许为空   | 0: 否，1: 是 |
| column_value          | 字段值             | 数组         |

### 请求地址

- 生产环境：http://sfi.daat.jdfin.local/WideDeepCharCNN
- 预发环境：http://10.221.30.127:9999/WideDeepCharCNN
- 测试环境：http://172.24.28.118:9999/WideDeepCharCNN

## 接口调用

### 结果 JSON 定义

一个结果 JSON 示例如下：

```json
{
    "status": 200,
    "error_msg": "",
    "label": 5,
    "raw_label": "身份证",
    "probability": 1,
    "predictions": [5, 5, 5]
}
```

其中字段说明如下：

| 字段        | 说明     | 备注                                             |
| ----------- | -------- | ------------------------------------------------ |
| status      | 状态码值 | 200: 成功，301: 入参字段缺失                     |
| error_msg   | 错误信息 |                                                  |
| label       | 标签     | 标签 ID                                          |
| raw_label   | 原始标签 | 标签可读值                                       |
| probability | 概率值   | 可信度                                           |
| predictions | 预测值   | 与入参 column_value 等长的数组，每个样本的预测值 |

当前版本 label 和 raw_label 对应关系为：

| raw_label | 卡号 | 固定电话 | 地址 | 姓名 | 手机 | 身份证 | 邮箱 | 非敏感 |
| --------- | ---- | -------- | ---- | ---- | ---- | ------ | ---- | ------ |
| label     | 0    | 1        | 2    | 3    | 4    | 5      | 6    | 7      |

### 结果 JSON 理解

在敏感字段识别中，我们更看中的是覆盖率，因此在使用过程中可以通过控制 probability 的阈值来确定最终结果，即将该阈值设置的越小覆盖率越高。

## 接口使用

### 注意事项

- 入参的 column_value 建议取该字段不为空的 n 个不同的值会效果更好。

### 后续事宜

- 根据新定义的敏感字段类型对训练数据打标后重新模型用于最终生产模型。