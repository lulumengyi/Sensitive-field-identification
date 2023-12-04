#!/usr/bin/env bash

# 配置信息
app_name="sensitive_field_identification"
python_version="3.5"
servers_dir="/export/servers"
python_dir=${servers_dir}/python_${python_version}_${app_name}

shell_dir="/export/Shell"
conda_deploy_shell=${shell_dir}/deploy-conda.sh

# 校验 Conda 部署脚本
if [ -f ${conda_deploy_shell} ]; then
    echo "Conda 部署脚本已存在：${conda_deploy_shell}"
else
    echo "正在下载 Conda 部署脚本至：${conda_deploy_shell}"
    curl http://repos.jd.com/conda/scripts/deploy-conda.sh > ${conda_deploy_shell}
fi

# 安装 Python 环境
if [ -d ${python_dir} ]; then
    echo "Python 环境已存在：${python_dir}"
else
    echo "正在安装 Python 环境至：${python_dir}"
    bash ${conda_deploy_shell} ${python_dir} ${python_version}
fi

# 激活 Python 环境
source ${python_dir}/setenv.sh

# 安装相关扩展包
pip install docopt==0.6.2 \
  flask==1.0.2 \
  flask_restful==0.3.6 \
  scipy==1.1.0 \
  pandas==0.23.4 \
  tableprint==0.8.0 \
  tensorflow==1.10.0 \
  scikit-learn==0.19.1 \
  jsonschema==2.6.0 \
  zhon==1.1.5 \
  wordsegment==1.3.1 \
  stanford-corenlp==3.8.0 \
  tqdm==4.25.0 \
  logzero==1.5.0 \
  gunicorn==19.9.0 \
  gevent==1.3.6
