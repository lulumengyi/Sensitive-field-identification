#!/usr/bin/env bash

# 程序目录
cd `dirname $0`
bin_dir=`pwd`

cd ..
deploy_dir=`pwd`
deploy_data_dir=${deploy_dir}/data
deploy_data_wide_deep_char_cnn_checkpoint_dir=${deploy_data_dir}/wide_deep_char_cnn_checkpoint

# 配置信息
app_name="sensitive_field_identification"
python_version="3.5"
servers_dir="/export/servers"
logs_dir="/export/Logs"
python_dir=${servers_dir}/python_${python_version}_${app_name}
app_log_dir=${logs_dir}/${app_name}
app_log_file=${logs_dir}/${app_name}/${app_name}_server.log
app_file="server"

# 激活 Python 环境
source ${python_dir}/setenv.sh

# 检查是否已经启动
pids=`ps -ef | grep python | grep ${app_name} | awk '{print $2}'`
if [ -n "${pids}" ]; then
    echo "错误: 服务 ${app_name} 已启动！"
    echo "PIDS: ${pids}"
    exit 1
fi

# 创建日志目录
if [ ! -d "${app_log_dir}" ]; then
    mkdir ${app_log_dir}
    chmod a+x+w+r ${app_log_dir}
fi

# 创建数据目录
if [ ! -d "${deploy_data_dir}" ]; then
    mkdir ${deploy_data_dir}
fi

if [ ! -d "${deploy_data_wide_deep_char_cnn_checkpoint_dir}" ]; then
    mkdir ${deploy_data_wide_deep_char_cnn_checkpoint_dir}
fi

# 下载数据文件
curl -o ${deploy_data_dir}/char_dict.pickle "http://storage.jd.local/sensitive-field-identification/data/char_dict.pickle?Expires=3683165219&AccessKey=6g7fvvgzrUTg7zns&Signature=2pUTcLmD1%2FL4aJUc26NDKefJbaU%3D"
curl -o ${deploy_data_wide_deep_char_cnn_checkpoint_dir}/checkpoint "http://storage.jd.local/sensitive-field-identification/data/wide_deep_char_cnn_checkpoint/checkpoint?Expires=3683165404&AccessKey=6g7fvvgzrUTg7zns&Signature=d4FNBF2z8eRaLPksUU4kwCmgMUk%3D"
curl -o ${deploy_data_wide_deep_char_cnn_checkpoint_dir}/wide_deep_char_cnn_1533287951.ckpt-6000.data-00000-of-00001 "http://storage.jd.local/sensitive-field-identification/data/wide_deep_char_cnn_checkpoint/wide_deep_char_cnn_1533287951.ckpt-6000.data-00000-of-00001?Expires=3683165428&AccessKey=6g7fvvgzrUTg7zns&Signature=wCasgdKAJlSk6ND0Q7GhqwDYuyE%3D"
curl -o ${deploy_data_wide_deep_char_cnn_checkpoint_dir}/wide_deep_char_cnn_1533287951.ckpt-6000.index "http://storage.jd.local/sensitive-field-identification/data/wide_deep_char_cnn_checkpoint/wide_deep_char_cnn_1533287951.ckpt-6000.index?Expires=3683165457&AccessKey=6g7fvvgzrUTg7zns&Signature=K42m%2FG9W%2FZfsvl9Jcr8MUmsrAS0%3D"
curl -o ${deploy_data_wide_deep_char_cnn_checkpoint_dir}/wide_deep_char_cnn_1533287951.ckpt-6000.meta "http://storage.jd.local/sensitive-field-identification/data/wide_deep_char_cnn_checkpoint/wide_deep_char_cnn_1533287951.ckpt-6000.meta?Expires=3683165475&AccessKey=6g7fvvgzrUTg7zns&Signature=VMGampDG%2F%2FmYD%2FGbzNIsm2GgsZg%3D"

# 启动服务
echo "正在启动 ${app_name} 服务 ..."
if [ -f "${app_log_file}" ]; then
    nohup gunicorn -w 4 -b 0.0.0.0:9999 ${app_file}:app >> ${app_log_file} 2>&1 &
else
    nohup gunicorn -w 4 -b 0.0.0.0:9999 ${app_file}:app > ${app_log_file} 2>&1 &
fi

# 等待程序启动
sleep 10

# 校验程序是否已经启动
counter=0
max_check_times=60
check_times=0
while [ ${counter} -lt 1 ]; do
    counter=`ps -ef | grep python | grep "${app_name}" |awk '{print $2}' | wc -l`

    if [ ${counter} -gt 0 ]; then
        break
    fi

    if [ ${check_times} -eq ${max_check_times} ]; then
        echo "服务在 ${max_check_times} 秒内仍未启动！"
        exit 1
    fi

    check_times=`expr ${check_times} + 1`
    sleep 1
done

echo "服务启动成功！"
pids=`ps -ef | grep python | grep "${app_name}" |awk '{print $2}'`
echo "PIDS: ${pids}"
