#!/usr/bin/env bash

# 程序目录
cd `dirname $0`
bin_dir=`pwd`

cd ..
deploy_dir=`pwd`

# 配置信息
app_name="sensitive_field_identification"
python_version="3.5"
servers_dir="/export/servers"
logs_dir="/export/Logs"
python_dir=${servers_dir}/python_${python_version}_${app_name}
app_log_dir=${logs_dir}/${app_name}
app_log_file=${logs_dir}/${app_name}/${app_name}_server.log
app_file="server"

function check_process_exit() {
    pids=`ps -ef | grep python | grep ${app_name} | awk '{print $2}'`
    if [ -z "${pids}" ]; then
        echo "错误: 服务 ${app_name} 未启动！"
        return 0
    fi
    return 1
}

check_process_exit
if_exist=$?

if [ "${if_exist}" = "1" ]; then
    echo "正在停止 ${app_name} 服务，PIDS: ${pids} ..."
    for pid in ${pids}; do
        kill ${pid} > /dev/null 2>&1
    done

    max_wait=10
    if [ "$1" = "force" ]; then
        if [ "$2" != "" ]; then
            max_wait=2
        fi
    fi
    if [ ${max_wait} -lt 5 ]; then
        max_wait=5
    fi

    counter=0
    while [ ${counter} -le ${max_wait} ]; do
        sleep 1
        ((counter=counter+1))

        for pid in ${pids}; do
            pid_exist=`ps -f -p ${pid} | grep python`
            if [ -n "${pid_exist}" ]; then
                if [ "$1" = "force" -a ${counter} -ge ${max_wait} ]; then
                    echo "正在强制停止服务 ${app_name}，PID: ${pid} ..."
                    kill -9 ${pid}
                fi
                break
            else
                ((counter=max_wait+1))
            fi
        done
    done

    echo "服务 ${app_name} 停止成功！"
    echo "PIDS: ${pids}"
fi
