#!/usr/bin/env bash

# Run SQLs (on 70/71 servers)

## Run all
nohup ls -1 *.sql | awk '{split($1, filename, "."); print filename[1]}' | xargs -i -n 1 -P 100 sh -c "/soft/hive/bin/hive -i /export/home/dmpuser/fanyeliang/udf/hive.config -f {}.sql > {}.tsv 2>{}.log" &

## Run with specific lines
nohup ls -1 *.sql | awk '{split($1, filename, "."); print filename[1]}' | head -n 1000 | tail -n +0 | xargs -i -n 1 -P 100 sh -c "hive -i /soft/hive_conf/productnew/init/hive.config -f {}.sql > {}.tsv 2>{}.log" &
nohup ls -1 *.sql | awk '{split($1, filename, "."); print filename[1]}' | head -n 2000 | tail -n +1000 | xargs -i -n 1 -P 100 sh -c "hive -i /soft/hive_conf/productnew/init/hive.config -f {}.sql > {}.tsv 2>{}.log" &

# Check Errors

## Check errors count
ls -1 *.log | xargs grep -H -c 'FAILED' | grep -v 0$ | cut -d':' -f1 | wc -l

## Check error tables
ls -1 *.log | xargs grep -H -c 'FAILED' | grep -v 0$ | cut -d':' -f1

## Copy errors tables info
ls -1 *.log | xargs grep -H -c 'FAILED' | grep -v 0$ | cut -d':' -f1 | xargs -i -n 1 sh -c "cp {} ../errors"

# Check Results

## Check results count
ls -1 *.tsv | wc -l

## Copy results
cp *.tsv ../tsvs
