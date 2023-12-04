-- 获取所有 STAGE 表 (该表作为基础表)
SELECT
table_name
FROM
nebulae_table
WHERE
database_name = 'stage' AND
table_status = '1' AND
table_name NOT REGEXP '.+_bak_.*' AND
table_name NOT REGEXP '.+_bak[0-9]+' AND
table_name NOT REGEXP '.+_bak$' AND
table_name NOT REGEXP '.+_bk$' AND
table_name NOT REGEXP '.+_bk[0-9]+' AND
table_name NOT REGEXP '.+[0-9]{8}.*' AND
table_name NOT REGEXP '^temp_' AND
table_name NOT REGEXP '^tmp_' AND
table_name NOT REGEXP '^s__' AND
table_name NOT REGEXP '_test$'
ORDER BY
table_name
;

-- 获取所有 STAGE 表是否包含 DT 字段
SELECT
DISTINCT(table_name) AS table_name
FROM
nebulae_column
WHERE
column_name = 'dt' AND
database_name = 'stage' AND
column_status = '1'
ORDER BY
table_name
;

-- 获取所有 STAGE 表所有字段信息
SELECT
stage_table_name AS table_name ,
stage_column_name AS column_name ,
column_is_sensitive
FROM
sensitive_column
WHERE
stage_table_name REGEXP '^[0-9a-zA-Z_]+$'
ORDER BY
table_name ,
column_name
;