library(tidyverse)
library(glue)

#' 分割数据集
#' 
#' @param column_meta Column Meta 数据框
#' @param majority_samples 数量最多的类的采样数量
#' @param random_seed 随机数种子
#' @param train_ratio 训练集比例
#' @param valid_ratio 验证集比例
#' @param identifier 结果文件标识
split_dataset <- function(
    column_meta, majority_samples=NA, random_seed=112358,
    train_ratio=0.7, valid_ratio=0.2, identifier='') {
    
    # Sentive columns meta
    sensitive_columns_meta <- column_meta %>%
        filter(column_is_sensitive == 1) %>%
        filter(column_sensitive_type != '0') %>%
        filter(column_sensitive_type != '1') %>%
        filter(column_sensitive_type != '京东pin') %>%
        filter(column_sensitive_type != '其他')
    
    # Insensitive columns meta
    insensitive_columns_meta <- column_meta %>%
        filter(column_is_sensitive == 0) %>%
        filter(is.na(column_sensitive_type))
    
    set.seed(random_seed)
    
    # Sample if needed
    if (!is.na(majority_samples)) {
        insensitive_columns_meta <- 
            sample_n(insensitive_columns_meta, majority_samples)
    }
    
    # Bind sensitive and insensitive columns
    column_meta <- bind_rows(sensitive_columns_meta, insensitive_columns_meta)
    
    set.seed(random_seed)
    
    # Train and test split
    column_meta_train <- column_meta %>%
        group_by(column_sensitive_type) %>%
        sample_frac(size = train_ratio)
    
    column_meta_test <- setdiff(column_meta, column_meta_train)
    
    # Train and valid split
    column_meta_valid <- column_meta_train %>%
        group_by(column_sensitive_type) %>%
        sample_frac(size = valid_ratio)
    
    column_meta_train <- setdiff(column_meta_train, column_meta_valid)
    
    # Save to files
    write_tsv(column_meta_train,
              glue('../data/column_meta{identifier}_train.tsv'))
    write_tsv(column_meta_valid,
              glue('../data/column_meta{identifier}_valid.tsv'))
    write_tsv(column_meta_test,
              glue('../data/column_meta{identifier}_test.tsv'))
    
    # Stats
    message('Train dataset:')
    print(table(column_meta_train$column_sensitive_type, useNA = 'always'))
    message('Valid dataset:')
    print(table(column_meta_valid$column_sensitive_type, useNA = 'always'))
    message('Test dataset:')
    print(table(column_meta_test$column_sensitive_type, useNA = 'always'))
    message('Cleaned dataset:')
    print(table(column_meta$column_sensitive_type, useNA = 'always'))
}

# Read column meta data
column_meta <- read_tsv('../data/column_meta.tsv',
                        col_names = T, col_types = 'cccicccccccccciiic')

# Demo dataset
split_dataset(column_meta, majority_samples = 3000, identifier = '_demo')

# Full dataset
split_dataset(column_meta)
