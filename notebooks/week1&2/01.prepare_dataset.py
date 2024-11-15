# Databricks notebook source

from pyspark.sql import SparkSession

from sleep_efficiency.config import ProjectConfig
from sleep_efficiency.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Load the sleep efficiency dataset
df = spark.read.csv(
    "/Volumes/dbw_mavencourse_e2emlops_weu_001/sleep_efficiency/data/data.csv", header=True, inferSchema=True
).toPandas()

# COMMAND ----------
data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
# COMMAND ----------
