# Databricks notebook source
# The 2 cells below is only when you are running from databricks UI, because of 'possible' not working locally in VS
# MAGIC %pip install ../mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC dbutils.library.restartPython()
# COMMAND ----------
from sleep_efficiency.utils import generate_synthetic_data
from pyspark.sql import SparkSession
from sleep_efficiency.config import ProjectConfig

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

try:
    table_path = f"{config.catalog_name}.{config.schema_name}.raw_{config.use_case_name}"
    full_data = spark.read.table(table_path)
except Exception as e:
    raise RuntimeError(f"Failed to read table {table_path}: {str(e)}")

# COMMAND ----------
existing_ids = full_data.select(config.primary_key).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------
synthetic_df = generate_synthetic_data(config, full_data)

# COMMAND ----------
synthetic_df.write.format("delta").mode("append").saveAsTable(f"{config.catalog_name}.{config.schema_name}.raw_{config.use_case_name}")

try:
    table_path = f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}"
    synthetic_df.write.format("delta").mode("append").saveAsTable(table_path)
    print(f"Successfully appended synthetic data to {table_path}")
except Exception as e:
    raise RuntimeError(f"Failed to write synthetic data: {str(e)}")