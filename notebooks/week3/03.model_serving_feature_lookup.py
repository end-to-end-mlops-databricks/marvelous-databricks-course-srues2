# Databricks notebook source
# MAGIC %pip install ../housing_price-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Online Table for sleep efficiency features
# MAGIC We already created sleep_features table as feature look up table.

# COMMAND ----------

import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from sleep_efficiency.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

online_table_name = f"{catalog_name}.{schema_name}.sleep_features_online"
spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.sleep_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------


config = ProjectConfig.from_yaml(config_path="/Volumes/mlops_dev/sleep_efficiencies/data/project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create endpoint

# COMMAND ----------

workspace.serving_endpoints.create(
    name="sleep-efficiencies-model-serving-fe",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.sleep_efficiencies_model_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Excluding "OverallQual", "GrLivArea", "GarageCars" because they will be taken from feature look up
required_columns = [
    "age",
    "sleep_duration",
    "rem_sleep_percentage",
    "deep_sleep_percentage",
    "light_sleep_percentage",
    "awakenings",
    "caffeine_consumption",
    "alcohol_consumption",
    "exercise_frequency",
    "gender",
    "smoking_status",
    "bedtime",
    "wakeup_time",
    "id"
]

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

train_set.dtypes

# COMMAND ----------

dataframe_records[0]

# COMMAND ----------
start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/sleep-efficiencies-model-serving-fe/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

sleep_features = spark.table(f"{catalog_name}.{schema_name}.sleep_features").toPandas()

# COMMAND ----------

sleep_features.dtypes