# Databricks notebook source
# The 2 cells below is only when you are running from databricks UI, because of 'possible' not working locally in VS
# MAGIC %pip install ../mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sleep_efficiency.config import ProjectConfig

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.temperature_features"
function_name = f"{catalog_name}.{schema_name}.calculate_sleep_duration"


# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")


# COMMAND ----------

# s
# df_temperature_nl_2021 = spark.read.csv(
#     "/Volumes/dbw_mavencourse_e2emlops_weu_001/sleep_efficiency/data/weather_netherlands_2021.csv",
#     header=True,
#     inferSchema=True).toPandas()

# df_temperature_with_timestamp = spark.createDataFrame(df_temperature_nl_2021)

# df_temperature_with_timestamp.write.mode("append").saveAsTable(
#             f"{catalog_name}.{schema_name}.temperatures_netherlands_2021")


# COMMAND ----------

# Create or replace the temperature_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.temperature_features
(Id STRING NOT NULL,
 Month INT NOT NULL,
 AverageTemperature INT);
""")

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.temperature_features " "ADD CONSTRAINT month_pk PRIMARY KEY(Month);"
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.temperature_features "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)

# Insert data into the feature table from both train and test sets
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.temperature_features "
    f"SELECT ID, Month_number, Average_temperature FROM {catalog_name}.{schema_name}.temperatures_netherlands_2021"
)


# COMMAND ----------

# Define a function to calculate the sleep duration from a person using the bedtime and the wake up time

# This is a test calculation. The sleep duration is already defined in the original table. However, to test if a function like this can work, here is a example. The orginal dataset shows not correct values for wakeup time and bedtime dates. Therefore the algortihm gives the wrong values. But this illustrates how it can work. To be sure to get the correct value, the algoritm returns the orginal value from the input dataset.
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(bed_time TIMESTAMP, wakeup_time TIMESTAMP, original_sleep_duration DOUBLE )
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
from datetime import datetime
# Calculate the difference and get the absolute value in hours
time_difference = abs(bed_time - wakeup_time)
hours_difference = time_difference.total_seconds() / 3600
# return hours_difference
return original_sleep_duration
$$
""")

# COMMAND ----------

# Load training and test sets
# train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("OverallQual", "GrLivArea", "GarageCars")
# test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Cast YearBuilt to int for the function input
# train_set = train_set.withColumn("age", train_set["age"].cast("int"))
# train_set = train_set.withColumn("bedtime", train_set["bedtime"].cast("timestamp"))
# train_set = train_set.withColumn("wakeup_time", train_set["wakeup_time"].cast("timestamp"))
train_set = train_set.withColumn("sleep_month", train_set["sleep_month"].cast("int"))
test_set = test_set.withColumn("sleep_month", test_set["sleep_month"].cast("int"))

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["AverageTemperature"],
            lookup_key="sleep_month",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="sleep_hours_duration",
            input_bindings={
                "bed_time": "bedtime",
                "wakeup_time": "wakeup_time",
                "original_sleep_duration": "sleep_duration",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

testing_set = fe.create_training_set(
    df=test_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["AverageTemperature"],
            lookup_key="sleep_month",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="sleep_hours_duration",
            input_bindings={
                "bed_time": "bedtime",
                "wakeup_time": "wakeup_time",
                "original_sleep_duration": "sleep_duration",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()
testing_df = testing_set.load_df().toPandas()

# Split features and target
X_train = training_df[num_features + cat_features + ["AverageTemperature"]]
# Don't use sleep_hours_duration, because it's covered in sleep_duration, but was a example to use feature function option
# X_train = training_df[num_features + cat_features + ["sleep_hours_duration"]]

y_train = training_df[target]
X_test = testing_df[num_features + cat_features + ["AverageTemperature"]]
# Don't use sleep_hours_duration, because it's covered in sleep_duration, but was a example to use feature function option
# X_test= testing_df[num_features + cat_features + ["sleep_hours_duration"]]
y_test = testing_df[target]

# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters))])

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name=" sleep-efficiency-fe")
git_sha = "nvt"

with mlflow.start_run(tags={"branch": "week1and2_stanruessink", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe",
    name=f"{catalog_name}.{schema_name}.sleep_efficiencies_model_fe",
)
