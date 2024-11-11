# The 2 cells below is only when you are running from databricks UI, because of 'possible' not working locally in VS
# Databricks notebook source
# MAGIC %pip install mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------
import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
from pyspark.sql import functions as F
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from sleep_efficiency.config import ProjectConfig


# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

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
df_temperature_nl_2021 = spark.read.csv(
    "/Volumes/dbw_mavencourse_e2emlops_weu_001/sleep_efficiency/data/weather_netherlands_2021.csv",
    header=True,
    inferSchema=True).toPandas()

df_temperature_with_timestamp = spark.createDataFrame(df_temperature_nl_2021)

df_temperature_with_timestamp.write.mode("append").saveAsTable(
            f"{catalog_name}.{schema_name}.temperatures_netherlands_2021")


# COMMAND ----------
# Create or replace the temperature_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.temperature_features
(Id STRING NOT NULL,
 Month INT,
 AverageTemperature INT);
""")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.temperature_features "
          "ADD CONSTRAINT month_pk PRIMARY KEY(Id);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.temperature_features "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data into the feature table from both train and test sets
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.temperature_features "
          f"SELECT ID, Month_number, Average_temperature FROM {catalog_name}.{schema_name}.temperatures_netherlands_2021")



# COMMAND ----------
# Define a function to calculate the sleep duration from a person using the bedtime and the wake up time
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(bed_time DATE, wakeup_time DATE )
RETURNS DECIMAL(2,0)
LANGUAGE PYTHON AS
$$
from datetime import datetime
#time_difference = wakeup_time - bed_time
# Convert the difference to hours
#hours = time_difference.total_seconds() / 3600
return 1

$$
""")
# COMMAND ----------
# Load training and test sets
#train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("OverallQual", "GrLivArea", "GarageCars")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Cast YearBuilt to int for the function input
#train_set = train_set.withColumn("YearBuilt", train_set["YearBuilt"].cast("int"))
train_set = train_set.withColumn("id", train_set["id"].cast("string"))

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        # FeatureLookup(
        #    table_name=feature_table_name,
        #    feature_names=["AverageTemperature"],
        #    lookup_key="Id",
        # ),
        FeatureFunction(
            udf_name=function_name,
            output_name="sleep_hours_duration",
            input_bindings={
                "bed_time": "bedtime", 
                "wakeup_time": "wakeup_time"
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate house_age for training and test set
test_set.withColumn(
    "sleep_hours_duration",
    F.round((F.col("wakeup_time").cast("long") - F.col("bedtime").cast("long")) / 3600, 2)  # convert seconds to hours
)

# Split features and target
X_train = training_df[num_features + cat_features + ["sleep_hours_duration"]]
y_train = training_df[target]
X_test = test_set[num_features + cat_features + ["sleep_hours_duration"]]
y_test = test_set[target]

# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters))]
)

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/sleep-efficiency-fe")
git_sha = "nvt"

with mlflow.start_run(tags={"branch": "week1and2_stanruessink",
                            "git_sha": f"{git_sha}"}) as run:
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
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=f"{catalog_name}.{schema_name}.house_prices_model_fe")
    


# COMMAND ----------
