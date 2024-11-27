import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from sleep_efficiency.config import ProjectConfig
from sleep_efficiency.featurisation import Featurisation


def featurisation():
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    config = ProjectConfig.from_yaml(config_path="../project_config.yml")

    sleep_efficiencies_data = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}")

    featurisation_instance = Featurisation(config, sleep_efficiencies_data, "temperature_features", config.primary_key)

    featurisation_instance.write_feature_table(spark)

    fe = feature_engineering.FeatureEngineeringClient()

    train_data = spark.read.table(
        f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_data"
    ).withColumn("sleep_hours_duration", col["sleep_hours_duration"].cast("double"))

    function_name = f"{config.catalog_name}.{config.schema_name}.calculate_sleep_duration"

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

    training_set = fe.create_training_set(
        df=train_data,
        label=config.target,
        feature_lookups=[
            FeatureLookup(
                table_name=f"{config.catalog_name}.{config.schema_name}.temperature_features",
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
    )

    display(training_set)  # type: ignore # noqa: F821


if __name__ == "__main__":
    featurisation()
