"""This script deploys the newest model predictions with Feature Serving, in case the model has been updated. The model version is taken from the evaluate_model task, the model with this version is loaded and then predictions are done. These predictions are written to the Feature Table, and the online table is synced such that the new predictions are present and can be retrieved by sending a request to the serving endpoint."""

import mlflow
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from sleep_efficiency.config import ProjectConfig
from sleep_efficiency.featurisation import Featurisation


def deploy_new_model_predictions():
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    model_version = dbutils.jobs.taskValues.get(taskKey="evaluate_model", key="model_version", debugValue=0)  # type: ignore # noqa: F821

    config = ProjectConfig.from_yaml(config_path="../project_config.yml")
    workspace = WorkspaceClient()

    train_data = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_set")
    test_data = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_test_set")

    predict = mlflow.pyfunc.spark_udf(
        spark, f"models:/{config.catalog_name}.{config.schema_name}.{config.use_case_name}_model_fe/{model_version}"
    )

    columns_to_serve = [config.target, "id", "sleep_duration", "awakenings", "sleep_month"]

    full_df = train_data.unionByName(test_data)

    predictions_df = full_df.withColumn("prediction", predict(*full_df.columns)).select("prediction", *columns_to_serve)

    featurisation_instance = Featurisation(config, predictions_df, "preds", config.target)

    featurisation_instance.write_feature_table(spark)

    workspace.pipelines.start_update(pipeline_id="c8011128-eb5f-4555-9704-0f6a0d47a336", full_refresh=True)


if __name__ == "__main__":
    deploy_new_model_predictions()
