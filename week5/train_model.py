"""
This script trains a LightGBM model for sleep efficiency prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses both numerical and categorical features, including a custom calculated house age feature.
"""

import argparse

import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sleep_efficiency.config import ProjectConfig
from sleep_efficiency.efficiency_model import EfficiencyModel
from sleep_efficiency.utils import check_repo_info, get_error_metrics


def train_model():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job_run_id",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    spark = SparkSession.builder.getOrCreate()

    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    mlflow.set_experiment(experiment_name=f"/{config.user_dir_path}/{config.use_case_name}")

    args = parser.parse_args()
    job_run_id = args.job_run_id

    model_instance = EfficiencyModel(config, spark)

    preprocessing_stages = model_instance.create_preprocessing_stages()

    git_branch, git_sha = check_repo_info(
        f"/Workspace/{config.user_dir_path}/{config.git_repo}",
        dbutils,  # type: ignore # noqa: F821
    )

    train_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_set")
    test_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_test_set")

    pipeline = Pipeline(
        stages=preprocessing_stages
        + [
            RandomForestRegressor(
                n_estimators=config.parameters.n_estimators,
                max_depth=config.parameters.max_depth,
                random_state=42,
            ),
        ]
    )

    with mlflow.start_run(
        tags={"git_sha": git_sha, "branch": git_branch},
    ) as run:
        run_id = run.info.run_id

        model = pipeline.fit(train_set)

        predictions = model.transform(test_set)

        error_metrics = get_error_metrics(predictions)
        # Print metrics
        print(f"Mean Squared Error: {error_metrics['mse']}")
        print(f"Mean Absolute Error: {error_metrics['mae']}")
        print(f"R2 Score: {error_metrics['r2']}")
        # Log model parameters, metrics, and model
        mlflow.log_param("model_type", "LightGBM with preprocessing")
        mlflow.log_metric("mse", error_metrics["mse"])
        mlflow.log_metric("mae", error_metrics["mae"])
        mlflow.log_metric("r2_score", error_metrics["r2"])
        signature = infer_signature(model_input=train_set, model_output=predictions.select("prediction"))
        # Log model with feature engineering
        mlflow.spark.log_model(
            model=model,
            floavor=mlflow.sklearn,
            artifact_path="lightgbm-pipeline-model-fe",
            signature=signature,
            training_set=train_set,
        )

    model_uri = f"runs:/{run_id}/lightgbm-pipeline-model-fe"

    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_client.get_registered_model("users.srues2@xs4all.nl.sleep_efficiencies_model_fe")
        print(
            "Model already exists, this run will be evaluated in the next task and it is registered as a new model version in case it performs better than the current version"
        )
    except mlflow.exceptions.RestException as e:
        print(
            f"This is the first time the training task is run on this workspace and the model will be registered: {str(e)}"
        )
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_model_fe",
            tags={"git_sha": git_sha, "branch": git_branch, "job_run_id": job_run_id},
        )
        print("New model registered with version:", model_version.version)

    dbutils.jobs.taskValues.set(key="git_sha", value=git_sha)  # type: ignore # noqa: F821
    dbutils.jobs.taskValues.set(key="job_run_id", value=job_run_id)  # type: ignore # noqa: F821
    dbutils.jobs.taskValues.set(key="model_uri", value=model_uri)  # type: ignore # noqa: F821


if __name__ == "__main__":
    train_model()
