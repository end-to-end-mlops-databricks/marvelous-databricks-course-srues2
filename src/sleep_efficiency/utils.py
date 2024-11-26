import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sleep_efficiency.config import ProjectConfig
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg as mean
from pyspark.sql.functions import stddev
from pyspark.sql.functions import current_timestamp
from typing import Optional, Any
import json
import requests
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.getOrCreate()

def generate_synthetic_data(config: ProjectConfig, input_data: DataFrame, num_rows: int = 1000) -> DataFrame:
    """Generates synthetic data in order to simulate data ingestion into the input data.

    Args:
        config (ProjectConfig): Project configuration file converted to dict, containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
        input_data (DataFrame): Current input DataFrame with real data
        num_rows (int, optional): Number of rows to add to the existing input data. Defaults to 1000.

    Returns:
        DataFrame: The added synthetic data, consisting of num_rows rows
    """
    num_rows = min(num_rows, 10)  # Cap the number of rows
    synthetic_data = {}

    # Save the original column names
    original_columns = input_data.columns
    column_dtypes = dict(input_data.dtypes)  # Convert list of tuples to a dictionary for easier access
    renamed_columns = {col: col.lower().replace(" ", "_") for col in original_columns}

    # Rename columns in input_data
    for original_col, renamed_col in renamed_columns.items():
        input_data = input_data.withColumnRenamed(original_col, renamed_col)
    
    # Loop through numerical features with constraints
    num_features = {key: {"min": feature.constraints.min} for key, feature in config.num_features.items()}

    # Loop through the columns and generate data based on constraints
    for col_name, constraints in num_features.items():
        mean_val, std_val = input_data.select(mean(col_name), stddev(col_name)).first()

        # Generate data and apply constraints
        synthetic_data[col_name] = np.round(np.random.normal(mean_val, std_val, num_rows))

        # Apply min constraints
        synthetic_data[col_name] = np.maximum(synthetic_data[col_name], constraints["min"])

        # Ensure the column's original dtype is preserved
        original_col_name = [k for k, v in renamed_columns.items() if v == col_name][0]  # Find the original column name
        dtype = column_dtypes.get(original_col_name)  # Access the original column's dtype
        if str(dtype).lower() in {"int", "bigint", "long"}:
            synthetic_data[col_name] = synthetic_data[col_name].astype(int)
        elif str(dtype).lower() in {"float", "double"}:
            synthetic_data[col_name] = synthetic_data[col_name].astype(float)


    # Loop through categorical features with allowed values
    cat_features = {
        key: [
            int(value) if isinstance(value, str) and value.isdigit() else value
            for value in feature.encoding or feature.allowed_values or []
        ]
        for key, feature in config.cat_features.items()
    }
    for col_name, allowed_values in cat_features.items():
        synthetic_data[col_name] = np.random.choice(allowed_values, num_rows)
        original_col_name = [k for k, v in renamed_columns.items() if v == col_name][0]  # Find the original column name
        dtype = column_dtypes.get(original_col_name)  # Access the original column's dtype
        if dtype == "string":
            synthetic_data[col_name] = synthetic_data[col_name].astype(str)


    # Loop through date features
    date_features = {key: feature for key, feature in config.date_features.items()}
    for col_name, date_feature in date_features.items():
        # Determine min and max datetime constraints
        min_datetime = date_feature.constraints.min
        max_datetime = date_feature.constraints.max

        # Handle null `max_datetime` by assigning a reasonable default
        if max_datetime is None:
            max_datetime = pd.Timestamp.now()

        # Ensure `min_datetime` and `max_datetime` are valid
        if min_datetime >= max_datetime:
            raise ValueError(
                f"Invalid constraints for {col_name}: min_datetime ({min_datetime}) must be earlier than max_datetime ({max_datetime})."
            )

        # Generate random datetimes within the range
        min_timestamp = int(min_datetime.timestamp())
        max_timestamp = int(max_datetime.timestamp())
        # Special handling for `bedtime` and `wakeup_time`. Bedtime must behore wakeuptime and maximum difference 18hours sleep
        if col_name == "bedtime":
            # Generate bedtime values first
            bedtime_values = np.random.randint(min_timestamp, max_timestamp, num_rows)
            synthetic_data[col_name] = pd.to_datetime(bedtime_values, unit="s")
        elif col_name == "wakeup_time":
            # Ensure wakeup times are generated after the bedtimes and within the 18-hour limit
            bedtime_values = synthetic_data.get("bedtime")
            if bedtime_values is None:
                raise ValueError("`bedtime` column must be generated before `wakeup_time`.")

            wakeup_times = []
            for bedtime in bedtime_values:
                max_wakeup_time = bedtime + pd.Timedelta(hours=18)
                # Ensure the max wakeup time does not exceed the configured max datetime
                max_wakeup_timestamp = min(max_wakeup_time.timestamp(), max_timestamp)
                # Ensure wakeup time is after bedtime and within range
                wakeup_time = np.random.randint(bedtime.timestamp(), max_wakeup_timestamp)
                # Ensure the generated wakeup time is always after bedtime (just to be extra safe)
                if wakeup_time <= bedtime.timestamp():
                    wakeup_time = bedtime.timestamp() + 1  # Just ensure it's after bedtime

                wakeup_times.append(wakeup_time)

            synthetic_data[col_name] = pd.to_datetime(wakeup_times, unit="s")
        else:
            # Handle other date features
            synthetic_data[col_name] = pd.to_datetime(
                np.random.randint(min_timestamp, max_timestamp, num_rows), unit="s"
            )
            # Ensure the column's original dtype is preserved
            original_col_name = [k for k, v in renamed_columns.items() if v == col_name][0]  # Find the original column name
            dtype = column_dtypes.get(original_col_name)  # Access the original column's dtype
            if dtype == "timestamp":
                synthetic_data[col_name] = synthetic_data[col_name].astype("datetime64[ns]")

            # Add the column with the correct data type to the final dataframe
            if dtype:
                synthetic_data[col_name] = synthetic_data[col_name].astype(dtype)

        # Create target variable (sleep_efficiency) as a random value between 0 and 1
        synthetic_data[config.target] = np.random.uniform(0, 1, num_rows)

    # Create unique id's
    existing_ids = input_data.select(config.primary_key).rdd.flatMap(lambda x: x).collect()
    new_ids = []
    i = max(existing_ids) + 1 if existing_ids else 1
    while len(new_ids) < num_rows:
        if i not in existing_ids:
            new_ids.append(i)  # Convert numeric ID to string
        i += 1
    synthetic_data[config.primary_key] = new_ids

    # Convert the synthetic data dictionary to a DataFrame
    synthetic_df = spark.createDataFrame(pd.DataFrame(synthetic_data))

     # Restore original column names
    renamed_to_original = {v: k for k, v in renamed_columns.items()}
    for renamed_col, original_col in renamed_to_original.items():
        synthetic_df = synthetic_df.withColumnRenamed(renamed_col, original_col)

    # Add update_timestamp_utc column with the current UTC timestamp
    synthetic_df = synthetic_df.withColumn("update_timestamp_utc", current_timestamp())

    return synthetic_df

def visualize_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Efficiency")
    plt.ylabel("Predicted Efficiency")
    plt.title("Actual vs Predicted Sleep Efficiency")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance, feature_names, top_n=10):
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx[-top_n:].shape[0]) + 0.5
    plt.barh(pos, feature_importance[sorted_idx[-top_n:]])
    plt.yticks(pos, feature_names[sorted_idx[-top_n:]])
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.show()


def adjust_predictions(predictions, scale_factor=1.3):
    return predictions * scale_factor

def check_repo_info(repo_path: str, dbutils: Optional[Any] = None) -> tuple[str, str]:
    """Retrieves the current branch and sha in the Databricks Git repos, based on the repo path.

    Args:
        repo_path (str): Full path to the Databricks Git repo
        dbutils (Optional[Any], optional): Databricks utilities, only available in Databricks. Defaults to None.

    Returns:
        git_branch (str)
            Current git branch
        git_sha (str)
            Current git sha
    """
    if dbutils is None:
        raise ValueError("dbutils cannot be None. Please pass the dbutils object when calling check_repo_info.")

    nb_context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())

    api_url = nb_context["extraContext"]["api_url"]

    api_token = nb_context["extraContext"]["api_token"]

    db_repo_data = requests.get(
        f"{api_url}/api/2.0/repos",
        headers={"Authorization": f"Bearer {api_token}"},
        params={"path_prefix": repo_path},
    ).json()

    repo_info = db_repo_data["repos"][0]
    db_repo_branch = repo_info.get("branch", "N/A")
    db_repo_head_commit = repo_info.get("head_commit_id", "N/A")

    return db_repo_branch, db_repo_head_commit

def get_error_metrics(
    predictions: DataFrame, label_col_name: str = "label", pred_col_name: str = "prediction"
) -> dict[str, float]:
    """Gets the mse, mae, and r2 error metrics.

    Args:
        predictions (DataFrame): DF containing the label and prediction column.
        label_col_name (str, optional): Name of the column containing the label. Defaults to "label".
        pred_col_name (str, optional): Name of the column containing the predictions. Defaults to "prediction".

    Raises:
        ValueError: If the specified label or prediction column is missing from the DataFrame.

    Returns:
        dict[str, float]: Dictionary containing the mse, mae, and r2.
    """
    # Check for presence of label and prediction columns
    if label_col_name not in predictions.columns:
        raise ValueError(f"Label column '{label_col_name}' is missing from the predictions DataFrame.")
    if pred_col_name not in predictions.columns:
        raise ValueError(f"Prediction column '{pred_col_name}' is missing from the predictions DataFrame.")

    evaluator = RegressionEvaluator(labelCol=label_col_name, predictionCol=pred_col_name)
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    error_metrics = {"mse": mse, "mae": mae, "r2": r2}

    return error_metrics