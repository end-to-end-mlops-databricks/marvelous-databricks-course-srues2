import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sleep_efficiency.config import ProjectConfig
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg as mean
from pyspark.sql.functions import stddev

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

    # Loop through numerical features with constraints
    num_features = {key: {"min": feature.constraints.min} for key, feature in config.num_features.items()}

    # change column names to lowercase and replace spaces with underscores
    for col in input_data.columns:
        input_data = input_data.withColumnRenamed(col, col.lower().replace(" ", "_"))
    
    # Loop through the columns and generate data based on constraints
    for col_name, constraints in num_features.items():
        mean_val, std_val = input_data.select(mean(col_name), stddev(col_name)).first()

        # Generate data and apply constraints
        synthetic_data[col_name] = np.round(np.random.normal(mean_val, std_val, num_rows))

        # Apply min constraints
        synthetic_data[col_name] = np.maximum(synthetic_data[col_name], constraints["min"])

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
