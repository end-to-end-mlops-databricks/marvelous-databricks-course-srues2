from typing import Optional, Tuple

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.ml import Pipeline
from sklearn.model_selection import train_test_split

from sleep_efficiency.config import ProjectConfig


class DataProcessor:
    """A class to preprocess the input data

    Attributes
    ----------
    config: ProjectConfig
        Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
    spark: SparkSession
        The Spark session is required for running Spark functionality outside of Databricks.

    Methods
    -------
    split_data:
        Splits the DataFrame into training and test sets
    """
    
    def __init__(self, config: ProjectConfig, spark: SparkSession) -> None:
        """Constructs all the necessary attributes for the preprocessing object

        Args:
            config (ProjectConfig): Project configuration file converted to dict, containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.
        """
        self.config: ProjectConfig = config # Store the configuration
        self.df: DataFrame = spark.read.table(f"{config.catalog}.{config.db_schema}.raw_{config.use_case_name}") # Store the DataFrame as self.df
        self.X: Optional[DataFrame] = None
        self.y: Optional[DataFrame] = None
        self.preprocessor: Optional[Pipeline] = None

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # change column names to lowercase and replace spaces with underscores
        self.df.columns = self.df.columns.str.lower().str.replace(" ", "_")

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Fill missing values with mean/max/min or default values
        self.df.fillna(
            {
                "awakenings": self.df["awakenings"].min(),
                "caffeine_consumption": self.df["caffeine_consumption"].mean(),
                "alcohol_consumption": self.df["alcohol_consumption"].mean(),
                "exercise_frequency": self.df["exercise_frequency"].mean(),
            },
            inplace=True,
        )

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Convert date features to the type datetime
        date_features = self.config.date_features
        for date_col in date_features:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format="%Y-%m-%d %H:%M:%S")  # Adjust format if needed

        # Add 'month' column based on 'bedtime' column if it exists
        if "bedtime" in self.df.columns:
            self.df["sleep_month"] = self.df["bedtime"].dt.month

        # Extract target and relevant features
        # Since bedtime and wakeup time is reflected in sleep duration, it will be omitted
        target = self.config.target
        relevant_columns = cat_features + num_features + date_features + [target] + ["id"] + ["sleep_month"]
        self.df = self.df[relevant_columns]
    
    def drop_missing_target(self) -> None:
        """Drops rows with missing target values"""
        target: str = self.config.target
        self.df = self.df.dropna(subset=[target])

    def split_data(self, test_size=0.2, random_state=42):
        """Splits the DataFrame into training and test sets, the missing target values are dropped.

        Args:
            test_size (float, optional): Proportion of the input data to be part of the test set. Defaults to 0.2.
            random_state (int, optional): Value of the state. Defaults to 42.

        Raises:
            ValueError: If `test_size` is not between 0 and 1.

        Returns:
            train_data (DataFrame): Data used for training the model
            test_data (DataFrame): Data used for testing the model
        """
        target: str = self.config.target
        self.df = self.df.dropna(subset=[target])

        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        if self.df.isEmpty():
            raise ValueError("Cannot split an empty DataFrame")

        self.drop_missing_target()

        train_data: DataFrame
        test_data: DataFrame
        train_data, test_data = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_data, test_data

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
