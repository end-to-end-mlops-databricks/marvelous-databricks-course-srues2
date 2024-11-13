import pandas as pd
from sleep_efficiency.config import ProjectConfig
import datetime
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import current_timestamp, to_utc_timestamp, to_timestamp
from datetime import datetime
from pyspark.sql import SparkSession

class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # change column names to lowercase and replace spaces with underscores
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Fill missing values with mean/max/min or default values
        self.df.fillna({
            'awakenings': self.df['awakenings'].min(),
            'caffeine_consumption': self.df['caffeine_consumption'].mean(),
            'alcohol_consumption': self.df['alcohol_consumption'].mean(),
            'exercise_frequency': self.df['exercise_frequency'].mean()
        }, inplace=True)

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype('category')

        # Convert date features to the type datetime
        date_features = self.config.date_features
        for date_col in date_features:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format="%Y-%m-%d %H:%M:%S")  # Adjust format if needed
        
        # Add 'month' column based on 'bedtime' column if it exists
        if 'bedtime' in self.df.columns:
            self.df['sleep_month'] = self.df['bedtime'].dt.month

        # Extract target and relevant features
        # Since bedtime and wakeup time is reflected in sleep duration, it will be omitted
        target = self.config.target
        relevant_columns = cat_features + num_features + date_features + [target] + ['id'] + ['sleep_month']
        self.df = self.df[relevant_columns]

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))   
        
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set")
        
        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set")

        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")