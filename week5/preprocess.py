"""This script preprocesses input data, the following scenarios are accounted for in this script:
- If the preprocessing has already been done on this workspace and no new input data has been ingested, no preprocessing is done.
- If the preprocessing has already been done on this workspace and new input data has been ingested, the preprocessing is done for the new rows and the train/test data is appended to the existing tables.
- If the preprocessing has not been done yet on this workspace, preprocessing is done on all input data and the train/test data is written to UC.

This ensures that the code also works after deploying to a new workspace, while only running the necessary compute.
"""

from pyspark.sql import SparkSession

from sleep_efficiency.config import ProjectConfig
from sleep_efficiency.data_processor import DataProcessor
from pyspark.sql.utils import AnalysisException, StreamingQueryException

def preprocessing():
    spark = SparkSession.builder.getOrCreate()

    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

    data_preprocessor = DataProcessor(config, spark)

    refreshed = False

    if spark.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_set") and spark.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_test_set"):
        train_data_sleep_ids = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_set").select(config.primary_key)
        test_data_sleep_ids = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_test_set").select(config.primary_key)

        input_data_sleep_ids = data_preprocessor.df.select(config.primary_key)

        new_sleeping_ids = input_data_sleep_ids.join(
            train_data_sleep_ids.unionByName(test_data_sleep_ids), config.primary_key, "left_anti"
        )

        if new_sleeping_ids.isEmpty():
            print(f"The input data {config.catalog_name}.{config.schema_name}.{config.use_case_name} has no new sleeping IDs and thus no further preprocessing is required")
        else:
            try:
                refreshed = True
                data_preprocessor.df = data_preprocessor.df.join(new_sleeping_ids, config.primary_key)
                train_new, test_new = data_preprocessor.split_data()
                train_new.write.format("delta").mode("append").saveAsTable(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_set")
                test_new.write.format("delta").mode("append").saveAsTable(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_test_set")
                print("The train and test set has been updated for the new booking IDs")
            except (AnalysisException, StreamingQueryException) as e:
                print(f"Error appending to Delta tables: {str(e)}")
                raise
    else:
        refreshed = True

        train, test = data_preprocessor.split_data()
        try:
            train.write.format("delta").mode("overwrite").saveAsTable(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_set")
            test.write.format("delta").mode("overwrite").saveAsTable(f"{config.catalog_name}.{config.schema}.{config.use_case_name}_test_set")
            print("The train and test data is created for the first time")
        except (AnalysisException, StreamingQueryException) as e:
            print(f"Error creating Delta tables: {str(e)}")
            raise

    dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)  # type: ignore # noqa: F821


if __name__ == "__main__":
    preprocessing()