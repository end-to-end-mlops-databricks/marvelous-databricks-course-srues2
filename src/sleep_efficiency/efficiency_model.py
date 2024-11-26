from sleep_efficiency.config import ProjectConfig
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from sklearn.metrics import mean_squared_error, r2_score

class EfficiencyModel:
    """A class for creating a PySpark ML model

    Attributes
    ----------
    config: ProjectConfig
        Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
    spark: SparkSession
        The Spark session is required for running Spark functionality outside of Databricks.

    Methods
    -------
    create_preprocessing_stages:
        Creates ML Pipeline preprocessing stages
    """

    def __init__(self, config: ProjectConfig, spark: SparkSession) -> None:
        """Constructs all the necessary attributes for the modelling object

        Args:
            config (ProjectConfig): Project configuration file converted to dict, containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.
        """
        self.config: ProjectConfig = config # Store the configuration
        try:
            self.train_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_train_set")
            self.test_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.{config.use_case_name}_test_set")
        except Exception as e:
            raise RuntimeError("Failed to read training or testing data tables") from e

        # self.model = Pipeline(
        #     steps=[
        #         ("preprocessor", preprocessor),
        #         (
        #             "regressor",
        #             RandomForestRegressor(
        #                 n_estimators=config["parameters"]["n_estimators"],
        #                 max_depth=config["parameters"]["max_depth"],
        #                 random_state=42,
        #             ),
        #         ),
        #     ]
        # )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def get_feature_importance(self):
        feature_importance = self.model.named_steps["regressor"].feature_importances_
        feature_names = self.model.named_steps["preprocessor"].get_feature_names_out()
        return feature_importance, feature_names

    def create_preprocessing_stages(self) -> list:
        """Creates the following ML Pipeline preprocessing stages:
        - Handling of categorical features
        - Use of the VectorAssembler to combine numeric and categorical features
        - Combine all numeric and categorical features into one feature column
        - Build the preprocessing pipeline

        Returns:
            List: Preprocessing stages required for the PySpark ML pipeline
        """
        target: str = self.config.target

        if target is None:
            raise ValueError("Target column not specified in config")

        # Extracting input column names from the config
        num_feature_cols = list(self.config.num_features.keys())
        cat_feature_cols = list(self.config.cat_features.keys())
        date_feature_cols = list(self.config.date_features.keys())

        if not num_feature_cols and not cat_feature_cols and not date_feature_cols:
            raise ValueError("No feature columns specified in config")

        target_indexer: StringIndexer = StringIndexer(inputCol=target, outputCol="label")

        # Set up the imputer
        numeric_imputer: Imputer = Imputer(
            inputCols=num_feature_cols, outputCols=[f"{c}_imputed" for c in num_feature_cols]
        )

        # Create the scaler
        scaler: StandardScaler = StandardScaler(inputCol="features_num", outputCol="scaled_features")

        # StringIndexer and OneHotEncoder for categorical features
        indexers: list[StringIndexer] = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed") for col in cat_feature_cols
        ]
        encoders: list[OneHotEncoder] = [
            OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded") for col in cat_feature_cols
        ]

        # Assemble numeric features
        assembler_numeric: VectorAssembler = VectorAssembler(
            inputCols=[f"{c}_imputed" for c in num_feature_cols], outputCol="features_num"
        )

        # Assemble categorical features
        assembler_categorical: VectorAssembler = VectorAssembler(
            inputCols=[f"{col}_encoded" for col in cat_feature_cols], outputCol="features_cat"
        )

        # Combine numeric and categorical features
        assembler_all: VectorAssembler = VectorAssembler(
            inputCols=["scaled_features", "features_cat"], outputCol="features"
        )

        # Building the pipeline
        preprocessing_stages: list = (
            [target_indexer, numeric_imputer, assembler_numeric, scaler]
            + indexers
            + encoders
            + [assembler_categorical, assembler_all]
        )

        return preprocessing_stages