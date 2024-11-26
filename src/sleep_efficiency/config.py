from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


class Constraints(BaseModel):
    min: Optional[Union[int, float]]  # Using Union for both int and float types
    max: Optional[Union[int, float]]  # Optional in case it is not defined


class DateConstraints(BaseModel):
    min: Optional[datetime]  # Minimum datetime constraint, optional
    max: Optional[datetime]  # Maximum datetime constraint, optional


class NumFeature(BaseModel):
    type: str
    constraints: Constraints


class CatFeature(BaseModel):
    type: str
    allowed_values: List[Union[str, bool]]  # Can include strings or booleans
    encoding: Optional[List[int]]  # Optional encoding


class DateFeature(BaseModel):
    type: str
    constraints: DateConstraints


class Parameters(BaseModel):
    """
    Holds model parameters.
    - `learning_rate`: Learning rate for the model (e.g., 0.01).
    - `n_estimators`: Number of estimators (e.g., 1000).
    - `max_depth`: Maximum tree depth (e.g., 6).
    """

    learning_rate: float = Field(..., gt=0, le=1)  # Use Field to set constraints
    n_estimators: int = Field(..., gt=0, le=10000)  # Use Field to set constraints
    max_depth: int = Field(..., gt=0, le=32)  # Use Field to set constraints


class ProjectConfig(BaseModel):
    """
    Defines the configuration for the project.
    - `catalog`: The data catalog name.
    - `db_schema`: The schema where the dataset resides. Alias to `schema`
    - `use_case_name`: The name of the use case.
    - `user_dir_path`: Path of the user folder in Databricks, this is used for the experiment, volume and git repo location
    - `git_repo`: Name of the Git repo
    - `volume_whl_path`: Volume path where the whl is stored
    - `parameters`: Model parameters such as learning rate and estimators.
    - `num_features`: Numerical features with details on type and constraints.
    - `cat_features`: Categorical features with details on type, allowed values, and constraints.
    - `target`: The target variable for model training (e.g., booking status).
    - `primary_key`: The priamry key in the input data, this is used for the registration of the data in the feature table
    """

    num_features: Dict[str, NumFeature]
    cat_features: Dict[str, CatFeature]
    date_features: Dict[str, DateFeature]
    target: str
    catalog_name: str
    schema_name: str
    use_case_name: str
    volume_whl_path: str
    user_dir_path: str
    git_repo: str
    primary_key: str
    parameters: Parameters
    ab_test: Dict[str, Any]  # Dictionary to hold A/B test parameters

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
