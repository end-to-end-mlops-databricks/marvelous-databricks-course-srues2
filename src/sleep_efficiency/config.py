from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
import yaml
from pydantic import BaseModel

class Constraints(BaseModel):
    min: Optional[Union[int, float]]  # Using Union for both int and float types
    max: Optional[Union[int, float]]  # Optional in case it is not defined


class NumFeature(BaseModel):
    type: str
    constraints: Constraints


class CatFeature(BaseModel):
    type: str
    allowed_values: List[Union[str, bool]]  # Can include strings or booleans
    encoding: Optional[List[int]]  # Optional encoding

class DateFeature(BaseModel):
    type: str

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
