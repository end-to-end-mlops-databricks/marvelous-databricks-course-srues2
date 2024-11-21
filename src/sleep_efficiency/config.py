from typing import Any, Dict, List

import yaml
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    date_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    use_case_name: str
    volume_whl_path: str
    user_dir_path: str
    git_repo: str
    primary_key: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    ab_test: Dict[str, Any]  # Dictionary to hold A/B test parameters

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
