<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the 25th of November.


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv venv -p 3.11.10
source .venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```

## Example of uploading package to the volume:
```
databricks auth login --host HOST
uv build

databricks fs cp dist/wheel dbfs:/Volumes/my_catalog/my_schema/my_volume/
# example upload data to Catalog (dbfs volume)
databricks fs cp data/data.csv dbfs:/Volumes/dbw_mavencourse_e2emlops_weu_001/sleep_efficiency/data/data.csv --profile maven_e2emlops_dbw_stan
# example upload package to Catalog (dbfs volume)
databricks fs cp dist/mlops_with_databricks-0.0.1-py3-none-any.whl dbfs:/Volumes/dbw_mavencourse_e2emlops_weu_001/sleep_efficiency/packages --profile maven_e2emlops_dbw_stan
```
