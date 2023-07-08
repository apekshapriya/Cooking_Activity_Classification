import mlflow

from activity_classifier.config import *

from tensorflow.keras.models import load_model
import tempfile
import json


def load_dict(filepath):
    with open(filepath, "r") as fp:
        d = json.load(fp)

    return d


# loading the experiment and search for successful runs


def get_artifacts(exp_name):
    experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    all_runs = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=["metrics.f1_score"]
    )

    best_run_id = all_runs.iloc[-1].run_id

    best_run = mlflow.get_run(run_id=best_run_id)

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as dp:
        client.download_artifacts(run_id=best_run_id, path="", dst_path=dp)

        model = load_model(Path(dp, "activity.model"))
        performance = load_dict(Path(dp, "performance.json"))

        # args = print(Path(dp, 'batch_size').read())
    # print (json.dumps(performance, indent=2))

    return {"model": model, "performance": performance}


if __name__ == """__main__""":
    get_artifacts("baselines_1")
