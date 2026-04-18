"""
Mock abstraction layer for Vertex AI Experiments and Google Cloud Monitoring.
Provides an interface identical to the official google-cloud python SDKs.
"""

import time
import random

# from google.cloud import aiplatform
# from google.cloud import monitoring_v3

class GCPObservability:
    def __init__(self, project_id: str, location: str, experiment_name: str):
        self.project_id = project_id
        self.location = location
        self.experiment_name = experiment_name
        print(f"[GCP STUB] Initializing Vertex AI Experiments in {project_id}/{location}")
        # aiplatform.init(project=project_id, location=location, experiment=experiment_name)

    def log_inference_metrics(self, prompt: str, model_mode: str, num_steps: int, latency: float, clip_score: float):
        """
        Logs a single run to Vertex AI Experiments.
        Replaces mlflow.start_run() and mlflow.log_metrics().
        """
        run_name = f"inference-{int(time.time())}"
        print(f"[GCP STUB] Vertex AI tracking run: {run_name}")
        # aiplatform.start_run(run=run_name)
        # aiplatform.log_params({
        #     "prompt": prompt,
        #     "model_mode": model_mode,
        #     "inference_steps": num_steps,
        # })
        # aiplatform.log_metrics({
        #     "latency_sec": latency,
        #     "clip_score": clip_score
        # })
        # aiplatform.end_run()

    def stream_custom_metric_to_cloud_monitoring(self, metric_type: str, value: float, labels: dict = None):
        """
        Pushes a real-time system health metric directly to GCP Cloud Monitoring
        (Google Managed Prometheus / Stackdriver).
        """
        print(f"[GCP STUB] Cloud Monitoring streamed {metric_type}: {value}")
        # client = monitoring_v3.MetricServiceClient()
        # project_name = f"projects/{self.project_id}"
        # series = monitoring_v3.TimeSeries()
        # series.metric.type = f"custom.googleapis.com/{metric_type}"
        # ... API dispatch code here ...
        pass

    def get_vertex_experiments_history(self):
        """
        Fetches the dataframe of past runs for the Admin Dashboard.
        Replaces mlflow.search_runs().
        """
        print("[GCP STUB] Fetching Vertex Experiments Dataset...")
        # df = aiplatform.get_experiment_df(experiment=self.experiment_name)
        # return df
        import pandas as pd
        return pd.DataFrame([
            {"run_name": "example-1", "metrics.latency_sec": 3.4, "metrics.clip_score": 25.1},
            {"run_name": "example-2", "metrics.latency_sec": 3.1, "metrics.clip_score": 24.8}
        ])
