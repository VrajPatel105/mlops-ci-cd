import dagshub
import mlflow

dagshub.init(repo_owner='VrajPatel105', repo_name='mlflow-dagshub', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/VrajPatel105/mlflow-dagshub.mlflow")

with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)