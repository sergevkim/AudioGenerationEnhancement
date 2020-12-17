try:
    import wandb
except:
    print('bad boy')

class WandbLogger:
    def __init__(self, project_name, experiment_name):
        wandb.init(project=project_name)
        self.experiment_name = experiment_name
    def log_metrics(
            self,
            metrics,
            step=None,
        ):
        for metric_name, metric_value in metrics.items():
            self.log_metric(
                metric_name=metric_name,
                metric_value=metric_value,
                step=step,
            )

    def log_metric(
            self,
            metric_name,
            metric_value,
            step=None,
        ):

        
        if step is None:
            wandb.log({metric_name: metric_value})
        else:
            wandb.log({metric_name: metric_value})
