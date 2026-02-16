# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for vLLM."""
import logging
import os
import sys
from typing import Any, Dict

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("openrlhf")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger


class WandbLogger:
    """Handle wandb setup and training-time logging."""

    def __init__(self, args) -> None:
        import wandb

        if not wandb.api.api_key:
            wandb.login(key=args.use_wandb)
        wandb.init(
            entity=args.wandb_org,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=args.__dict__,
            reinit=True,
        )

        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("eval/epoch")
        wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
        self.handle = wandb
        self.samples_table = wandb.Table(columns=["global_step", "text", "reward", "gts"])

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)

        generated_samples = logs_dict.pop("generated_samples", None)
        if generated_samples:
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
            new_table = self.handle.Table(columns=self.samples_table.columns, data=self.samples_table.data)
            new_table.add_data(global_step, *generated_samples)
            self.samples_table = new_table
            self.handle.log({"train/generated_samples": new_table})

        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"train/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
        self.handle.log(logs)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)

        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"eval/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
        self.handle.log(logs)

    def close(self) -> None:
        self.handle.finish()


class TensorboardLogger:
    """Handle tensorboard setup and training-time logging."""

    def __init__(self, args) -> None:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(args.use_tensorboard, exist_ok=True)
        log_dir = os.path.join(args.use_tensorboard, args.wandb_run_name)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        generated_samples = logs_dict.get("generated_samples")
        for k, v in logs_dict.items():
            if k == "generated_samples" and v is not None:
                if len(generated_samples) == 2:
                    text, reward = generated_samples
                    formatted_text = f"Sample:\\n{text}\\n\\nReward: {reward:.4f}"
                else:
                    text, reward, label = generated_samples
                    formatted_text = f"Sample:\\n{text}\\n\\nReward: {reward:.4f}\\n\\nGround truth: {label}"
                    
                formatted_text = f"Sample:\\n{text}\\n\\nReward: {reward:.4f}"
                self.writer.add_text("train/generated_samples", formatted_text, global_step)
            elif v is not None:
                self.writer.add_scalar(f"train/{k}", v, global_step)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        for k, v in logs_dict.items():
            self.writer.add_scalar(f"eval/{k}", v, global_step)

    def close(self) -> None:
        self.writer.close()

class MLflowLogger:
    """Handle MLflow setup and training-time logging."""

    def __init__(self, args) -> None:
        import mlflow
        import transformers
        self.mlflow = mlflow

        if hasattr(args, "mlflow_tracking_uri") and args.mlflow_tracking_uri:
            self.mlflow.set_tracking_uri(args.mlflow_tracking_uri)

        if hasattr(args, "mlflow_experiment_name") and args.mlflow_experiment_name:
            exp_name = args.mlflow_experiment_name
        
            s3_root = os.environ.get('S3_ARTIFACT_ROOT')
            if not s3_root:
                raise ValueError("[MLflowLogger] S3_ARTIFACT_ROOT is not set in environment!")
            
            # Формируем путь: s3://mlflow/бакет/имя-эксперимента
            artifact_location = f"{s3_root}/{exp_name}"
            
            client = self.mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(exp_name)
            
            if experiment is None:
                print(f"[MLflowLogger] Creating new experiment: {exp_name} at {artifact_location}")
                self.mlflow.create_experiment(name=exp_name, artifact_location=artifact_location)
            else:
                print(f"[MLflowLogger] Using existing experiment: {exp_name}")
    
        self.mlflow.set_experiment(exp_name)
        
        # Determine run name (prioritize mlflow_run_name, fallback to wandb_run_name)
        run_name = getattr(args, "mlflow_run_name", None)
        if not run_name:
            run_name = getattr(args, "wandb_run_name", "openrlhf_run")
        
        self.mlflow.enable_system_metrics_logging()  
        self.mlflow.set_system_metrics_sampling_interval(1)
        self.mlflow.start_run(run_name=run_name)
        
        # Log model and dataset as tags so they appear in Details section
        if hasattr(args, "pretrain") and args.pretrain:
            self.mlflow.set_tag("model", args.pretrain)
            # self.mlflow.transformers.log_model(args.pretrain, "model")
        if hasattr(args, "prompt_data") and args.prompt_data:
            self.mlflow.set_tag("dataset", args.prompt_data)

        if hasattr(args, "__dict__"):
            params = {k: v for k, v in args.__dict__.items() if isinstance(v, (int, float, str, bool))}
            self.mlflow.log_params(params)

        # Log Hyperparameters (filter for serializable types)
        if hasattr(args, "__dict__"):
            params = {k: v for k, v in args.__dict__.items() if isinstance(v, (int, float, str, bool))}
            self.mlflow.log_params(params)

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)
        
        # Handle generated samples: Log as text artifact + metric for reward
        generated_samples = logs_dict.pop("generated_samples", None)
        if generated_samples:
            if len(generated_samples) > 2:
                text, reward, label = generated_samples
                # Log the sample text to a file in artifacts
                self.mlflow.log_text(text+f"\nground_truth: {label}\n", f"generated_samples/step_{global_step}.txt")
            else:
                text, reward = generated_samples
                # Log the sample text to a file in artifacts
                self.mlflow.log_text(text, f"generated_samples/step_{global_step}.txt")
            # Log the reward of this sample
            self.mlflow.log_metric("train/sample_reward", reward, step=global_step)

        # Log standard metrics
        metrics = {}
        for k, v in logs_dict.items():
            if v is not None and isinstance(v, (int, float)):
                metrics[f"train/{k}"] = v
        
        self.mlflow.log_metrics(metrics, step=global_step)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        metrics = {}
        for k, v in logs_dict.items():
            if v is not None and isinstance(v, (int, float)):
                metrics[f"eval/{k}"] = v
        
        self.mlflow.log_metrics(metrics, step=global_step)

    def close(self) -> None:
        self.mlflow.end_run()
