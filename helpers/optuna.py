import os
from typing import List, Dict, Optional
import optuna
from optuna.exceptions import OptunaError
from optuna.visualization._plotly_imports import go
from logging import getLogger

from helpers.context import get_context_id, with_context

logger = getLogger(__name__)

def load_study(
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
    direction: Optional[str] = None,
) -> optuna.Study:
    """
    Load an existing Optuna study (using a persistent storage backend),
    or create it if it doesn't exist.
    """
    study_name = study_name or os.getenv("OPTUNA_STUDY_NAME", "default_study")
    storage_url = storage_url or os.getenv(
        "OPTUNA_STORAGE_URL", "sqlite:///optuna_studies.db"
    )
    direction = (direction or os.getenv("OPTUNA_DIRECTION", "minimize")).lower()
    if direction not in ("minimize", "maximize"):
        logger.warning(
            f"Invalid optimization direction {direction!r}; falling back to 'minimize'"
        )
        direction = "minimize"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        logger.info(f"Loaded existing study '{study_name}' from {storage_url}")
    except OptunaError:
        logger.info(f"Study '{study_name}' not found at {storage_url}; creating new one")
        study = optuna.create_study(
            study_name=study_name, storage=storage_url, direction=direction
        )
        logger.info(f"Created study '{study_name}' with direction='{direction}'")
    return study

def get_opt_history_graphs(
    study: optuna.Study,
    metrics: List[str] = None
) -> Dict[str, go.Figure]:
    if metrics is None:
        metrics = ["power", "cp", "area"]
    figs: Dict[str, go.Figure] = {}
    for idx, name in enumerate(metrics):
        figs[name] = optuna.visualization.plot_optimization_history(
            study,
            target=lambda t, idx=idx: t.values[idx],
            target_name=name,
        )
    return figs

def get_param_importance_graphs(
    study: optuna.Study,
    metrics: List[str] = None
) -> Dict[str, go.Figure]:
    if metrics is None:
        metrics = ["power", "cp", "area"]
    figs: Dict[str, go.Figure] = {}
    for idx, name in enumerate(metrics):
        figs[name] = optuna.visualization.plot_param_importances(
            study,
            target=lambda t, idx=idx: t.values[idx],
            target_name=name,
        )
    return figs

# DEMO
@with_context
def main():
    # Retrieve or create the study
    # ctx = get_context_id()
    ctx = "context"
    study = load_study(
        study_name="bfs_motpe_fl_dse",
        storage_url="sqlite:///bfs_motpe_fl_dse.db",
    )

    # Ensure a folder for this context
    os.makedirs(ctx, exist_ok=True)

    # 1) Export optimization-history plots
    opt_his = get_opt_history_graphs(
        study=study, metrics=["power", "cp", "area"]
    )
    for name, fig in opt_his.items():
        out_path = os.path.join(ctx, f"opt_history_{name}.svg")
        fig.write_image(out_path)
        print(f"Saved optimization history for {name} at {out_path}")

    # 2) Export parameter-importance plots
    param_imp = get_param_importance_graphs(
        study=study, metrics=["power", "cp", "area"]
    )
    for name, fig in param_imp.items():
        out_path = os.path.join(ctx, f"param_importance_{name}.svg")
        fig.write_image(out_path)
        print(f"Saved parameter importance for {name} at {out_path}")

if __name__ == "__main__":
    main()
