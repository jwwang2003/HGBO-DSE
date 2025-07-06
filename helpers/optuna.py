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

    Parameters
    ----------
    study_name : str, optional
        Name of the Optuna study; overrides OPTUNA_STUDY_NAME.
    storage_url : str, optional
        Storage backend URL; overrides OPTUNA_STORAGE_URL.
    direction : str, optional
        "minimize" or "maximize"; overrides OPTUNA_DIRECTION.

    Environment variables (used only if corresponding arg is None):
      OPTUNA_STUDY_NAME    - default "default_study"
      OPTUNA_STORAGE_URL   - default "sqlite:///optuna_studies.db"
      OPTUNA_DIRECTION     - default "minimize"

    Returns
    -------
    optuna.Study
        The loaded or newly created study.
    """
    
    # 1) Determine study_name
    study_name = study_name or os.getenv("OPTUNA_STUDY_NAME", "default_study")

    # 2) Determine storage_url
    storage_url = storage_url or os.getenv(
        "OPTUNA_STORAGE_URL", "sqlite:///optuna_studies.db"
    )

    # 3) Determine direction
    direction = (direction or os.getenv("OPTUNA_DIRECTION", "minimize")).lower()
    if direction not in ("minimize", "maximize"):
        logger.warning(
            f"Invalid optimization direction {direction!r}; falling back to 'minimize'"
        )
        direction = "minimize"

    # Try to load, else create
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
    """
    Generate optimization-history plots for each objective in a multi-objective study.

    Parameters
    ----------
    study : optuna.Study
        The Optuna Study instance to visualize.
    metrics : list of str, optional
        Names of each objective in the same order as trial.values.
        Defaults to ["power", "cp", "area"].

    Returns
    -------
    Dict[str, go.Figure]
        A mapping from metric name to its plotly Figure.
    """
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
    """
    Generate parameter-importance plots for each objective in a multi-objective study.

    Parameters
    ----------
    study : optuna.Study
        The Optuna Study instance to analyze.
    metrics : list of str, optional
        Names of each objective in the same order as trial.values.
        Defaults to ["power", "cp", "area"].

    Returns
    -------
    Dict[str, go.Figure]
        A mapping from metric name to its plotly Figure showing parameter importances.
    """
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
    print(get_context_id())
    loaded_study = load_study(study_name="bfs_motpe_fl_dse", storage_url="sqlite:///bfs_motpe_fl_dse.db")
    print(loaded_study)
    # print(loaded_study.trials)
    
    opt_his_figs = get_opt_history_graphs(study=loaded_study)
    
    for (name, fig) in opt_his_figs.items():
        fig.write_image(name + ".svg")
        print(name)

if __name__ == "__main__":
    main()