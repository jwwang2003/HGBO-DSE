import optuna

from optuna.visualization._plotly_imports import go

def main():
    loaded_study = optuna.load_study(study_name="bfs_motpe_fl_dse", storage="sqlite:///bfs_motpe_fl_dse.db")
    print(loaded_study)
    print(loaded_study.trials)
    
    fig_pwr_h: go.Figure = optuna.visualization.plot_optimization_history(
        loaded_study,
        target=lambda t: t.values[0],
        target_name="power"
    )
    fig_cp_h = optuna.visualization.plot_optimization_history(
        loaded_study,
        target=lambda t: t.values[1],
        target_name="cp"
    )
    fig_area_h = optuna.visualization.plot_optimization_history(
        loaded_study,
        target=lambda t: t.values[2],
        target_name="area"
    )
    
    fig_pwr_h.write_image("test.svg")

if __name__ == "__main__":
    main()