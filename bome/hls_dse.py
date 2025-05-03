import optuna
import argparse
import pyDOE
from functools import partial

from bome.alg.sa_sampler import SimulatedAnnealingSampler
from bome.hls_basic import HLSBasic
from bome.tdm.gen_config import *
from bome.tdm.design_space import *
from bome.hgp_pred import *
from bome.get_ppa import *
from bome.save_report import *

from bome.vitis_hls import VitisHLSRunner


noLatList = ['bfs', 'fft', 'nw', 'stencil']

class IterationCallback:
    def __init__(self, log: Logger):
        self.caseNumber = 0
        self.log = log
        
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.caseNumber = self.caseNumber + 1
        self.log.info(f"[Inference] Iteration: {trial.number}, Duration: {trial.duration}, Params: {trial.params}")
        # socketio.emit('progress_update', {'current': self.caseNumber})

def objective(trial, basic: HLSBasic):
    log = basic.log

    # get the global variables from basic
    dataset_path = basic.dataset_path
    
    # parsed versions of config.yaml & params.yaml
    static_config = basic.static_config
    params = basic.params

    ori_prj_path = basic.ori_prj_path
    hls_temp = basic.hls_temp
    hls_script_path = basic.hls_script_path
    case = basic.case
    top = basic.top
    alg = basic.alg
    encode = basic.encode
    process = basic.process
    mode = basic.mode

    tempDir, paraDict = config_tree_space(static_config, encode, trial, params, basic.log)
    # log.info(paraDict)

    # generate tcl for HLS
    iterNum = trial.number
    dir_json = os.path.join(hls_script_path, "dir_%d.json" % iterNum)
    dir_tcl = os.path.join(hls_script_path, "dir_%d.tcl" % iterNum)
    hls_tcl = os.path.join(hls_script_path, "hls_%d.tcl" % iterNum)
    genDirConfig(encode, params, static_config, paraDict, dir_tcl, tempDir, dir_json, basic.log)
    f_script = open(hls_temp, "r")
    content = f_script.read()
    f_script.close()
    content = content.replace('dir_test.tcl', 'dir_%d.tcl' % iterNum)
    f_script = open(hls_tcl, "w")
    f_script.write(content)
    f_script.close()
    if mode == 'hgp':
        log.info("Running Vitis HLS to get adb and adb.xml files...")
        
        hls_runner = VitisHLSRunner(
            tcl_script=hls_tcl,
            context=basic.get_cwd(),
            check=False
        )
        hls_runner.run()
        
        basic.log.info("Using HGP to predict PPA values...")
        rpt_list, prj_path = get_adb_rpt_verilog(case, top, alg, ori_prj_path, dataset_path, iterNum, process, mode)
        dictPPA, fail_flag = getHLS(params, rpt_list, log)  # get results from HLS
        if fail_flag:
            dictPPA['IMPL'] = {'LUT': 1e8, 'FF': 1e8, 'DSP': 1e8, 'BRAM': 1e8, 'CP': 1e8, 'PWR': 1e8}
        else:
            dict_hls = dictPPA['HLS']
            hls_attr = list(dict_hls.values())
            dictPPA['IMPL'] = getGNNPred(prj_path, hls_attr, case)
    else:
        basic.log.info("Running Vitis HLS and Vivado to get PPA...")
        
        hls_runner = VitisHLSRunner(
            tcl_script=hls_tcl,
            context=basic.get_cwd(),
            check=False
        )
        hls_runner.run()
        
        log.info("Collecting adb files, hls/syn/impl report and Verilog files...")
        rpt_list, _ = get_adb_rpt_verilog(case, top, alg, ori_prj_path, dataset_path, iterNum, process, mode)
        dictPPA, _ = getPPA(params, rpt_list)

    ppa_rpt = os.path.join(hls_script_path, "ppa_%d.json" % iterNum)
    with open(ppa_rpt, "w") as fout:
        fout.write(json.dumps(dictPPA, indent=4))

    if case in noLatList:
        npower, ncp, narea = normalizePCA(params, dictPPA)
        if alg == 'sa':
            ppa = npower + ncp + narea
        else:
            ppa = [npower, ncp, narea]
    else:
        npower, nlat, ncp, narea = normalizePLCA(params, dictPPA)
        if alg == 'sa':
            ppa = npower + nlat + ncp + narea
        else:
            ppa = [npower, nlat, ncp, narea]

    return ppa


def runDSE(basic: HLSBasic, progress_callback=None):
    case = basic.case
    alg = basic.alg
    num = basic.num
    params = basic.params
    mode = basic.mode
    encode = basic.encode
    parallel = basic.parallel
    isolated_path = basic.isolated

    # run lhs to pre-sample initial design points
    init_params = basic.paraDict
    dim = len(init_params)
    n_startup_trials = 10
    init_params_arr = pyDOE.lhs(n=dim, samples=n_startup_trials, criterion='maximin')
    init_params_arr = init_params_arr.T
    idx = 0
    for key in init_params:
        init_params[key] = init_params_arr[idx]
        idx += 1

    if encode == 'discrete':
        init_params = map_to_discrete(init_params, params, n_startup_trials)

    # specify the experiment name
    if mode == 'impl':
        study_name = case + "_" + mode + "_dse"
    else:
        study_name = case + "_" + alg + "_dse"
        
    storage_path = os.path.join(
        isolated_path if isolated_path else "", study_name + ".db"
    )
    if parallel:
        # Important: must create the corresponding database in MySQL.
        # Here is the method:
        # mysql -u root -p
        # CREATE DATABASE 'study_name';
        # SHOW DATABASES; (to see if the database is created successfully)
        storage = "mysql+pymysql://root:password@localhost/" + storage_path
        basic.log.info('Using MySQL Database to store the distributed running data!')
    else:
        # Sqlite is not suitable for distributed running.
        storage = "sqlite:///" + storage_path
        basic.log.info('Using Sqlite Database to store data!')
    
    # specify random number seed
    seed = 12345
    # choose the algorithm for DSE
    if alg == "sa":
        basic.log.info("Using Simulated Annealing for HLS DSE")
        sampler = SimulatedAnnealingSampler(seed=seed)
        study = optuna.create_study(storage=storage, study_name=study_name, sampler=sampler, direction="minimize",
                                    load_if_exists=True)
    else:
        if alg == 'motpe_d':
            basic.log.info("Using Discrete Encoding and MOTPE based Bayesian Optimization for HLS DSE")
            n_ei_candidates = 24
            sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, n_ei_candidates=n_ei_candidates,
                                                 seed=seed)
        elif alg == "motpe_f":
            basic.log.info("Using Float Encoding and MOTPE based Bayesian Optimization for HLS DSE")
            n_ei_candidates = 24
            sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, n_ei_candidates=n_ei_candidates,
                                                 seed=seed)
        elif alg == "motpe_fl":
            basic.log.info("Using Float Encoding and Latin MOTPE based Bayesian Optimization for HLS DSE")
            from bome.alg.tpe_sampler import TPESampler
            n_ei_candidates = 24
            sampler = TPESampler(n_startup_trials=n_startup_trials, n_ei_candidates=n_ei_candidates,
                                 seed=seed, init_method='lhs', 
                                 init_params=init_params)
        elif alg == "nsga":
            basic.log.info("Using NSGA-II for HLS DSE")
            sampler = optuna.samplers.NSGAIISampler(seed=seed)
        elif alg == "random":  # usually used to collect dataset
            basic.log.info("Using Random Sampling for HLS DSE")
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            basic.log.info("Using Float Encoding and MOTPE based Bayesian Optimization for HLS DSE")
            n_ei_candidates = 24
            sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, n_ei_candidates=n_ei_candidates,
                                                 seed=seed)

        if case in noLatList:
            study = optuna.create_study(storage=storage, study_name=study_name, sampler=sampler,
                                        directions=["minimize", "minimize", "minimize"], load_if_exists=False)
        else:
            study = optuna.create_study(storage=storage, study_name=study_name, sampler=sampler,
                                        directions=["minimize", "minimize", "minimize", "minimize"],
                                        load_if_exists=False)

    obj = partial(objective, basic=basic)

    if progress_callback:
        study.optimize(obj, n_trials=num, show_progress_bar=True, callbacks=[progress_callback])
    else:
        study.optimize(obj, n_trials=num, show_progress_bar=True)
    
    basic.log.info(f"Number of finished trials: {len(study.trials)}")

    if alg == "sa":
        optuna.visualization.plot_optimization_history(study)
        optuna.visualization.plot_parallel_coordinate(study)
        optuna.visualization.plot_param_importances(study)
        optuna.visualization.plot_contour(study)
        optuna.visualization.plot_slice(study)

        basic.log.info("Best trial:")
        basic.log.info(f"Value: {study.best_trial.value}")
        basic.log.info("Params: ")
        for key, value in study.best_trial.params.items():
            basic.log.info("{}: {}".format(key, value))
    else:
        basic.log.info("Pareto front:")
        trials = sorted(study.best_trials, key=lambda t: t.values)
        for trial in trials:
            basic.log.info("Trial#{}".format(trial.number))
            basic.log.info("Params: {}".format(trial.params))
        basic.log.info(f"Number of trials on the Pareto front: {len(study.best_trials)}")

        # Visualization
        if case in noLatList:
            trial_with_lowest_power = min(study.best_trials, key=lambda t: t.values[0])
            basic.log.info(f"Trial with lowest power: ")
            basic.log.info(f"\tnumber: {trial_with_lowest_power.number}")
            basic.log.info(f"\tparams: {trial_with_lowest_power.params}")
            basic.log.info(f"\tvalues: {trial_with_lowest_power.values}")

            trial_with_best_cp = min(study.best_trials, key=lambda t: t.values[1])
            basic.log.info(f"Trial with best cp: ")
            basic.log.info(f"\tnumber: {trial_with_best_cp.number}")
            basic.log.info(f"\tparams: {trial_with_best_cp.params}")
            basic.log.info(f"\tvalues: {trial_with_best_cp.values}")

            trial_with_smallest_area = min(study.best_trials, key=lambda t: t.values[2])
            basic.log.info(f"Trial with smallest area: ")
            basic.log.info(f"\tnumber: {trial_with_smallest_area.number}")
            basic.log.info(f"\tparams: {trial_with_smallest_area.params}")
            basic.log.info(f"\tvalues: {trial_with_smallest_area.values}")
            
            if isolated_path == "":
                fig_pwr_h = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0],
                                                                        target_name="power")
                fig_cp_h = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[1],
                                                                        target_name="cp")
                fig_area_h = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[2],
                                                                            target_name="area")
                fig_pwr_h.show()
                fig_cp_h.show()
                fig_area_h.show()

                fig_pwr_i = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0],
                                                                        target_name="power")
                fig_cp_i = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1],
                                                                    target_name="cp")
                fig_area_i = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[2],
                                                                        target_name="area")
                fig_pwr_i.show()
                fig_cp_i.show()
                fig_area_i.show()
        else:
            trial_with_lowest_power = min(study.best_trials, key=lambda t: t.values[0])
            basic.log.info(f"Trial with lowest power: ")
            basic.log.info(f"\tnumber: {trial_with_lowest_power.number}")
            basic.log.info(f"\tparams: {trial_with_lowest_power.params}")
            basic.log.info(f"\tvalues: {trial_with_lowest_power.values}")

            trial_with_best_lat = min(study.best_trials, key=lambda t: t.values[1])
            basic.log.info(f"Trial with best lat: ")
            basic.log.info(f"\tnumber: {trial_with_best_lat.number}")
            basic.log.info(f"\tparams: {trial_with_best_lat.params}")
            basic.log.info(f"\tvalues: {trial_with_best_lat.values}")

            trial_with_best_cp = min(study.best_trials, key=lambda t: t.values[2])
            basic.log.info(f"Trial with best cp: ")
            basic.log.info(f"\tnumber: {trial_with_best_cp.number}")
            basic.log.info(f"\tparams: {trial_with_best_cp.params}")
            basic.log.info(f"\tvalues: {trial_with_best_cp.values}")

            trial_with_smallest_area = min(study.best_trials, key=lambda t: t.values[3])
            basic.log.info(f"Trial with smallest area: ")
            basic.log.info(f"\tnumber: {trial_with_smallest_area.number}")
            basic.log.info(f"\tparams: {trial_with_smallest_area.params}")
            basic.log.info(f"\tvalues: {trial_with_smallest_area.values}")

            if isolated_path == "":
                fig_pwr_h = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0],
                                                                        target_name="power")
                fig_lat_h = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[1],
                                                                        target_name="lat")
                fig_cp_h = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[2],
                                                                        target_name="cp")
                fig_area_h = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[3],
                                                                            target_name="area")
                fig_pwr_h.show()
                fig_lat_h.show()
                fig_cp_h.show()
                fig_area_h.show()

                fig_pwr_i = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0],
                                                                        target_name="power")
                fig_lat_i = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1],
                                                                        target_name="lat")
                fig_cp_i = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[2],
                                                                    target_name="cp")
                fig_area_i = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[3],
                                                                        target_name="area")
                fig_pwr_i.show()
                fig_lat_i.show()
                fig_cp_i.show()
                fig_area_i.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HLS Design Space Exploration")
    parser.add_argument("--mode", type=str, help="The running mode of dse flow: hgp, impl.", default="hgp")
    parser.add_argument("--bench", type=str, help="The public benchmark name.", default="MachSuite")
    parser.add_argument("--case", type=str, help="The name of the benchmark.", default="bfs")
    parser.add_argument("--ver", type=str, help="The version of the benchmark.", default="bulk")
    parser.add_argument("--num", type=int, help="The number of optimization iterations.", default=100)
    parser.add_argument("--alg", type=str, help="The DSE algorithm.", default="motpe_fl")
    parser.add_argument("--device", type=str, help="FPGA device for implementation.", default="xc7vx485tffg1761-2")
    parser.add_argument("--clk", type=str, help="Clock period for implementation.", default="10")
    parser.add_argument("--encode", type=str, help="Float or discrete encoding style.", default="float")
    parser.add_argument("--space", type=str, help="Tree-structured or homo-structured design space.", default="tree")
    parser.add_argument("--parallel", type=bool, help="Using parallel running or not.", default=False)
    parser.add_argument("--process", type=int, help="The process number of current running.", default=1)

    args = parser.parse_args()

    mode = args.mode
    bench = args.bench
    case = args.case
    ver = args.ver
    num = args.num
    alg = args.alg
    encode = args.encode
    space = args.space
    parallel = args.parallel
    process = args.process

    root = os.path.abspath("./")
    # basic = HLSBasic(root, mode, bench, case, "", encode, num, alg, space, parallel, process, isolated="id123456")
    basic = HLSBasic(root, mode, bench, case, ver, encode, num, alg, space, parallel, process)
    
    iterationCallback = IterationCallback(basic.log)
    runDSE(basic, iterationCallback)

    basic.log.info("HLS Design Space Exploration is Done!")