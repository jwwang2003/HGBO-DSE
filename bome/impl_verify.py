import argparse
import json
import os
from copy import deepcopy

from bome.get_ppa import getPPA, normalizePCA, normalizePLCA
from bome.hls_basic import HLSBasic
from bome.save_report import get_adb_rpt_verilog
from bome.tdm.design_space import config_tree_space
from bome.tdm.gen_config import genDirConfig
from bome.vitis_hls import VitisHLSRunner


NO_LATENCY_CASES = {"bfs", "fft", "nw", "stencil"}


class FixedTrial:
    def __init__(self, number, params):
        self.number = number
        self.params = params

    def suggest_float(self, name, low, high):
        value = self._get_param(name)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Trial {} parameter {} must be numeric.".format(self.number, name)) from exc

        if numeric < low or numeric > high:
            raise ValueError(
                "Trial {} parameter {}={} is outside [{}, {}].".format(
                    self.number,
                    name,
                    numeric,
                    low,
                    high,
                )
            )
        return numeric

    def suggest_categorical(self, name, choices):
        value = self._get_param(name)
        if value not in choices:
            raise ValueError(
                "Trial {} parameter {}={!r} is not in {}.".format(
                    self.number,
                    name,
                    value,
                    choices,
                )
            )
        return value

    def _get_param(self, name):
        if name not in self.params:
            raise ValueError("Trial {} is missing parameter {}.".format(self.number, name))
        return self.params[name]


def parse_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError("Invalid boolean value {!r}; expected true/false.".format(value))


def load_selection(selection_path):
    with open(selection_path, "r") as source:
        payload = json.load(source)

    trials = payload.get("trials", [])
    if not isinstance(trials, list) or not trials:
        raise ValueError("selected_trials.json must contain a non-empty trials list.")

    return payload


def replay_trial(basic, entry):
    trial_number = int(entry["trial"])
    fixed_trial = FixedTrial(trial_number, entry.get("params", {}))
    temp_dir, para_dict = config_tree_space(
        deepcopy(basic.static_config),
        basic.encode,
        fixed_trial,
        basic.params,
        basic.log,
    )

    dir_json = os.path.join(basic.hls_script_path, "dir_%d.json" % trial_number)
    dir_tcl = os.path.join(basic.hls_script_path, "dir_%d.tcl" % trial_number)
    hls_tcl = os.path.join(basic.hls_script_path, "hls_%d.tcl" % trial_number)
    genDirConfig(
        basic.encode,
        basic.params,
        deepcopy(basic.static_config),
        para_dict,
        dir_tcl,
        temp_dir,
        dir_json,
        basic.log,
    )
    write_hls_script(basic.hls_temp, hls_tcl, trial_number)

    basic.log.info("[ImplVerify] Running implementation for trial {}.".format(trial_number))
    hls_runner = VitisHLSRunner(
        tcl_script=hls_tcl,
        context=os.path.join(
            basic.isolated_folder_path or "./",
            "{}_trial_{}_vitis_hls".format(basic.context, trial_number),
        ),
        check=False,
    )
    hls_runner.run()
    if hls_runner.process is not None and hls_runner.process.returncode != 0:
        basic.log.error(hls_runner.get_stderr())
        raise RuntimeError("Vitis HLS failed for trial {}.".format(trial_number))

    rpt_list, _ = get_adb_rpt_verilog(
        basic.case,
        basic.top,
        "impl",
        basic.ori_prj_path,
        basic.dataset_path,
        trial_number,
        basic.process,
        "impl",
    )
    dict_ppa, _ = getPPA(basic.params, rpt_list, basic.log)
    ppa_path = os.path.join(basic.hls_script_path, "ppa_%d.json" % trial_number)
    with open(ppa_path, "w") as fout:
        fout.write(json.dumps(dict_ppa, indent=4))

    return {
        "trial": trial_number,
        "actual": normalized_actual_metrics(basic.case, basic.params, dict_ppa),
        "rawPpa": dict_ppa,
    }


def write_hls_script(template_path, output_path, trial_number):
    with open(template_path, "r") as source:
        content = source.read()

    content = content.replace("dir_test.tcl", "dir_%d.tcl" % trial_number)
    with open(output_path, "w") as target:
        target.write(content)


def normalized_actual_metrics(case, params, dict_ppa):
    if case in NO_LATENCY_CASES:
        npower, ncp, narea = normalizePCA(params, dict_ppa)
        return {
            "power": float(npower),
            "cp": float(ncp),
            "area": float(narea),
        }

    npower, nlat, ncp, narea = normalizePLCA(params, dict_ppa)
    return {
        "power": float(npower),
        "latency": float(nlat),
        "cp": float(ncp),
        "area": float(narea),
    }


def write_results(output_path, source_run_id, selected_trials, results):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    existing_runs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as source:
            try:
                existing_payload = json.load(source)
                existing_runs = existing_payload.get("verificationRuns", [])
            except json.JSONDecodeError:
                existing_runs = []

    study_trials = build_study_trials(selected_trials, results)
    verification_run = {
        "sourceRunId": source_run_id,
        "results": results,
        "study": {
            "sourceRunId": source_run_id,
            "trials": study_trials,
        },
    }
    merged_results = merge_results(existing_runs, results)
    merged_study_trials = merge_study_trials(existing_runs, study_trials)
    with open(output_path, "w") as target:
        target.write(json.dumps({
            "results": merged_results,
            "study": {
                "sourceRunId": source_run_id,
                "trials": merged_study_trials,
            },
            "verificationRuns": existing_runs + [verification_run],
        }, indent=4))


def build_study_trials(selected_trials, results):
    result_by_trial = {result.get("trial"): result for result in results}
    study_trials = []
    for entry in selected_trials:
        trial_number = int(entry["trial"])
        study_entry = deepcopy(entry)
        if trial_number in result_by_trial:
            study_entry["implementation"] = result_by_trial[trial_number]
        study_trials.append(study_entry)

    return sorted(study_trials, key=lambda entry: int(entry["trial"]))


def merge_study_trials(existing_runs, new_study_trials):
    by_trial = {}
    for run in existing_runs:
        for entry in run.get("study", {}).get("trials", []):
            by_trial[entry.get("trial")] = entry
    for entry in new_study_trials:
        by_trial[entry.get("trial")] = entry

    return [by_trial[trial] for trial in sorted(trial for trial in by_trial if trial is not None)]


def merge_results(existing_runs, new_results):
    by_trial = {}
    for run in existing_runs:
        for result in run.get("results", []):
            by_trial[result.get("trial")] = result
    for result in new_results:
        by_trial[result.get("trial")] = result

    return [by_trial[trial] for trial in sorted(trial for trial in by_trial if trial is not None)]


def main():
    parser = argparse.ArgumentParser(
        description="Replay selected HGBO-DSE trials through implementation for prediction verification."
    )
    parser.add_argument("--bench", type=str, default="MachSuite")
    parser.add_argument("--case", type=str, default="bfs")
    parser.add_argument("--ver", type=str, default="bulk")
    parser.add_argument("--alg", type=str, default="motpe_fl")
    parser.add_argument("--device", type=str, default="xc7vx485tffg1761-2")
    parser.add_argument("--clk", type=str, default="10")
    parser.add_argument("--encode", type=str, default="float")
    parser.add_argument("--space", type=str, default="tree")
    parser.add_argument("--parallel", type=parse_bool, default=False)
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--isolated", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--params-path", type=str, default=None)
    parser.add_argument("--project-path", type=str, default=None)
    parser.add_argument("--selection-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    root = os.path.abspath("./")
    selection = load_selection(args.selection_path)
    trials = selection["trials"]
    basic = HLSBasic(
        root,
        "impl",
        args.bench,
        args.case,
        args.ver,
        args.encode,
        len(trials),
        args.alg,
        args.space,
        args.parallel,
        args.process,
        args.device,
        args.clk,
        isolated=args.isolated,
        config_path=args.config_path,
        params_path=args.params_path,
        project_path=args.project_path,
    )

    results = []
    for entry in trials:
        result = replay_trial(basic, entry)
        results.append(result)
        basic.log.info("[ImplVerify] Completed implementation for trial {}.".format(result["trial"]))

    write_results(args.output_path, selection.get("sourceRunId"), trials, results)
    basic.log.info("[ImplVerify] Wrote impl_verification.json to {}.".format(args.output_path))


if __name__ == "__main__":
    main()
