import json
import os
import logging
from logging import Logger
from bome.log import setup_logger
from bome.common import *

from helpers.context import with_context, get_context_id

@with_context
class HLSBasic(object):
    def __init__(
        self,
        root: str,
        mode: str,
        bench: str,
        case: str,
        ver: str,           # (optional) 
        encode:str,
        num=100,
        alg='motpe_f',
        space='tree',
        parallel=False,
        process=1,
        device='xc7vx485tffg1761-2',
        clk='10',
        log: Logger=None,
        isolated: str=None  # (optional) specifies a temporary directory for the working directory
    ):
        self.root = root
        self.mode = mode
        self.bench = bench
        self.case = case
        self.ver = ver
        self.num = num
        self.alg = alg
        self.space = space
        self.parallel = parallel
        self.process = process
        self.device = device
        self.clk = clk
        self.encode = encode
        self.config_path = None
        self.params_path = None
        self.ori_prj_path = None
        self.dataset_path = None
        self.temp_path = None
        self.hls_temp = None
        self.hls_script_path = None
        self.static_config = None
        self.params = None
        self.top = None
        self.tempDir = None
        self.paraDict = None
        self.log = log
        self.isolated = isolated  # New isolated folder parameter
        self.isolated_folder_path = None
        self.context: str = None

        # Initialization
        if self.isolated:
            self.create_isolated_folder()  # Create isolated folder if specified
        
        if not self.context:
            self.context = get_context_id()
        
        if not self.log:
            self.log = setup_logger(context=self.context)
        
        self.get_config_path()
        self.get_params_path()
        self.get_ori_prj_path()
        self.get_dataset_path()
        self.get_temp_path()
        self.get_hls_temp_path()
        self.get_hls_script_path()
        self.static_config = getYaml(self.config_path, self.log)
        self.params = getYaml(self.params_path, self.log)
        self.config_space()
        self.gen_space_temp()
        self.gen_hls_temp_script()

    def create_isolated_folder(self):
        """Creates a folder for isolated files if specified"""
        self.isolated_folder_path = os.path.join(self.root, self.isolated)
        createFolder(self.isolated_folder_path)
    
    def get_cwd(self):
        if self.isolated:
            return self.isolated_folder_path
        return self.root

    def get_config_path(self):
        if self.ver == "":
            self.config_path = os.path.join(
                self.get_cwd(), 
                'config', 
                self.bench, 
                self.case + '_config.yaml'
            )
        else:
            self.config_path = os.path.join(
                self.get_cwd(), 
                'config', 
                self.bench, 
                self.case + '_' + self.ver + '_config.yaml'
            )
        return self.config_path

    def get_params_path(self):
        if self.ver == "":
            self.params_path = os.path.join(
                self.get_cwd(), 
                'config', 
                self.bench,
                self.case + '_params.yaml'
            )
        else:
            self.params_path = os.path.join(
                self.get_cwd(), 
                'config', 
                self.bench, 
                self.case + '_' + self.ver + '_params.yaml'
            )
        return self.params_path

    def get_ori_prj_path(self):
        self.ori_prj_path = os.path.join(
            self.isolated_folder_path if self.isolated else os.path.join(self.root, 'benchmark'), 
            self.bench,
            self.case
        )
        if self.ver != "":
            self.ori_prj_path = os.path.join(self.ori_prj_path, self.ver)
        return self.ori_prj_path

    def get_dataset_path(self):
        if self.mode == 'impl':
            self.dataset_path = os.path.join(
                os.path.join(
                    self.isolated_folder_path if self.isolated else self.root,
                    "artifacts" if self.isolated else 'dse_ds'
                ),
                self.bench, 
                self.mode + '_ds', self.case
            )
        else:
            self.dataset_path = os.path.join(
                os.path.join(
                    self.isolated_folder_path if self.isolated else self.root,
                    "artifacts" if self.isolated else 'dse_ds'
                ),
                self.bench, 
                self.alg + '_ds', self.case
            )
        if self.ver != "":
            self.dataset_path = os.path.join(self.dataset_path, self.ver, 'p' + str(self.process))
        else:
            self.dataset_path = os.path.join(self.dataset_path, 'p' + str(self.process))
        createFolder(self.dataset_path)
        return self.dataset_path

    def get_temp_path(self):
        self.temp_path = os.path.join(self.dataset_path, 'temp')
        createFolder(self.temp_path)
        return self.temp_path

    def gen_space_temp(self):
        with open(os.path.join(self.temp_path, 'template.json'), "w") as fout:
            fout.write(json.dumps(self.tempDir, indent=4))

    def get_hls_temp_path(self):
        self.hls_temp = os.path.join(self.temp_path, 'hls_temp.tcl')
        return self.hls_temp

    def get_hls_script_path(self):
        self.hls_script_path = os.path.join(self.dataset_path, 'script')
        createFolder(self.hls_script_path)
        return self.hls_script_path

    def gen_hls_temp_script(self):
        fw = open(self.hls_temp, "w")
        fw.write('cd {}\n'.format(self.ori_prj_path))
        if self.mode == 'impl':
            fw.write('open_project {}_{}_prj_p{}\n'.format(self.case, self.mode, str(self.process)))
        else:
            fw.write('open_project {}_{}_prj_p{}\n'.format(self.case, self.alg, str(self.process)))
        fw.write('add_files {}.c\n'.format(self.case))
        if self.bench == 'MachSuite':
            fw.write('add_files local_support.c\n')
        elif self.bench == 'Polybench':
            fw.write('add_files {}.h\n'.format(self.case))
        else:
            pass

        fw.write('set_top {}\n'.format(self.top))

        fw.write('open_solution -reset solution\n')
        fw.write('set_part {}\n'.format(self.device))
        fw.write('create_clock -period {}\n'.format(self.clk))
        fw.write('source {}/dir_test.tcl\n'.format(self.hls_script_path))

        fw.write('csynth_design\n')
        if self.mode == 'impl':
            fw.write('export_design -flow impl -rtl verilog -format ip_catalog\n')
        fw.write('exit\n')
        fw.close()

    # Method of parsing the configuration space (TDM)
    def config_space(self):
        self.log.info("[INFO] Configuring the design space...")
        tempDir = {"Option": basicOption(), "Function": {}, "Loop": {}, "Array": {}, "Interface": {}, "Operation": {}}

        config = self.static_config
        top = config["top"][0]
        funcList = config["funcList"]
        loopList = config["loopList"]
        arrList = config["arrList"]
        interList = config["interList"]
        dictOp = config["dictOp"]

        paraFunc = {}
        paraLoop = {}
        paraArr = {}
        paraInter = {}
        paraOp = {}
        paraDict = {}

        cntF = cntL = cntA = cntI = cntO = 0

        paraFunc.update({'TFB_0': 0.0})
        for func in funcList:
            if func is None:
                break
            else:
                dictFunc = {
                    func: {
                        "INLINE": [],
                        "BALANCE": []
                    }
                }
                tempDir["Function"].update(dictFunc)
                paraFunc.update({('FI_' + str(cntF)): 0.0})
                paraFunc.update({('FB_' + str(cntF)): 0.0})
                cntF = cntF + 1
        paraDict.update(paraFunc)

        for group in loopList:
            if group is None:
                break
            else:
                for loop in loopList[group]['level']:
                    dictLoop = {
                        loop: {}
                    }
                    if loop in loopList[group]['unroll']:
                        unroll = {
                            "UNROLL": {
                                "-factor": []
                            }
                        }
                        dictLoop[loop].update(unroll)
                        paraLoop.update({('LU_' + str(cntL)): 0.0})
                    if loop in loopList[group]['pipeline']:
                        ppl = {
                            "PIPELINE": {
                                "-style": [],
                                "-state": []
                            }
                        }
                        dictLoop[loop].update(ppl)
                        paraLoop.update({('LP_' + str(cntL)): 0.0})
                    if loop in loopList[group]['flatten']:
                        flt = {
                            "FLATTEN": {
                                "-state": []
                            }
                        }
                        dictLoop[loop].update(flt)
                        paraLoop.update({('LF_' + str(cntL)): 0.0})
                    cntL = cntL + 1
                    loopList[group].update(dictLoop)
        tempDir["Loop"].update(loopList)
        paraDict.update(paraLoop)

        for arr in arrList:
            if arr is None:
                break
            else:
                dictArr = {
                    arr: {
                        "PARTITION": {
                            "-factor": [],
                            "-type": []},
                        "RESHAPE": {
                            "-factor": [],
                            "-type": []},
                        "STORAGE": {
                            "-type": [],
                            "-impl": [],
                            "-latency": []},
                    }
                }
                tempDir["Array"].update(dictArr)
                paraArr.update({('APF_' + str(cntA)): 0.0})
                paraArr.update({('APT_' + str(cntA)): 0.0})
                paraArr.update({('ARF_' + str(cntA)): 0.0})
                paraArr.update({('ART_' + str(cntA)): 0.0})
                paraArr.update({('AST_' + str(cntA)): 0.0})
                paraArr.update({('ASI_' + str(cntA)): 0.0})
                paraArr.update({('ASL_' + str(cntA)): 0.0})
                cntA = cntA + 1
        paraDict.update(paraArr)

        for inter in interList:
            if inter is None:
                break
            else:
                dictInter = {
                    inter: {
                        "PARTITION": {
                            "-factor": [],
                            "-type": []},
                        "RESHAPE": {
                            "-factor": [],
                            "-type": []},
                    }
                }
                tempDir["Interface"].update(dictInter)
                paraInter.update({('IPF_' + str(cntI)): 0.0})
                paraInter.update({('IPT_' + str(cntI)): 0.0})
                paraInter.update({('IRF_' + str(cntI)): 0.0})
                paraInter.update({('IRT_' + str(cntI)): 0.0})
                cntI = cntI + 1
        paraDict.update(paraInter)

        for optype in dictOp:
            for key in dictOp[optype]:
                if key is None:
                    break
                else:
                    tempDir["Operation"].update({key: {}})
                    opList = dictOp[optype][key]
                    for op in opList:
                        if op is None:
                            break
                        else:
                            itemOp = {
                                op: {
                                    "-impl": [],
                                    "-latency": []
                                }
                            }
                            tempDir["Operation"][key].update(itemOp)
                            if optype == 'int':
                                paraOp.update({('OPI_' + str(cntO)): 0.0})
                                paraOp.update({('OPL_' + str(cntO)): 0.0})
                            elif optype == 'float':
                                paraOp.update({('FOPI_' + str(cntO)): 0.0})
                                paraOp.update({('FOPL_' + str(cntO)): 0.0})
                            elif optype == 'double':
                                paraOp.update({('DOPI_' + str(cntO)): 0.0})
                                paraOp.update({('DOPL_' + str(cntO)): 0.0})
                            elif optype == 'half':
                                paraOp.update({('HOPI_' + str(cntO)): 0.0})
                                paraOp.update({('HOPL_' + str(cntO)): 0.0})
                            cntO = cntO + 1
        paraDict.update(paraOp)

        self.log.info("[INFO] Total parameters: " + str(len(paraDict)))
        self.log.info("[INFO] Design space configuration done!")
        self.top = top
        self.tempDir = tempDir
        self.paraDict = paraDict
        return self.top, self.tempDir, self.paraDict
