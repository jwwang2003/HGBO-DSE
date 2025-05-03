"""
All inference related tasks will be imlpemented here:
- HGBO-DSE inference
"""

import optuna
from bome.hls_basic import HLSBasic
from bome import hls_dse

from backend.celery_app import app

@app.task(bind=True)
def run_inference_task(self, basic_config: HLSBasic):
    
    progress = 0

    class IterationCallback:
        def __init__(self):
            self.caseNumber = 0
            
        def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            self.caseNumber = self.caseNumber + 1
            # socketio.emit('progress_update', {'current': self.caseNumber})
            print("[DEBUG] Case:", self.caseNumber)
        
        def getCaseNumber(self):
            return self.caseNumber
    
    iterationCallback = IterationCallback()

    hls_dse.runDSE(basic_config, iterationCallback)
        
    return {'status': 'Inference completed!'}