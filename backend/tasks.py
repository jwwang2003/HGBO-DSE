from helpers.celery_app import app


@app.task(name="backend.tasks.predict_impl_ppa")
def predict_impl_ppa_task(prj_path, hls_attr, case):
    from bome.hgp_pred import getGNNPred

    return getGNNPred(prj_path, list(hls_attr), case)
