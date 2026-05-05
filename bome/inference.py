import os


HOST_INFERENCE = "host"
REMOTE_INFERENCE = "remote"
VALID_INFERENCE_MODES = {HOST_INFERENCE, REMOTE_INFERENCE}


def normalize_inference_mode(mode=None):
    value = (mode or os.getenv("HGBO_INFERENCE_MODE", HOST_INFERENCE)).strip().lower()
    if value not in VALID_INFERENCE_MODES:
        raise ValueError(
            "Invalid inference mode {!r}; expected one of {}".format(
                value, sorted(VALID_INFERENCE_MODES)
            )
        )
    return value


def dispatch_remote_inference(prj_path, hls_attr, case, timeout=None):
    from backend.mcp_client import predict_impl_ppa

    return predict_impl_ppa(prj_path, list(hls_attr), case, timeout=timeout)
