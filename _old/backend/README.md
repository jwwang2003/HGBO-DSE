# Compass Backend

The Compass backend is based on Python, Flash, Celery, & Redis.

## Features

* API authentication backend
   * API key generation
   * API key management
* User session authentication
   * Session management via Redis
   * Session auto expire / renew
* Celery task broker \(for inference tasks\)
* Full fledged REST API for:
   * API user authentication
   * Creating inference tickets
   * Uploading project files & configuring project settings
   * Running inference & transfering artifacts

## Backend Worflow

1. **Submit Inference Ticket**

   * The user submits an inference ticket to the backend.
   * The backend generates a **ticket ID** and creates a temporary folder based on this ID.
   * The inference ticket is placed into a **Redis queue**, acting as a task broker for the Celery backend.

2. **Submit Project Settings**

   * Project settings are submitted and associated with the **ticket ID**.

3. **Upload Project Files**

   * The user uploads the necessary project files, including:

     * Source code files (.c)
     * `config.yaml`
     * `params.yaml`

4. **Start Inference**

   * The system checks if all required files are present.
   * A **websocket** is opened, and the inference process begins.
   * Redis continues to manage the inference tasks via the Celery backend.

5. **Download Artifacts and Logs**

   * The user can download inference results, artifacts, and log files through the client-side interface.

**Redis Manager**

* During development, **AnotherRedisDesktopManager** was used to manage Redis tasks and monitor queues.
  ([GitHub Link to AnotherRedisDesktopManager](https://github.com/qishibo/AnotherRedisDesktopManager))

**End of Workflow**

## Configuration\(s\)

### Environment variables

```shell
SECRET_KEY=
MASTER_API_KEY=

AUTH_TOKEN_EXP=3600

REDIS_HOST='localhost'
REDIS_PORT=6379

AUTH_SESSION_MANAGER_REDIS_DB=0
CELERY_BROKER_REDIS_DB=1
```

## Planned improvements / features

* At the moment there is only one master API key which can login to the system via the REST API
   * This could be a single point of failure in terms of security
   * It also bad if multiple users share the same API key
   * In the future there should be a API key management system associated with 
   some sort of user authentication \(maybe Github OAuth?\) to improve security
   and controlled access to the backend
   * We should be able to associate API keys to "users" and also revoke API keys, etc.
   * __TL;DR implement more administrative tools & features__
* Somehow get whole backend server & inference python script running on separate Docker containers

## Problems & Solutions

The following is a small compilation of problems I have ran into along the way and how I solved them \(or overcome, because some problems we can only compromise\).

### Unstable conenction to external resources \(Github & Docker\)

Resources such as Github & Docker services connections are very unstable for some reason. They bascially do not work. So instead of using Github for version tracking and deployment, I am using Gitlab. Then, I am building my Docker images on a separate x86 platform machine and then manually downloading it to the remote server via `scp`.

The main reason for this issue is because of it being a school server and it is somewhat isolated from the outside \(for security purposes of course\), which makes sense why some things might not be accessible. But this kind of security measures just add soo much extra hassle into development tasks.

### `RuntimeError: CUDA error: initialization error` with celery worker

```shell
[2025-05-02 16:43:12,720: ERROR/ForkPoolWorker-56] Task backend.tasks.run_inference_task[05cb3ad3-dcc6-492d-b08c-b26a6c9b9347] raised unexpected: RuntimeError('CUDA error: initialization error\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with TORCH_USE_CUDA_DSA to enable device-side assertions.\n')
Traceback (most recent call last):
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/celery/app/trace.py", line 453, in trace_task
    R = retval = fun(*args, **kwargs)
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/celery/app/trace.py", line 736, in __protected_call__
    return self.run(*args, **kwargs)
  File "/home/jwwang/HGBO-DSE/backend/tasks.py", line 46, in run_inference_task
    hls_dse.runDSE(
  File "/home/jwwang/HGBO-DSE/bome/hls_dse.py", line 197, in runDSE
    study.optimize(obj, n_trials=num, show_progress_bar=True, callbacks=[progress_callback])
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/optuna/study/study.py", line 443, in optimize
    _optimize(
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/optuna/study/_optimize.py", line 66, in _optimize
    _optimize_sequential(
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/optuna/study/_optimize.py", line 163, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/optuna/study/_optimize.py", line 251, in _run_trial
    raise func_err
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/optuna/study/_optimize.py", line 200, in _run_trial
    value_or_values = func(trial)
  File "/home/jwwang/HGBO-DSE/bome/hls_dse.py", line 70, in objective
    dictPPA['IMPL'] = getGNNPred(prj_path, hls_attr, case)
  File "/home/jwwang/HGBO-DSE/bome/hgp_pred.py", line 72, in getGNNPred
    lut = lut_pred(std_dataframe)
  File "/home/jwwang/HGBO-DSE/bome/pred/pred_lut.py", line 8, in lut_pred
    data.to(device)
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/torch_geometric/data/data.py", line 362, in to
    return self.apply(
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/torch_geometric/data/data.py", line 342, in apply
    store.apply(func, *args)
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/torch_geometric/data/storage.py", line 201, in apply
    self[key] = recursive_apply(value, func)
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/torch_geometric/data/storage.py", line 897, in recursive_apply
    return func(data)
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/torch_geometric/data/data.py", line 363, in <lambda>
    lambda x: x.to(device=device, non_blocking=non_blocking), *args)
  File "/home/jwwang/.conda/envs/hgbo/lib/python3.9/site-packages/torch/cuda/__init__.py", line 319, in _lazy_init
    torch._C._cuda_init()
RuntimeError: CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.
```

**Solution reference**: https://stackoverflow.com/a/76945563

**Solution**: add the following parameter, `--pool=threads`,  when launching the celery worker instance. A full example of the shell command would be like so: `celery -A backend.main worker --pool=threads  --loglevel=DEBUG` .

This problem mainly stems from using Celery as an asynchronous task broker. The exact issue what causes this is unclear.

> When Celery tasks, especially those involving CUDA and PyTorch, encounter asynchronous errors, **it's often related to issues with the way CUDA is used within the Celery worker process, particularly when combining it with different concurrency models like Gevent or Eventlet**. These errors can stem from asynchronous operations not completing before subsequent operations are initiated, leading to problems like device-side asserts or memory access violations.
— Google AI Overview


## References

* Python Docker image
   * https://hub.docker.com/_/python
   * `3.9-bookworm`
   * https://github.com/docker-library/python/blob/5f041dab48cbaa33eef235fb94ddf07c61a53ad7/3.9/bookworm/Dockerfile

* https://github.com/miguelgrinberg/flask-celery-example/tree/master
   * A reference repository that integrates both Flask & Celery together and a good starting point

* Celery
   * https://github.com/celery/celery
   * https://docs.celeryq.dev/en/stable/getting-started/introduction.html
   * https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html
   * https://lip17.medium.com/hands-on-learn-python-celery-in-30-minutes-9544aabb70b1
   * https://medium.com/@Aman-tech/celery-with-flask-d1f1c555ceb7
* https://redis.io/docs/latest/commands/setex/
   * Redis documentation
   * https://github.com/qishibo/AnotherRedisDesktopManager

* REST API / Flask related
   * https://www.geeksforgeeks.org/upload-multiple-files-with-flask/