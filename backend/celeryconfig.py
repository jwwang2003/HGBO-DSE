# broker_url = 'redis://localhost:6379/0'
# result_backend = 'redis://localhost:6379/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Asia/Shanghai'
enable_utc = True

task_routes = {
    'backend.worker.run_hls_dse_inference': 'high-priority',
}

task_annotations = {
    'backend.worker.run_hls_dse_inference': {'rate_limit': '10/m'}
}