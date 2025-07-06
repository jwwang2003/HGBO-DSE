from .celery_app import report_context

if __name__ == "__main__":
    # fire off 3 tasks
    results = [report_context.delay() for _ in range(3)]
    print("Submitted 3 tasks, waiting for results…")

    for i, r in enumerate(results, 1):
        cid = r.get(timeout=10)
        print(f" Task #{i} returned UUID = {cid}")