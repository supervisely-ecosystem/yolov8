import os
import time
from dotenv import load_dotenv
import supervisely as sly
from supervisely.nn.benchmark import ObjectDetectionBenchmark, InstanceSegmentationBenchmark

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api()

gt_project_id = 39099
gt_dataset_ids = [64270]
model_session_id = "http://localhost:8000"
# model_session_id = 52141

# 1. Initialize benchmark
while True:
    try:
        bench = ObjectDetectionBenchmark(api, gt_project_id, gt_dataset_ids)
        bench.api.retry_count = 1
        bench.run_speedtest(model_session_id, gt_project_id, batch_sizes=[1])
    except Exception as e:
        print(f"Failed: {e}")
        continue
    finally:
        time.sleep(2)



# 2. Run evaluation
# This will run inference with the model and calculate metrics.
# Evaluation results will be saved in the "./benchmark" directory locally.
bench.run_evaluation(model_session=model_session_id)

# 3. Generate charts and dashboards
# This will generate visualization files and save them locally.
bench.visualize()

# 4. Upload to Supervisely Team Files
# To open the generated visualizations in the web interface, you need to upload them to Team Files.
bench.upload_visualizations(f"/model-benchmark/{gt_project_id}/visualizations")
