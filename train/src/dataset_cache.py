import json
import os
from pathlib import Path
from typing import List
import supervisely as sly
from supervisely.app.widgets import Progress
import src.globals as g


def download_project(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    dataset_infos: List[sly.DatasetInfo],
    use_cache: bool,
    progress: Progress,
):
    if not use_cache:
        dataset_ids = [dataset_info.id for dataset_info in dataset_infos]
        total = sum([dataset_info.images_count for dataset_info in dataset_infos])
        with progress(message="Downloading input data...", total=total) as pbar:
            sly.download(
                api=api,
                project_id=project_info.id,
                dest_dir=g.project_dir,
                dataset_ids=dataset_ids,
                log_progress=True,
                progress_cb=pbar.update,
            )
        return

    # get datasets to download and cached
    dirs = [p.resolve().as_posix() for p in Path("/apps_cache", str(project_info.id)).iterdir()]
    sly.logger.info("apps_cache/%s contents:\n%s", str(project_info.id), "\n".join(dirs))
    debug_data = {
        info.name: {
            "isdir": os.path.isdir(os.path.join("/apps_cache", str(project_info.id), info.name)),
            "path": os.path.join("/apps_cache", str(project_info.id), info.name)
        }
        for info in dataset_infos
    }
    debug_msg = json.dumps(debug_data, indent=4)
    sly.logger.info(debug_msg)
    to_download = [info for info in dataset_infos if not sly.is_cached(project_info.id, info.name)]
    cached = [info for info in dataset_infos if sly.is_cached(project_info.id, info.name)]
    if len(cached) == 0:
        log_msg = "No cached datasets found"
    else:
        log_msg = "Using cached datasets: " + ", ".join(
            f"{ds_info.name} ({ds_info.id})"
            for ds_info in cached
        )
    sly.logger.info(log_msg)
    if len(to_download) == 0:
        log_msg = "All datasets are cached. No datasets to download"
    else:
        log_msg = "Downloading datasets: " + ", ".join(
            f"{ds_info.name} ({ds_info.id})"
            for ds_info in to_download
        )
    sly.logger.info(log_msg)
    # get images count
    total = sum([ds_info.images_count for ds_info in to_download])
    # clean project dir
    if os.path.exists(g.project_dir):
        sly.fs.clean_dir(g.project_dir)
    # download
    with progress(message="Downloading input data...", total=total) as pbar:
        sly.download_to_cache(
            api=api,
            project_id=project_info.id,
            dataset_infos=to_download,
            log_progress=True,
            progress_cb=pbar.update,
        )
    # copy datasets from cache
    total = sum([sly.get_cache_size(project_info.id, ds.name) for ds in dataset_infos])
    with progress(message="Retreiving data from cache...", total=total) as pbar:
        for ds_info in dataset_infos:
            sly.copy_from_cache(
                project_id=project_info.id,
                dest_dir=g.project_dir,
                dataset_name=ds_info.name,
                progress_cb=pbar.update
            )
