import os
from typing import List
import supervisely as sly
from supervisely.app.widgets import Progress
import src.globals as g
from supervisely.project.download import (
    download_to_cache,
    copy_from_cache,
    is_cached,
    get_cache_size
)


def download_project(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    dataset_infos: List[sly.DatasetInfo],
    use_cache: bool,
    progress: Progress,
):
    if os.path.exists(g.project_dir):
        sly.fs.clean_dir(g.project_dir)
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
    to_download = [info for info in dataset_infos if not is_cached(project_info.id, info.name)]
    cached = [info for info in dataset_infos if is_cached(project_info.id, info.name)]
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
    # download
    with progress(message="Downloading input data...", total=total) as pbar:
        download_to_cache(
            api=api,
            project_id=project_info.id,
            dataset_infos=to_download,
            log_progress=True,
            progress_cb=pbar.update,
        )
    # copy datasets from cache
    total = sum([get_cache_size(project_info.id, ds.name) for ds in dataset_infos])
    with progress(
        message="Retreiving data from cache...",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as pbar:
        dataset_names = [ds_info.name for ds_info in dataset_infos]
        copy_from_cache(
            project_id=project_info.id,
            dest_dir=g.project_dir,
            dataset_names=dataset_names,
            progress_cb=pbar.update
        )
