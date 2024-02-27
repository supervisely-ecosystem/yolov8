import os
from typing import Tuple, List
import supervisely as sly
from supervisely.app.widgets import Progress
import src.globals as g


def is_project_cached(project_id):
    cache_project_dir = os.path.join(g.cache_dir, str(project_id))
    return sly.fs.dir_exists(cache_project_dir)


def split_by_cache(project_id, dataset_ids) -> Tuple[set, set]:
    cache_project_dir = os.path.join(g.cache_dir, str(project_id))
    to_download = set(dataset_ids)
    cached = set()
    if not sly.fs.dir_exists(cache_project_dir):
        return to_download, cached
    for dataset_id in dataset_ids:
        cache_dataset_dir = os.path.join(cache_project_dir, str(dataset_id))
        if sly.fs.dir_exists(cache_dataset_dir):
            cached.add(dataset_id)
            to_download.remove(dataset_id)
    
    return to_download, cached


def download_project(api: sly.Api, project_info: sly.ProjectInfo, dataset_infos: List[sly.DatasetInfo], use_cache: bool, progress: Progress):
    dataset_ids = [dataset_info.id for dataset_info in dataset_infos]
    if not use_cache:
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

    dataset_infos_dict = {dataset_info.id:dataset_info for dataset_info in dataset_infos}
    # get datasets to download and cached
    to_download, cached = split_by_cache(project_info.id, dataset_ids)
    # get images count
    total = sum([dataset_infos_dict[dataset_id].items_count for dataset_id in to_download])
    # clean project dir
    if os.path.exists(g.project_dir):
        sly.fs.clean_dir(g.project_dir)
    
    # TODO Check if to_download is empty
    
    # download
    with progress(message="Downloading input data...", total=total) as pbar:
        sly.download(
            api=api,
            project_id=project_info.id,
            dest_dir=g.project_dir,
            dataset_ids=to_download,
            log_progress=True,
            progress_cb=pbar.update,
        )
    # cache downloaded datasets
    for dataset_id in to_download:
        dataset_name = dataset_infos_dict[dataset_id].name
        dataset_dir = os.path.join(g.project_dir, project_info.name, dataset_name)
        cache_dataset_dir = os.path.join(g.cache_dir, str(project_info.id), str(dataset_id))
        with progress(message="Saving data to cache...") as pbar:
            sly.fs.copy_dir_recursively(dataset_dir, cache_dataset_dir, progress_cb=pbar.update)
    # copy cached datasets
    for dataset_id in cached:
        dataset_name = dataset_infos_dict[dataset_id].name
        cache_dataset_dir = os.path.join(g.cache_dir, str(project_info.id), str(dataset_id))
        dataset_dir = os.path.join(g.project_dir, project_info.name, dataset_name)
        with progress(message="Retreiving data from cache...") as pbar:
            sly.fs.copy_dir_recursively(cache_dataset_dir, dataset_dir, progress_cb=pbar.update)
