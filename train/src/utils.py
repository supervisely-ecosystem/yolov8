import supervisely as sly


def get_train_val_sets(project_dir, split_method, split_counts):
    sly.logger.info(f"Split method for train/val is '{split_method}'")
    if split_method == "Random":
        val_part = split_counts["val"] / split_counts["total"]
        project = sly.Project(project_dir, sly.OpenMode.READ)
        n_images = project.total_items
        val_count = round(val_part * n_images)
        train_count = n_images - val_count
        train_set, val_set = sly.Project.get_train_val_splits_by_count(
            project_dir, train_count, val_count
        )
        return train_set, val_set
    elif split_method == "Based on image tags":
        # train_tag_name = state["trainTagName"]
        # val_tag_name = state["valTagName"]
        # add_untagged_to = state["untaggedImages"]
        # train_set, val_set = sly.Project.get_train_val_splits_by_tag(
        #     project_dir, train_tag_name, val_tag_name, add_untagged_to
        # )
        # return train_set, val_set
        pass
    elif split_method == "Based on datasets":
        # train_datasets = state["trainDatasets"]
        # val_datasets = state["valDatasets"]
        # train_set, val_set = sly.Project.get_train_val_splits_by_dataset(
        #     project_dir, train_datasets, val_datasets
        # )
        # return train_set, val_set
        pass
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")
