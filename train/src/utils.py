import supervisely as sly


def get_train_val_sets(project_dir, train_val_split_widget, api, project_id):
    split_method = train_val_split_widget._content.get_active_tab()
    sly.logger.info(f"Split method for train/val is '{split_method}'")
    if split_method == "Random":
        random_split_table = train_val_split_widget._get_random_content()._widgets[0]
        split_counts = random_split_table.get_splits_counts()
        val_part = split_counts["val"] / split_counts["total"]
        project = sly.Project(project_dir, sly.OpenMode.READ)
        n_images = project.total_items
        val_count = round(val_part * n_images)
        train_count = n_images - val_count
        train_set, val_set = sly.Project.get_train_val_splits_by_count(
            project_dir, train_count, val_count
        )
        return train_set, val_set
    elif split_method == "Based on item tags":
        train_tag_name = train_val_split_widget._train_tag_select.get_selected_name()
        val_tag_name = train_val_split_widget._val_tag_select.get_selected_name()
        add_untagged_to = train_val_split_widget._untagged_select.get_value()
        train_set, val_set = sly.Project.get_train_val_splits_by_tag(
            project_dir, train_tag_name, val_tag_name, add_untagged_to
        )
        return train_set, val_set
    elif split_method == "Based on datasets":
        train_ds_ids = train_val_split_widget._train_ds_select.get_selected_ids()
        val_ds_ids = train_val_split_widget._val_ds_select.get_selected_ids()
        ds_infos = api.dataset.get_list(project_id)
        train_ds_names, val_ds_names = [], []
        for ds_info in ds_infos:
            if ds_info.id in train_ds_ids:
                train_ds_names.append(ds_info.name)
            if ds_info.id in val_ds_ids:
                val_ds_names.append(ds_info.name)
        train_set, val_set = sly.Project.get_train_val_splits_by_dataset(
            project_dir, train_ds_names, val_ds_names
        )
        return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")
