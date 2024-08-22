import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import supervisely as sly


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")


# monkey patching for ConfusionMatrix plot
def custom_plot(self, normalize=True, save_dir="", names=(), on_plot=None):
    """Modified plot function to handle long class names"""
    if any(len(name) > 70 for name in names):
        sly.logger.warn("Some class names are too long, they will be truncated...")
        names = [f"{name[:50]}..." for name in names]

    long_names = any(len(name) > 25 for name in names)
    super_long_names = any(len(name) > 40 for name in names)

    import seaborn  # scope for faster 'import ultralytics'

    array = self.matrix / (
        (self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1
    )  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc, nn = self.nc, len(names)  # number of classes, names
    seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    if long_names:
        plt.tick_params(axis="both", which="major", labelsize=7)
    elif super_long_names:
        plt.tick_params(axis="both", which="major", labelsize=5)
    labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
    ticklabels = (list(names) + ["background"]) if labels else "auto"
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        seaborn.heatmap(
            array,
            ax=ax,
            annot=nc < 30,
            annot_kws={"size": 8},
            cmap="Blues",
            fmt=".2f" if normalize else ".0f",
            square=True,
            vmin=0.0,
            xticklabels=ticklabels,
            yticklabels=ticklabels,
        ).set_facecolor((1, 1, 1))
    title = "Confusion Matrix" + " Normalized" * normalize
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
    fig.savefig(plot_fname, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(plot_fname)


# def deploy_the_best_model():
#     from serve.src.main import YOLOv8Model

#     m = YOLOv8Model(
#         model_dir="app_data",
#         use_gui=True,
#         custom_inference_settings=os.path.join(root_source_path, "serve", "custom_settings.yaml"),
#     )

#     pass


def get_eval_results_dir(api, task_id, project_info):
    task_info = api.task.get_info_by_id(task_id)
    task_dir = f"{task_id}_task_{task_info['meta']['app']['name']}"
    eval_res_dir = f"/model-benchmark/evaluation/{project_info.id}_{project_info.name}/{task_dir}/"
    eval_res_dir = api.storage.get_free_dir_name(sly.env.team_id(), eval_res_dir)
    return eval_res_dir
