from pathlib import Path
import supervisely as sly
from ultralytics import YOLO
import globals as g


def export_weights(format: str):
    # Download model
    remote_file_path = Path(g.STATE.selected_file)
    local_file_path = Path(g.LOCAL_WEIGHTS_DIR) / Path(remote_file_path).name
    g.Api.file.download(g.STATE.selected_team, remote_file_path.as_posix(), local_file_path)

    # Load a model
    model = YOLO(local_file_path)

    # Export the model
    output_file_path = model.export(format=format, dynamic=False)
    output_file_path = Path(output_file_path)

    # Upload the model
    dst_path = Path(remote_file_path).parent / output_file_path.name
    g.Api.file.upload(g.STATE.selected_team, output_file_path, dst_path.as_posix())

    # Set output
    sly.output.set_directory(dst_path.parent.as_posix())


def main():
    export_weights(g.STATE.format)

if __name__ == "__main__":
    sly.main_wrapper("main", main)
