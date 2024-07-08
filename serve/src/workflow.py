import supervisely as sly


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            self.is_compatible = self.check_instance_ver_compatibility()
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self._min_instance_version = (
            "6.9.31" if min_instance_version is None else min_instance_version
        )
    
    def check_instance_ver_compatibility(self):
        if self.api.instance_version < self._min_instance_version:
            sly.logger.info(
                f"Supervisely instance version does not support workflow and versioning features. To use them, please update your instance minimum to version {self._min_instance_version}."
            )
            return False
        return True

    @check_compatibility
    def add_input(self, model_params: dict):
        checkpoint_url = model_params.get("checkpoint_url")
        checkpoint_name = model_params.get("checkpoint_name")
        if checkpoint_name and "v8" in checkpoint_name:
            model_name = "YOLOv8"
        elif checkpoint_name and "v9" in checkpoint_name:
            model_name = "YOLOv9"
        else:
            model_name = "Custom Model"
        meta = {"customNodeSettings": {"title": f"<h4>Serve {model_name}</h4>"}}
        if checkpoint_url and self.api.file.exists(sly.env.team_id(), checkpoint_url):
            self.api.app.workflow.add_input_file(checkpoint_url, model_weight=True, meta=meta)
        else:
            sly.logger.warn(f"Checkpoint {checkpoint_url} not found. Cannot set workflow input")

    @check_compatibility
    def add_output(self):
        raise NotImplementedError("add_output is not implemented in this workflow")
