from supervisely.app.widgets import SelectString


def create_layout():
    device_values = []
    device_names = []
    try:
        import torch

        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            for i in range(gpus):
                device_values.append(f"cuda:{i}")
                device_names.append(f"{torch.cuda.get_device_name(i)} (cuda:{i})")
    except:
        pass
    device_values.append("cpu")
    device_names.append("CPU")

    device_select = SelectString(
        values=device_values,
        labels=device_names,
        width_percent=30,
    )
