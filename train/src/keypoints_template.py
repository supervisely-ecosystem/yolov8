from supervisely.geometry.graph import KeypointsTemplate

# build human template
human_template = KeypointsTemplate()
# add nodes
human_template.add_point(label="nose", row=635, col=427)
human_template.add_point(label="left_eye", row=597, col=404)
human_template.add_point(label="right_eye", row=685, col=401)
human_template.add_point(label="left_ear", row=575, col=431)
human_template.add_point(label="right_ear", row=723, col=425)
human_template.add_point(label="left_shoulder", row=502, col=614)
human_template.add_point(label="right_shoulder", row=794, col=621)
human_template.add_point(label="left_elbow", row=456, col=867)
human_template.add_point(label="right_elbow", row=837, col=874)
human_template.add_point(label="left_wrist", row=446, col=1066)
human_template.add_point(label="right_wrist", row=845, col=1073)
human_template.add_point(label="left_hip", row=557, col=1035)
human_template.add_point(label="right_hip", row=743, col=1043)
human_template.add_point(label="left_knee", row=541, col=1406)
human_template.add_point(label="right_knee", row=751, col=1421)
human_template.add_point(label="left_ankle", row=501, col=1760)
human_template.add_point(label="right_ankle", row=774, col=1765)
# add edges
human_template.add_edge(src="left_ankle", dst="left_knee")
human_template.add_edge(src="left_knee", dst="left_hip")
human_template.add_edge(src="right_ankle", dst="right_knee")
human_template.add_edge(src="right_knee", dst="right_hip")
human_template.add_edge(src="left_hip", dst="right_hip")
human_template.add_edge(src="left_shoulder", dst="left_hip")
human_template.add_edge(src="right_shoulder", dst="right_hip")
human_template.add_edge(src="left_shoulder", dst="right_shoulder")
human_template.add_edge(src="left_shoulder", dst="left_elbow")
human_template.add_edge(src="right_shoulder", dst="right_elbow")
human_template.add_edge(src="left_elbow", dst="left_wrist")
human_template.add_edge(src="right_elbow", dst="right_wrist")
human_template.add_edge(src="left_eye", dst="right_eye")
human_template.add_edge(src="nose", dst="left_eye")
human_template.add_edge(src="nose", dst="right_eye")
human_template.add_edge(src="left_eye", dst="left_ear")
human_template.add_edge(src="right_eye", dst="right_ear")
human_template.add_edge(src="left_ear", dst="left_shoulder")
human_template.add_edge(src="right_ear", dst="right_shoulder")


def dict_to_template(geometry_config):
    template = KeypointsTemplate()
    id_to_label = {}
    for key, value in geometry_config["nodes"].items():
        id_to_label[key] = value["label"]
        template.add_point(
            label=value["label"],
            row=value["loc"][0],
            col=value["loc"][1],
        )
    for edge in geometry_config["edges"]:
        template.add_edge(
            src=id_to_label[edge["src"]],
            dst=id_to_label[edge["dst"]],
        )
    return template
