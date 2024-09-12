# Monkey patching to avoid a problem with installing wrong onnxruntime requirements
# issue: https://github.com/ultralytics/ultralytics/issues/5093
def monkey_patching_fix():
    import importlib
    import ultralytics.nn.autobackend
    import ultralytics.utils.checks as checks
    check_requirements = checks.check_requirements  # save original function
    def check_requirements_dont_install(*args, **kwargs):
        kwargs["install"] = False
        return check_requirements(*args, **kwargs)
    checks.check_requirements = check_requirements_dont_install
    importlib.reload(ultralytics.nn.autobackend)