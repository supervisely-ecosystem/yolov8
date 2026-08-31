"""Progress reporting for artifact uploads.

The SDK changed what an upload ``progress_cb`` receives: since 6.74.16 a plain callable
gets a byte increment, before that it got the multipart encoder monitor. An app that
handles only one of the two breaks the moment the SDK is upgraded, so this handles both.
"""


def make_upload_monitor(progress, pbar):
    """
    Build a progress callback for ``api.file.upload_bulk``.

    :param progress: Progress to report bytes into.
    :type progress: :class:`supervisely.Progress`
    :param pbar: Widget progress bar to keep in sync with it.
    :type pbar: tqdm like
    :returns: Callback accepting either a byte increment or an encoder monitor.
    :rtype: Callable
    """

    def upload_monitor(value):
        if hasattr(value, "bytes_read"):
            current, total = value.bytes_read, value.len
        else:
            current, total = progress.current + value, progress.total
        if progress.total == 0:
            progress.set(current, total, report=False)
        else:
            progress.set_current_value(current, report=False)
        pbar.update(progress.current - pbar.n)

    return upload_monitor
