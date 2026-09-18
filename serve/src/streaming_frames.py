"""Serve a track's frames from one streamed decode instead of one request per frame.

Why this exists
---------------
Every tracking app in the Ecosystem gets its frames through
``InferenceImageCache``, which ends at ``POST videos.download-frame`` — one HTTP
round trip per frame, proxied to ``cdn-app/image-converter``. That service
decodes at most two frames concurrently per process (a hardcoded semaphore added
to stop it OOM-killing), so under load the queue in front of it is the
bottleneck: ``/v3/videos.download-frame`` was measured at p95 120.8 s against a
59 s istio route timeout. Tracking stops with a 504 that names nothing.

Auto Track makes this worse rather than better. It fans out to one serving
session per geometry, so a single track can ask seven apps for frames at once,
and any one of them stalling fails the whole track.

Serve Segment Anything 2.1 was moved off that path and is the only tracker that
kept working. This does the same thing here, without touching the SDK: it wraps
``self.cache`` and replaces only how *video frames* are obtained — decoding the
requested span in one pass, straight from the video, with PyAV. Everything else
about the cache (images, persistence, the ``/smart_cache`` endpoint) is
delegated untouched.

Design notes
------------
* **Wrapping the cache, not overriding a method.** The SDK builds its
  ``TrackerInterface`` with ``frame_loader=self.cache.download_frame`` and
  ``frames_loader=self.cache.download_frames``, in six places across three base
  classes. Replacing the object catches all of them and needs no SDK change.

* **Memory is bounded by what was asked for.** The frames are consumed from an
  async generator one at a time and only the requested indexes are retained, so
  the peak is the caller's own result plus a single frame — not the span.

* **It gives up rather than guesses.** Missing PyAV, a video PyAV cannot open, a
  short read, anything at all: it logs once, marks itself off, and delegates to
  the cache it wraps for the rest of the session. A tracker that falls back to
  the old path is slow; one that returns the wrong frames is a silent wrong
  answer on a customer's annotations.
"""

import asyncio
import concurrent.futures
import os
from typing import Any, Dict, List, Optional

import numpy as np
import supervisely as sly

# Import at module load so a missing/renamed SDK symbol is a startup error rather
# than a failure on the first track, which is how image 1.0.20 got to production.
from supervisely.video.sampling import async_stream_video_frames

#: ``auto`` streams when it can and falls back silently; ``api`` keeps the old
#: per-frame path; ``stream`` refuses to fall back, for testing that streaming is
#: actually the thing being exercised.
MODE_ENV = "SLY_TRACKING_FRAME_SOURCE"

#: Streaming decodes a contiguous span. Asking for frames 10 and 50,000 would
#: decode everything between them to return two, so beyond this ratio the old
#: loader is the cheaper answer. Tracking asks for contiguous runs, so this is a
#: guard against pathological input rather than a limit on normal use.
SPAN_RATIO_ENV = "SLY_TRACKING_MAX_SPAN_RATIO"
DEFAULT_SPAN_RATIO = 4.0
SPAN_FLOOR = 64


def _run_coroutine(coro):
    """Run ``coro`` to completion from sync code, event loop running or not.

    Tracking is called from FastAPI, so whether a loop is already running in this
    thread depends on which endpoint arrived. Same guard the SDK's own
    ``stream_video_frames_to_dir`` uses.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None and running.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class StreamingFrameCache:
    """An ``InferenceImageCache`` that streams video frames instead of fetching them.

    Every attribute this class does not define is delegated, so it is a drop-in
    replacement for the cache the SDK created.
    """

    def __init__(self, cache: Any, logger: Optional[Any] = None):
        self._cache = cache
        self._logger = logger if logger is not None else sly.logger
        mode = os.getenv(MODE_ENV, "auto").strip().lower()
        if mode not in ("auto", "api", "stream"):
            self._logger.warning(
                "Unknown %s=%r, falling back to 'auto'.", MODE_ENV, mode
            )
            mode = "auto"
        self._mode = mode
        self._streaming_off = mode == "api"
        try:
            self._span_ratio = float(os.getenv(SPAN_RATIO_ENV, DEFAULT_SPAN_RATIO))
        except ValueError:
            self._span_ratio = DEFAULT_SPAN_RATIO
        self._logger.info(
            "Tracking frame source: %s (streaming from the video, not videos.download-frame)"
            if mode != "api"
            else "Tracking frame source: api (per-frame videos.download-frame)",
            mode,
        )

    # Anything not overridden below is the wrapped cache's own behaviour:
    # download_image(s), run_cache_task, is_persistent, the /smart_cache
    # endpoint, eviction, all of it.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)

    @property
    def streaming_enabled(self) -> bool:
        return not self._streaming_off

    def _give_up(self, reason: str, error: Optional[BaseException] = None) -> None:
        """Turn streaming off for the rest of this session, once, loudly."""
        if self._mode == "stream":
            # Explicitly asked to prove streaming works; surface the failure
            # instead of hiding it behind a fallback that looks like success.
            raise RuntimeError(f"Frame streaming failed and {MODE_ENV}=stream: {reason}") from error
        if self._streaming_off:
            return
        self._streaming_off = True
        self._logger.warning(
            "Frame streaming disabled for this session (%s). Falling back to "
            "videos.download-frame, which is slower and is the path that 504s under load.",
            reason,
            exc_info=error,
        )

    def _worth_streaming(self, frame_indexes: List[int]) -> bool:
        if self._streaming_off or not frame_indexes:
            return False
        span = max(frame_indexes) - min(frame_indexes) + 1
        if span <= SPAN_FLOOR:
            return True
        return span <= len(frame_indexes) * self._span_ratio

    def _stream_span(
        self, api: sly.Api, video_id: int, wanted: List[int]
    ) -> Dict[int, np.ndarray]:
        """Decode ``min(wanted)..max(wanted)`` once, keeping only ``wanted``."""
        needed = set(wanted)
        collected: Dict[int, np.ndarray] = {}

        async def _gather():
            async for frame_index, img in async_stream_video_frames(
                api=api,
                video_id=video_id,
                start=min(wanted),
                end=max(wanted),
            ):
                if frame_index in needed:
                    collected[frame_index] = img
                    if len(collected) == len(needed):
                        break
            return collected

        return _run_coroutine(_gather())

    def download_frames(
        self, api: sly.Api, video_id: int, frame_indexes: List[int], **kwargs
    ) -> List[np.ndarray]:
        frame_indexes = list(frame_indexes)
        if not self._worth_streaming(frame_indexes):
            return self._cache.download_frames(api, video_id, frame_indexes, **kwargs)

        try:
            decoded = self._stream_span(api, video_id, frame_indexes)
        except ImportError as error:
            self._give_up("PyAV is not installed", error)
            return self._cache.download_frames(api, video_id, frame_indexes, **kwargs)
        except Exception as error:  # noqa: BLE001 - any decode failure falls back
            self._give_up(f"streaming video {video_id} failed", error)
            return self._cache.download_frames(api, video_id, frame_indexes, **kwargs)

        missing = [index for index in frame_indexes if index not in decoded]
        if missing:
            # A short read is not a partial success: returning the frames it did
            # get, in the caller's order, would silently track the wrong frames.
            self._give_up(
                f"video {video_id} produced {len(decoded)}/{len(frame_indexes)} requested frames"
            )
            return self._cache.download_frames(api, video_id, frame_indexes, **kwargs)

        progress_cb = kwargs.get("progress_cb", None)
        if progress_cb is not None:
            progress_cb(len(frame_indexes))

        # The caller's order, which is descending for a backward track.
        return [decoded[index] for index in frame_indexes]

    def download_frame(self, api: sly.Api, video_id: int, frame_index: int) -> np.ndarray:
        if self._streaming_off:
            return self._cache.download_frame(api, video_id, frame_index)
        return self.download_frames(api, video_id, [frame_index])[0]

    def download_frames_to_paths(
        self,
        api: sly.Api,
        video_id: int,
        frame_indexes: List[int],
        paths: List[str],
        progress_cb: Optional[Any] = None,
    ) -> None:
        """Write the frames to disk, from the video rather than the endpoint.

        The apps that ask for files rather than arrays -- SAM 3 and ClickSeg --
        go through here. The wrapped implementation fetches each frame with its
        own request and copies it out of the cache; this decodes the span once
        and writes straight out, so the same span costs one decode rather than
        N round trips.
        """
        frame_indexes = list(frame_indexes)
        paths = list(paths)
        if not self._worth_streaming(frame_indexes) or len(frame_indexes) != len(paths):
            return self._cache.download_frames_to_paths(
                api, video_id, frame_indexes, paths, progress_cb=progress_cb
            )

        try:
            decoded = self._stream_span(api, video_id, frame_indexes)
        except ImportError as error:
            self._give_up("PyAV is not installed", error)
            return self._cache.download_frames_to_paths(
                api, video_id, frame_indexes, paths, progress_cb=progress_cb
            )
        except Exception as error:  # noqa: BLE001 - any decode failure falls back
            self._give_up(f"streaming video {video_id} failed", error)
            return self._cache.download_frames_to_paths(
                api, video_id, frame_indexes, paths, progress_cb=progress_cb
            )

        if any(index not in decoded for index in frame_indexes):
            self._give_up(
                f"video {video_id} produced {len(decoded)}/{len(frame_indexes)} requested frames"
            )
            return self._cache.download_frames_to_paths(
                api, video_id, frame_indexes, paths, progress_cb=progress_cb
            )

        for frame_index, path in zip(frame_indexes, paths):
            sly.image.write(path, decoded[frame_index])
            if progress_cb is not None:
                progress_cb()
        return None

    def run_cache_task_manually(
        self,
        api: sly.Api,
        list_of_ids_ranges_or_hashes,
        *,
        dataset_id: Optional[int] = None,
        video_id: Optional[int] = None,
    ) -> None:
        """Skip the whole-video prefetch while streaming is doing the work.

        The tracking base classes call this with ``None`` to pull the entire
        video into the persistent cache before a track. That downloads a whole
        file to serve a twenty-frame track, and streaming already reads only the
        span it needs — so while streaming is healthy this is pure cost. Every
        other form of the call (image ids, hashes, explicit frame ranges) is
        passed through untouched.
        """
        if (
            not self._streaming_off
            and list_of_ids_ranges_or_hashes is None
            and video_id is not None
        ):
            self._logger.debug(
                "Skipping whole-video prefetch for video %s: frames are streamed on demand.",
                video_id,
            )
            return
        return self._cache.run_cache_task_manually(
            api, list_of_ids_ranges_or_hashes, dataset_id=dataset_id, video_id=video_id
        )


def use_streaming_frames(inference: Any) -> Any:
    """Point an inference app's cache at the video instead of the frame endpoint.

    Call once, after ``Inference.__init__`` has created ``self.cache``.
    """
    if isinstance(inference.cache, StreamingFrameCache):
        return inference.cache
    inference.cache = StreamingFrameCache(inference.cache, logger=sly.logger)
    return inference.cache
