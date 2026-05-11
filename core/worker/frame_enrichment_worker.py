"""
Long-lived thread pool that performs per-frame enrichment work
(embedding, window OCR, sub_frame persistence, cluster assignment,
batch-write buffering) off the ``/api/store_frame`` HTTP request path.

Design
------
``store_frame``'s synchronous (fast) path only:

1. decodes the full-screen base64 image,
2. runs the solid-color dedup check,
3. persists the full-screen PNG via ``temp_frame_buffer``,
4. reads the currently focused window metadata,
5. enqueues an :class:`EnrichmentJob`,
6. returns ``frame_summary`` to the frontend (usually <500 ms).

Full-screen embedding, per-window embedding + OCR + SQLite writes,
fullscreen sub_frame creation, cluster assignment and the main-frame
``batch_write_buffer.add_frame`` call all run inside this worker.

Lifecycle
---------
Started once in ``@app.on_event('startup')`` and shut down in
``@app.on_event('shutdown')``. ``drain(timeout)`` is invoked by
``/api/recording/stop`` to let in-flight jobs finish before the
video/batch-write buffers flush.

Diagnostics
-----------
Heartbeat logging and thread dumps are disabled unless the caller passes
``enable_diagnostics=True``. They are useful for stuck-job investigation but
should not run in the normal startup path.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class EnrichmentJob:
    """A unit of background work for a single captured frame.

    ``processor`` must be a zero-argument callable that closes over all
    data needed to finish processing this frame (image, request, focused
    window, etc.). The worker only tracks timing and failures; it does
    not introspect the closure.
    """

    frame_id: str
    processor: Callable[[], None]
    submitted_at: float = field(default_factory=time.monotonic)


class FrameEnrichmentWorker:
    """Fixed-size thread pool consuming an unbounded FIFO of jobs.

    Queue is intentionally unbounded — frontend backpressure
    (``recording.ts`` send-queue caps) is the primary producer-side
    throttle, and rejecting frames from the backend would drop data that
    has already been captured + persisted to disk. Queue depth is logged
    so operators can tune ``FRAME_ENRICHMENT_WORKERS`` if it grows.
    """

    def __init__(
        self,
        num_workers: int = 2,
        name: str = "frame-enrich",
        *,
        enable_diagnostics: bool = False,
        heartbeat_interval_s: float = 30.0,
        stuck_job_threshold_s: float = 60.0,
    ):
        if num_workers < 1:
            num_workers = 1
        self.num_workers = num_workers
        self.name = name
        self.enable_diagnostics = enable_diagnostics
        self.heartbeat_interval_s = max(1.0, float(heartbeat_interval_s))
        self.stuck_job_threshold_s = max(1.0, float(stuck_job_threshold_s))

        self._queue: "queue.Queue[Optional[EnrichmentJob]]" = queue.Queue()
        self._threads: List[threading.Thread] = []

        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._drain_cv = threading.Condition()

        self._stopped = False
        self._start_lock = threading.Lock()

        self._stats_lock = threading.Lock()
        self._stats: Dict[str, float] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "total_wait_s": 0.0,
            "total_run_s": 0.0,
        }

        # Per-worker Work-In-Progress tracking. Maps thread name -> (frame_id,
        # run_start_monotonic). Used by the heartbeat thread to detect hung
        # jobs and dump stacks.
        self._wip: Dict[str, Tuple[str, float]] = {}
        self._wip_lock = threading.Lock()
        # Remember which (thread_name, frame_id) already had their stack dumped
        # in the current "stuck" episode — avoid log spam every heartbeat.
        self._stuck_dumped: Dict[str, str] = {}

        self._heartbeat_thread: Optional[threading.Thread] = None

    # ----- lifecycle -----

    def start(self) -> None:
        with self._start_lock:
            if self._threads:
                return
            self._stopped = False
            for i in range(self.num_workers):
                t = threading.Thread(
                    target=self._run,
                    name=f"{self.name}-{i}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
            # Heartbeat/thread-dump diagnostics are test/debug instrumentation
            # and must stay disabled unless explicitly configured.
            if (
                self.enable_diagnostics
                and (self._heartbeat_thread is None or not self._heartbeat_thread.is_alive())
            ):
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name=f"{self.name}-heartbeat",
                    daemon=True,
                )
                self._heartbeat_thread.start()
        logger.info(
            f"FrameEnrichmentWorker '{self.name}' started with {self.num_workers} workers "
            f"(diagnostics={'on' if self.enable_diagnostics else 'off'}, "
            f"heartbeat={self.heartbeat_interval_s}s, "
            f"stuck_threshold={self.stuck_job_threshold_s}s)"
        )

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._start_lock:
            if self._stopped:
                return
            self._stopped = True
            threads = list(self._threads)
            for _ in threads:
                self._queue.put(None)  # poison pill
            self._threads.clear()

        for t in threads:
            t.join(timeout=timeout)

        # Heartbeat is daemon + self-exits on ``_stopped``; give it a chance
        # but don't block shutdown.
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None

        logger.info(f"FrameEnrichmentWorker '{self.name}' stopped")

    # ----- producer API -----

    def submit(self, job: EnrichmentJob) -> None:
        if self._stopped:
            raise RuntimeError("FrameEnrichmentWorker is stopped")
        self._queue.put(job)
        with self._stats_lock:
            self._stats["submitted"] += 1

    # ----- introspection -----

    def queue_depth(self) -> int:
        # ``qsize`` is approximate on some platforms but fine for logging / drain checks.
        return self._queue.qsize()

    def inflight(self) -> int:
        with self._inflight_lock:
            return self._inflight

    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            s = dict(self._stats)
        s["queue_depth"] = self.queue_depth()
        s["inflight"] = self.inflight()
        completed = s.get("completed", 0) or 0
        if completed > 0:
            s["avg_run_ms"] = (s["total_run_s"] / completed) * 1000.0
            s["avg_wait_ms"] = (s["total_wait_s"] / completed) * 1000.0
        else:
            s["avg_run_ms"] = 0.0
            s["avg_wait_ms"] = 0.0
        return s

    # ----- consumer helpers -----

    def drain(self, timeout: float) -> bool:
        """Block until queue empty AND no inflight jobs, up to ``timeout`` seconds.

        Returns ``True`` if fully drained, ``False`` on timeout. Safe to
        call from request handlers (e.g. ``/api/recording/stop``) — does
        not acquire the job queue mutex except via ``qsize``.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._drain_cv:
            while True:
                if self._queue.empty() and self.inflight() == 0:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                # Wake periodically to re-check ``qsize`` (which ``notify_all``
                # only covers for inflight transitions, not new submits).
                self._drain_cv.wait(timeout=min(remaining, 0.2))

    # ----- internal -----

    def _run(self) -> None:
        thread_name = threading.current_thread().name
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return

                wait_s = time.monotonic() - job.submitted_at
                with self._inflight_lock:
                    self._inflight += 1

                run_t0 = time.monotonic()
                with self._wip_lock:
                    self._wip[thread_name] = (job.frame_id, run_t0)

                try:
                    job.processor()
                    with self._stats_lock:
                        self._stats["completed"] += 1
                        self._stats["total_wait_s"] += wait_s
                        self._stats["total_run_s"] += time.monotonic() - run_t0
                except Exception as e:  # pragma: no cover - defensive
                    logger.error(
                        f"FrameEnrichmentWorker '{self.name}': job {job.frame_id} "
                        f"failed after {(time.monotonic() - run_t0):.2f}s: {e}",
                        exc_info=True,
                    )
                    with self._stats_lock:
                        self._stats["failed"] += 1
                finally:
                    with self._wip_lock:
                        self._wip.pop(thread_name, None)
                        self._stuck_dumped.pop(thread_name, None)
                    with self._inflight_lock:
                        self._inflight -= 1
                    with self._drain_cv:
                        self._drain_cv.notify_all()
            finally:
                self._queue.task_done()

    # ----- heartbeat / stuck-job diagnostics -----

    def _heartbeat_loop(self) -> None:
        """Emit periodic stats + dump stacks when a single job stalls.

        The *only* reason this exists is that silent worker hangs (e.g.
        synchronous VLM HTTP call with no timeout, SQLite lock deadlock,
        ffmpeg subprocess wait) leave zero trace in the log — the fast
        path keeps returning 200 to the frontend while nothing advances
        in the SQLite ``frames`` table. Dumping all Python thread stacks
        the moment a job crosses the stuck threshold is what lets the
        user go from "backend mysteriously froze" to a specific line of
        code, in a single log entry.
        """
        while not self._stopped:
            # Use a short sleep + loop rather than ``sleep(INTERVAL)`` so
            # shutdown returns promptly.
            waited = 0.0
            while waited < self.heartbeat_interval_s and not self._stopped:
                time.sleep(0.5)
                waited += 0.5
            if self._stopped:
                return

            try:
                s = self.stats()
                logger.info(
                    f"EnrichHeartbeat '{self.name}': "
                    f"queue_depth={int(s.get('queue_depth', 0))} "
                    f"inflight={int(s.get('inflight', 0))} "
                    f"submitted={int(s.get('submitted', 0))} "
                    f"completed={int(s.get('completed', 0))} "
                    f"failed={int(s.get('failed', 0))} "
                    f"avg_run_ms={s.get('avg_run_ms', 0.0):.0f} "
                    f"avg_wait_ms={s.get('avg_wait_ms', 0.0):.0f}"
                )

                now = time.monotonic()
                with self._wip_lock:
                    wip_snapshot = list(self._wip.items())
                    stuck_already = dict(self._stuck_dumped)

                stuck: List[Tuple[str, str, float]] = []
                for tname, (fid, t0) in wip_snapshot:
                    age = now - t0
                    if age >= self.stuck_job_threshold_s:
                        stuck.append((tname, fid, age))

                for tname, fid, age in stuck:
                    logger.warning(
                        f"EnrichStuck '{self.name}': thread={tname} frame={fid} "
                        f"running {age:.1f}s (threshold={self.stuck_job_threshold_s}s)"
                    )
                    # Dump stacks only on the *first* heartbeat that sees this
                    # (thread, frame) pair stuck — re-dump if the frame changes
                    # (new job also stuck) but not every 30s for the same hang.
                    if stuck_already.get(tname) == fid:
                        continue
                    self._dump_all_thread_stacks(reason=f"{tname}/{fid} stuck {age:.1f}s")
                    with self._wip_lock:
                        self._stuck_dumped[tname] = fid
            except Exception as e:  # pragma: no cover - never crash heartbeat
                logger.debug(f"EnrichHeartbeat error (non-fatal): {e}")

    @staticmethod
    def _dump_all_thread_stacks(reason: str) -> None:
        """Dump every Python thread's stack to the logger at WARNING level.

        Manually formatted (rather than calling ``faulthandler.dump_traceback``)
        so output lands in ``backend_server.log`` via the normal logging
        pipeline instead of stderr.
        """
        try:
            frames_by_tid = sys._current_frames()
            name_by_tid = {t.ident: t.name for t in threading.enumerate()}
            lines = [f"===== THREAD DUMP ({reason}) ====="]
            for tid, frame in frames_by_tid.items():
                tname = name_by_tid.get(tid, f"tid-{tid}")
                lines.append(f"--- Thread {tname} (tid={tid}) ---")
                for fl in traceback.format_stack(frame):
                    lines.extend(line.rstrip() for line in fl.rstrip("\n").splitlines())
            lines.append("===== END THREAD DUMP =====")
            logger.warning("\n".join(lines))
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to dump thread stacks: {e}", exc_info=True)
