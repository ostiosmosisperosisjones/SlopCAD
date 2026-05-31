"""
cad/profiler.py

Lightweight in-app performance profiler.

Goals
-----
* Near-zero overhead when disabled (the common case): every entry point
  short-circuits on a single boolean check before touching the clock.
* When enabled, time arbitrary named "sections" and accumulate per-section
  stats (total time, call count, last/avg/max ms) plus a rolling frame-time
  window for a live FPS / frame-time readout.

Usage
-----
    from cad.profiler import profiler

    profiler.frame_begin()                 # once per painted frame
    with profiler.section("hover.rebuild"):
        ...
    profiler.frame_end()

    profiler.dump()                        # print a sorted table to stdout

The viewport draws `profiler.hud_lines()` on screen when enabled, and a
console dump can be triggered from a keybind. Toggle with `profiler.toggle()`.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

# Monotonic, high-resolution clock. perf_counter is unaffected by NTP/system
# clock changes and is the right tool for measuring elapsed durations.
_clock = time.perf_counter


@dataclass
class _Stat:
    calls: int = 0
    total: float = 0.0          # cumulative seconds
    last: float = 0.0           # most recent sample, seconds
    max: float = 0.0            # worst sample, seconds
    # Per-frame accumulation: a section may run several times within one frame
    # (e.g. once per body). We sum those into the current frame, then fold the
    # frame total into a rolling window so the HUD shows per-frame cost.
    _frame_accum: float = 0.0
    frame_samples: deque = field(default_factory=lambda: deque(maxlen=120))

    @property
    def avg_frame_ms(self) -> float:
        if not self.frame_samples:
            return 0.0
        return 1000.0 * sum(self.frame_samples) / len(self.frame_samples)

    @property
    def max_frame_ms(self) -> float:
        return 1000.0 * max(self.frame_samples) if self.frame_samples else 0.0


class Profiler:
    def __init__(self):
        self.enabled: bool = False
        self._stats: dict[str, _Stat] = {}
        # Rolling window of whole-frame durations for FPS / frame-time readout.
        self._frame_times: deque = deque(maxlen=120)
        self._frame_start: float | None = None
        # Names whose accumulator needs folding at frame_end (touched this frame).
        self._touched_this_frame: set[str] = set()

    # -- enable / disable ---------------------------------------------------

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if self.enabled:
            self.reset()
        return self.enabled

    def reset(self):
        self._stats.clear()
        self._frame_times.clear()
        self._frame_start = None
        self._touched_this_frame.clear()

    # -- frame timing -------------------------------------------------------

    def frame_begin(self):
        if not self.enabled:
            return
        self._frame_start = _clock()
        for name in self._touched_this_frame:
            self._stats[name]._frame_accum = 0.0
        self._touched_this_frame.clear()

    def frame_end(self):
        if not self.enabled or self._frame_start is None:
            return
        self._frame_times.append(_clock() - self._frame_start)
        self._frame_start = None
        # Fold each touched section's per-frame total into its rolling window.
        for name in self._touched_this_frame:
            st = self._stats[name]
            st.frame_samples.append(st._frame_accum)

    # -- section timing -----------------------------------------------------

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = _clock()
        try:
            yield
        finally:
            self._record(name, _clock() - t0)

    def _record(self, name: str, dt: float):
        st = self._stats.get(name)
        if st is None:
            st = self._stats[name] = _Stat()
        st.calls += 1
        st.total += dt
        st.last = dt
        if dt > st.max:
            st.max = dt
        st._frame_accum += dt
        self._touched_this_frame.add(name)

    # -- reporting ----------------------------------------------------------

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 1e-9 else 0.0

    @property
    def avg_frame_ms(self) -> float:
        if not self._frame_times:
            return 0.0
        return 1000.0 * sum(self._frame_times) / len(self._frame_times)

    @property
    def max_frame_ms(self) -> float:
        return 1000.0 * max(self._frame_times) if self._frame_times else 0.0

    def _sorted_sections(self):
        """Sections ranked by average per-frame cost (the lag culprits first)."""
        return sorted(self._stats.items(),
                      key=lambda kv: kv[1].avg_frame_ms, reverse=True)

    def hud_lines(self) -> list[str]:
        """Compact lines for the on-screen overlay."""
        lines = [
            f"FPS {self.fps:5.1f}   frame {self.avg_frame_ms:5.1f}ms "
            f"(max {self.max_frame_ms:5.1f})"
        ]
        for name, st in self._sorted_sections()[:8]:
            lines.append(
                f"{name:<22} {st.avg_frame_ms:6.2f}ms  x{st.calls:<5} "
                f"max {st.max_frame_ms:6.2f}"
            )
        return lines

    def dump(self):
        """Print a full sorted table to stdout."""
        print("\n=== Profiler ===")
        print(f"FPS {self.fps:.1f}  avg frame {self.avg_frame_ms:.2f}ms  "
              f"max frame {self.max_frame_ms:.2f}ms  "
              f"window={len(self._frame_times)} frames")
        print(f"{'section':<28}{'avg/frame':>11}{'max/frame':>11}"
              f"{'calls':>9}{'total':>10}")
        print("-" * 79)
        for name, st in self._sorted_sections():
            print(f"{name:<28}{st.avg_frame_ms:>9.2f}ms{st.max_frame_ms:>9.2f}ms"
                  f"{st.calls:>9}{st.total * 1000:>8.1f}ms")
        print("=" * 79 + "\n")


# Process-wide singleton.
profiler = Profiler()
