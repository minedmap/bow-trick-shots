"""matplotlib 기반 시뮬레이션들이 공유하는 재생 제어 유틸리티."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
from matplotlib import font_manager
from matplotlib.widgets import Button

_KOREAN_FONT_CANDIDATES = [
    "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
    "Malgun Gothic", "AppleGothic", "UnDotum",
]


def set_korean_font() -> None:
    """한글이 깨지지 않도록 설치된 한글 폰트를 찾아 matplotlib 기본 폰트로 지정한다.

    시스템에 한글 폰트가 없으면 조용히 넘어간다(글자가 네모로 보일 수 있음).
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


@dataclass
class PlaybackState:
    dt: float = 0.02
    t: float = 0.0
    running: bool = True

    def step(self) -> float:
        if self.running:
            self.t += self.dt
        return self.t

    def reset(self) -> None:
        self.t = 0.0
        self.running = True


@dataclass
class PlaybackButtons:
    """Button 객체들을 보관해 GC로 사라지지 않도록 붙잡아 둔다."""
    play: Button
    reset: Button


def add_playback_buttons(fig, state: PlaybackState, rect_play=(0.82, 0.02, 0.08, 0.05),
                          rect_reset=(0.90, 0.02, 0.08, 0.05), on_reset=None) -> PlaybackButtons:
    ax_play = fig.add_axes(rect_play)
    btn_play = Button(ax_play, "Pause")

    def toggle(_event):
        state.running = not state.running
        btn_play.label.set_text("Pause" if state.running else "Play")

    btn_play.on_clicked(toggle)

    ax_reset = fig.add_axes(rect_reset)
    btn_reset = Button(ax_reset, "Reset")

    def reset(_event):
        state.reset()
        btn_play.label.set_text("Pause")
        if on_reset is not None:
            on_reset()

    btn_reset.on_clicked(reset)
    return PlaybackButtons(play=btn_play, reset=btn_reset)
