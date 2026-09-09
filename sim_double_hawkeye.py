"""1. 더블 하위처 샷: 같은 v0, 상보각(theta, 90-theta)으로 동시 착탄시키는 시뮬레이션."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

import physics as ph
from sim_common import PlaybackState, add_playback_buttons, set_korean_font

ARRIVAL_EPS = 0.05  # "동시 착탄" 문구를 표시할 시간 오차 허용치(초)
LOOP_PAUSE = 1.0     # 착탄 후 잠시 멈췄다가 재시작하기까지의 여유 시간(초)


class DoubleHawkeyeSim:
    def __init__(self, v0=30.0, theta1=70.0, g=ph.G_DEFAULT):
        set_korean_font()
        self.v0, self.theta1, self.g = v0, theta1, g
        self.result = ph.double_hawkeye(v0, theta1, g)
        self.state = PlaybackState(dt=0.02)

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        plt.subplots_adjust(bottom=0.32)
        self.ax.set_xlabel("x (m)  — 수평 거리")
        self.ax.set_ylabel("y (m)  — 높이")
        self.ax.set_title("더블 하위처 샷 (동시 착탄)")
        self.ax.grid(True, alpha=0.3)

        (self.guide1,) = self.ax.plot([], [], "--", color="tab:blue", alpha=0.4, lw=1,
                                       label=f"고각 θ1={theta1:.0f}°")
        (self.guide2,) = self.ax.plot([], [], "--", color="tab:red", alpha=0.4, lw=1,
                                       label=f"저각 θ2={90-theta1:.0f}°")
        (self.trail1,) = self.ax.plot([], [], "-", color="tab:blue", lw=2)
        (self.trail2,) = self.ax.plot([], [], "-", color="tab:red", lw=2)
        (self.arrow1,) = self.ax.plot([], [], "o", color="tab:blue", ms=10)
        (self.arrow2,) = self.ax.plot([], [], "o", color="tab:red", ms=10)
        (self.target,) = self.ax.plot([], [], "X", color="black", ms=14)
        self.ax.legend(loc="upper right")

        self.info_text = self.ax.text(0.02, 0.97, "", transform=self.ax.transAxes,
                                       va="top", ha="left", fontsize=10,
                                       bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        self.hit_text = self.ax.text(0.5, 0.5, "", transform=self.ax.transAxes,
                                      va="center", ha="center", fontsize=20, color="green",
                                      fontweight="bold", alpha=0.0)

        ax_v0 = self.fig.add_axes((0.26, 0.18, 0.55, 0.03))
        ax_theta1 = self.fig.add_axes((0.26, 0.12, 0.55, 0.03))
        self.slider_v0 = Slider(ax_v0, "v0 (m/s)", 10.0, 50.0, valinit=v0, valfmt="%.1f")
        self.slider_theta1 = Slider(ax_theta1, "theta1 (deg, 45~90)", 45.5, 89.5, valinit=theta1,
                                     valfmt="%.1f")
        self.slider_v0.on_changed(self._on_slider_change)
        self.slider_theta1.on_changed(self._on_slider_change)

        self.buttons = add_playback_buttons(self.fig, self.state, on_reset=self._on_reset)

        self._recompute(reset_time=True)
        self.anim = FuncAnimation(self.fig, self._on_frame, interval=20, blit=False,
                                   cache_frame_data=False)

    def _recompute(self, reset_time: bool = False):
        self.v0 = self.slider_v0.val
        self.theta1 = self.slider_theta1.val
        self.result = ph.double_hawkeye(self.v0, self.theta1, self.g)
        r = self.result

        t1 = np.linspace(0, r.T1, 200)
        t2 = np.linspace(0, r.T2, 200)
        gx1, gy1 = ph.projectile_position(r.v0, r.theta1, t1, self.g)
        gx2, gy2 = ph.projectile_position(r.v0, r.theta2, t2, self.g)
        self.guide1.set_data(gx1, gy1)
        self.guide2.set_data(gx2, gy2)
        self.guide1.set_label(f"고각 θ1={r.theta1:.1f}°")
        self.guide2.set_label(f"저각 θ2={r.theta2:.1f}°")
        self.ax.legend(loc="upper right")

        self.target.set_data([r.R], [0])
        margin = max(r.R * 0.1, 3)
        self.ax.set_xlim(-margin * 0.2, r.R + margin)
        y_max = max(gy1.max(), gy2.max())
        self.ax.set_ylim(-y_max * 0.05, y_max * 1.2)

        if reset_time:
            self.state.reset()

    def _on_slider_change(self, _val):
        self._recompute(reset_time=True)

    def _on_reset(self):
        pass  # PlaybackState.reset()이 이미 t=0으로 되돌림

    def _on_frame(self, _frame):
        r = self.result
        t = self.state.step()
        loop_end = r.T1 + LOOP_PAUSE
        if t > loop_end:
            self.state.t = 0.0
            t = 0.0

        # 화살 1 (고각): t=0에 발사, T1에 착탄
        t1 = np.clip(t, 0, r.T1)
        tt1 = np.linspace(0, t1, max(int(t1 / max(r.T1, 1e-6) * 200), 2))
        x1, y1 = ph.projectile_position(r.v0, r.theta1, tt1, self.g)
        self.trail1.set_data(x1, y1)
        if t <= r.T1:
            self.arrow1.set_data([x1[-1]], [y1[-1]])
            self.arrow1.set_alpha(1.0)
        else:
            self.arrow1.set_alpha(0.0)

        # 화살 2 (저각): t=delay에 발사, T1(=delay+T2)에 착탄
        t2_local = np.clip(t - r.delay, 0, r.T2)
        if t >= r.delay:
            tt2 = np.linspace(0, t2_local, max(int(t2_local / max(r.T2, 1e-6) * 200), 2))
            x2, y2 = ph.projectile_position(r.v0, r.theta2, tt2, self.g)
            self.trail2.set_data(x2, y2)
            if t <= r.T1:
                self.arrow2.set_data([x2[-1]], [y2[-1]])
                self.arrow2.set_alpha(1.0)
            else:
                self.arrow2.set_alpha(0.0)
        else:
            self.trail2.set_data([], [])
            self.arrow2.set_alpha(0.0)

        near_arrival = abs(t - r.T1) < ARRIVAL_EPS
        if near_arrival:
            self.hit_text.set_text("동시 착탄!")
            self.hit_text.set_alpha(1.0)
        elif t > r.T1:
            self.hit_text.set_alpha(1.0)
            self.hit_text.set_text("동시 착탄!")
        else:
            self.hit_text.set_alpha(0.0)

        self.info_text.set_text(
            f"v0 = {r.v0:.1f} m/s\n"
            f"θ1(고각) = {r.theta1:.1f}°   θ2(저각) = {r.theta2:.1f}°\n"
            f"R = {r.R:.2f} m\n"
            f"T1 = {r.T1:.2f} s   T2 = {r.T2:.2f} s\n"
            f"delay(저각 발사 지연) = {r.delay:.2f} s\n"
            f"t = {t:.2f} s"
        )
        return (self.trail1, self.trail2, self.arrow1, self.arrow2,
                self.target, self.info_text, self.hit_text)

    def show(self):
        plt.show()


def main():
    sim = DoubleHawkeyeSim()
    sim.show()


if __name__ == "__main__":
    main()
