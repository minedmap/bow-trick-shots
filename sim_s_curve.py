"""2. S자 우회 샷: top-down 평면에서 장애물 2개를 피해가는 매그누스 궤적 시뮬레이션."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button

import physics as ph
from sim_common import PlaybackState, add_playback_buttons, set_korean_font

LOOP_PAUSE = 0.8


class SCurveSim:
    def __init__(self, vx=20.0, A=1.5, omega=1.2, target_x=40.0,
                 obstacles=None):
        set_korean_font()
        self.vx = vx
        self.target_x = target_x
        self.obstacles = obstacles or [
            {"x": 14.0, "y": 1.2, "radius": 0.6},
            {"x": 28.0, "y": -1.2, "radius": 0.6},
        ]
        self.t_end = target_x / vx
        self.state = PlaybackState(dt=self.t_end / 250)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.42)
        self.ax.set_xlabel("x (m) — 다운레인지 (발사점 → 표적)")
        self.ax.set_ylabel("y (m) — 좌우 위치")
        self.ax.set_title("S자 우회 샷 (Top-down)")
        self.ax.grid(True, alpha=0.3)

        (self.launch_pt,) = self.ax.plot([0], [0], "s", color="black", ms=8)
        (self.target_pt,) = self.ax.plot([target_x], [0], "X", color="green", ms=14)
        (self.guide,) = self.ax.plot([], [], "--", color="gray", alpha=0.5, lw=1)
        (self.trail,) = self.ax.plot([], [], "-", color="tab:orange", lw=2)
        (self.arrow_pt,) = self.ax.plot([], [], "o", color="tab:orange", ms=10)
        self.vel_arrow = None

        self.obstacle_patches = [
            Circle((o["x"], o["y"]), o["radius"], color="tab:red", alpha=0.4)
            for o in self.obstacles
        ]
        for p in self.obstacle_patches:
            self.ax.add_patch(p)

        self.info_text = self.ax.text(0.02, 0.97, "", transform=self.ax.transAxes,
                                       va="top", ha="left", fontsize=9,
                                       bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        self.warn_text = self.ax.text(0.5, 0.9, "", transform=self.ax.transAxes,
                                       va="top", ha="center", fontsize=16, color="red",
                                       fontweight="bold")

        s_left = 0.15
        s_w = 0.32
        self.slider_A = Slider(self.fig.add_axes((s_left, 0.30, s_w, 0.03)),
                                "A (횡방향 진폭, m)", 0.0, 5.0, valinit=A)
        self.slider_omega = Slider(self.fig.add_axes((s_left, 0.25, s_w, 0.03)),
                                    "omega (rad/s)", 0.05, 4.0, valinit=omega)
        self.slider_obs1_x = Slider(self.fig.add_axes((s_left, 0.20, s_w, 0.03)),
                                     "장애물1 x", 5.0, target_x - 5, valinit=self.obstacles[0]["x"])
        self.slider_obs1_y = Slider(self.fig.add_axes((s_left, 0.15, s_w, 0.03)),
                                     "장애물1 y", -4.0, 4.0, valinit=self.obstacles[0]["y"])
        self.slider_obs2_x = Slider(self.fig.add_axes((s_left, 0.10, s_w, 0.03)),
                                     "장애물2 x", 5.0, target_x - 5, valinit=self.obstacles[1]["x"])
        self.slider_obs2_y = Slider(self.fig.add_axes((s_left, 0.05, s_w, 0.03)),
                                     "장애물2 y", -4.0, 4.0, valinit=self.obstacles[1]["y"])
        for s in (self.slider_A, self.slider_omega, self.slider_obs1_x,
                  self.slider_obs1_y, self.slider_obs2_x, self.slider_obs2_y):
            s.on_changed(self._on_slider_change)

        ax_auto = self.fig.add_axes((0.60, 0.05, 0.15, 0.05))
        self.btn_auto = Button(ax_auto, "자동으로 A,ω 계산")
        self.btn_auto.on_clicked(self._on_auto_solve)

        self.buttons = add_playback_buttons(self.fig, self.state, on_reset=self._on_reset)

        self._recompute(reset_time=True)
        self.anim = FuncAnimation(self.fig, self._on_frame, interval=20, blit=False,
                                   cache_frame_data=False)

    def _current_A_omega(self):
        return self.slider_A.val, self.slider_omega.val

    def _sync_obstacles_from_sliders(self):
        self.obstacles[0]["x"] = self.slider_obs1_x.val
        self.obstacles[0]["y"] = self.slider_obs1_y.val
        self.obstacles[1]["x"] = self.slider_obs2_x.val
        self.obstacles[1]["y"] = self.slider_obs2_y.val
        for patch, obs in zip(self.obstacle_patches, self.obstacles):
            patch.center = (obs["x"], obs["y"])

    def _recompute(self, reset_time: bool = False):
        self._sync_obstacles_from_sliders()
        A, omega = self._current_A_omega()
        t = np.linspace(0, self.t_end, 400)
        x, y = ph.s_curve_position(self.vx, A, omega, t)
        self.guide.set_data(x, y)

        y_extent = max(A, max(abs(o["y"]) + o["radius"] for o in self.obstacles), 1.0) * 1.4
        self.ax.set_xlim(-2, self.target_x + 2)
        self.ax.set_ylim(-y_extent, y_extent)

        if reset_time:
            self.state.reset()

    def _on_slider_change(self, _val):
        self._recompute(reset_time=True)

    def _on_reset(self):
        pass

    def _on_auto_solve(self, _event):
        self._sync_obstacles_from_sliders()
        try:
            A_sol, omega_sol = ph.solve_s_curve_avoidance(self.vx, self.obstacles, margin=0.3)
            self.slider_A.set_val(min(A_sol, self.slider_A.valmax))
            self.slider_omega.set_val(min(omega_sol, self.slider_omega.valmax))
        except Exception as exc:  # noqa: BLE001 - UI 피드백용
            self.warn_text.set_text(f"자동 계산 실패: {exc}")

    def _on_frame(self, _frame):
        A, omega = self._current_A_omega()
        t = self.state.step()
        if t > self.t_end + LOOP_PAUSE:
            self.state.t = 0.0
            t = 0.0

        t_clamped = min(t, self.t_end)
        tt = np.linspace(0, t_clamped, max(int(t_clamped / self.t_end * 400), 2))
        x, y = ph.s_curve_position(self.vx, A, omega, tt)
        self.trail.set_data(x, y)

        if t <= self.t_end:
            self.arrow_pt.set_data([x[-1]], [y[-1]])
            self.arrow_pt.set_alpha(1.0)
            if self.vel_arrow is not None:
                self.vel_arrow.remove()
                self.vel_arrow = None
            vy = ph.lateral_velocity(A, omega, t_clamped)
            scale = 0.6
            self.vel_arrow = self.ax.annotate(
                "", xy=(x[-1] + self.vx * scale, y[-1] + vy * scale), xytext=(x[-1], y[-1]),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5),
            )
        else:
            self.arrow_pt.set_alpha(0.0)

        collisions = [ph.check_collision(A, omega, self.vx, obs, t_end=self.t_end)
                      for obs in self.obstacles]
        for patch, hit in zip(self.obstacle_patches, collisions):
            patch.set_color("tab:red" if not hit else "crimson")
            patch.set_alpha(0.4 if not hit else 0.8)
        self.warn_text.set_text("충돌 위험!" if any(collisions) else "")

        self.info_text.set_text(
            f"vx = {self.vx:.1f} m/s\n"
            f"A = {A:.2f} m   omega = {omega:.2f} rad/s\n"
            f"t = {t:.2f} s / {self.t_end:.2f} s\n"
            f"장애물1: ({self.obstacles[0]['x']:.1f}, {self.obstacles[0]['y']:.1f})\n"
            f"장애물2: ({self.obstacles[1]['x']:.1f}, {self.obstacles[1]['y']:.1f})"
        )
        return (self.trail, self.arrow_pt, self.info_text, self.warn_text, *self.obstacle_patches)

    def show(self):
        plt.show()


def main():
    sim = SCurveSim()
    sim.show()


if __name__ == "__main__":
    main()
