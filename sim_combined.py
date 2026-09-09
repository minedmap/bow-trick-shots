"""3. 결합 모델: 더블 하위처 + S자 우회, 3D 궤적으로 동일 탄착점 검증."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection 등록용)

import physics as ph
from sim_common import PlaybackState, add_playback_buttons, set_korean_font

ARRIVAL_EPS = 0.05
LOOP_PAUSE = 1.0
HIT_THRESHOLD = 0.05  # "명중 일치" 판정 임계값(화살 지름 정도, m)


class CombinedSim:
    def __init__(self, v0=30.0, theta1=70.0, g=ph.G_DEFAULT, A=1.0, omega=None, phi=0.0):
        set_korean_font()
        self.v0, self.theta1, self.g, self.phi = v0, theta1, g, phi
        self.result = ph.double_hawkeye(v0, theta1, g)
        if omega is None:
            omega = ph.suggest_omega(self.result.T1, self.result.T2, n=0)
        self.state = PlaybackState(dt=0.02)

        self.fig = plt.figure(figsize=(10, 7.5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=0.32, top=0.78)
        self.ax.set_xlabel("x (m) — 다운레인지")
        self.ax.set_ylabel("y (m) — 좌우")
        self.ax.set_zlabel("z (m) — 높이")
        self.ax.set_title("더블 하위처 + S자 우회 결합 모델 (동일 탄착점 검증)", pad=20)

        (self.guide1,) = self.ax.plot([], [], [], "--", color="tab:blue", alpha=0.4, lw=1)
        (self.guide2,) = self.ax.plot([], [], [], "--", color="tab:red", alpha=0.4, lw=1)
        (self.trail1,) = self.ax.plot([], [], [], "-", color="tab:blue", lw=2, label="화살1 (고각)")
        (self.trail2,) = self.ax.plot([], [], [], "-", color="tab:red", lw=2, label="화살2 (저각)")
        (self.arrow1,) = self.ax.plot([], [], [], "o", color="tab:blue", ms=8)
        (self.arrow2,) = self.ax.plot([], [], [], "o", color="tab:red", ms=8)
        self.ax.legend(loc="upper left")

        self.info_text = self.fig.text(0.02, 0.95, "", va="top", ha="left", fontsize=9,
                                        bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        self.verdict_text = self.fig.text(0.5, 0.95, "", va="top", ha="center", fontsize=14,
                                           fontweight="bold")

        s_left = 0.20
        s_w = 0.26
        self.slider_v0 = Slider(self.fig.add_axes((s_left, 0.24, s_w, 0.03)),
                                 "v0 (m/s)", 10.0, 50.0, valinit=v0, valfmt="%.1f")
        self.slider_theta1 = Slider(self.fig.add_axes((s_left, 0.19, s_w, 0.03)),
                                     "theta1 (deg)", 45.5, 89.5, valinit=theta1, valfmt="%.1f")
        self.slider_A = Slider(self.fig.add_axes((s_left, 0.14, s_w, 0.03)),
                                "A (m)", 0.0, 3.0, valinit=A, valfmt="%.2f")
        self.slider_omega = Slider(self.fig.add_axes((s_left, 0.09, s_w, 0.03)),
                                    "omega (rad/s)", 0.0, 3.0, valinit=omega, valfmt="%.2f")
        for s in (self.slider_v0, self.slider_theta1, self.slider_A, self.slider_omega):
            s.on_changed(self._on_slider_change)

        ax_snap = self.fig.add_axes((0.55, 0.09, 0.18, 0.05))
        self.btn_snap = Button(ax_snap, "제안 ω로 스냅")
        self.btn_snap.on_clicked(self._on_snap_omega)

        self.buttons = add_playback_buttons(self.fig, self.state, on_reset=self._on_reset)

        self._recompute(reset_time=True)
        self.anim = FuncAnimation(self.fig, self._on_frame, interval=20, blit=False,
                                   cache_frame_data=False)

    def _on_reset(self):
        pass

    def _on_slider_change(self, _val):
        self._recompute(reset_time=True)

    def _on_snap_omega(self, _event):
        r = self.result
        best = ph.nearest_suggested_omega(r.T1, r.T2, self.slider_omega.val)
        self.slider_omega.set_val(min(best, self.slider_omega.valmax))

    def _recompute(self, reset_time: bool = False):
        self.v0 = self.slider_v0.val
        self.theta1 = self.slider_theta1.val
        self.A = self.slider_A.val
        self.omega = self.slider_omega.val
        self.result = ph.double_hawkeye(self.v0, self.theta1, self.g)
        r = self.result

        t1 = np.linspace(0, r.T1, 300)
        t2 = np.linspace(0, r.T2, 300)
        gx1, gy1, gz1 = ph.arrow3d_position(r.v0, r.theta1, self.A, self.omega, t1, self.g, self.phi)
        gx2, gy2, gz2 = ph.arrow3d_position(r.v0, r.theta2, self.A, self.omega, t2, self.g, self.phi)
        self.guide1.set_data_3d(gx1, gy1, gz1)
        self.guide2.set_data_3d(gx2, gy2, gz2)

        self.lateral_err = ph.lateral_error(r.T1, r.T2, self.A, self.omega, self.phi,
                                             self.A, self.omega, self.phi)
        self.suggested_omega = ph.suggest_omega(r.T1, r.T2, n=0)

        x_max = max(gx1.max(), gx2.max())
        y_span = max(self.A, 0.5) * 1.3
        z_max = max(gz1.max(), gz2.max())
        self.ax.set_xlim(0, x_max * 1.05)
        self.ax.set_ylim(-y_span, y_span)
        self.ax.set_zlim(0, z_max * 1.2)

        if reset_time:
            self.state.reset()

    def _on_frame(self, _frame):
        r = self.result
        t = self.state.step()
        loop_end = r.T1 + LOOP_PAUSE
        if t > loop_end:
            self.state.t = 0.0
            t = 0.0

        t1c = np.clip(t, 0, r.T1)
        n1 = max(int(t1c / max(r.T1, 1e-6) * 300), 2)
        tt1 = np.linspace(0, t1c, n1)
        x1, y1, z1 = ph.arrow3d_position(r.v0, r.theta1, self.A, self.omega, tt1, self.g, self.phi)
        self.trail1.set_data_3d(x1, y1, z1)
        if t <= r.T1:
            self.arrow1.set_data_3d([x1[-1]], [y1[-1]], [z1[-1]])
            self.arrow1.set_alpha(1.0)
        else:
            self.arrow1.set_alpha(0.0)

        if t >= r.delay:
            t2c = np.clip(t - r.delay, 0, r.T2)
            n2 = max(int(t2c / max(r.T2, 1e-6) * 300), 2)
            tt2 = np.linspace(0, t2c, n2)
            x2, y2, z2 = ph.arrow3d_position(r.v0, r.theta2, self.A, self.omega, tt2, self.g, self.phi)
            self.trail2.set_data_3d(x2, y2, z2)
            if t <= r.T1:
                self.arrow2.set_data_3d([x2[-1]], [y2[-1]], [z2[-1]])
                self.arrow2.set_alpha(1.0)
            else:
                self.arrow2.set_alpha(0.0)
        else:
            self.trail2.set_data_3d([], [], [])
            self.arrow2.set_alpha(0.0)

        y1_hit = self.A * np.sin(self.omega * r.T1 + self.phi)
        y2_hit = self.A * np.sin(self.omega * r.T2 + self.phi)
        hit_match = self.lateral_err <= HIT_THRESHOLD
        if t >= r.T1 - ARRIVAL_EPS:
            self.verdict_text.set_text("명중 일치" if hit_match else "어긋남")
            self.verdict_text.set_color("green" if hit_match else "red")
        else:
            self.verdict_text.set_text("")

        self.info_text.set_text(
            f"v0={r.v0:5.1f} m/s  th1={r.theta1:5.1f}  th2={r.theta2:5.1f}\n"
            f"R={r.R:6.2f} m  T1={r.T1:5.2f}s  T2={r.T2:5.2f}s  delay={r.delay:5.2f}s\n"
            f"A={self.A:4.2f} m  omega={self.omega:5.2f} rad/s  (제안 omega0={self.suggested_omega:5.2f})\n"
            f"y1(T1)={y1_hit:6.3f}  y2(T2)={y2_hit:6.3f}  lateral_error={self.lateral_err:6.3f} m\n"
            f"t = {t:5.2f} s"
        )
        return (self.trail1, self.trail2, self.arrow1, self.arrow2,
                self.info_text, self.verdict_text)

    def show(self):
        plt.show()


def main():
    sim = CombinedSim()
    sim.show()


if __name__ == "__main__":
    main()
