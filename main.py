"""활 트릭샷 시뮬레이션 실행 진입점.

사용법:
    python main.py --mode 1   # 더블 하위처 샷 (수직면, 동시 착탄)
    python main.py --mode 2   # S자 우회 샷 (수평면, 장애물 회피)
    python main.py --mode 3   # 결합 모델 (3D, 동일 탄착점 검증)
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="활 트릭샷 벡터 물리 시뮬레이션")
    parser.add_argument("--mode", "-m", type=int, choices=(1, 2, 3), default=None,
                         help="1: 더블 하위처 샷, 2: S자 우회 샷, 3: 결합 모델(3D)")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        print("실행할 시뮬레이션을 선택하세요:")
        print("  1) 더블 하위처 샷 (수직면, 동시 착탄)")
        print("  2) S자 우회 샷 (수평면, 장애물 회피)")
        print("  3) 결합 모델 (3D, 동일 탄착점 검증)")
        mode = int(input("번호 입력 (1/2/3): ").strip())

    if mode == 1:
        from sim_double_hawkeye import main as run
    elif mode == 2:
        from sim_s_curve import main as run
    elif mode == 3:
        from sim_combined import main as run
    else:
        raise SystemExit(f"알 수 없는 모드: {mode}")

    run()


if __name__ == "__main__":
    main()
