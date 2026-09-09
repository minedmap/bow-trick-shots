# bow-trick-shots

활을 이용한 두 가지 트릭샷을 벡터 물리로 모델링한 파이썬 시뮬레이션입니다.

1. **더블 하위처 샷** — 같은 속력, 상보각(θ, 90°−θ)으로 쏘아 두 화살이 동시에 착탄하는 궤적
2. **S자 우회 샷** — 매그누스 힘에 의한 횡방향 진동으로 장애물 2개를 피해가는 궤적
3. **결합 모델** — 위 두 가지를 합쳐 3차원(다운레인지·높이·좌우)에서 두 화살의 탄착점이 실제로 일치하는지 검증

모든 물리 공식과 파라미터는 `bow_trick_shot_simulation_spec.md` 스펙을 따릅니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python main.py --mode 1   # 더블 하위처 샷
python main.py --mode 2   # S자 우회 샷
python main.py --mode 3   # 결합 모델 (3D)
```

`--mode`를 생략하면 실행 시 번호를 물어봅니다. 각 창에는 슬라이더, Play/Pause·Reset 버튼이 있으며 값을 바꾸면 궤적과 수치(사거리 R, 체공시간 T1/T2, 지연시간 delay, lateral_error 등)가 실시간으로 갱신됩니다.

## 파일 구성

- `physics.py` — 벡터 물리 공식(사거리, 체공시간, 포물선 위치, S자 궤적, 장애물 회피 연립방정식, 동일 탄착 omega 제안 등) 순수 함수 모음
- `sim_common.py` — 재생 제어(Play/Pause/Reset)와 한글 폰트 설정 등 시뮬레이션 공통 유틸리티
- `sim_double_hawkeye.py` — 1번 시뮬레이션 (matplotlib 애니메이션)
- `sim_s_curve.py` — 2번 시뮬레이션
- `sim_combined.py` — 3번 시뮬레이션 (3D)
- `main.py` — 실행 진입점
- `tests/test_physics.py` — 스펙의 검증용 예시값(v0=30, θ1=70° → R≈59m, T1≈5.75s, T2≈2.09s, delay≈3.66s)을 포함한 단위 테스트

## 테스트

```bash
pip install pytest
pytest
```

## 한글 폰트 안내

그래프에 한글이 네모(□)로 보이면 시스템에 한글 폰트가 없는 것입니다. 예를 들어 우분투/데비안 계열에서는 다음으로 설치할 수 있습니다.

```bash
sudo apt-get install fonts-nanum
```
