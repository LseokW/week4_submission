# Week 4 제작 과정 소개

**프로젝트명**: Neural Network Physics Labs — PySide6 GUI
**작성일**: 2026-04-07
**제작 방식**: Claude Code (Sonnet 4.6) + Superpowers Skills

---

## 제작 과정 개요

이 앱은 단순히 코드를 생성한 것이 아니라, **PRD → TRD → 구현 계획 → 코드 → 테스트** 순서의 소프트웨어 개발 프로세스를 따라 AI와 협업하여 만들었습니다.

```
1. 요구사항 분석 (Brainstorming)
        ↓
2. PRD 작성 (Product Requirements Document)
        ↓
3. TRD 작성 (Technical Requirements Document)
        ↓
4. 구현 계획 작성 (Writing Plans)
        ↓
5. 코드 구현 (Executing Plans)
        ↓
6. 테스트 실행 (7 tests passed)
```

---

## Step 1: Brainstorming (superpowers:brainstorming)

`superpowers:brainstorming` 스킬을 사용해 사용자와 질문-답변을 통해 제품 방향을 결정했습니다.

**결정된 핵심 사항:**
- 사용자: 코딩/ML 수업 학생
- 플랫폼: PySide6 데스크톱 GUI
- 구조: 단일 창 + 4탭 (Option A 선택)
- 파라미터 수준: 중급 (에폭, 학습률, 네트워크 크기, 활성화 함수, Dropout)

---

## Step 2: PRD — 제품 요구사항 정의

`PRD_week4.md` 작성. 주요 내용:

| 구분 | 내용 |
|------|------|
| 목적 | 4개 Lab을 통한 ML 개념 직관적 학습 |
| 사용자 | 코딩/ML 수업 대학생 |
| 핵심 기능 | 실시간 학습 시각화, 하이퍼파라미터 조작 |
| 비기능 요구사항 | UI 블로킹 없음, 1280×800 해상도, 학습 3분 이내 |

총 **22개 기능 요구사항 (F-01~F-45)** + **4개 비기능 요구사항 (NF-01~04)** 정의.

---

## Step 3: TRD — 기술 명세 정의

`TRD_week4.md` 작성. 주요 내용:

**기술 스택:**
```
PySide6 + TensorFlow/Keras + NumPy + Matplotlib (QtAgg backend)
```

**아키텍처:**
```
QMainWindow
└── QTabWidget (4탭)
    └── 각 탭: BaseLabWidget
        ├── QSplitter
        │   ├── 좌측: 컨트롤 패널 (QScrollArea)
        │   └── 우측: MplCanvas (Matplotlib 임베딩)
        └── TrainingWorker (QThread — 비동기 학습)
```

**Lab별 모델 구조, 파라미터 범위, 플롯 구성** 등 구현 전 완전히 명세화.

---

## Step 4: 구현 계획 (superpowers:writing-plans)

`docs/superpowers/plans/2026-04-07-week4-nn-labs.md` 작성.

6개 Task로 분해:

| Task | 내용 |
|------|------|
| 1 | 물리 함수 뼈대 + 테스트 |
| 2 | 모델 팩토리 + TrainingWorker + BaseLabWidget |
| 3 | Lab1Widget + MainWindow |
| 4 | Lab2Widget (포물선 운동) |
| 5 | Lab3Widget (과적합 데모) |
| 6 | Lab4Widget (진자 예측) |

PRD 요구사항 vs 구현 태스크 커버리지 테이블 포함 (22/22 요구사항 커버).

---

## Step 5: 코드 구현 (superpowers:executing-plans)

`week4_app.py` 단일 파일 (~470 lines). 구현된 주요 컴포넌트:

### 공유 인프라
- **`MplCanvas`**: Matplotlib Figure를 Qt 위젯으로 래핑
- **`LiveLossCallback`**: Keras 콜백 → epoch마다 Qt Signal emit
- **`TrainingWorker`**: QThread 기반 비동기 학습 실행기
- **`BaseLabWidget`**: 4개 Lab의 공통 레이아웃 골격

### 물리 함수
- `calculate_pendulum_period()` — 타원 적분 근사 주기 계산
- `generate_projectile_data()` — 포물선 운동 데이터 생성
- `simulate_pendulum_rk4()` — Runge-Kutta 4차 수치 적분

### 모델 팩토리
- `build_1d_model()` — 가변 레이어 1D 근사 모델
- `build_projectile_model()` — 포물선 회귀 모델 (입력 3, 출력 2)
- `build_underfit_model()` / `build_good_model()` / `build_overfit_model()`
- `build_pendulum_model()` — 진자 주기 예측 (입력 2, 출력 1)

---

## Step 6: 테스트

```
7 passed in 6.95s
```

| 테스트 | 내용 |
|--------|------|
| test_pendulum_small_angle | 작은 각도 근사 공식 검증 |
| test_pendulum_large_angle_longer | 큰 각도에서 주기 증가 검증 |
| test_projectile_data_shape | 데이터 shape (N, 3) / (N, 2) 검증 |
| test_1d_function_keys | 함수 딕셔너리 키 3개 검증 |
| test_build_1d_model_output_shape | 모델 출력 (1,1) 검증 |
| test_build_projectile_model_output_shape | 모델 출력 (1,2) 검증 |
| test_build_pendulum_model_output_shape | 모델 출력 (1,1) 검증 |

---

## 생성된 파일 목록

```
week4/
├── week4_app.py              # 메인 앱 (PySide6 GUI, ~470 lines)
├── PRD_week4.md              # 제품 요구사항 문서
├── TRD_week4.md              # 기술 명세 문서
├── PROCESS_week4.md          # 이 파일 (제작 과정)
├── tests/
│   ├── __init__.py
│   ├── test_physics.py       # 물리 함수 유닛 테스트 (4개)
│   └── test_models.py        # 모델 팩토리 유닛 테스트 (3개)
└── docs/superpowers/plans/
    └── 2026-04-07-week4-nn-labs.md   # 구현 계획
```

---

## 실행 방법

```bash
cd week4
python week4_app.py
```

**테스트 실행:**
```bash
cd week4
python -m pytest tests/ -v
```

---

## 참고: 각 Lab의 ML 개념

| Lab | ML 핵심 개념 |
|-----|------------|
| Lab 1 | Universal Approximation Theorem — 충분한 뉴런으로 임의 함수 근사 가능 |
| Lab 2 | 물리 법칙을 데이터로부터 학습 (회귀) |
| Lab 3 | Underfitting / Overfitting / Good Fit — 모델 복잡도와 일반화 |
| Lab 4 | 비선형 물리 관계 학습 + RK4 수치 해석과의 비교 |

---

**제작**: Claude Code (Sonnet 4.6) × Superpowers Skills v5.0.7
**워크플로우**: brainstorming → writing-plans → executing-plans
