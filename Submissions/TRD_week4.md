# TRD: Week 4 — Neural Network Physics Labs (PySide6 GUI)

**버전**: 1.0
**작성일**: 2026-04-07
**작성**: AI (Claude Sonnet 4.6) + 사용자 협업
**연관 문서**: PRD_week4.md

---

## 1. 기술 스택

| 구분 | 라이브러리 | 버전 |
|------|-----------|------|
| GUI | PySide6 | 6.x |
| ML | TensorFlow / Keras | 2.x |
| 수치 계산 | NumPy | 1.x |
| 시각화 | Matplotlib (QtAgg backend) | 3.x |
| 런타임 | Python | 3.10+ |

---

## 2. 파일 구조

```
week4/
├── week4_app.py          # 메인 진입점 (QApplication 실행)
├── PRD_week4.md
├── TRD_week4.md
├── PROCESS_week4.md      # 제작 과정 소개 (완성 후 작성)
└── outputs/              # 기존 standalone 스크립트 결과물
```

> 단일 파일(`week4_app.py`)로 구현한다. 각 Lab 클래스를 파일 내 섹션으로 구분해 관리한다.

---

## 3. 전체 아키텍처

```
QApplication
└── MainWindow(QMainWindow)
    └── QTabWidget
        ├── Lab1Widget  ──┐
        ├── Lab2Widget    ├── 공통 BaseLabWidget 상속
        ├── Lab3Widget    │   - 좌측: ControlPanel (QScrollArea)
        └── Lab4Widget  ──┘   - 우측: MplCanvas (FigureCanvasQTAgg)
                              - TrainingWorker (QThread)
```

---

## 4. 공통 컴포넌트

### 4.1 MplCanvas

```python
class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib Figure를 Qt 위젯으로 래핑"""
    - Figure 생성 (figsize 파라미터화)
    - tight_layout 적용
    - 외부에서 fig, axes 직접 접근 가능
```

### 4.2 TrainingWorker (QThread)

```python
class TrainingWorker(QThread):
    progress = Signal(int, float)   # (epoch, loss)
    finished = Signal(object)       # 결과 데이터 dict
    error = Signal(str)             # 오류 메시지

    def __init__(self, config: dict): ...
    def run(self): ...              # Keras 학습 실행
```

- `config` dict에 lab_id, 모든 하이퍼파라미터 포함
- Keras `LambdaCallback`으로 epoch마다 `progress` signal emit
- 완료 시 결과(history, predictions) dict를 `finished` signal로 전달

### 4.3 LiveLossCallback

```python
class LiveLossCallback(keras.callbacks.Callback):
    """epoch 종료마다 Qt Signal 발행"""
    def __init__(self, signal, total_epochs): ...
    def on_epoch_end(self, epoch, logs): ...
```

### 4.4 BaseLabWidget (공통 레이아웃)

```python
class BaseLabWidget(QWidget):
    """각 Lab 탭의 공통 골격"""
    - QSplitter: 좌측 300px 컨트롤 / 우측 캔버스
    - ProgressBar + 상태 레이블
    - [Train] 버튼 (학습 중 비활성화)
    - _build_controls(): 서브클래스에서 구현
    - _on_train_clicked(): TrainingWorker 시작
    - _on_progress(epoch, loss): ProgressBar 업데이트
    - _on_finished(result): 플롯 렌더링 (서브클래스 구현)
```

---

## 5. Lab별 기술 명세

### 5.1 Lab1Widget — 1D 함수 근사

**입력 파라미터 (컨트롤 패널):**

| 컨트롤 | 타입 | 값 범위 | 기본값 |
|--------|------|---------|--------|
| 함수 선택 | QComboBox | sin(x) / cos(x)+0.5sin(2x) / x·sin(x) | sin(x) |
| 네트워크 크기 | QComboBox | Small/Medium/Large/Very Large | Large |
| 활성화 함수 | QComboBox | tanh / relu | tanh |
| Epochs | QSpinBox | 500~5000, step 500 | 2000 |
| 학습률 | QComboBox | 0.01 / 0.001 / 0.0001 | 0.01 |

**모델 구조:**
```python
architectures = {
    'Small':      [32],
    'Medium':     [64, 64],
    'Large':      [128, 128],
    'Very Large': [128, 128, 64],
}
model = Sequential([Input(1), *[Dense(n, activation) for n in arch], Dense(1)])
```

**결과 플롯 (3-panel, 1×3):**
- Panel 1: True vs Predicted 곡선
- Panel 2: Training Loss (log scale)
- Panel 3: Absolute Error 곡선

---

### 5.2 Lab2Widget — 포물선 운동 회귀

**입력 파라미터:**

| 컨트롤 | 타입 | 값 범위 | 기본값 |
|--------|------|---------|--------|
| 초기 속도 v₀ | QSlider | 10~50 m/s | 30 |
| 발사 각도 θ | QSlider | 10~80° | 45 |
| 학습 샘플 수 | QComboBox | 500/1000/2000 | 1000 |
| Epochs | QSpinBox | 50~300 | 100 |

**물리 수식 (데이터 생성):**
```
x(t) = v₀·cos(θ)·t
y(t) = v₀·sin(θ)·t - 0.5·g·t²    (g = 9.81)
```

**모델 구조:** Input(3) → Dense(128,relu) → Dropout(0.1) → Dense(64) → Dense(32) → Dense(2)

**결과 플롯 (1×2):**
- Panel 1: 예측 궤적 vs 실제 궤적
- Panel 2: Training/Validation Loss 곡선

---

### 5.3 Lab3Widget — 과적합 vs 과소적합

**입력 파라미터:**

| 컨트롤 | 타입 | 값 범위 | 기본값 |
|--------|------|---------|--------|
| 노이즈 레벨 | QSlider (×0.1) | 0.0~1.0 | 0.3 |
| Dropout 비율 | QSlider (×0.1) | 0.0~0.5 | 0.2 |
| Epochs | QSpinBox | 50~500 | 200 |

**3개 모델 동시 학습:**

| 모델 | 구조 | 특징 |
|------|------|------|
| Underfit | Dense(4) | 너무 단순 |
| Good | Dense(32)→Dropout→Dense(16)→Dropout | 적절 |
| Overfit | Dense(256)→Dense(128)→Dense(64)→Dense(32) | 너무 복잡 |

- 3개 모델을 순차 학습 (progress: 0→33→66→100%)
- True function: `y = sin(2x) + 0.5x`

**결과 플롯 (1×2):**
- Panel 1: 3개 모델 예측 + True function 오버레이
- Panel 2: Train/Val Loss 3개 모델 학습 곡선

---

### 5.4 Lab4Widget — 진자 주기 예측

**입력 파라미터:**

| 컨트롤 | 타입 | 값 범위 | 기본값 |
|--------|------|---------|--------|
| 진자 길이 L | QSlider (×0.1) | 0.5~2.0 m | 1.0 |
| 초기 각도 θ₀ | QSlider | 5~75° | 30 |
| Epochs | QSpinBox | 50~300 | 100 |

**물리 수식:**
```
T_small = 2π√(L/g)
T_true  = T_small × [1 + θ²/16 + 11θ⁴/3072]   (θ in radians)
RK4: d²θ/dt² = -(g/L)·sin(θ)
```

**모델 구조:** Input(2) → Dense(64,relu) → Dropout(0.1) → Dense(32) → Dense(16) → Dense(1)

**결과 플롯 (1×2):**
- Panel 1: 각도 범위 5~80°에서 예측 주기 vs 실제 주기 (슬라이더 L 기준)
- Panel 2: RK4 시뮬레이션 — 슬라이더 θ₀, L 기준 각도 vs 시간

**추가 UI:** 학습 완료 후 "예측 주기: X.XXXs | 실제 주기: X.XXXs | 오차: X.X%" 텍스트 레이블 표시

---

## 6. 스레드 안전성

- Keras 학습은 반드시 `QThread.run()` 내에서 실행
- Qt 위젯 업데이트는 반드시 메인 스레드에서 (Signal/Slot 통해 전달)
- `QMetaObject.invokeMethod` 직접 호출 금지 — Signal만 사용

---

## 7. 오류 처리

| 상황 | 처리 방법 |
|------|-----------|
| 학습 중 예외 발생 | `error` Signal로 메시지 전달 → 상태 레이블에 빨간 글씨 표시 |
| 학습 중 [Train] 재클릭 | 버튼 비활성화로 방지 |
| TensorFlow import 실패 | 앱 시작 시 import 오류 메시지 출력 후 종료 |

---

## 8. 실행 방법

```bash
cd week4
python week4_app.py
```

**의존성 설치:**
```bash
uv sync
# 또는
pip install pyside6 tensorflow numpy matplotlib
```

---

## 9. 제약사항

- TensorFlow는 CPU 모드로 실행 (GPU 설정 미지원)
- `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'` 로 TF 로그 억제
- Matplotlib backend: `matplotlib.use('QtAgg')` 필수 (import 전 설정)
