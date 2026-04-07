"""
Week 4: Neural Network Physics Labs — PySide6 GUI
PRD: PRD_week4.md | TRD: TRD_week4.md | Plan: docs/superpowers/plans/2026-04-07-week4-nn-labs.md

4개 탭 구성:
  Lab 1 — 1D 함수 근사 (Universal Approximation Theorem)
  Lab 2 — 포물선 운동 회귀
  Lab 3 — 과적합 vs 과소적합 데모
  Lab 4 — 진자 주기 예측 + RK4 시뮬레이션
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QLabel, QSlider, QSpinBox, QComboBox, QPushButton,
    QProgressBar, QScrollArea, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal

import tensorflow as tf
from tensorflow import keras

g = 9.81  # 중력 가속도 (m/s²)

# ─── 공유 상수 ────────────────────────────────────────────────────────────────

FUNCTIONS_1D = {
    'sin(x)':            lambda x: np.sin(x),
    'cos(x)+0.5sin(2x)': lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
    'x·sin(x)':          lambda x: x * np.sin(x),
}

ARCHITECTURES = {
    'Small [32]':              [32],
    'Medium [64,64]':          [64, 64],
    'Large [128,128]':         [128, 128],
    'Very Large [128,128,64]': [128, 128, 64],
}

# ─── 물리 함수 ────────────────────────────────────────────────────────────────

def calculate_pendulum_period(L: float, theta0_deg: float) -> float:
    """타원 적분 근사로 진자 주기 계산"""
    theta0_rad = np.deg2rad(theta0_deg)
    T_small = 2 * np.pi * np.sqrt(L / g)
    correction = (1 + (1 / 16) * theta0_rad ** 2 + (11 / 3072) * theta0_rad ** 4)
    return T_small * correction


def generate_projectile_data(n_samples: int = 1000, noise: float = 0.5):
    """포물선 운동 학습 데이터 생성"""
    v0 = np.random.uniform(10, 50, n_samples)
    theta = np.random.uniform(20, 70, n_samples)
    theta_rad = np.deg2rad(theta)
    t_max = 2 * v0 * np.sin(theta_rad) / g
    t = np.random.uniform(0, t_max * 0.9, n_samples)
    x = v0 * np.cos(theta_rad) * t + np.random.normal(0, noise, n_samples)
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t ** 2 + np.random.normal(0, noise, n_samples)
    mask = y >= 0
    X = np.column_stack([v0[mask], theta[mask], t[mask]])
    Y = np.column_stack([x[mask], y[mask]])
    return X, Y


def simulate_pendulum_rk4(L: float, theta0_deg: float, t_max: float, dt: float = 0.01):
    """RK4 수치 적분으로 진자 운동 시뮬레이션"""
    theta = np.deg2rad(theta0_deg)
    omega = 0.0
    t_arr = np.arange(0, t_max, dt)
    th_arr = np.zeros_like(t_arr)
    om_arr = np.zeros_like(t_arr)
    for i in range(len(t_arr)):
        th_arr[i] = theta
        om_arr[i] = omega
        k1t, k1o = omega, -(g / L) * np.sin(theta)
        k2t, k2o = omega + 0.5 * dt * k1o, -(g / L) * np.sin(theta + 0.5 * dt * k1t)
        k3t, k3o = omega + 0.5 * dt * k2o, -(g / L) * np.sin(theta + 0.5 * dt * k2t)
        k4t, k4o = omega + dt * k3o,       -(g / L) * np.sin(theta + dt * k3t)
        theta += (dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t)
        omega += (dt / 6) * (k1o + 2 * k2o + 2 * k3o + k4o)
    return t_arr, np.rad2deg(th_arr), om_arr


# ─── 모델 팩토리 ──────────────────────────────────────────────────────────────

def build_1d_model(hidden_layers: list, activation: str = 'tanh', lr: float = 0.01):
    model = keras.Sequential([keras.layers.Input(shape=(1,))])
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation=activation))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    return model


def build_projectile_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2, activation='linear'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_underfit_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='linear'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_good_model(dropout: float = 0.2):
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation='linear'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_overfit_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='linear'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model


def build_pendulum_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='linear'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model


# ─── 공유 UI 컴포넌트 ─────────────────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(12, 4)):
        self.fig = Figure(figsize=figsize)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class LiveLossCallback(keras.callbacks.Callback):
    """epoch 종료마다 Qt Signal로 진행률·loss 전달"""
    def __init__(self, signal, total_epochs: int, start_pct: int = 0, end_pct: int = 100):
        super().__init__()
        self._signal = signal
        self._total = total_epochs
        self._s = start_pct
        self._e = end_pct

    def on_epoch_end(self, epoch, logs=None):
        frac = (epoch + 1) / self._total
        pct = int(self._s + frac * (self._e - self._s))
        loss = float(logs.get('loss', 0.0)) if logs else 0.0
        self._signal.emit(pct, loss)


class TrainingWorker(QThread):
    progress = Signal(int, float)   # (percent, loss)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            result = self._dispatch()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _dispatch(self) -> dict:
        lab = self.config['lab']
        if lab == 1:
            return self._train_lab1()
        if lab == 2:
            return self._train_lab2()
        if lab == 3:
            return self._train_lab3()
        if lab == 4:
            return self._train_lab4()
        raise ValueError(f"Unknown lab: {lab}")

    # ── Lab 1 ──────────────────────────────────────────────────────────────────
    def _train_lab1(self) -> dict:
        c = self.config
        x = np.linspace(-2 * np.pi, 2 * np.pi, 300).reshape(-1, 1)
        idx = np.random.permutation(len(x))
        x_train = x[idx]
        y_train = FUNCTIONS_1D[c['func']](x_train)
        x_test = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
        y_test = FUNCTIONS_1D[c['func']](x_test)

        model = build_1d_model(ARCHITECTURES[c['arch']], c['activation'], c['lr'])
        cb = LiveLossCallback(self.progress, c['epochs'])
        history = model.fit(
            x_train, y_train,
            epochs=c['epochs'], batch_size=32, verbose=0,
            callbacks=[cb],
        )
        y_pred = model.predict(x_test, verbose=0)
        return {
            'x_test': x_test, 'y_test': y_test, 'y_pred': y_pred,
            'loss': history.history['loss'],
        }

    # ── Lab 2 ──────────────────────────────────────────────────────────────────
    def _train_lab2(self) -> dict:
        c = self.config
        X_train, Y_train = generate_projectile_data(c['n_samples'])
        model = build_projectile_model()
        cb = LiveLossCallback(self.progress, c['epochs'])
        history = model.fit(
            X_train, Y_train,
            validation_split=0.2, epochs=c['epochs'],
            batch_size=32, verbose=0, callbacks=[cb],
        )
        # 테스트 궤적
        v0, theta = c['v0'], c['theta']
        theta_rad = np.deg2rad(theta)
        t_max = 2 * v0 * np.sin(theta_rad) / g
        t_pts = np.linspace(0, t_max, 60)
        X_traj = np.column_stack([np.full(60, v0), np.full(60, theta), t_pts])
        pred = model.predict(X_traj, verbose=0)
        x_true = v0 * np.cos(theta_rad) * t_pts
        y_true = v0 * np.sin(theta_rad) * t_pts - 0.5 * g * t_pts ** 2
        return {
            't': t_pts,
            'x_true': x_true, 'y_true': y_true,
            'x_pred': pred[:, 0], 'y_pred': pred[:, 1],
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'v0': v0, 'theta': theta,
        }

    # ── Lab 3 ──────────────────────────────────────────────────────────────────
    def _train_lab3(self) -> dict:
        c = self.config
        np.random.seed(42)
        x_train = np.random.uniform(-2, 2, 100).reshape(-1, 1)
        y_train = np.sin(2 * x_train) + 0.5 * x_train + np.random.normal(0, c['noise'], (100, 1))
        x_val = np.random.uniform(-2, 2, 50).reshape(-1, 1)
        y_val = np.sin(2 * x_val) + 0.5 * x_val + np.random.normal(0, c['noise'], (50, 1))
        x_test = np.linspace(-2, 2, 200).reshape(-1, 1)
        y_test = np.sin(2 * x_test) + 0.5 * x_test

        models_cfg = {
            'underfit': (build_underfit_model(), 0, 33),
            'good':     (build_good_model(c['dropout']), 33, 66),
            'overfit':  (build_overfit_model(), 66, 100),
        }
        histories, preds = {}, {}
        for name, (model, s, e) in models_cfg.items():
            cb = LiveLossCallback(self.progress, c['epochs'], s, e)
            h = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=c['epochs'], batch_size=16, verbose=0, callbacks=[cb],
            )
            histories[name] = h.history
            preds[name] = model.predict(x_test, verbose=0).flatten()
        return {
            'x_test': x_test, 'y_test': y_test.flatten(),
            'x_train': x_train.flatten(), 'y_train': y_train.flatten(),
            'histories': histories, 'preds': preds,
        }

    # ── Lab 4 ──────────────────────────────────────────────────────────────────
    def _train_lab4(self) -> dict:
        c = self.config
        L_arr = np.random.uniform(0.5, 3.0, 2000)
        th_arr = np.random.uniform(5, 80, 2000)
        T_arr = np.array([calculate_pendulum_period(l, t) for l, t in zip(L_arr, th_arr)])
        T_arr *= (1 + np.random.normal(0, 0.01, 2000))
        X = np.column_stack([L_arr, th_arr])
        Y = T_arr.reshape(-1, 1)

        model = build_pendulum_model()
        cb = LiveLossCallback(self.progress, c['epochs'])
        history = model.fit(
            X, Y, validation_split=0.2,
            epochs=c['epochs'], batch_size=32, verbose=0, callbacks=[cb],
        )
        # 각도 범위 예측
        angles = np.linspace(5, 80, 60)
        L_val = c['L']
        T_pred = model.predict(np.column_stack([np.full(60, L_val), angles]), verbose=0).flatten()
        T_true = np.array([calculate_pendulum_period(L_val, a) for a in angles])
        # RK4 시뮬레이션
        T_ref = calculate_pendulum_period(c['L'], c['theta0'])
        t_sim, th_sim, _ = simulate_pendulum_rk4(c['L'], c['theta0'], T_ref * 3)
        T_single_pred = float(model.predict(np.array([[c['L'], c['theta0']]]), verbose=0)[0, 0])
        return {
            'angles': angles, 'T_pred': T_pred, 'T_true': T_true,
            't_sim': t_sim, 'th_sim': th_sim,
            'T_pred_single': T_single_pred, 'T_true_single': T_ref,
            'train_loss': history.history['loss'],
        }


# ─── BaseLabWidget ────────────────────────────────────────────────────────────

class BaseLabWidget(QWidget):
    def __init__(self, figsize=(12, 4)):
        super().__init__()
        self._worker = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # 좌측: 스크롤 가능한 컨트롤 패널
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        ctrl_container = QWidget()
        self._ctrl_layout = QVBoxLayout(ctrl_container)
        self._ctrl_layout.setAlignment(Qt.AlignTop)
        self._build_controls()

        # 하단 고정: 버튼 + 진행률
        self._btn_train = QPushButton("Train")
        self._btn_train.setStyleSheet("font-size: 14px; padding: 6px;")
        self._btn_train.clicked.connect(self._on_train_clicked)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._status = QLabel("파라미터를 설정하고 Train을 클릭하세요.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #555; font-size: 11px;")

        self._ctrl_layout.addStretch()
        self._ctrl_layout.addWidget(self._btn_train)
        self._ctrl_layout.addWidget(self._progress)
        self._ctrl_layout.addWidget(self._status)
        scroll.setWidget(ctrl_container)
        splitter.addWidget(scroll)

        # 우측: Matplotlib 캔버스
        self._canvas = MplCanvas(figsize=figsize)
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(1, 1)

    def _build_controls(self):
        raise NotImplementedError

    def _get_config(self) -> dict:
        raise NotImplementedError

    def _render(self, result: dict):
        raise NotImplementedError

    def _add_group(self, title: str) -> QVBoxLayout:
        grp = QGroupBox(title)
        lay = QVBoxLayout(grp)
        self._ctrl_layout.addWidget(grp)
        return lay

    def _on_train_clicked(self):
        self._btn_train.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText("학습 중...")
        self._worker = TrainingWorker(self._get_config())
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, loss: float):
        self._progress.setValue(pct)
        self._status.setText(f"Epoch 진행 중... Loss: {loss:.6f}")

    def _on_finished(self, result: dict):
        try:
            self._render(result)
        except Exception as e:
            self._on_error(f"렌더링 오류: {e}")
            return
        self._progress.setValue(100)
        self._status.setText("학습 완료!")
        self._btn_train.setEnabled(True)

    def _on_error(self, msg: str):
        self._status.setStyleSheet("color: red; font-size: 11px;")
        self._status.setText(f"오류: {msg}")
        self._btn_train.setEnabled(True)


# ─── Lab 1: 1D 함수 근사 ──────────────────────────────────────────────────────

class Lab1Widget(BaseLabWidget):
    def __init__(self):
        super().__init__(figsize=(12, 4))

    def _build_controls(self):
        lay = self._add_group("근사할 함수")
        self._combo_func = QComboBox()
        self._combo_func.addItems(list(FUNCTIONS_1D.keys()))
        lay.addWidget(self._combo_func)

        lay2 = self._add_group("네트워크 크기")
        self._combo_arch = QComboBox()
        self._combo_arch.addItems(list(ARCHITECTURES.keys()))
        self._combo_arch.setCurrentIndex(2)  # Large 기본값
        lay2.addWidget(self._combo_arch)

        lay3 = self._add_group("활성화 함수")
        self._combo_act = QComboBox()
        self._combo_act.addItems(['tanh', 'relu'])
        lay3.addWidget(self._combo_act)

        lay4 = self._add_group("학습 파라미터")
        lay4.addWidget(QLabel("Epochs"))
        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(500, 5000)
        self._spin_epochs.setSingleStep(500)
        self._spin_epochs.setValue(2000)
        lay4.addWidget(self._spin_epochs)
        lay4.addWidget(QLabel("학습률 (Learning Rate)"))
        self._combo_lr = QComboBox()
        self._combo_lr.addItems(['0.01', '0.001', '0.0001'])
        lay4.addWidget(self._combo_lr)

    def _get_config(self) -> dict:
        return {
            'lab': 1,
            'func': self._combo_func.currentText(),
            'arch': self._combo_arch.currentText(),
            'activation': self._combo_act.currentText(),
            'epochs': self._spin_epochs.value(),
            'lr': float(self._combo_lr.currentText()),
        }

    def _render(self, result: dict):
        self._canvas.fig.clear()
        axes = self._canvas.fig.subplots(1, 3)
        x = result['x_test']
        y_true = result['y_test']
        y_pred = result['y_pred']

        # Panel 1: 함수 근사
        axes[0].plot(x, y_true, 'b-', lw=2.5, label='True', alpha=0.7)
        axes[0].plot(x, y_pred, 'r--', lw=2, label='NN 예측')
        mse = float(np.mean((y_pred - y_true) ** 2))
        axes[0].set_title(f'함수 근사\nMSE: {mse:.6f}', fontweight='bold')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # Panel 2: Loss 곡선
        axes[1].plot(result['loss'], 'g-', lw=1.5)
        axes[1].set_title('Training Loss', fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Epoch'); axes[1].grid(alpha=0.3)

        # Panel 3: 절대 오차
        err = np.abs(y_pred - y_true)
        axes[2].plot(x, err, 'r-', lw=1.5)
        axes[2].fill_between(x.flatten(), 0, err.flatten(), color='r', alpha=0.3)
        axes[2].set_title(f'절대 오차\nMax: {err.max():.5f}', fontweight='bold')
        axes[2].grid(alpha=0.3)

        self._canvas.fig.tight_layout()
        self._canvas.draw()


# ─── Lab 2: 포물선 운동 회귀 ──────────────────────────────────────────────────

class Lab2Widget(BaseLabWidget):
    def __init__(self):
        super().__init__(figsize=(12, 5))

    def _build_controls(self):
        lay = self._add_group("발사 조건 (테스트용)")
        lay.addWidget(QLabel("초기 속도 v₀ (m/s)"))
        self._slider_v0 = QSlider(Qt.Horizontal)
        self._slider_v0.setRange(10, 50)
        self._slider_v0.setValue(30)
        self._label_v0 = QLabel("30 m/s")
        self._slider_v0.valueChanged.connect(lambda v: self._label_v0.setText(f"{v} m/s"))
        lay.addWidget(self._slider_v0)
        lay.addWidget(self._label_v0)

        lay.addWidget(QLabel("발사 각도 θ (°)"))
        self._slider_theta = QSlider(Qt.Horizontal)
        self._slider_theta.setRange(10, 80)
        self._slider_theta.setValue(45)
        self._label_theta = QLabel("45 °")
        self._slider_theta.valueChanged.connect(lambda v: self._label_theta.setText(f"{v} °"))
        lay.addWidget(self._slider_theta)
        lay.addWidget(self._label_theta)

        lay2 = self._add_group("학습 설정")
        lay2.addWidget(QLabel("학습 샘플 수"))
        self._combo_samples = QComboBox()
        self._combo_samples.addItems(['500', '1000', '2000'])
        self._combo_samples.setCurrentIndex(1)
        lay2.addWidget(self._combo_samples)
        lay2.addWidget(QLabel("Epochs"))
        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(50, 300)
        self._spin_epochs.setValue(100)
        lay2.addWidget(self._spin_epochs)

    def _get_config(self) -> dict:
        return {
            'lab': 2,
            'v0': self._slider_v0.value(),
            'theta': self._slider_theta.value(),
            'n_samples': int(self._combo_samples.currentText()),
            'epochs': self._spin_epochs.value(),
        }

    def _render(self, result: dict):
        self._canvas.fig.clear()
        axes = self._canvas.fig.subplots(1, 2)

        # Panel 1: 궤적 비교
        ax = axes[0]
        ax.plot(result['x_true'], result['y_true'], 'b-', lw=2.5, label='물리 공식', alpha=0.7)
        ax.plot(result['x_pred'], result['y_pred'], 'r--', lw=2, label='NN 예측')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(f"포물선 궤적\nv₀={result['v0']} m/s, θ={result['theta']}°", fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

        # Panel 2: 학습 곡선
        ax2 = axes[1]
        ax2.plot(result['train_loss'], 'b-', lw=2, label='Train Loss')
        ax2.plot(result['val_loss'], 'r--', lw=2, label='Val Loss')
        ax2.set_title('학습 곡선', fontweight='bold')
        ax2.set_yscale('log'); ax2.set_xlabel('Epoch')
        ax2.legend(); ax2.grid(alpha=0.3)

        self._canvas.fig.tight_layout()
        self._canvas.draw()


# ─── Lab 3: 과적합 vs 과소적합 ───────────────────────────────────────────────

class Lab3Widget(BaseLabWidget):
    def __init__(self):
        super().__init__(figsize=(12, 5))

    def _build_controls(self):
        lay = self._add_group("데이터 설정")
        lay.addWidget(QLabel("노이즈 레벨"))
        self._slider_noise = QSlider(Qt.Horizontal)
        self._slider_noise.setRange(0, 10)
        self._slider_noise.setValue(3)
        self._label_noise = QLabel("0.3")
        self._slider_noise.valueChanged.connect(
            lambda v: self._label_noise.setText(f"{v * 0.1:.1f}")
        )
        lay.addWidget(self._slider_noise)
        lay.addWidget(self._label_noise)

        lay2 = self._add_group("Good 모델 설정")
        lay2.addWidget(QLabel("Dropout 비율"))
        self._slider_dropout = QSlider(Qt.Horizontal)
        self._slider_dropout.setRange(0, 5)
        self._slider_dropout.setValue(2)
        self._label_dropout = QLabel("0.2")
        self._slider_dropout.valueChanged.connect(
            lambda v: self._label_dropout.setText(f"{v * 0.1:.1f}")
        )
        lay2.addWidget(self._slider_dropout)
        lay2.addWidget(self._label_dropout)

        lay3 = self._add_group("학습 설정")
        lay3.addWidget(QLabel("Epochs (3개 모델 공통)"))
        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(50, 500)
        self._spin_epochs.setValue(200)
        lay3.addWidget(self._spin_epochs)

    def _get_config(self) -> dict:
        return {
            'lab': 3,
            'noise': self._slider_noise.value() * 0.1,
            'dropout': self._slider_dropout.value() * 0.1,
            'epochs': self._spin_epochs.value(),
        }

    def _render(self, result: dict):
        self._canvas.fig.clear()
        axes = self._canvas.fig.subplots(1, 2)
        colors = {'underfit': '#2196F3', 'good': '#4CAF50', 'overfit': '#F44336'}
        labels = {'underfit': 'Underfit (너무 단순)', 'good': 'Good Fit', 'overfit': 'Overfit (너무 복잡)'}

        # Panel 1: 예측 비교
        ax = axes[0]
        ax.scatter(result['x_train'], result['y_train'],
                   alpha=0.4, s=20, c='gray', label='학습 데이터')
        ax.plot(result['x_test'], result['y_test'],
                'k-', lw=2.5, label='실제 함수', alpha=0.8)
        for name, pred in result['preds'].items():
            ax.plot(result['x_test'], pred, color=colors[name],
                    lw=2, linestyle='--', label=labels[name])
        ax.set_title('모델 예측 비교', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Panel 2: 학습 곡선
        ax2 = axes[1]
        for name, h in result['histories'].items():
            ax2.plot(h['loss'], color=colors[name], lw=2,
                     linestyle='-', label=f'{labels[name]} Train')
            ax2.plot(h['val_loss'], color=colors[name], lw=1.5,
                     linestyle='--', alpha=0.6, label=f'Val')
        ax2.set_title('Train vs Val Loss', fontweight='bold')
        ax2.set_yscale('log'); ax2.set_xlabel('Epoch')
        ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

        self._canvas.fig.tight_layout()
        self._canvas.draw()


# ─── Lab 4: 진자 주기 예측 ────────────────────────────────────────────────────

class Lab4Widget(BaseLabWidget):
    def __init__(self):
        super().__init__(figsize=(12, 5))

    def _build_controls(self):
        lay = self._add_group("진자 파라미터")
        lay.addWidget(QLabel("길이 L (m)"))
        self._slider_L = QSlider(Qt.Horizontal)
        self._slider_L.setRange(5, 20)   # ×0.1 → 0.5~2.0 m
        self._slider_L.setValue(10)
        self._label_L = QLabel("1.0 m")
        self._slider_L.valueChanged.connect(
            lambda v: self._label_L.setText(f"{v * 0.1:.1f} m")
        )
        lay.addWidget(self._slider_L)
        lay.addWidget(self._label_L)

        lay.addWidget(QLabel("초기 각도 θ₀ (°)"))
        self._slider_theta = QSlider(Qt.Horizontal)
        self._slider_theta.setRange(5, 75)
        self._slider_theta.setValue(30)
        self._label_theta = QLabel("30 °")
        self._slider_theta.valueChanged.connect(
            lambda v: self._label_theta.setText(f"{v} °")
        )
        lay.addWidget(self._slider_theta)
        lay.addWidget(self._label_theta)

        lay2 = self._add_group("학습 설정")
        lay2.addWidget(QLabel("Epochs"))
        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(50, 300)
        self._spin_epochs.setValue(100)
        lay2.addWidget(self._spin_epochs)

        # 예측 결과 표시
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        self._result_label.setStyleSheet(
            "font-size: 12px; color: #1a237e; background: #e8eaf6;"
            "padding: 8px; border-radius: 4px;"
        )
        self._ctrl_layout.addWidget(self._result_label)

    def _get_config(self) -> dict:
        return {
            'lab': 4,
            'L': self._slider_L.value() * 0.1,
            'theta0': self._slider_theta.value(),
            'epochs': self._spin_epochs.value(),
        }

    def _on_finished(self, result: dict):
        T_pred = result['T_pred_single']
        T_true = result['T_true_single']
        err_pct = abs(T_pred - T_true) / T_true * 100
        self._result_label.setText(
            f"예측 주기: {T_pred:.4f} s\n"
            f"실제 주기: {T_true:.4f} s\n"
            f"오차: {err_pct:.2f}%"
        )
        super()._on_finished(result)

    def _render(self, result: dict):
        self._canvas.fig.clear()
        axes = self._canvas.fig.subplots(1, 2)

        # Panel 1: 주기 예측 vs 실제
        ax = axes[0]
        ax.plot(result['angles'], result['T_true'],
                'b-', lw=2.5, label='실제 주기 (물리 공식)', alpha=0.7)
        ax.plot(result['angles'], result['T_pred'],
                'r--', lw=2, label='NN 예측')
        ax.set_xlabel('초기 각도 (°)'); ax.set_ylabel('주기 T (s)')
        ax.set_title('진자 주기 예측', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)

        # Panel 2: RK4 시뮬레이션
        ax2 = axes[1]
        ax2.plot(result['t_sim'], result['th_sim'], '#4CAF50', lw=1.5)
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('각도 (°)')
        ax2.set_title('RK4 운동 시뮬레이션', fontweight='bold')
        ax2.axhline(0, color='k', lw=0.8, linestyle='--', alpha=0.4)
        ax2.grid(alpha=0.3)

        self._canvas.fig.tight_layout()
        self._canvas.draw()


# ─── MainWindow ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 4: Neural Network Physics Labs")
        self.resize(1280, 800)

        tabs = QTabWidget()
        tabs.addTab(Lab1Widget(), "Lab 1: 1D 함수 근사")
        tabs.addTab(Lab2Widget(), "Lab 2: 포물선 운동")
        tabs.addTab(Lab3Widget(), "Lab 3: 과적합 데모")
        tabs.addTab(Lab4Widget(), "Lab 4: 진자 예측")
        self.setCentralWidget(tabs)


# ─── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
