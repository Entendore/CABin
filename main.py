#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA TEXT TRANSLATOR — single-file application
============================================
A cellular-automata search laboratory with a *guaranteed* any-text -> any-text
translation mode.

Highlights
----------
* Universal Translator (Guaranteed): any Unicode -> any Unicode, any lengths,
  built as a windowed micro-rule CA path, peephole-optimized and independently
  verified — 100% exact match on every run.
* Hybrid mode: Beam search does the CA work, constructive fix-up guarantees exactness.
* Classic searches: Forward Simulation, Exhaustive 256, A*, Beam, MCTS,
  Genetic Algorithm, Simulated Annealing, Hill Climb — all with live telemetry.
* Full dark-themed GUI: header, splitter layout, tabbed telemetry deck
  (match strip, dynamics chart, clickable step timeline, candidates/bars,
  256-rule heatmap, MCTS tree, GA population, preview), result inspector with
  VERIFIED badge, exports (PNG / path JSON / telemetry JSONL / session JSON),
  persistent settings (geometry, splitters, last algorithm).
* Headless API + CLI: translate_text(), batch_translate_file(), selftest, bench.
"""

# =====================================================================
#  1. IMPORTS
# =====================================================================
import sys
import time
import json
import math
import random
import heapq
import logging
import datetime
import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, List, Dict, Tuple

import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QLabel, QSpinBox, QTextEdit,
    QGroupBox, QFormLayout, QSlider, QComboBox, QDoubleSpinBox, QProgressBar,
    QFileDialog, QMessageBox, QSplitter, QScrollArea, QTabWidget,
    QStatusBar, QFrame, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QStackedWidget, QListWidget, QListWidgetItem)
from PySide6.QtCore import (Qt, QTimer, QThread, Signal, QRectF, QObject,
    QElapsedTimer, QPoint, QPointF, QSettings)
from PySide6.QtGui import (QPainter, QColor, QFont, QPen, QLinearGradient,
    QRadialGradient, QImage, QPainterPath, QKeySequence, QShortcut,
    QMouseEvent, QWheelEvent, QFontMetrics)

# =====================================================================
#  2. CONSTANTS, THEME & LOGGING
# =====================================================================
UNICODE_BITS = 32
MAX_CODE_POINT = 0x10FFFF
DEFAULT_GENOME_SIZE = 32
REPLACEMENT_CHAR = '\ufffd'

EXPLORATION_C = math.sqrt(2)
ROLLOUT_GREEDY_P = 0.8
GA_STAGNATION_PATIENCE = 5
GA_MUT_MAX = 0.25
ELITE_FRACTION = 0.10
SA_T_END = 0.02

TELEMETRY_INTERVALS = {'default': 50, 'series': 60, 'heat': 60, 'bits': 90,
                       'stat': 100, 'candidates': 120, 'population': 120,
                       'preview': 150, 'tree': 250}

SERIES_STYLE = {  # name: (color, width, dashed)
    "best":    ("#00ff66", 2, False),
    "current": ("#00aaff", 2, False),
    "mean":    ("#ffcc00", 1, True),
    "worst":   ("#ff5555", 1, True),
    "temp":    ("#ff55ff", 2, True),
}


class Theme:
    BG      = "#050505"
    PANEL   = "#0a0a0a"
    PANEL2  = "#101010"
    BORDER  = "#1d1d1d"
    INPUT   = "#0d0d0d"
    TEXT    = "#eaeaea"
    DIM     = "#8a8a8a"
    ACCENT  = "#00ff66"
    ACCENT2 = "#00aaff"
    WARN    = "#ffcc00"
    DANGER  = "#ff5555"
    MAGENTA = "#ff55ff"


ALGORITHMS = ["Forward Simulation", "Exhaustive (256 Rules)", "A* (Backtracking)",
              "Beam Search", "MCTS", "Genetic Algorithm", "Simulated Annealing",
              "Hill Climb (Greedy)", "Universal Translator (Guaranteed)",
              "Hybrid (Search + Fix-up)"]

ALGO_PAGE = {"Forward Simulation": 0, "Exhaustive (256 Rules)": 1, "MCTS": 2,
             "Genetic Algorithm": 3, "A* (Backtracking)": 4, "Beam Search": 4,
             "Simulated Annealing": 4, "Hill Climb (Greedy)": 4,
             "Universal Translator (Guaranteed)": 4, "Hybrid (Search + Fix-up)": 4}

TRANSLATION_PRESETS = [
    ("classic",       "HELLO", "WORLD"),
    ("reorder-proof", "AAAA", "ABAB"),
    ("length-change", "CELLULAR AUTOMATA", "CA"),
    ("unicode",       "héllo wörld", "你世界 ⚙️"),
    ("emoji",         "plain text", "🚀🔥✨"),
]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ca_engine.{name}")


def _configure_logging() -> None:
    root = logging.getLogger("ca_engine")
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler("ca_search_engine.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)


_configure_logging()
logger = get_logger("app")


def apply_theme(app: QApplication) -> None:
    t = Theme
    qss = """
    QWidget { background:@BG; color:@TEXT; font-family:'Consolas','Courier New',monospace; font-size:12px; }
    QLabel#headerTitle { font-size:16px; font-weight:bold; color:@ACCENT; background:transparent; }
    QLabel#headerSub   { font-size:11px; color:@DIM; background:transparent; }
    QLabel#chip        { color:@DIM; border:1px solid @BORDER; border-radius:9px; padding:3px 10px; background:@PANEL; }
    QLabel#counter     { color:@DIM; font-size:10px; }
    QLabel#statLine    { color:@WARN; font-size:11px; }
    QFrame#header { background:@PANEL; border-bottom:1px solid @BORDER; }
    QFrame#card   { background:@PANEL; border:1px solid @BORDER; border-radius:8px; }
    QGroupBox { background:@PANEL; border:1px solid @BORDER; border-radius:8px; margin-top:14px; padding-top:4px; }
    QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:@ACCENT; }
    QPushButton { background:#111111; border:1px solid #2a2a2a; border-radius:6px; padding:6px 14px; color:@TEXT; }
    QPushButton:hover { border-color:@ACCENT2; }
    QPushButton:pressed { background:#1a1a1a; }
    QPushButton:disabled { color:#555; border-color:#222; background:#0c0c0c; }
    QPushButton#primary { background:#0b2e1c; border-color:@ACCENT; color:#7dffb0; font-weight:bold; }
    QPushButton#primary:hover { background:#0e3a24; }
    QPushButton#danger { border-color:@DANGER; color:#ffb0b0; }
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background:@INPUT; border:1px solid #262626; border-radius:4px;
        padding:3px 6px; selection-background-color:@ACCENT2; }
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color:@ACCENT2; }
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width:14px; }
    QComboBox QAbstractItemView { background:@INPUT; selection-background-color:#12324a; border:1px solid @BORDER; }
    QTabWidget::pane { border:1px solid @BORDER; border-radius:6px; top:-1px; }
    QTabBar::tab { background:@INPUT; color:@DIM; padding:6px 16px; border:1px solid @BORDER; border-bottom:none;
                   border-top-left-radius:6px; border-top-right-radius:6px; margin-right:2px; }
    QTabBar::tab:selected { color:@ACCENT; background:@PANEL2; border-color:@ACCENT; }
    QProgressBar { background:@INPUT; border:1px solid #262626; border-radius:4px; text-align:center; max-height:16px; }
    QProgressBar::chunk { border-radius:3px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 @ACCENT, stop:1 @ACCENT2); }
    QSlider::groove:horizontal { height:4px; background:#262626; border-radius:2px; }
    QSlider::sub-page:horizontal { background:#0f4d2f; border-radius:2px; }
    QSlider::handle:horizontal { width:12px; margin:-5px 0; border-radius:6px; background:@ACCENT; }
    QSlider::handle:horizontal:hover { background:#66ffb2; }
    QSplitter::handle { background:@BG; }
    QSplitter::handle:horizontal { width:5px; }
    QSplitter::handle:vertical { height:5px; }
    QSplitter::handle:hover { background:#1d3a4d; }
    QStatusBar { background:@PANEL; border-top:1px solid @BORDER; }
    QStatusBar QLabel { color:@DIM; }
    QCheckBox::indicator { width:13px; height:13px; border:1px solid #333; border-radius:3px; background:@INPUT; }
    QCheckBox::indicator:checked { background:@ACCENT; border-color:@ACCENT; }
    QScrollBar:vertical { width:10px; background:transparent; }
    QScrollBar::handle:vertical { background:#262626; border-radius:5px; min-height:30px; }
    QScrollBar::handle:vertical:hover { background:@ACCENT2; }
    QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
    QScrollArea { border:none; background:transparent; }
    QListWidget { background:@PANEL; border:1px solid @BORDER; border-radius:4px; }
    QListWidget::item { padding:2px 4px; }
    QListWidget::item:selected { background:#12324a; }
    RunStatsPanel QLabel { font-family:'Consolas'; font-size:11px; }
    """
    tokens = {"@BG": t.BG, "@PANEL": t.PANEL, "@PANEL2": t.PANEL2, "@BORDER": t.BORDER,
              "@INPUT": t.INPUT, "@TEXT": t.TEXT, "@DIM": t.DIM, "@ACCENT": t.ACCENT,
              "@ACCENT2": t.ACCENT2, "@WARN": t.WARN, "@DANGER": t.DANGER,
              "@MAGENTA": t.MAGENTA}
    for k, v in tokens.items():
        qss = qss.replace(k, v)
    app.setStyleSheet(qss)

# =====================================================================
#  3. BIT CODEC & CA ENGINE
# =====================================================================
def text_to_bits(text: str, target_chars: Optional[int] = None, pad_char: str = ' ') -> np.ndarray:
    if target_chars is not None:
        text = text[:target_chars].ljust(target_chars, pad_char)
    if not text:
        return np.zeros(0, dtype=np.uint8)
    cp = np.fromiter((ord(c) & MAX_CODE_POINT for c in text), dtype='<u4', count=len(text))
    return np.unpackbits(cp.view(np.uint8)).astype(np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    if bits.size == 0:
        return ""
    usable = bits.size - (bits.size % UNICODE_BITS)
    cp = np.packbits(bits[:usable]).reshape(-1, 4).view('<u4').ravel()
    return "".join(REPLACEMENT_CHAR if (0xD800 <= c <= 0xDFFF or c > MAX_CODE_POINT
                                       or c in (0xFFFE, 0xFFFF)) else chr(c)
                   for c in cp.tolist())


class CAEngine:
    def __init__(self, cols: int, rows: int, neighborhood: int = 3):
        self.logger = get_logger("CAEngine")
        self.cols, self.rows, self.neighborhood = cols, rows, neighborhood
        self.grid: List[np.ndarray] = []
        self.current_row = 0
        self.rule_table = np.zeros(1 << neighborhood, dtype=np.uint8)

    def set_rule_table(self, table) -> None:
        self.rule_table = np.array(table, dtype=np.uint8)  # copy: never alias cached tables

    def seed_from_bits(self, bits: np.ndarray) -> None:
        self.grid = [np.asarray(bits, dtype=np.uint8).copy()]
        self.current_row = 1

    def _step_vectorized(self, prev_row: np.ndarray) -> np.ndarray:
        if self.neighborhood == 3:
            L = np.roll(prev_row, 1); L[0] = 0
            R = np.roll(prev_row, -1); R[-1] = 0
            idx = (L << 2) | (prev_row << 1) | R
        else:
            L2 = np.roll(prev_row, 2); L2[:2] = 0
            L1 = np.roll(prev_row, 1); L1[0] = 0
            R1 = np.roll(prev_row, -1); R1[-1] = 0
            R2 = np.roll(prev_row, -2); R2[-2:] = 0
            idx = (L2 << 4) | (L1 << 3) | (prev_row << 2) | (R1 << 1) | R2
        return self.rule_table[idx]

    def step(self) -> Optional[np.ndarray]:
        if self.current_row >= self.rows:
            return None
        new_row = self._step_vectorized(self.grid[-1])
        self.grid.append(new_row)
        self.current_row += 1
        return new_row

    def run_full(self) -> np.ndarray:
        while self.current_row < self.rows:
            self.step()
        return self.grid[-1]


def apply_rule_fast(bits: np.ndarray, rule_table: np.ndarray, cols: int) -> np.ndarray:
    L = np.roll(bits, 1); L[0] = 0
    R = np.roll(bits, -1); R[-1] = 0
    idx = (L << 2) | (bits << 1) | R
    return np.asarray(rule_table, dtype=np.uint8)[idx]


def apply_rule_wrap(bits: np.ndarray, rule_table: np.ndarray) -> np.ndarray:
    """Toroidal boundaries."""
    L = np.roll(bits, 1)
    R = np.roll(bits, -1)
    idx = (L << 2) | (bits << 1) | R
    return np.asarray(rule_table, dtype=np.uint8)[idx]


def _neighborhoods5(rows: np.ndarray) -> np.ndarray:
    L2 = np.zeros_like(rows); L2[:, 2:] = rows[:, :-2]
    L1 = np.zeros_like(rows); L1[:, 1:] = rows[:, :-1]
    R1 = np.zeros_like(rows); R1[:, :-1] = rows[:, 1:]
    R2 = np.zeros_like(rows); R2[:, :-2] = rows[:, 2:]
    return (L2 << 4) | (L1 << 3) | (rows << 2) | (R1 << 1) | R2


def batch_evolve(population: np.ndarray, seed_row: np.ndarray, steps: int) -> np.ndarray:
    """Evolve a whole GA population simultaneously. population: (P, 32)."""
    P = population.shape[0]
    rows = np.repeat(seed_row[None, :], P, axis=0)
    prow = np.arange(P)[:, None]
    for _ in range(steps):
        rows = population[prow, _neighborhoods5(rows)]
    return rows


def _genome_grid(genome: np.ndarray, input_bits: np.ndarray, rows: int) -> Optional[List[np.ndarray]]:
    if genome is None:
        return None
    ca = CAEngine(input_bits.size, rows, 5)
    ca.set_rule_table(genome)
    ca.seed_from_bits(input_bits)
    ca.run_full()
    return ca.grid

# =====================================================================
#  4. RULE TABLES
# =====================================================================
def elementary_table(rule: int) -> np.ndarray:
    """table[i] = output for neighborhood index i = (L<<2)|(C<<1)|R."""
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)


_RULE_NUMBERS = {
    "OR": 254, "AND": 128, "XOR": 150, "NAND": 127, "NOR": 1, "XNOR": 105,
    "MAJORITY": 232, "LEFT": 240, "RIGHT": 170, "IDENTITY": 204, "INVERT": 51,
    "R30": 30, "R90": 90, "R110": 110, "R184": 184,
}
RULE_NAME_BY_NUMBER = {v: k for k, v in _RULE_NUMBERS.items()}


@lru_cache(maxsize=1)
def get_basic_rule_tables() -> Dict[str, np.ndarray]:
    tables = {name: elementary_table(num) for name, num in _RULE_NUMBERS.items()}
    tables["TOP"] = tables["IDENTITY"]      # back-compat aliases
    tables["NOT_TOP"] = tables["INVERT"]
    return tables


def rule_name(rule_num: int) -> str:
    return RULE_NAME_BY_NUMBER.get(rule_num, f"Rule {rule_num}")

# =====================================================================
#  4b. UNIVERSAL TRANSLATOR (guaranteed any-text -> any-text)
# =====================================================================
@lru_cache(maxsize=1)
def get_micro_rule_tables() -> Dict[str, np.ndarray]:
    """Windowed micro-rules. All three are neighbor-INDEPENDENT (output depends
    only on the center cell), which is what makes windowed application exact and
    path merging/optimization provably safe."""
    return {"SET0": elementary_table(0),     # always write 0
            "SET1": elementary_table(255),   # always write 1
            "INV":  elementary_table(51)}    # flip center (NOT)


def apply_rule_window(bits: np.ndarray, table: np.ndarray,
                      start: int, end: int, wrap: bool = False) -> np.ndarray:
    """Apply a rule only inside [start, end). Zero boundaries, or toroidal."""
    out = np.asarray(bits, np.uint8).copy()
    a = max(0, min(out.size, int(start)))
    b = max(0, min(out.size, int(end)))
    if b <= a:
        return out
    seg = out[a:b]
    L = np.roll(seg, 1); R = np.roll(seg, -1)
    if not wrap:
        L[0] = 0; R[-1] = 0
    idx = (L << 2) | (seg << 1) | R
    out[a:b] = np.asarray(table, dtype=np.uint8)[idx]
    return out


def _parse_span(s: str) -> int:
    s = s.strip()
    return int(s[:-1]) * UNICODE_BITS if (s and s[-1] in "cC") else int(s)


def parse_step(step: str) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """'RULE' | 'SET1@12:44' (bit units) | 'SET1@2c:5c' (char units)."""
    if "@" in step:
        name, _, rng = step.partition("@")
        a_s, _, b_s = rng.partition(":")
        tables = get_micro_rule_tables()
        if name not in tables:
            raise KeyError(f"Unknown micro-rule: {name}")
        return tables[name], _parse_span(a_s), _parse_span(b_s)
    return get_basic_rule_tables()[step], None, None


def align_bits_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Equalize lengths so ANY pair of texts (any sizes) can be transformed.
    Whole-character arrays are space-padded per character; otherwise zero-padded."""
    a = np.asarray(a, np.uint8)
    b = np.asarray(b, np.uint8)
    if a.size == b.size:
        return a.copy(), b.copy()
    if a.size % UNICODE_BITS == 0 and b.size % UNICODE_BITS == 0:
        w = max(a.size, b.size) // UNICODE_BITS
        return (text_to_bits(bits_to_text(a).ljust(w)),
                text_to_bits(bits_to_text(b).ljust(w)))
    w = max(a.size, b.size)
    pa = np.zeros(w, dtype=np.uint8); pa[:a.size] = a
    pb = np.zeros(w, dtype=np.uint8); pb[:b.size] = b
    return pa, pb

# =====================================================================
#  4c. REPAIR ENGINE v2  (per-run strategy => short guaranteed paths)
# =====================================================================
def _runs_of(mask: np.ndarray):
    """Yield (start, end) spans of True runs — vectorized."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    splits = np.flatnonzero(np.diff(idx) > 1)
    for s, e in zip(np.r_[idx[0], idx[splits + 1]], np.r_[idx[splits], idx[-1] + 1]):
        yield int(s), int(e)


def repair_bits(state, target, reporter=None, progress_from: int = 0) -> Tuple[List[str], np.ndarray]:
    """Constructively build a rule path mapping `state` onto `target` EXACTLY.
    Per mismatched run: constant target -> one SET; exact complement -> one INV;
    otherwise split by target runs and SET each."""
    a = np.asarray(state, np.uint8).copy()
    b = np.asarray(target, np.uint8)
    n = min(a.size, b.size)
    path: List[str] = []
    span = max(1, 100 - progress_from)

    def report(frac: float) -> None:
        if reporter:
            fit = int(np.count_nonzero(a[:n] == b[:n]))
            reporter.progress(min(99, progress_from + int(frac * span)),
                              f"repairing… {len(path)} steps | {fit}/{n} bits match")
            reporter.emit_telemetry({'type': 'bits', 'bits': a.copy(), 'target': b})
            reporter.emit_telemetry({'type': 'series', 'best': fit, 'current': fit})

    for _attempt in range(2):                     # 2nd pass = paranoid safety net
        for i, j in _runs_of(a[:n] != b[:n]):
            if reporter and reporter.isInterruptionRequested():
                return path, a
            seg_a, seg_b = a[i:j], b[i:j]
            vals = np.unique(seg_b)
            if vals.size == 1:                             # constant target → 1 write
                v = int(vals[0]); path.append(f"SET{v}@{i}:{j}"); a[i:j] = v
            elif np.array_equal(seg_a, seg_b ^ 1):         # exact complement → 1 flip
                path.append(f"INV@{i}:{j}"); a[i:j] = seg_b
            else:                                          # split by target runs
                s = i
                while s < j:
                    e, v = s, int(seg_b[s - i])
                    while e < j and int(seg_b[e - i]) == v:
                        e += 1
                    path.append(f"SET{v}@{s}:{e}"); a[s:e] = v
                    s = e
            report(j / max(1, n))
        if np.array_equal(a[:n], b[:n]):
            break

    if reporter:
        fit = int(np.count_nonzero(a == b))
        reporter.progress(100, f"repair done: {len(path)} steps | {fit}/{a.size}")
        reporter.emit_telemetry({'type': 'stat',
                                 'text': f"steps {len(path)} · exact {fit == a.size}"})
        reporter.emit_telemetry({'type': 'bars',
                                 'data': dict(Counter(s.split('@', 1)[0] for s in path))})
    logger.info("Repair built %d steps (exact=%s).",
                len(path), bool(np.array_equal(a[:n], b[:n])))
    return path, a


def guaranteed_fixup(state, target, reporter=None, progress_from: int = 0):
    """Back-compat wrapper around repair_bits."""
    return repair_bits(state, target, reporter, progress_from)


def universal_translate(input_bits, target_bits, reporter=None):
    """GUARANTEED any-text -> any-text translation. Returns (path, final_bits)."""
    logger.info("Universal translator starting (guaranteed mode).")
    a, b = align_bits_pair(input_bits, target_bits)
    if np.array_equal(a, b):
        if reporter:
            reporter.progress(100, "Input already equals target.")
        return [], a
    return repair_bits(a, b, reporter, progress_from=0)


def hybrid_translate(input_bits, target_bits, max_depth, beam_width,
                     basic_rules, reporter=None):
    """CA search first (Beam), then constructive fix-up -> exact result."""
    logger.info("Hybrid translate starting (beam %d, depth %d).", beam_width, max_depth)
    a, b = align_bits_pair(input_bits, target_bits)
    if np.array_equal(a, b):
        return [], a
    path: List[str] = []
    try:
        spath, sbits = beam_search(a, b, max_depth, beam_width, basic_rules, reporter)
    except Exception as e:
        logger.warning("Hybrid beam stage failed: %s", e)
        spath, sbits = None, None
    if spath:
        path.extend(spath)
    fpath, final = repair_bits(sbits if sbits is not None else a, b,
                               reporter, progress_from=70)
    path.extend(fpath)
    return path, final

# =====================================================================
#  4d. PATH OPTIMIZER & INDEPENDENT VERIFIER
# =====================================================================
def _parse_full(step: str):
    name, _, rng = step.partition("@")
    if rng:
        a_s, _, b_s = rng.partition(":")
        return name, get_micro_rule_tables()[name], _parse_span(a_s), _parse_span(b_s)
    return name, get_basic_rule_tables()[name], None, None


def _fmt_step(name: str, a: Optional[int], b: Optional[int]) -> str:
    return f"{name}@{a}:{b}" if a is not None else name


def _apply_parsed(st, bits: np.ndarray) -> np.ndarray:
    _, table, a, b = st
    return (apply_rule_window(bits, table, a, b) if a is not None
            else apply_rule_fast(bits, table, bits.size))


def optimize_path(path: List[str], seed_bits: Optional[np.ndarray] = None,
                  passes: int = 3) -> List[str]:
    """Safe peephole optimizer (result is always semantically identical):
       · drops IDENTITY no-ops
       · same window: SET,SET→last · SET,INV→SET{1-v} · INV,SET→SET · INV,INV→gone
       · merges contiguous windows of the same micro-rule (neighbor-independent
         micro-rules make merging provably equivalent)
       · with seed_bits: simulation-based removal of any step that changes nothing"""
    micro = get_micro_rule_tables()
    steps = [_parse_full(s) for s in path]
    for _ in range(max(1, passes)):
        before = len(steps)
        steps = [s for s in steps if not (s[2] is None and s[0] == "IDENTITY")]
        out: List[tuple] = []
        for st in steps:
            name, table, a, b = st
            if a is not None and out and (out[-1][2], out[-1][3]) == (a, b):
                pn = out[-1][0]
                if pn.startswith("SET") and name.startswith("SET"):
                    out[-1] = st; continue
                if pn.startswith("SET") and name == "INV":
                    nm = f"SET{1 - int(pn[3:])}"
                    out[-1] = (nm, micro[nm], a, b); continue
                if pn == "INV" and name.startswith("SET"):
                    out[-1] = st; continue
                if pn == "INV" and name == "INV":
                    out.pop(); continue
            if (a is not None and out and out[-1][0] == name
                    and out[-1][3] == a and name in micro):
                out[-1] = (name, table, out[-1][2], b); continue
            out.append(st)
        steps = out
        if len(steps) == before:
            break
    if seed_bits is not None:
        bits = np.asarray(seed_bits, np.uint8).copy()
        kept = []
        for st in steps:
            nb = _apply_parsed(st, bits)
            if not np.array_equal(nb, bits):
                kept.append(st); bits = nb
        steps = kept
    return [_fmt_step(s[0], s[2], s[3]) for s in steps]


def verify_path(input_bits, target_bits, path: List[str]) -> Tuple[bool, np.ndarray]:
    """Independent re-application of every step. Trust, not hope."""
    bits = np.asarray(input_bits, np.uint8).copy()
    for step in path:
        table, a, b = parse_step(step)
        bits = (apply_rule_window(bits, table, a, b) if a is not None
                else apply_rule_fast(bits, table, bits.size))
    return bool(np.array_equal(bits, np.asarray(target_bits, np.uint8))), bits

# =====================================================================
#  4e. HIGH-LEVEL TRANSLATION API
# =====================================================================
@dataclass
class TranslationResult:
    source: str
    target: str
    path: List[str]
    verified: bool
    steps: int
    elapsed_ms: float
    algo: str
    final_text: str


def align_texts(src: str, dst: str, pad_char: str = ' ') -> Tuple[str, str]:
    w = max(len(src), len(dst))
    return src.ljust(w, pad_char), dst.ljust(w, pad_char)


def translate_bits(src_bits, dst_bits, *, optimize: bool = True, verify: bool = True,
                   algo: str = "Universal Translator (Guaranteed)") -> TranslationResult:
    t0 = time.perf_counter()
    a, b = align_bits_pair(src_bits, dst_bits)
    src_txt, tgt_txt = bits_to_text(a), bits_to_text(b)
    if np.array_equal(a, b):
        return TranslationResult(src_txt, tgt_txt, [], True, 0,
                                 (time.perf_counter() - t0) * 1000, algo, tgt_txt)
    path, final = repair_bits(a, b)
    if optimize and path:
        path = optimize_path(path, seed_bits=a)
    verified = np.array_equal(final, b)
    if verify and path:
        ok, final = verify_path(a, b, path)
        verified = verified and ok
    return TranslationResult(src_txt, tgt_txt, path, bool(verified), len(path),
                             (time.perf_counter() - t0) * 1000, algo, bits_to_text(final))


def translate_text(src: str, dst: str, *, optimize: bool = True, verify: bool = True,
                   pad_char: str = ' ') -> TranslationResult:
    s, d = align_texts(src, dst, pad_char)
    return translate_bits(text_to_bits(s), text_to_bits(d),
                          optimize=optimize, verify=verify)


def batch_translate_file(in_path: str, out_path: str, *, optimize: bool = True,
                         encoding: str = "utf-8") -> int:
    """Input: one 'source<TAB>target' pair per line. Output: JSONL results."""
    n = 0
    with open(in_path, encoding=encoding) as fin, \
         open(out_path, "w", encoding=encoding) as fout:
        for ln, line in enumerate(fin, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            src, sep, dst = line.partition("\t")
            if not sep:
                logger.warning("batch line %d: missing TAB separator — skipped", ln)
                continue
            res = translate_text(src, dst, optimize=optimize)
            fout.write(json.dumps({"line": ln, "source": src, "target": dst,
                                   "verified": res.verified, "steps": res.steps,
                                   "ms": round(res.elapsed_ms, 2), "path": res.path},
                                  ensure_ascii=False) + "\n")
            n += 1
    logger.info("Batch translated %d pairs -> %s", n, out_path)
    return n


def save_session(out_path: str, *, source: str, target: str, algo: str,
                 params: Optional[dict] = None, path: Optional[List[str]] = None) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "source": source, "target": target, "algo": algo,
                   "params": params or {}, "path": path or []},
                  f, ensure_ascii=False, indent=2)


def load_session(in_path: str) -> dict:
    with open(in_path, encoding="utf-8") as f:
        return json.load(f)


def export_telemetry_jsonl(log, out_path: str) -> int:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist() if o.size <= 128 else {"len": int(o.size), "sum": int(o.sum())}
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, bytes):
            return o.hex()[:96]
        return str(o)
    with open(out_path, "w", encoding="utf-8") as f:
        for t, rec in log:
            f.write(json.dumps({"t_ms": t, **rec}, default=default, ensure_ascii=False) + "\n")
    return len(log)

# =====================================================================
#  5. SEARCH ALGORITHMS (instrumented for live telemetry)
# =====================================================================
# NOTE: A*'s Hamming heuristic is INADMISSIBLE (one rule can flip many bits),
# so found paths are not guaranteed shortest. Kept for speed/UX.
def backtracking_search(input_bits, target_bits, max_depth, basic_rules, reporter=None):
    logger.info("A* search starting (max_depth=%d)", max_depth)
    cols = input_bits.size
    target = np.asarray(target_bits, np.uint8)
    if np.array_equal(input_bits, target):
        return [], input_bits
    heap = [(int(np.count_nonzero(input_bits != target)), 0, 0, input_bits.copy(), [])]
    visited = {input_bits.tobytes()}
    nodes, counter = 0, 1
    while heap:
        if reporter and reporter.isInterruptionRequested():
            return None, None
        h, depth, _, current_bits, path = heapq.heappop(heap)
        nodes += 1
        cur_fit = cols - h
        if reporter and nodes % 25 == 0:
            reporter.progress(-1, f"A* … nodes: {nodes} | depth: {depth}")
            reporter.emit_telemetry({'type': 'series', 'best': cur_fit, 'current': cur_fit})
            reporter.emit_telemetry({'type': 'stat',
                                     'text': f"nodes {nodes} · depth {depth} · heap {len(heap)}"})
            reporter.emit_telemetry({'type': 'bits', 'bits': current_bits, 'target': target})
            cur_label = "▶ " + ("→".join(path[-4:]) or "start")
            items = [(cur_label, cur_fit, cols)]
            items += [("→".join(t[4][-4:]) or "start", cols - (t[0] - t[1]), cols)
                      for t in heapq.nsmallest(4, heap, key=lambda t: t[0])]
            reporter.emit_telemetry({'type': 'candidates', 'items': items})
            reporter.emit_telemetry({'type': 'bars', 'data': dict(Counter(path))})
        if depth >= max_depth:
            continue
        for r_name, table in basic_rules.items():
            nb = apply_rule_fast(current_bits, table, cols)
            if np.array_equal(nb, target):
                return path + [r_name], nb
            hb = nb.tobytes()
            if hb not in visited:
                visited.add(hb)
                nh = int(np.count_nonzero(nb != target))
                heapq.heappush(heap, (depth + 1 + nh, depth + 1, counter, nb, path + [r_name]))
                counter += 1
    logger.warning("A* exhausted all states without a match.")
    return None, None


def beam_search(input_bits, target_bits, max_depth, beam_width, basic_rules, reporter=None):
    logger.info("Beam search starting (depth=%d, width=%d)", max_depth, beam_width)
    cols = input_bits.size
    target = np.asarray(target_bits, np.uint8)
    if np.array_equal(input_bits, target):
        return [], input_bits
    fit0 = int(np.count_nonzero(input_bits == target))
    beam = [(fit0, input_bits.tobytes(), input_bits, [])]
    visited = {input_bits.tobytes()}
    for depth in range(max_depth):
        if reporter and reporter.isInterruptionRequested():
            break
        candidates = []
        for _, _, bits, path in beam:
            for r_name, table in basic_rules.items():
                nb = apply_rule_fast(bits, table, cols)
                hb = nb.tobytes()
                if hb in visited:
                    continue
                visited.add(hb)
                nfit = int(np.count_nonzero(nb == target))
                if nfit == cols:
                    return path + [r_name], nb
                candidates.append((nfit, hb, nb, path + [r_name]))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[0], reverse=True)
        beam = candidates[:beam_width]
        if reporter:
            reporter.progress(int((depth + 1) / max_depth * 100),
                              f"Beam depth {depth + 1}/{max_depth} | best {beam[0][0]}/{cols}")
            reporter.emit_telemetry({'type': 'series', 'best': beam[0][0],
                                     'mean': float(np.mean([c[0] for c in beam]))})
            reporter.emit_telemetry({'type': 'stat',
                                     'text': f"depth {depth + 1}/{max_depth} · frontier {len(candidates)}"})
            reporter.emit_telemetry({'type': 'bits', 'bits': beam[0][2], 'target': target})
            reporter.emit_telemetry({'type': 'candidates',
                                     'items': [("→".join(c[3][-4:]) or "start", c[0], cols)
                                               for c in beam[:8]]})
            reporter.emit_telemetry({'type': 'bars',
                                     'data': dict(Counter(a for *_, p in beam for a in p[-3:]))})
    if beam:
        logger.info("Beam finished. Best partial fitness %d/%d", beam[0][0], cols)
        return beam[0][3], beam[0][2]
    return None, None


@dataclass(eq=False)
class MCTSNode:
    state_bits: np.ndarray
    parent: Optional['MCTSNode']
    action: Optional[str]
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[str] = field(default_factory=list)


def _reconstruct_path(node: MCTSNode) -> List[str]:
    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    path.reverse()
    return path


def mcts_search(input_bits, target_bits, max_depth, iterations, reporter=None, rng=None):
    logger.info("MCTS starting (iters=%d, depth=%d)", iterations, max_depth)
    rng = rng or random.Random()
    basic_rules = get_basic_rule_tables()
    rule_names = list(basic_rules.keys())
    cols = input_bits.size
    target = np.asarray(target_bits, np.uint8)
    if np.array_equal(input_bits, target):
        return [], True
    root = MCTSNode(input_bits.copy(), None, None, list(rule_names))
    best_roll = 0.0

    for i in range(iterations):
        if reporter and reporter.isInterruptionRequested():
            break
        node, depth, visited = root, 0, []
        # Selection (bounded by max_depth)
        while node.children and not node.untried_actions and depth < max_depth:
            if node.visits == 0:
                break
            log_n = math.log(node.visits)
            best_child, best_ucb = None, -float('inf')
            for child in node.children.values():
                ucb = ((child.value / child.visits) + EXPLORATION_C * math.sqrt(log_n / child.visits)
                       if child.visits else float('inf'))
                if ucb > best_ucb:
                    best_ucb, best_child = ucb, child
            if best_child is None:
                break
            node = best_child; visited.append(node); depth += 1
            if np.array_equal(node.state_bits, target):
                return _reconstruct_path(node), True
        # Expansion
        if node.untried_actions and depth < max_depth:
            action = rng.choice(node.untried_actions)
            node.untried_actions.remove(action)
            child = MCTSNode(apply_rule_fast(node.state_bits, basic_rules[action], cols),
                             node, action, list(rule_names))
            node.children[action] = child
            node = child; visited.append(node); depth += 1
            if np.array_equal(node.state_bits, target):
                return _reconstruct_path(node), True
        # Rollout
        rollout_path: List[str] = []
        temp_bits = node.state_bits.copy()
        seen = {temp_bits.tobytes()}
        for _ in range(max(0, max_depth - depth)):
            if rng.random() < ROLLOUT_GREEDY_P:
                best_action, best_fit = None, -1
                for act in rule_names:
                    tb = apply_rule_fast(temp_bits, basic_rules[act], cols)
                    f = int(np.count_nonzero(tb == target))
                    if f > best_fit:
                        best_fit, best_action = f, act
                act = best_action
            else:
                act = rng.choice(rule_names)
            rollout_path.append(act)
            temp_bits = apply_rule_fast(temp_bits, basic_rules[act], cols)
            if temp_bits.tobytes() in seen:
                break
            seen.add(temp_bits.tobytes())
            if np.array_equal(temp_bits, target):
                return _reconstruct_path(node) + rollout_path, True
        rollout_reward = float(np.count_nonzero(temp_bits == target)) / cols
        best_roll = max(best_roll, rollout_reward)
        for n in visited:
            n.visits += 1; n.value += rollout_reward
        root.visits += 1; root.value += rollout_reward

        if reporter and i % 20 == 0:
            avg = root.value / root.visits if root.visits else 0.0
            reporter.progress(int(i / iterations * 100),
                              f"MCTS iter {i + 1}/{iterations} | avg reward {avg:.3f}")
            reporter.emit_telemetry({'type': 'series', 'best': best_roll, 'mean': avg})
            snap: List[tuple] = []
            def walk(n: MCTSNode, pid: int) -> None:
                nid = len(snap)
                snap.append((nid, pid, n.action, n.visits,
                             n.value / n.visits if n.visits else 0.0))
                for c in n.children.values():
                    walk(c, nid)
            walk(root, -1)
            reporter.emit_telemetry({'type': 'tree', 'nodes': snap})
            reporter.emit_telemetry({'type': 'stat',
                                     'text': f"iter {i + 1} · tree {len(snap)} nodes"})
            leaf = root
            while leaf.children:
                leaf = max(leaf.children.values(), key=lambda c: c.visits)
            reporter.emit_telemetry({'type': 'bits', 'bits': leaf.state_bits, 'target': target})
            if root.children:
                reporter.emit_telemetry({'type': 'bars',
                                         'data': {a: c.visits for a, c in root.children.items()}})
                ranked = sorted(root.children.items(), key=lambda kv: kv[1].visits, reverse=True)[:8]
                reporter.emit_telemetry({'type': 'candidates',
                                         'items': [(name, int(round(c.value / c.visits * cols)) if c.visits else 0, cols)
                                                   for name, c in ranked]})

    best_path, node = [], root
    while node.children:
        node = max(node.children.values(), key=lambda c: c.visits)
        best_path.append(node.action)
    logger.info("MCTS done. Best path by visits: %s", " -> ".join(best_path) or "None")
    return best_path, False


def simulated_annealing(input_bits, target_bits, seq_len, iters, t_init,
                        basic_rules, reporter=None, rng=None):
    """SA over fixed-length rule sequences. Neighbor = 1-3 random position mutations."""
    logger.info("SA starting (iters=%d, len=%d, T0=%.2f)", iters, seq_len, t_init)
    rng = rng or random.Random()
    names = list(basic_rules.keys())
    cols = input_bits.size
    target = np.asarray(target_bits, np.uint8)
    if np.array_equal(input_bits, target):
        return [], input_bits

    def evaluate(seq):
        bits = input_bits
        for r in seq:
            bits = apply_rule_fast(bits, basic_rules[r], cols)
        return int(np.count_nonzero(bits == target)), bits

    seq = [rng.choice(names) for _ in range(seq_len)]
    fit, bits = evaluate(seq)
    best_seq, best_fit, best_bits = seq[:], fit, bits
    n_acc = 0

    for i in range(iters):
        if reporter and reporter.isInterruptionRequested():
            break
        T = t_init * (SA_T_END / t_init) ** (i / max(1, iters - 1))
        cand = seq[:]
        for _ in range(rng.randint(1, max(1, seq_len // 3))):
            cand[rng.randrange(seq_len)] = rng.choice(names)
        cfit, cbits = evaluate(cand)
        if cfit >= fit or (T > 1e-9 and rng.random() < math.exp((cfit - fit) / T)):
            seq, fit, bits = cand, cfit, cbits
            n_acc += 1
        if fit > best_fit:
            best_seq, best_fit, best_bits = seq[:], fit, bits
        if best_fit == cols:
            break
        if reporter and i % 25 == 0:
            reporter.progress(int(i / iters * 100),
                              f"SA iter {i + 1}/{iters} | T={T:.2f} | best {best_fit}/{cols}")
            reporter.emit_telemetry({'type': 'series', 'best': best_fit,
                                     'current': fit, 'temp': T})
            reporter.emit_telemetry({'type': 'stat',
                                     'text': f"iter {i + 1} · T {T:.2f} · acc {100 * n_acc / (i + 1):.0f}%"})
            reporter.emit_telemetry({'type': 'bits', 'bits': bits, 'target': target})
            reporter.emit_telemetry({'type': 'candidates', 'items': [
                ("best " + "→".join(best_seq[-4:]), best_fit, cols),
                ("cur  " + "→".join(seq[-4:]), fit, cols)]})
    logger.info("SA done. Best fitness %d/%d", best_fit, cols)
    return best_seq, best_bits


def exhaustive_search_256(input_bits, target_bits, rows, reporter=None):
    logger.info("Exhaustive search starting (256 rules).")
    cols = input_bits.size
    ca = CAEngine(cols, rows, 3)
    ca.seed_from_bits(input_bits)
    target = np.asarray(target_bits, np.uint8)
    fits = np.zeros(256, dtype=np.int32)
    best_fit, best_rule, best_grid = -1, None, None
    for rule_num in range(256):
        if reporter and reporter.isInterruptionRequested():
            return best_rule, best_grid
        ca.set_rule_table(elementary_table(rule_num))
        final = ca.run_full()
        fit = int(np.count_nonzero(final == target))
        fits[rule_num] = fit
        if fit > best_fit:
            best_fit, best_rule, best_grid = fit, rule_num, ca.grid
            logger.debug("New best rule %d (fitness %d/%d)", rule_num, fit, cols)
        if reporter:
            reporter.progress(int((rule_num + 1) / 256 * 100),
                              f"Rule {rule_num}/255… best {best_fit}/{cols}")
            reporter.emit_telemetry({'type': 'heat', 'data': fits.copy(),
                                     'current': rule_num, 'best': best_rule, 'cols': cols})
            reporter.emit_telemetry({'type': 'series', 'best': best_fit, 'current': fit})
            reporter.emit_telemetry({'type': 'stat',
                                     'text': f"rule {rule_num}/255 · best {best_fit}/{cols}"})
            reporter.emit_telemetry({'type': 'bits', 'bits': final, 'target': target})
            reporter.emit_telemetry({'type': 'preview', 'grid': ca.grid})
        if fit == cols:
            logger.info("Exact match at Rule %d.", rule_num)
            return rule_num, ca.grid
    logger.info("Exhaustive done. Best partial: Rule %s (%d/%d)", best_rule, best_fit, cols)
    return best_rule, best_grid


def genetic_algorithm_search(input_bits, target_bits, rows, pop_size=100, ga_gens=50,
                             mut_rate=0.02, reporter=None, nprng=None):
    logger.info("GA starting (pop=%d, gens=%d)", pop_size, ga_gens)
    cols = input_bits.size
    target = np.asarray(target_bits, np.uint8)
    nprng = nprng or np.random.default_rng()
    pop = nprng.integers(0, 2, (pop_size, DEFAULT_GENOME_SIZE), dtype=np.uint8)
    best_fit, best_rule, mut, stagn = -1, None, mut_rate, 0
    for gen in range(ga_gens):
        if reporter and reporter.isInterruptionRequested():
            break
        finals = batch_evolve(pop, input_bits, rows - 1)
        fits = (finals == target).sum(axis=1).astype(np.int32)
        bi = int(np.argmax(fits))
        p = pop.mean(axis=0, dtype=np.float64)
        diversity = float(np.mean(2.0 * p * (1.0 - p)))
        if int(fits[bi]) > best_fit:
            best_fit, best_rule, stagn, mut = int(fits[bi]), pop[bi].copy(), 0, mut_rate
        else:
            stagn += 1
        if stagn > GA_STAGNATION_PATIENCE:
            mut = min(GA_MUT_MAX, mut * 1.5)
        if reporter:
            reporter.progress(int((gen + 1) / ga_gens * 100),
                              f"GA gen {gen + 1}/{ga_gens} | best {best_fit}/{cols} | mut {mut:.3f}")
            reporter.emit_telemetry({'type': 'population',
                                     'fits': fits.copy(),
                                     'genome': best_rule if best_rule is not None else pop[bi].copy(),
                                     'gen': gen + 1, 'mut': mut, 'div': diversity, 'cols': cols})
            reporter.emit_telemetry({'type': 'series', 'best': best_fit,
                                     'mean': float(np.mean(fits)), 'worst': int(np.min(fits))})
            reporter.emit_telemetry({'type': 'stat',
                                     'text': f"gen {gen + 1}/{ga_gens} · div {diversity:.2f} · mut {mut:.3f}"})
            reporter.emit_telemetry({'type': 'bits', 'bits': finals[bi], 'target': target})
        if best_fit == cols:
            break
        if gen % 3 == 0 and best_rule is not None:
            g = _genome_grid(best_rule, input_bits, rows)
            if g:
                reporter.emit_telemetry({'type': 'preview', 'grid': g})
        order = np.argsort(fits)[::-1]
        elite = max(1, int(pop_size * ELITE_FRACTION))
        newpop = [pop[i].copy() for i in order[:elite]]
        while len(newpop) < pop_size:
            def tournament():
                idxs = nprng.choice(pop_size, 3, replace=False)
                return pop[int(idxs[int(np.argmax(fits[idxs]))])]
            p1, p2 = tournament(), tournament()
            a, b = np.sort(nprng.choice(np.arange(1, DEFAULT_GENOME_SIZE), 2, replace=False))
            child = np.empty(DEFAULT_GENOME_SIZE, dtype=np.uint8)
            child[:a] = p1[:a]; child[a:b] = p2[a:b]; child[b:] = p1[b:]
            child[nprng.random(DEFAULT_GENOME_SIZE) < mut] ^= 1
            newpop.append(child)
        pop = np.asarray(newpop)
    logger.info("GA done. Best fitness %d/%d", best_fit, cols)
    best_grid = _genome_grid(best_rule, input_bits, rows)
    return best_rule, best_grid, max(best_fit, 0)


def hill_climb_search(input_bits, target_bits, max_steps, basic_rules,
                      reporter=None, rng=None):
    """Steepest-ascent with random plateau escape. Very fast, best-effort."""
    logger.info("Hill climb starting (max_steps=%d)", max_steps)
    rng = rng or random.Random()
    names = list(basic_rules)
    cols = input_bits.size
    target = np.asarray(target_bits, np.uint8)
    if np.array_equal(input_bits, target):
        return [], input_bits
    bits = input_bits.copy()
    fit = int(np.count_nonzero(bits == target))
    path: List[str] = []
    for i in range(max_steps):
        if reporter and reporter.isInterruptionRequested():
            break
        best_name, best_bits, best_fit = None, None, fit
        for name in names:
            nb = apply_rule_fast(bits, basic_rules[name], cols)
            f = int(np.count_nonzero(nb == target))
            if f > best_fit:
                best_name, best_bits, best_fit = name, nb, f
        if best_name is None:                       # plateau → one random escape step
            best_name = rng.choice(names)
            best_bits = apply_rule_fast(bits, basic_rules[best_name], cols)
        bits = best_bits
        fit = int(np.count_nonzero(bits == target))
        path.append(best_name)
        if reporter and i % 10 == 0:
            reporter.progress(int(i / max_steps * 100),
                              f"Hill climb {i + 1}/{max_steps} | fit {fit}/{cols}")
            reporter.emit_telemetry({'type': 'series', 'best': fit, 'current': fit})
            reporter.emit_telemetry({'type': 'bits', 'bits': bits, 'target': target})
        if fit == cols:
            logger.info("Hill climb exact in %d steps.", len(path))
            return path, bits
    return path, bits

# =====================================================================
#  6. THREADING: REPORTER + WORKER
# =====================================================================
class _Reporter(QObject):
    """Wraps the worker so searches get progress-with-ETA + throttled telemetry."""
    def __init__(self, worker: "SearchWorker"):
        super().__init__()
        self._w = worker
        self._t0 = QElapsedTimer()
        self._t0.start()

    def isInterruptionRequested(self) -> bool:
        return self._w.isInterruptionRequested()

    def emit_telemetry(self, data: dict) -> None:
        self._w.emit_telemetry(data)

    def progress(self, pct: int, msg: str) -> None:
        if pct < 0:
            self._w.progress.emit(-1, msg)
            return
        pct = max(0, min(100, int(pct)))
        eta = ""
        if pct > 2:
            rem = self._t0.elapsed() / pct * (100 - pct)
            eta = f" | ETA {int(rem // 60):02d}:{int(rem % 60):02d}"
        self._w.progress.emit(pct, msg + eta)


class SearchWorker(QObject):
    finished = Signal(object, object, str)
    progress = Signal(int, str)
    telemetry = Signal(object)

    def __init__(self, algo, input_bits, target_bits, rows, params):
        super().__init__()
        self.logger = get_logger("SearchWorker")
        self.algo, self.input_bits, self.target_bits, self.rows, self.params = \
            algo, input_bits, target_bits, rows, params
        self._is_interrupted = False
        self._timer = QElapsedTimer(); self._timer.start()
        self._last_emit_ms: Dict[str, float] = {}
        self.record = False
        self.telemetry_log: List[Tuple[float, dict]] = []
        self.result_table: Optional[np.ndarray] = None
        self.result_neighborhood = 3
        self.result_path: List[str] = []
        self.result_final_bits: Optional[np.ndarray] = None

    def request_interruption(self): self._is_interrupted = True
    def isInterruptionRequested(self): return self._is_interrupted

    def emit_telemetry(self, data: dict) -> None:
        now = self._timer.elapsed()
        kind = data.get("type", "default")
        iv = TELEMETRY_INTERVALS.get(kind, TELEMETRY_INTERVALS['default'])
        if now - self._last_emit_ms.get(kind, -(1 << 30)) >= iv:
            self._last_emit_ms[kind] = now
            if self.record:
                self.telemetry_log.append((now, data))
            self.telemetry.emit(data)

    def _grid_from_path(self, path: List[str]) -> List[np.ndarray]:
        ca = CAEngine(self.input_bits.size, len(path) + 1, 3)
        ca.seed_from_bits(self.input_bits)
        for step in path:
            table, a, b = parse_step(step)
            if a is None:
                ca.set_rule_table(table); ca.step()
            else:
                ca.grid.append(apply_rule_window(ca.grid[-1], table, a, b))
                ca.current_row += 1
        return ca.grid

    def _path_analysis(self, path: List[str]) -> str:
        bits = self.input_bits.copy()
        cols = bits.size
        out = [f"start {int((bits == self.target_bits).sum())}/{cols}"]
        for i, step in enumerate(path[:40], 1):
            table, a, b = parse_step(step)
            bits = (apply_rule_window(bits, table, a, b) if a is not None
                    else apply_rule_fast(bits, table, cols))
            out.append(f"{i:02d} {step}: {int((bits == self.target_bits).sum())}/{cols}")
        if len(path) > 40:
            out.append(f"… (+{len(path) - 40} more steps)")
        return " | ".join(out)

    def run(self):
        self.logger.info("Worker started: %s", self.algo)
        try:
            label, grid, msg, table, n = self._run_algo()
        except Exception as e:
            self.logger.exception("Critical error in %s", self.algo)
            label, grid, msg, table, n = None, None, f"Critical Error: {e}", None, 3
        self.result_table, self.result_neighborhood = table, n
        self.result_final_bits = (grid[-1].copy() if grid is not None and len(grid) else None)
        if grid is not None and self.isInterruptionRequested():
            msg = "Interrupted — " + msg
        self.finished.emit(label, grid, msg)
        self.logger.info("Worker finished: %s", self.algo)

    def _run_algo(self):
        p, algo = self.params, self.algo
        # Equalize sizes so ANY text pair works (also fixes crashes when input
        # and target have different lengths in the other algorithms).
        self.input_bits, self.target_bits = align_bits_pair(self.input_bits, self.target_bits)
        cols = self.input_bits.size
        rules = get_basic_rule_tables()
        seed = p['seed']
        rep = _Reporter(self)

        if algo == "Universal Translator (Guaranteed)":
            path, _final = universal_translate(self.input_bits, self.target_bits, rep)
            if path:
                path = optimize_path(path, seed_bits=self.input_bits)
        elif algo == "Hybrid (Search + Fix-up)":
            path, _final = hybrid_translate(self.input_bits, self.target_bits,
                                            p.get('max_depth', 48), p.get('beam_width', 64),
                                            rules, rep)
            if path:
                path = optimize_path(path, seed_bits=self.input_bits)
        elif algo == "Forward Simulation":
            rn = int(p.get('fwd_rule', 30))
            ca = CAEngine(cols, p['rows'], 3)
            ca.set_rule_table(elementary_table(rn))
            ca.seed_from_bits(self.input_bits)
            while ca.current_row < p['rows']:
                if self.isInterruptionRequested():
                    break
                ca.step()
                if ca.current_row % 8 == 0:
                    rep.progress(int(ca.current_row / p['rows'] * 100),
                                 f"Forward rule {rn}: row {ca.current_row}/{p['rows']}")
                    self.emit_telemetry({'type': 'preview', 'grid': list(ca.grid)})
            fit = int((ca.grid[-1] == self.target_bits).sum())
            return (f"Rule {rn}", ca.grid,
                    f"Forward simulation of Rule {rn} — final match {fit}/{cols} (seed {seed}).",
                    elementary_table(rn), 3)
        elif algo == "Exhaustive (256 Rules)":
            rn, grid = exhaustive_search_256(self.input_bits, self.target_bits, p['rows'], rep)
            if grid is None:
                return None, None, "No result (interrupted).", None, 3
            fit = int((grid[-1] == self.target_bits).sum())
            kind = "Exact match" if fit == cols else "Best partial"
            return (f"Rule {rn}", grid, f"{kind}: Rule {rn} ({fit}/{cols}) — seed {seed}.",
                    elementary_table(rn), 3)
        elif algo == "A* (Backtracking)":
            path, _ = backtracking_search(self.input_bits, self.target_bits, p['max_depth'], rules, rep)
        elif algo == "Beam Search":
            path, _ = beam_search(self.input_bits, self.target_bits, p['max_depth'], p['beam_width'], rules, rep)
        elif algo == "MCTS":
            path, _exact = mcts_search(self.input_bits, self.target_bits, p['max_depth'],
                                       p['mcts_iters'], rep, random.Random(seed))
        elif algo == "Simulated Annealing":
            path, _ = simulated_annealing(self.input_bits, self.target_bits, p['max_depth'],
                                          p['sa_iters'], p['sa_temp'], rules, rep, random.Random(seed))
        elif algo == "Hill Climb (Greedy)":
            path, _ = hill_climb_search(self.input_bits, self.target_bits,
                                        p['max_depth'], rules, rep, random.Random(seed))
        elif algo == "Genetic Algorithm":
            rule, grid, fit = genetic_algorithm_search(self.input_bits, self.target_bits, p['rows'],
                                                       p['pop_size'], p['ga_gens'], p['mut_rate'],
                                                       rep, np.random.default_rng(seed))
            if grid is None:
                return None, None, "No result (interrupted).", None, 5
            lbl = "Evolved" if fit == cols else "Best evolved"
            return (lbl, grid,
                    f"{lbl} genome {''.join(map(str, rule.tolist()))} — fitness {fit}/{cols} (seed {seed}).",
                    rule, 5)
        else:
            return None, None, f"Unknown algorithm: {algo}", None, 3

        # Shared tail for path-based algorithms:
        if path is None:
            return None, None, f"{algo}: no sequence found within limits.", None, 3
        if not path:
            return ("Identity", [self.input_bits.copy()],
                    "Input already equals target.", rules["IDENTITY"], 3)
        self.result_path = list(path)
        grid = self._grid_from_path(path)
        exact = int((grid[-1] == self.target_bits).sum()) == cols
        pretty = (" -> ".join(path) if len(path) <= 12
                  else " -> ".join(path[:6]) + f" … (+{len(path) - 6} more steps)")
        msg = f"{algo} {'exact' if exact else 'best-effort'} path: {pretty}\n{self._path_analysis(path)}"
        first_table, _, _ = parse_step(path[0])
        return algo.split()[0], grid, msg + f" — seed {seed}.", first_table, 3

# =====================================================================
#  7. VISUALIZATION WIDGETS
# =====================================================================
class CAView2D(QWidget):
    animation_finished = Signal()
    mouse_pos_changed = Signal(str)
    reveal_changed = Signal(int)
    zoom_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.full_grid: List[np.ndarray] = []
        self.target_bits: Optional[np.ndarray] = None
        self.rows, self.cols = 0, 0
        self.reveal_row = 0
        self.anim_timer: Optional[QTimer] = None
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.is_panning = False
        self.pan_start_pos = QPoint()
        self.color_mode = "Neon"
        self.show_grid = True
        self.live_mode = False
        self._image: Optional[QImage] = None
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)

    # ---- data ----
    def set_full_grid(self, grid, target_bits=None, reveal_all=False):
        if self.anim_timer and self.anim_timer.isActive():
            self.anim_timer.stop()
        self.full_grid = [np.asarray(r, dtype=np.uint8) for r in grid]
        self.target_bits = target_bits
        self.rows = len(self.full_grid)
        self.cols = self.full_grid[0].size if self.rows > 0 else 0
        self.reveal_row = self.rows if reveal_all else 0
        self._rebuild_image()
        self.update()

    def set_color_mode(self, mode: str):
        self.color_mode = mode
        self._rebuild_image()
        self.update()

    def set_show_grid(self, show: bool):
        self.show_grid = show
        self.update()

    def set_live(self, on: bool):
        self.live_mode = on
        self.update()

    def start_reveal_animation(self, speed_ms: int):
        self.reveal_row = 0
        if self.anim_timer:
            self.anim_timer.stop()
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._reveal_step)
        self.anim_timer.start(max(1, speed_ms))

    def set_reveal_row(self, r: int):
        if self.anim_timer:
            self.anim_timer.stop()
        self.reveal_row = int(np.clip(r, 0, self.rows))
        self.update()

    def _reveal_step(self):
        if self.reveal_row < self.rows:
            self.reveal_row += 1
            self.reveal_changed.emit(self.reveal_row)
            self.update()
        else:
            if self.anim_timer:
                self.anim_timer.stop()
            self.animation_finished.emit()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.zoom_changed.emit(1.0)
        self.update()

    # ---- rendering (QImage based) ----
    def _row_colors(self) -> np.ndarray:
        n = max(1, self.rows)
        t = np.linspace(0.0, 1.0, n)[:, None]
        if self.color_mode == "Matrix":
            c = np.hstack([np.zeros((n, 1)), 50 + 205 * t, np.zeros((n, 1))])
        elif self.color_mode == "Amber":
            c = np.hstack([200 + 55 * t, 100 + 100 * t, np.zeros((n, 1))])
        elif self.color_mode == "Heatmap":
            u = t * 2
            c = np.hstack([np.clip(u, 0, 1), np.clip(u - 1, 0, 1), np.clip(1 - u, 0, 1)]) * 255
        else:  # Neon
            u, v = t * 2, np.clip((t - 0.5) * 2, 0, 1)
            first = np.hstack([170 * u, 170 * (1 - u), np.full((n, 1), 255.0)])
            second = np.hstack([170 * (1 - v), 255 * v, 255 - 153 * v])
            c = np.where(t < 0.5, first, second)
        return np.clip(c, 0, 255).astype(np.uint8)

    def _rebuild_image(self):
        if not self.full_grid or self.rows == 0 or self.cols == 0:
            self._image = None
            return
        grid = np.stack(self.full_grid)
        buf = (grid[:, :, None] * self._row_colors()[:, None, :]).astype(np.uint8)
        img = QImage(buf.tobytes(), self.cols, self.rows,
                     self.cols * 3, QImage.Format.Format_RGB888)
        self._image = img.copy()

    def _grid_geometry(self) -> Tuple[float, float, float]:
        cell = min(self.width() / self.cols, self.height() / self.rows)
        return cell, (self.width() - cell * self.cols) / 2, (self.height() - cell * self.rows) / 2

    def _cell_at(self, pos: QPointF) -> Tuple[int, int]:
        if not self.full_grid or not self.cols or not self.rows:
            return -1, -1
        cx, cy = self.width() / 2, self.height() / 2
        x = (pos.x() - self.pan_offset.x() - cx) / self.zoom_factor + cx
        y = (pos.y() - self.pan_offset.y() - cy) / self.zoom_factor + cy
        cell, ox, oy = self._grid_geometry()
        if cell <= 0:
            return -1, -1
        return int((y - oy) / cell), int((x - ox) / cell)

    # ---- interaction ----
    def wheelEvent(self, event: QWheelEvent):
        factor = 1.1 ** (event.angleDelta().y() / 120)
        old = self.zoom_factor
        self.zoom_factor = float(np.clip(old * factor, 0.1, 10.0))
        m, c = event.position(), QPointF(self.width() / 2, self.height() / 2)
        r = self.zoom_factor / old
        self.pan_offset = (m - c) * (1.0 - r) + self.pan_offset * r
        self.zoom_changed.emit(self.zoom_factor)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_panning = True
            self.pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_panning:
            delta = event.position().toPoint() - self.pan_start_pos
            self.pan_offset += QPointF(delta)
            self.pan_start_pos = event.position().toPoint()
            self.update()
        r, c = self._cell_at(event.position())
        if 0 <= r < min(self.reveal_row, self.rows) and 0 <= c < self.cols:
            val = int(self.full_grid[r][c])
            k = c // UNICODE_BITS
            seg = self.full_grid[r][k * UNICODE_BITS:(k + 1) * UNICODE_BITS]
            ch = bits_to_text(seg) if seg.size == UNICODE_BITS else "?"
            disp = ch if ch.isprintable() and not ch.isspace() else "·"
            self.mouse_pos_changed.emit(f"R{r} C{c} = {val}   char: '{disp}'")
        else:
            self.mouse_pos_changed.emit("")

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)

    # ---- paint ----
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(Theme.BG))
        if not self.full_grid or self.rows == 0 or self.cols == 0:
            painter.setPen(QColor("#444")); painter.setFont(QFont("Consolas", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Space-Time Diagram will appear here…")
            self._draw_live_badge(painter)
            return

        cell, offset_x, offset_y = self._grid_geometry()
        painter.save()
        cx, cy = self.width() / 2, self.height() / 2
        painter.translate(cx + self.pan_offset.x(), cy + self.pan_offset.y())
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(-cx, -cy)

        if self._image is not None and self.reveal_row > 0:
            painter.drawImage(QRectF(offset_x, offset_y, self.cols * cell, self.reveal_row * cell),
                              self._image, QRectF(0, 0, self.cols, self.reveal_row))

        if self.show_grid and cell * self.zoom_factor > 5:
            painter.setPen(QPen(QColor("#151515"), 0))
            for r in range(self.rows + 1):
                y = offset_y + r * cell
                painter.drawLine(QPointF(offset_x, y), QPointF(offset_x + self.cols * cell, y))
            for c in range(self.cols + 1):
                x = offset_x + c * cell
                painter.drawLine(QPointF(x, offset_y), QPointF(x, offset_y + self.rows * cell))

        match_count = 0
        if (self.target_bits is not None and self.reveal_row >= self.rows and self.rows > 1):
            size = max(2, int(cell))
            for r_off in range(min(3, self.rows)):
                r = self.rows - 1 - r_off
                last = self.full_grid[r]
                opacity = 120 - r_off * 40
                for c in range(min(self.cols, self.target_bits.size)):
                    is_match = bool(last[c] == self.target_bits[c])
                    if r_off == 0 and is_match:
                        match_count += 1
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 255, 102, opacity) if is_match
                                     else QColor(255, 50, 50, opacity))
                    painter.drawRect(int(offset_x + c * cell), int(offset_y + r * cell), size, size)
        painter.restore()

        if match_count or (self.target_bits is not None and self.reveal_row >= self.rows and self.rows > 1):
            pct = (match_count / self.cols) * 100 if self.cols else 0
            painter.setPen(QColor("#fff")); painter.setFont(QFont("Consolas", 12, QFont.Bold))
            painter.drawText(10, self.height() - 10,
                             f"Final Row Match: {pct:.1f}% ({match_count}/{self.cols})")

        if 0 < self.reveal_row <= self.rows and not self.live_mode:
            scene_y = offset_y + self.reveal_row * cell
            cy = self.height() / 2
            y_dev = self.zoom_factor * (scene_y - cy) + cy + self.pan_offset.y()
            sg = QLinearGradient(0, y_dev - 15, 0, y_dev + 15)
            sg.setColorAt(0, QColor(0, 255, 102, 0))
            sg.setColorAt(0.5, QColor(0, 255, 102, 120))
            sg.setColorAt(1, QColor(0, 255, 102, 0))
            painter.fillRect(0, int(y_dev - 15), self.width(), 30, sg)

        vignette = QRadialGradient(self.width() / 2, self.height() / 2,
                                   max(self.width(), self.height()) / 1.2)
        vignette.setColorAt(0, QColor(0, 0, 0, 0))
        vignette.setColorAt(1, QColor(0, 0, 0, 150))
        painter.fillRect(self.rect(), vignette)
        self._draw_live_badge(painter)

    def _draw_live_badge(self, painter: QPainter):
        if not self.live_mode:
            return
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(220, 40, 40, 230))
        painter.drawEllipse(self.width() - 88, 10, 10, 10)
        painter.setPen(QColor("#ff6b6b"))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(self.width() - 72, 20, "LIVE")


class MatchStripWidget(QWidget):
    """Live 'state vs target' comparison strip — green bit = match, red = mismatch."""
    def __init__(self):
        super().__init__()
        self.bits: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.setMinimumHeight(88)

    def set_bits(self, bits, target):
        self.bits = None if bits is None else np.asarray(bits, np.uint8)
        self.target = None if target is None else np.asarray(target, np.uint8)
        self.update()

    def reset(self):
        self.bits = None
        self.update()

    @staticmethod
    def _disp(txt: str) -> str:
        return "".join(ch if (ch.isprintable() and not ch.isspace()) else "·" for ch in txt)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        p.setPen(QPen(QColor("#333"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.setFont(QFont("Consolas", 9, QFont.Bold))
        p.setPen(QColor(Theme.ACCENT))
        p.drawText(8, 14, "state vs target (live)")
        if self.bits is None or self.target is None or self.bits.size == 0 or self.target.size == 0:
            p.setPen(QColor("#555"))
            p.drawText(self.rect().adjusted(0, 18, 0, 0), Qt.AlignCenter, "waiting for data…")
            return
        n = min(self.bits.size, self.target.size)
        matches = int(np.count_nonzero(self.bits[:n] == self.target[:n]))
        pct = 100.0 * matches / max(1, n)
        x0, x1 = 10, self.width() - 10
        cw = (x1 - x0) / max(1, n)
        cy, ch = 20, 12
        for i in range(n):
            ok = bool(self.bits[i] == self.target[i])
            p.fillRect(QRectF(x0 + i * cw, cy, max(1.0, cw - 0.5), ch),
                       QColor(0, 230, 110) if ok else QColor(255, 70, 70))
        p.setFont(QFont("Consolas", 10))
        p.setPen(QColor(Theme.ACCENT2))
        p.drawText(10, cy + ch + 16, "state:  " + self._disp(bits_to_text(self.bits)))
        p.setPen(QColor(Theme.WARN))
        p.drawText(10, cy + ch + 32, "target: " + self._disp(bits_to_text(self.target)))
        p.setFont(QFont("Consolas", 11, QFont.Bold))
        p.setPen(QColor(Theme.ACCENT) if pct == 100 else QColor("#eee"))
        p.drawText(self.width() - 84, cy + ch + 27, f"{pct:.1f}%")


class MultiSeriesChart(QWidget):
    """Multi-series line chart with legend. 'temp' gets its own scale (dashed)."""
    def __init__(self):
        super().__init__()
        self.data: Dict[str, deque] = {name: deque(maxlen=300) for name in SERIES_STYLE}
        self.title = "Search dynamics"
        self.setMinimumHeight(170)

    def add_point(self, name: str, val):
        if name in self.data:
            self.data[name].append(float(val))
            self.update()

    def reset(self):
        for d in self.data.values():
            d.clear()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        p.setPen(QPen(QColor("#1d1d1d"), 1))
        for i in range(1, 5):
            y = int(self.height() * i / 5)
            p.drawLine(0, y, self.width(), y)
        main = [(n, d) for n, d in self.data.items() if d and n != "temp"]
        has_temp = bool(self.data["temp"])
        if not main and not has_temp:
            p.setPen(QColor("#444")); p.setFont(QFont("Consolas", 9))
            p.drawText(self.rect(), Qt.AlignCenter, self.title + " — waiting…")
            return
        top = 26  # room for the legend
        if main:
            allv = [v for _, d in main for v in d]
            mx, mn = max(allv), min(allv)
            if mx == mn:
                mx = mn + 1
            def to_y(v):
                return self.height() - ((v - mn) / (mx - mn)) * (self.height() - top - 15) - 15
            for name, d in main:
                color, wdt, dash = SERIES_STYLE[name]
                pts = list(d)
                path = QPainterPath()
                step = self.width() / max(1, len(pts) - 1)
                for i, v in enumerate(pts):
                    if i == 0:
                        path.moveTo(0, to_y(v))
                    else:
                        path.lineTo(i * step, to_y(v))
                pen = QPen(QColor(color), wdt)
                if dash:
                    pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.drawPath(path)
            p.setPen(QColor("#666")); p.setFont(QFont("Consolas", 7))
            p.drawText(4, self.height() - 4, f"{mn:.3g}")
            p.drawText(4, 14, f"{mx:.3g}")
        if has_temp:
            pts = list(self.data["temp"])
            lo, hi = min(pts), max(pts)
            if hi == lo:
                hi = lo + 1
            path = QPainterPath()
            step = self.width() / max(1, len(pts) - 1)
            for i, v in enumerate(pts):
                y = self.height() - ((v - lo) / (hi - lo)) * (self.height() - top - 15) - 15
                if i == 0:
                    path.moveTo(0, y)
                else:
                    path.lineTo(i * step, y)
            pen = QPen(QColor(SERIES_STYLE["temp"][0]), 1)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawPath(path)
        # legend
        p.setFont(QFont("Consolas", 8))
        x = self.width() - 12
        for name in reversed(list(SERIES_STYLE)):
            if not self.data[name]:
                continue
            color, _, dash = SERIES_STYLE[name]
            fm_w = p.fontMetrics().horizontalAdvance(name) + 16
            x -= fm_w
            pen = QPen(QColor(color), 2)
            if dash:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(x + 2, 12, x + 12, 12)
            p.setPen(QColor("#aaa"))
            p.drawText(x + 15, 16, name)


class HeatmapWidget(QWidget):
    """256 elementary-rule fitness map (16x16) for the Exhaustive search."""
    def __init__(self):
        super().__init__()
        self.fits: Optional[np.ndarray] = None
        self.current = -1
        self.best = -1
        self.cols = 1
        self.setMinimumHeight(150)

    def set_data(self, d: dict):
        self.fits = d.get('data')
        self.current = int(d.get('current', -1))
        self.best = int(d.get('best', -1))
        self.cols = max(1, int(d.get('cols', 1)))
        self.update()

    def reset(self):
        self.fits = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        p.setFont(QFont("Consolas", 9))
        p.setPen(QColor(Theme.DIM))
        p.drawText(8, 14, "exhaustive 256-rule fitness map")
        if self.fits is None or len(self.fits) != 256:
            p.setPen(QColor("#555"))
            p.drawText(self.rect().adjusted(0, 18, 0, 0), Qt.AlignCenter, "waiting for data…")
            return
        x0, y0 = 10, 22
        w = (self.width() - 20) / 16.0
        h = min(18.0, (self.height() - 46) / 16.0)
        p.setPen(Qt.NoPen)
        for i in range(256):
            r, c = divmod(i, 16)
            v = float(self.fits[i]) / self.cols
            p.setBrush(QColor(int(20), int(40 + 200 * v), int(30 + 40 * v)))
            p.drawRect(QRectF(x0 + c * w, y0 + r * h, max(1, w - 1), max(1, h - 1)))
        for idx, color in ((self.best, Theme.ACCENT), (self.current, Theme.ACCENT2)):
            if 0 <= idx < 256:
                r, c = divmod(idx, 16)
                p.setPen(QPen(QColor(color), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x0 + c * w, y0 + r * h, max(1, w - 1), max(1, h - 1)))
        best_txt = (f"best rule {self.best}: {int(self.fits[self.best])}/{self.cols}"
                    if self.best >= 0 else "best: —")
        p.setPen(QColor(Theme.TEXT)); p.setFont(QFont("Consolas", 9, QFont.Bold))
        p.drawText(8, self.height() - 6, f"{best_txt}    current {self.current}")


class TreeViewWidget(QWidget):
    """MCTS tree snapshot: radius ∝ visits, color = avg value (red→green)."""
    def __init__(self):
        super().__init__()
        self.nodes: List[tuple] = []
        self.setMinimumHeight(150)

    def set_nodes(self, d: dict):
        self.nodes = d.get('nodes') or []
        self.update()

    def reset(self):
        self.nodes = []
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        p.setFont(QFont("Consolas", 9)); p.setPen(QColor(Theme.DIM))
        p.drawText(8, 14, "MCTS search tree (radius ∝ visits, green = high value)")
        if not self.nodes:
            p.setPen(QColor("#555"))
            p.drawText(self.rect().adjusted(0, 18, 0, 0), Qt.AlignCenter, "waiting for data…")
            return
        children: Dict[int, List[int]] = {}
        for nid, pid, _act, _vis, _val in self.nodes:
            children.setdefault(pid, []).append(nid)
        depth: Dict[int, int] = {}
        xpos: Dict[int, float] = {}
        leaf_counter = [0]

        def dfs(nid, d):
            depth[nid] = d
            kids = children.get(nid, [])
            if not kids:
                xpos[nid] = leaf_counter[0]; leaf_counter[0] += 1
            else:
                for k in kids:
                    dfs(k, d + 1)
                xs = [xpos[k] for k in kids]
                xpos[nid] = sum(xs) / len(xs)

        roots = [n[0] for n in self.nodes if n[1] < 0]
        for r in roots:
            dfs(r, 0)
        maxd = (max(depth.values()) if depth else 1) or 1
        maxleaf = max(1, leaf_counter[0] - 1)
        maxvis = max(1, max(n[3] for n in self.nodes))
        W, H = self.width() - 20, self.height() - 44

        def px(x): return 10 + (x / maxleaf) * W
        def py(d): return 28 + (d / maxd) * max(10, H - 20)

        p.setPen(QPen(QColor("#2a4a3a"), 1))
        for nid, pid, _a, _v, _val in self.nodes:
            if pid >= 0 and nid in xpos and pid in xpos:
                p.drawLine(QPointF(px(xpos[pid]), py(depth[pid])),
                           QPointF(px(xpos[nid]), py(depth[nid])))
        p.setPen(Qt.NoPen)
        for nid, _pid, _a, vis, val in self.nodes:
            r = 2 + 6 * math.sqrt(vis / maxvis)
            p.setBrush(QColor(int(255 * (1 - val)), int(120 + 135 * val), 60))
            p.drawEllipse(QPointF(px(xpos[nid]), py(depth[nid])), r, r)


class PopulationWidget(QWidget):
    """GA population: fitness histogram + best-genome bit strip + stats line."""
    def __init__(self):
        super().__init__()
        self.fits: Optional[np.ndarray] = None
        self.genome: Optional[np.ndarray] = None
        self.gen = 0; self.mut = 0.0; self.div = 0.0; self.cols = 1
        self.setMinimumHeight(150)

    def set_data(self, d: dict):
        self.fits = d.get('fits')
        self.genome = d.get('genome')
        self.gen = d.get('gen', 0)
        self.mut = d.get('mut', 0)
        self.div = d.get('div', 0)
        self.cols = max(1, int(d.get('cols', 1)))
        self.update()

    def reset(self):
        self.fits = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        p.setFont(QFont("Consolas", 9)); p.setPen(QColor(Theme.DIM))
        p.drawText(8, 14, "GA population")
        if self.fits is None:
            p.setPen(QColor("#555"))
            p.drawText(self.rect().adjusted(0, 18, 0, 0), Qt.AlignCenter, "waiting for data…")
            return
        fits = np.asarray(self.fits)
        cols = max(1, self.cols)
        x0, y0 = 10, 24
        w = self.width() - 20
        h = max(30, self.height() - 80)
        hist = np.bincount(fits, minlength=cols + 1)[:cols + 1]
        mx = max(1, int(hist.max()))
        bw = w / max(1, (cols + 1))
        p.setPen(Qt.NoPen)
        for f_i, cnt in enumerate(hist):
            bh = (cnt / mx) * h
            p.setBrush(QColor(0, 170, 255) if f_i < cols else QColor(0, 255, 102))
            p.drawRect(QRectF(x0 + f_i * bw, y0 + h - bh, max(1.0, bw - 1), bh))
        gy = y0 + h + 8
        if self.genome is not None and len(self.genome) > 0:
            g = np.asarray(self.genome, np.uint8)
            gw = (self.width() - 20) / len(g)
            p.setPen(Qt.NoPen)
            for i, bit in enumerate(g):
                p.setBrush(QColor(0, 255, 102) if bit else QColor("#202020"))
                p.drawRect(QRectF(10 + i * gw, gy, max(1.0, gw - 1), 12))
        p.setPen(QColor(Theme.TEXT)); p.setFont(QFont("Consolas", 9, QFont.Bold))
        best = int(fits.max()) if fits.size else 0
        p.drawText(8, self.height() - 6,
                   f"gen {self.gen} · best {best}/{cols} · diversity {self.div:.2f} · mutation {self.mut:.3f}")


class PreviewWidget(QWidget):
    """Scaled live preview of the full CA grid."""
    def __init__(self):
        super().__init__()
        self._img: Optional[QImage] = None
        self.setMinimumHeight(120)

    def set_data(self, d: dict):
        grid = d.get('grid')
        self._img = None
        if grid:
            try:
                g = np.stack([np.asarray(r, np.uint8) for r in grid])
                R, C = g.shape
                if R > 0 and C > 0:
                    rgb = (g[:, :, None] * _row_colors_static(R)[:, None, :]).astype(np.uint8)
                    self._img = QImage(rgb.tobytes(), C, R, C * 3,
                                       QImage.Format.Format_RGB888).copy()
            except Exception:
                self._img = None
        self.update()

    def reset(self):
        self._img = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        p.setPen(QColor(Theme.DIM)); p.setFont(QFont("Consolas", 9))
        p.drawText(8, 14, "live grid preview")
        if self._img is None:
            p.setPen(QColor("#555"))
            p.drawText(self.rect().adjusted(0, 18, 0, 0), Qt.AlignCenter, "waiting for data…")
            return
        area = self.rect().adjusted(10, 22, -10, -8)
        s = min(area.width() / self._img.width(), area.height() / self._img.height())
        w, h = self._img.width() * s, self._img.height() * s
        x = area.x() + (area.width() - w) / 2
        y = area.y() + (area.height() - h) / 2
        p.drawImage(QRectF(x, y, w, h), self._img)

# =====================================================================
#  7b. EXPORT HELPERS
# =====================================================================
def _row_colors_static(n: int, mode: str = "Neon") -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(1, n))[:, None]
    if mode == "Matrix":
        c = np.hstack([np.zeros((n, 1)), 50 + 205 * t, np.zeros((n, 1))])
    elif mode == "Amber":
        c = np.hstack([200 + 55 * t, 100 + 100 * t, np.zeros((n, 1))])
    elif mode == "Heatmap":
        u = t * 2
        c = np.hstack([np.clip(u, 0, 1), np.clip(u - 1, 0, 1), np.clip(1 - u, 0, 1)]) * 255
    else:
        u, v = t * 2, np.clip((t - 0.5) * 2, 0, 1)
        first = np.hstack([170 * u, 170 * (1 - u), np.full((n, 1), 255.0)])
        second = np.hstack([170 * (1 - v), 255 * v, 255 - 153 * v])
        c = np.where(t < 0.5, first, second)
    return np.clip(c, 0, 255).astype(np.uint8)


def render_grid_png(grid, out_path: str, mode: str = "Neon", cell: int = 6) -> str:
    """Save any grid (list of bit rows) as a PNG without needing the widget."""
    rows = [np.asarray(r, np.uint8) for r in grid]
    g = np.stack(rows)
    R, C = g.shape
    rgb = (g[:, :, None] * _row_colors_static(R, mode)[:, None, :]).astype(np.uint8)
    big = np.repeat(np.repeat(rgb, max(1, cell), axis=0), max(1, cell), axis=1)
    W = C * max(1, cell)
    QImage(big.tobytes(), W, R * max(1, cell), W * 3,
           QImage.Format.Format_RGB888).save(out_path, "PNG")
    return out_path


def export_path_json(path: List[str], out_path: str, meta: Optional[dict] = None) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "steps": list(path), "meta": meta or {}}, f, ensure_ascii=False, indent=2)


def copy_to_clipboard(text: str) -> None:
    QApplication.clipboard().setText(text)

# =====================================================================
#  8. RESULT INSPECTOR WIDGETS (timeline, stats, toolbar)
# =====================================================================
class PathTimelineWidget(QWidget):
    """Clickable chips, one per path step. Click → jump the reveal to that step."""
    stepSelected = Signal(int)
    CHIP_COLORS = {"SET0": "#ff5555", "SET1": "#00ff66", "INV": "#ff55ff"}

    def __init__(self):
        super().__init__()
        self.steps: List[str] = []
        self.selected = -1
        self._scroll = 0
        self._widths: List[int] = []
        self.setMinimumHeight(50)

    def set_path(self, path: List[str]):
        self.steps = list(path)
        self.selected = -1
        self._scroll = 0
        fm = QFontMetrics(QFont("Consolas", 9))
        self._widths = [fm.horizontalAdvance(s) + 14 for s in self.steps]
        self.update()

    def wheelEvent(self, e):
        total = sum(self._widths) + 6 * len(self._widths)
        self._scroll = int(np.clip(self._scroll - e.angleDelta().y() // 2,
                                   0, max(0, total - self.width())))
        self.update()

    def mousePressEvent(self, e):
        x = 6 - self._scroll
        for i, w in enumerate(self._widths):
            if x <= e.position().x() < x + w:
                self.selected = i
                self.update()
                self.stepSelected.emit(i)
                return
            x += w + 6

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        if not self.steps:
            p.setPen(QColor("#444"))
            p.drawText(self.rect(), Qt.AlignCenter, "no path steps")
            return
        p.setFont(QFont("Consolas", 9))
        x = 6 - self._scroll
        for i, (s, w) in enumerate(zip(self.steps, self._widths)):
            if x + w < 0 or x > self.width():
                x += w + 6
                continue
            hexc = self.CHIP_COLORS.get(s.split('@', 1)[0], "#00aaff")
            base = QColor(*(int(hexc[j:j + 2], 16) for j in (1, 3, 5)))
            sel = (i == self.selected)
            p.setPen(QPen(QColor("#fff" if sel else base), 2 if sel else 1))
            p.setBrush(QColor(255, 255, 255, 35) if sel else QColor("#111"))
            p.drawRoundedRect(QRectF(x, 6, w, 22), 5, 5)
            p.setPen(base)
            p.drawText(QRectF(x, 6, w, 22), Qt.AlignCenter, s)
            p.setPen(QColor("#666"))
            p.drawText(QRectF(x, 30, w, 14), Qt.AlignCenter, str(i + 1))
            x += w + 6


class RunStatsPanel(QFrame):
    """Compact result summary with a verified badge."""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        lay = QFormLayout(self)
        lay.setLabelAlignment(Qt.AlignRight)
        self._lab = {k: QLabel("—") for k in
                     ("algo", "steps", "chars", "match", "time", "verified")}
        for k in ("algo", "steps", "chars", "match", "time", "verified"):
            lay.addRow(k.capitalize() + ":", self._lab[k])

    def set_result(self, res: Optional[TranslationResult]):
        if res is None:
            for l in self._lab.values():
                l.setText("—")
            return
        self._lab["algo"].setText(res.algo)
        self._lab["steps"].setText(str(res.steps))
        self._lab["chars"].setText(f"{len(res.source)} → {len(res.target)}")
        self._lab["match"].setText("100%" if res.verified else "partial")
        self._lab["time"].setText(f"{res.elapsed_ms:.1f} ms")
        v = self._lab["verified"]
        if res.verified:
            v.setText("✓ VERIFIED"); v.setStyleSheet("color:#00ff66;font-weight:bold;")
        else:
            v.setText("✗ UNVERIFIED"); v.setStyleSheet("color:#ff5555;font-weight:bold;")


class TranslationInspector(QObject):
    """Glue: feed a finished run; it verifies, fills timeline + stats, exposes exports."""
    def __init__(self, timeline: PathTimelineWidget, stats: RunStatsPanel, parent=None):
        super().__init__(parent)
        self.timeline, self.stats = timeline, stats
        self.last: Optional[TranslationResult] = None

    def load_bits(self, input_bits, target_bits, path: List[str],
                  algo: str = "", elapsed_ms: float = 0.0,
                  final_bits: Optional[np.ndarray] = None) -> TranslationResult:
        a = np.asarray(input_bits, np.uint8)
        b = np.asarray(target_bits, np.uint8)
        if path:
            ok, final = verify_path(a, b, path)
        elif final_bits is not None:
            final = np.asarray(final_bits, np.uint8)
            ok = bool(np.array_equal(final, b))
        else:
            ok, final = bool(np.array_equal(a, b)), a.copy()
        res = TranslationResult(bits_to_text(a), bits_to_text(b), list(path), ok,
                                len(path), float(elapsed_ms), algo or "—", bits_to_text(final))
        self.last = res
        self.timeline.set_path(res.path)
        self.stats.set_result(res)
        return res


def build_result_toolbar(parent, inspector: TranslationInspector, ca_view=None,
                         telemetry_getter=None, session_saver=None) -> QWidget:
    bar = QWidget(parent)
    h = QHBoxLayout(bar); h.setContentsMargins(2, 2, 2, 2)

    def add(txt, fn):
        btn = QPushButton(txt); btn.clicked.connect(fn); h.addWidget(btn)

    has = lambda: inspector.last is not None

    def copy_text():
        if has(): copy_to_clipboard(inspector.last.final_text)
    def copy_path():
        if has(): copy_to_clipboard(" -> ".join(inspector.last.path))

    def save_png():
        img = getattr(ca_view, "_image", None) if ca_view else None
        if img is None:
            QMessageBox.information(bar, "Save PNG", "Nothing rendered yet."); return
        fn, _ = QFileDialog.getSaveFileName(bar, "Save space-time diagram",
                                            "ca_diagram.png", "PNG (*.png)")
        if fn:
            img.save(fn, "PNG")

    def save_json():
        if not has():
            return
        fn, _ = QFileDialog.getSaveFileName(bar, "Export path", "ca_path.json", "JSON (*.json)")
        if fn:
            export_path_json(inspector.last.path, fn,
                             meta={"algo": inspector.last.algo,
                                   "verified": inspector.last.verified})

    def save_tl():
        log = telemetry_getter() if telemetry_getter else []
        if not log:
            QMessageBox.information(bar, "Export telemetry",
                                    "No telemetry recorded.\nEnable 'Record telemetry' before the run.")
            return
        fn, _ = QFileDialog.getSaveFileName(bar, "Export telemetry",
                                            "ca_telemetry.jsonl", "JSONL (*.jsonl)")
        if fn:
            export_telemetry_jsonl(log, fn)

    def save_sess():
        if not has() or session_saver is None:
            return
        fn, _ = QFileDialog.getSaveFileName(bar, "Save session", "ca_session.json", "JSON (*.json)")
        if fn:
            session_saver(fn)

    add("Copy text", copy_text)
    add("Copy path", copy_path)
    add("Save PNG", save_png)
    add("Export path JSON", save_json)
    add("Export telemetry", save_tl)
    add("Save session", save_sess)
    h.addStretch(1)
    return bar

# =====================================================================
#  9. LAYOUT BUILDING BLOCKS
# =====================================================================
class HeaderBar(QFrame):
    """App title + running-state chip."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("header")
        h = QHBoxLayout(self); h.setContentsMargins(14, 9, 14, 9)
        left = QVBoxLayout(); left.setSpacing(1)
        title = QLabel("CA TEXT TRANSLATOR"); title.setObjectName("headerTitle")
        sub = QLabel("universal bit-space translation · guaranteed exact mode")
        sub.setObjectName("headerSub")
        left.addWidget(title); left.addWidget(sub)
        h.addLayout(left); h.addStretch(1)
        self.chip = QLabel("IDLE"); self.chip.setObjectName("chip")
        h.addWidget(self.chip)

    def set_running(self, running: bool):
        self.chip.setText("● RUNNING" if running else "IDLE")
        self.chip.setStyleSheet(f"color:{Theme.DANGER};" if running
                                else f"color:{Theme.DIM};")


class TextCard(QGroupBox):
    """Editor with live chars/bits counter."""
    def __init__(self, title: str, placeholder: str):
        super().__init__(title)
        v = QVBoxLayout(self); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(4)
        self.edit = QTextEdit(); self.edit.setPlaceholderText(placeholder)
        self.edit.setMinimumHeight(56)
        v.addWidget(self.edit)
        row = QHBoxLayout()
        self.counter = QLabel("0 chars · 0 bits"); self.counter.setObjectName("counter")
        btn = QPushButton("Clear"); btn.setFixedWidth(64)
        btn.clicked.connect(self.edit.clear)
        row.addWidget(self.counter); row.addStretch(1); row.addWidget(btn)
        v.addLayout(row)
        self.edit.textChanged.connect(self._update)

    def _update(self):
        n = len(self.text())
        self.counter.setText(f"{n} chars · {n * UNICODE_BITS} bits")

    def text(self) -> str: return self.edit.toPlainText()
    def set_text(self, t: str): self.edit.setPlainText(t)


class CandidatesWidget(QListWidget):
    """Telemetry 'candidates' items as colored rows."""
    def set_items(self, items):
        self.clear()
        for label, fit, cols in (items or [])[:12]:
            it = QListWidgetItem(f"{label}   [{fit}/{cols}]")
            it.setForeground(QColor(Theme.ACCENT if fit >= cols else Theme.TEXT))
            self.addItem(it)


class BarsWidget(QWidget):
    """Compact horizontal bar chart for 'bars' telemetry."""
    PALETTE = ["#00ff66", "#00aaff", "#ffcc00", "#ff55ff", "#ff5555", "#7dd3fc"]

    def __init__(self):
        super().__init__()
        self.data: Dict[str, float] = {}
        self.setMinimumHeight(78)

    def set_data(self, d: dict):
        self.data = {k: float(v) for k, v in (d or {}).items() if v}
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(Theme.PANEL))
        if not self.data:
            p.setPen(QColor("#444")); p.setFont(QFont("Consolas", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "action distribution — waiting…")
            return
        mx = max(self.data.values())
        bh = max(10, min(16, (self.height() - 8) // max(1, len(self.data))))
        y = 6
        p.setFont(QFont("Consolas", 9))
        for i, (k, v) in enumerate(sorted(self.data.items(), key=lambda kv: -kv[1])):
            w = (self.width() - 96) * (v / mx)
            p.setPen(QColor(Theme.DIM)); p.drawText(4, y + bh - 3, k[:10])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self.PALETTE[i % len(self.PALETTE)]))
            p.drawRoundedRect(QRectF(80, y + 2, max(2, w), bh - 5), 2, 2)
            p.setPen(QColor(Theme.TEXT))
            p.drawText(QPointF(88 + w, y + bh - 3), f"{int(v)}")
            y += bh


class TelemetryRouter(QObject):
    """Dispatch telemetry dicts by 'type' to registered slots; ignores extras."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.map: Dict[str, object] = {}

    def register(self, kind: str, fn): self.map[kind] = fn

    def handle(self, data: dict):
        fn = self.map.get(data.get("type"))
        if fn:
            try:
                fn(data)
            except Exception:
                logger.debug("telemetry slot failed", exc_info=True)


class ParamPages(QStackedWidget):
    """One page per ALGO_PAGE index. rows/seed live outside (shared)."""
    def __init__(self, parent=None):
        super().__init__(parent)

        def hint(s):
            lab = QLabel(s); lab.setObjectName("counter"); return lab

        def page(*rows):
            w = QWidget(); f = QFormLayout(w)
            f.setLabelAlignment(Qt.AlignRight); f.setContentsMargins(2, 2, 2, 2)
            for label, widget, tip in rows:
                widget.setToolTip(tip); f.addRow(label, widget)
            return w

        self.fwd_rule = QSpinBox(); self.fwd_rule.setRange(0, 255); self.fwd_rule.setValue(30)
        self.max_depth = self._spin(4, 512, 32)
        self.beam_width = self._spin(4, 4096, 64)
        self.mcts_iters = self._spin(100, 500000, 4000)
        self.sa_iters = self._spin(100, 500000, 4000)
        self.sa_temp = QDoubleSpinBox(); self.sa_temp.setRange(0.1, 100.0)
        self.sa_temp.setValue(8.0); self.sa_temp.setSingleStep(0.5)
        self.pop_size = self._spin(8, 2000, 100)
        self.ga_gens = self._spin(1, 2000, 60)
        self.mut_rate = QDoubleSpinBox(); self.mut_rate.setRange(0.0, 1.0)
        self.mut_rate.setValue(0.02); self.mut_rate.setSingleStep(0.01)
        self.mut_rate.setDecimals(3)

        self.addWidget(page(("elementary rule", self.fwd_rule, "rule 0-255 to stream"),
                            ("—", hint("streams the seed row through this rule."), "")))
        self.addWidget(page(("—", hint("tests all 256 elementary rules end-to-end."), "")))
        self.addWidget(page(("max depth", self.max_depth, "tree depth limit"),
                            ("iterations", self.mcts_iters, "MCTS rollouts")))
        self.addWidget(page(("population", self.pop_size, "GA population size"),
                            ("generations", self.ga_gens, "GA generations"),
                            ("mutation", self.mut_rate, "per-bit mutation rate")))
        self.addWidget(page(("max depth / seq len", self.max_depth, "rule chain length"),
                            ("beam width", self.beam_width, "beam search width"),
                            ("SA iterations", self.sa_iters, "simulated annealing steps"),
                            ("SA start temp", self.sa_temp, "initial temperature")))

    @staticmethod
    def _spin(lo, hi, val) -> QSpinBox:
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); return s

    def params(self) -> dict:
        return {"fwd_rule": self.fwd_rule.value(),
                "max_depth": self.max_depth.value(),
                "beam_width": self.beam_width.value(),
                "mcts_iters": self.mcts_iters.value(),
                "sa_iters": self.sa_iters.value(),
                "sa_temp": self.sa_temp.value(),
                "pop_size": self.pop_size.value(),
                "ga_gens": self.ga_gens.value(),
                "mut_rate": self.mut_rate.value()}

# =====================================================================
#  10. MAIN WINDOW
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CA Text Translator")
        self.worker = None
        self.thread = None
        self.last_telemetry_log: List[Tuple[float, dict]] = []
        self.router = TelemetryRouter(self)

        self.resize(1280, 820)
        self._build_ui()
        self._wire_telemetry()
        self._wire_shortcuts()
        self._restore_settings()
        self.statusBar().showMessage("Ready.")

    # ------------------------------------------------ UI construction
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        self.header = HeaderBar(); root.addWidget(self.header)

        self.main_split = QSplitter(Qt.Horizontal)
        root.addWidget(self.main_split, 1)

        self.main_split.addWidget(self._build_left_panel())
        self.main_split.addWidget(self._build_right_panel())
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)

        self._build_status_bar()

    def _build_left_panel(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        panel = QWidget(); scroll.setWidget(panel)
        v = QVBoxLayout(panel); v.setContentsMargins(10, 10, 8, 10); v.setSpacing(10)

        # presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for name, s, t in TRANSLATION_PRESETS:
            self.preset_combo.addItem(name, (s, t))
        self.preset_combo.addItem("— custom —", None)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.activated.connect(self._load_preset)
        preset_row.addWidget(self.preset_combo, 1)
        swap = QPushButton("⇅ Swap"); swap.setToolTip("Exchange source and target")
        swap.clicked.connect(self._swap_texts)
        preset_row.addWidget(swap)
        v.addLayout(preset_row)

        # editors
        self.source_card = TextCard("Source text", "text to translate from…")
        self.target_card = TextCard("Target text", "text to translate to…")
        v.addWidget(self.source_card); v.addWidget(self.target_card)

        # algorithm
        algo_box = QGroupBox("Algorithm")
        av = QVBoxLayout(algo_box); av.setSpacing(8)
        grid = QFormLayout(); grid.setLabelAlignment(Qt.AlignRight)
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(2, 4096)
        self.rows_spin.setValue(64)
        self.rows_spin.setToolTip("CA grid height (grid-based algorithms only)")
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(random.randrange(2**31))
        dice = QPushButton("🎲"); dice.setFixedWidth(34); dice.setToolTip("Random seed")
        dice.clicked.connect(lambda: self.seed_spin.setValue(random.randrange(2**31)))
        srow = QHBoxLayout(); srow.addWidget(self.seed_spin); srow.addWidget(dice)
        grid.addRow("rows:", self.rows_spin); grid.addRow("seed:", srow)
        av.addLayout(grid)

        self.algo_combo = QComboBox(); self.algo_combo.addItems(ALGORITHMS)
        self.algo_combo.currentTextChanged.connect(
            lambda a: self.param_stack.setCurrentIndex(ALGO_PAGE.get(a, 4)))
        av.addWidget(self.algo_combo)
        self.param_stack = ParamPages(); av.addWidget(self.param_stack)
        v.addWidget(algo_box)

        # run controls
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run  (Ctrl+Return)")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.start_run)
        self.stop_btn = QPushButton("■ Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_run)
        run_row.addWidget(self.run_btn, 1); run_row.addWidget(self.stop_btn)
        v.addLayout(run_row)

        self.record_chk = QCheckBox("Record telemetry (for JSONL export)")
        v.addWidget(self.record_chk)
        v.addStretch(1)
        scroll.setMinimumWidth(380)
        return scroll

    def _build_right_panel(self):
        self.vert_split = QSplitter(Qt.Vertical)

        # --- space-time view card ---
        card = QFrame(); card.setObjectName("card")
        cv = QVBoxLayout(card); cv.setContentsMargins(8, 8, 8, 8); cv.setSpacing(6)

        self.ca_view = CAView2D()

        opt = QHBoxLayout()
        opt.addWidget(QLabel("Palette:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Neon", "Matrix", "Amber", "Heatmap"])
        self.color_combo.currentTextChanged.connect(self.ca_view.set_color_mode)
        opt.addWidget(self.color_combo)
        self.grid_chk = QCheckBox("Grid"); self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(self.ca_view.set_show_grid)
        self.live_chk = QCheckBox("Live badge")
        self.live_chk.toggled.connect(self.ca_view.set_live)
        opt.addWidget(self.grid_chk); opt.addWidget(self.live_chk); opt.addStretch(1)
        self.zoom_lab = QLabel("zoom 1.00×"); self.zoom_lab.setObjectName("counter")
        opt.addWidget(self.zoom_lab)
        rst = QPushButton("⟲ Reset"); rst.setFixedWidth(80)
        rst.clicked.connect(self.ca_view.reset_view)
        opt.addWidget(rst)
        cv.addLayout(opt)
        cv.addWidget(self.ca_view, 1)

        rev = QHBoxLayout()
        rev.addWidget(QLabel("Reveal:"))
        self.reveal_slider = QSlider(Qt.Horizontal)
        self.reveal_slider.setRange(0, 0)
        self.reveal_slider.valueChanged.connect(self.ca_view.set_reveal_row)
        self.ca_view.reveal_changed.connect(self._sync_reveal_slider)
        rev.addWidget(self.reveal_slider, 1)
        self.speed_spin = QSpinBox(); self.speed_spin.setRange(1, 200)
        self.speed_spin.setValue(12); self.speed_spin.setSuffix(" ms/row")
        self.speed_spin.setFixedWidth(110)
        rev.addWidget(self.speed_spin)
        self.anim_btn = QPushButton("▶ Animate"); self.anim_btn.setCheckable(True)
        self.anim_btn.setFixedWidth(96)
        self.anim_btn.toggled.connect(self._toggle_reveal)
        self.ca_view.animation_finished.connect(lambda: self.anim_btn.setChecked(False))
        rev.addWidget(self.anim_btn)
        cv.addLayout(rev)
        self.vert_split.addWidget(card)

        # --- bottom deck: toolbar + tabs ---
        self.tabs = QTabWidget()
        self.match_strip = MatchStripWidget()
        self.chart = MultiSeriesChart()
        self.timeline = PathTimelineWidget()
        self.timeline.stepSelected.connect(lambda i: self.ca_view.set_reveal_row(i + 1))
        self.candidates = CandidatesWidget()
        self.bars = BarsWidget()
        self.heat = HeatmapWidget()
        self.tree_view = TreeViewWidget()
        self.pop_view = PopulationWidget()
        self.preview = PreviewWidget()
        self.stat_line = QLabel("—"); self.stat_line.setObjectName("statLine")
        self.stat_line.setWordWrap(True)

        tele_page = QWidget(); tv = QVBoxLayout(tele_page)
        tv.setContentsMargins(6, 4, 6, 6); tv.setSpacing(4)
        tv.addWidget(self.stat_line); tv.addWidget(self.bars)
        tv.addWidget(QLabel("candidates:"))
        tv.addWidget(self.candidates, 1)

        self.tabs.addTab(self.match_strip, "Match")
        self.tabs.addTab(self.chart, "Dynamics")
        self.tabs.addTab(self.timeline, "Steps")
        self.tabs.addTab(tele_page, "Telemetry")
        self.tabs.addTab(self.heat, "Heat 256")
        self.tabs.addTab(self.tree_view, "MCTS Tree")
        self.tabs.addTab(self.pop_view, "GA Pop")
        self.tabs.addTab(self.preview, "Preview")

        self.inspector = None
        try:
            self.stats_panel = RunStatsPanel()
            self.inspector = TranslationInspector(self.timeline, self.stats_panel, self)
            self.tabs.addTab(self.stats_panel, "Result")
        except NameError:
            pass

        self.toolbar = build_result_toolbar(
            self, self.inspector, self.ca_view,
            telemetry_getter=lambda: self.last_telemetry_log,
            session_saver=self._save_session)

        bottom = QWidget(); bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0); bv.setSpacing(4)
        bv.addWidget(self.toolbar); bv.addWidget(self.tabs, 1)
        self.vert_split.addWidget(bottom)

        self.vert_split.setStretchFactor(0, 3)
        self.vert_split.setStretchFactor(1, 1)
        self.vert_split.setSizes([520, 300])
        return self.vert_split

    def _build_status_bar(self):
        sb = self.statusBar()
        self.status_msg = QLabel(""); sb.addWidget(self.status_msg, 1)
        self.progress = QProgressBar(); self.progress.setFixedWidth(220)
        self.progress.setRange(0, 100); self.progress.setValue(0)
        sb.addPermanentWidget(self.progress)
        self.pos_lab = QLabel(""); self.pos_lab.setFixedWidth(240)
        sb.addPermanentWidget(self.pos_lab)
        self.ca_view.mouse_pos_changed.connect(self.pos_lab.setText)
        self.ca_view.zoom_changed.connect(
            lambda z: self.zoom_lab.setText(f"zoom {z:.2f}×"))

    # ------------------------------------------------ wiring
    def _wire_telemetry(self):
        r = self.router
        r.register('bits', lambda d: self.match_strip.set_bits(d.get('bits'), d.get('target')))
        r.register('series', lambda d: [self.chart.add_point(k, d[k])
                    for k in ('best', 'current', 'mean', 'worst', 'temp') if k in d])
        r.register('stat', lambda d: self.stat_line.setText(str(d.get('text', ''))))
        r.register('candidates', lambda d: self.candidates.set_items(d.get('items')))
        r.register('bars', lambda d: self.bars.set_data(d.get('data')))
        r.register('heat', lambda d: self.heat.set_data(d))
        r.register('tree', lambda d: self.tree_view.set_nodes(d))
        r.register('population', lambda d: self.pop_view.set_data(d))
        r.register('preview', lambda d: self.preview.set_data(d))

    def _wire_shortcuts(self):
        sc1 = QShortcut(QKeySequence("Ctrl+Return"), self); sc1.activated.connect(self.start_run)
        sc2 = QShortcut(QKeySequence("Ctrl+."), self); sc2.activated.connect(self.stop_run)
        sc3 = QShortcut(QKeySequence("Ctrl+R"), self); sc3.activated.connect(self.ca_view.reset_view)

    def _sync_reveal_slider(self, row: int):
        self.reveal_slider.blockSignals(True)
        self.reveal_slider.setValue(row)
        self.reveal_slider.blockSignals(False)

    def _toggle_reveal(self, on: bool):
        if on and self.ca_view.rows:
            self.ca_view.start_reveal_animation(self.speed_spin.value())
        elif not on and self.ca_view.anim_timer:
            self.ca_view.anim_timer.stop()

    def _load_preset(self, idx: int):
        data = self.preset_combo.itemData(idx)
        if data:
            s, t = data
            self.source_card.set_text(s); self.target_card.set_text(t)

    def _swap_texts(self):
        s, t = self.source_card.text(), self.target_card.text()
        self.source_card.set_text(t); self.target_card.set_text(s)

    # ------------------------------------------------ run lifecycle
    def start_run(self):
        src = self.source_card.text()
        if not src:
            QMessageBox.warning(self, "Run", "Source text is empty.")
            return
        tgt = self.target_card.text()
        params = self.param_stack.params()
        params.update({"seed": self.seed_spin.value(), "rows": self.rows_spin.value()})

        # reset telemetry surfaces
        self.chart.reset(); self.match_strip.reset()
        self.candidates.clear(); self.bars.set_data({})
        self.heat.reset(); self.tree_view.reset(); self.pop_view.reset()
        self.preview.reset(); self.timeline.set_path([])
        self.stat_line.setText("—")
        if self.inspector:
            self.inspector.last = None
            self.stats_panel.set_result(None)
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.header.set_running(True); self.status_msg.setText("starting…")

        self.thread = QThread(self)
        self.worker = SearchWorker(self.algo_combo.currentText(),
                                   text_to_bits(src), text_to_bits(tgt),
                                   self.rows_spin.value(), params)
        self.worker.moveToThread(self.thread)
        self.worker.record = self.record_chk.isChecked()
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.telemetry.connect(self.router.handle)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def stop_run(self):
        if self.worker:
            self.worker.request_interruption()
            self.status_msg.setText("stopping…")

    def _on_progress(self, pct: int, msg: str):
        self.status_msg.setText(msg.splitlines()[0][:160] if msg else "")
        if pct < 0:
            self.progress.setRange(0, 0)            # busy indicator
        else:
            self.progress.setRange(0, 100); self.progress.setValue(pct)

    def _on_finished(self, label, grid, msg):
        worker = self.worker
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.header.set_running(False)
        self.progress.setRange(0, 100); self.progress.setValue(100)
        self.status_msg.setText((msg or "done").splitlines()[0][:160])
        if grid:
            tb = worker.target_bits if worker else None
            self.ca_view.set_full_grid(grid, tb, reveal_all=False)
            self.reveal_slider.setRange(0, len(grid))
            self.reveal_slider.setValue(0)
            self.anim_btn.setChecked(bool(self.ca_view.rows))
        if self.inspector and worker:
            self.inspector.load_bits(worker.input_bits, worker.target_bits,
                                     getattr(worker, "result_path", []),
                                     algo=worker.algo,
                                     elapsed_ms=float(worker._timer.elapsed()),
                                     final_bits=getattr(worker, "result_final_bits", None))
        self.last_telemetry_log = getattr(worker, "telemetry_log", []) if worker else []
        self.worker = None; self.thread = None

    # ------------------------------------------------ session & settings
    def _save_session(self, fn: str):
        save_session(fn,
                     source=self.source_card.text(),
                     target=self.target_card.text(),
                     algo=self.algo_combo.currentText(),
                     params=self.param_stack.params(),
                     path=(self.inspector.last.path
                           if (self.inspector and self.inspector.last) else []))

    def _restore_settings(self):
        s = QSettings("CAEngine", "Translator")
        geo = s.value("main/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        hs = s.value("main/splitH"); vs = s.value("main/splitV")
        if hs: self.main_split.setSizes([int(x) for x in hs])
        if vs: self.vert_split.setSizes([int(x) for x in vs])
        i = s.value("run/lastAlgo")
        idx = self.algo_combo.findText(i) if i else -1
        if idx >= 0: self.algo_combo.setCurrentIndex(idx)
        self.record_chk.setChecked(s.value("run/record", "0") in ("1", True))
        c = s.value("view/color")
        j = self.color_combo.findText(c) if c else -1
        if j >= 0: self.color_combo.setCurrentIndex(j)
        self.grid_chk.setChecked(s.value("view/grid", "1") in ("1", True))

    def closeEvent(self, e):
        s = QSettings("CAEngine", "Translator")
        s.setValue("main/geometry", self.saveGeometry())
        s.setValue("main/splitH", self.main_split.sizes())
        s.setValue("main/splitV", self.vert_split.sizes())
        s.setValue("run/lastAlgo", self.algo_combo.currentText())
        s.setValue("run/record", int(self.record_chk.isChecked()))
        s.setValue("view/color", self.color_combo.currentText())
        s.setValue("view/grid", int(self.grid_chk.isChecked()))
        super().closeEvent(e)

# =====================================================================
#  11. SELF-TEST / BENCHMARK / CLI
# =====================================================================
def self_test(rounds: int = 200, seed: int = 0xC0FFEE) -> bool:
    """Fuzz: random Unicode pairs -> translate -> optimize -> independently verify."""
    import string
    rng = random.Random(seed)
    alphabets = [string.ascii_letters + string.digits + string.punctuation + " ",
                 "áéíóúüñçßøæå", "你好世界，测试！", "😀🎉🚀☕🧬", "∵∴≈≠≤≥∑∆Ω"]
    fails = []
    t0 = time.perf_counter()
    for k in range(rounds):
        al = rng.choice(alphabets)
        src = "".join(rng.choice(al) for _ in range(rng.randint(0, 24)))
        dst = "".join(rng.choice(al) for _ in range(rng.randint(0, 24)))
        try:
            res = translate_text(src, dst)
            if not (res.verified and res.final_text == align_texts(src, dst)[1]):
                fails.append((k, src, dst))
        except Exception as e:
            fails.append((k, src, dst, repr(e)))
    dt = (time.perf_counter() - t0) / max(1, rounds) * 1000
    if fails:
        logger.error("SELF-TEST FAILED %d/%d — first: %r", len(fails), rounds, fails[0])
        return False
    logger.info("Self-test OK: %d rounds · avg %.2f ms/translation", rounds, dt)
    return True


def benchmark(n: int = 20, seed: int = 7) -> None:
    rng = random.Random(seed)
    al = "ABCDEFGH0123456789 "
    pairs = []
    for _ in range(n):
        s = "".join(rng.choice(al) for _ in range(rng.randint(8, 16)))
        t = "".join(rng.choice(al) for _ in range(rng.randint(8, 16)))
        pairs.append((s, t))

    def timed(fn):
        t0 = time.perf_counter(); ex = 0
        for s, t in pairs:
            ex += bool(fn(s, t))
        return ex, (time.perf_counter() - t0) / n * 1000

    def beam_run(s, t):
        a, b = align_texts(s, t)
        sb, tb = text_to_bits(a), text_to_bits(b)
        _, fb = beam_search(sb, tb, 24, 64, get_basic_rule_tables())
        return np.array_equal(fb, tb)

    def hc_run(s, t):
        a, b = align_texts(s, t)
        sb, tb = text_to_bits(a), text_to_bits(b)
        _, fb = hill_climb_search(sb, tb, 200, get_basic_rule_tables())
        return np.array_equal(fb, tb)

    def hyb_run(s, t):
        a, b = align_texts(s, t)
        sb, tb = text_to_bits(a), text_to_bits(b)
        _, fb = hybrid_translate(sb, tb, 24, 64, get_basic_rule_tables())
        return np.array_equal(fb, tb)

    rows = [("Universal Translator (guaranteed)", timed(lambda s, t: translate_text(s, t).verified)),
            ("Hybrid (beam + fix-up)", timed(hyb_run)),
            ("Beam Search (best-effort)", timed(beam_run)),
            ("Hill Climb (best-effort)", timed(hc_run))]
    print(f"{'algorithm':<36}{'exact':>8}{'avg ms':>10}")
    for name, (ex, ms) in rows:
        print(f"{name:<36}{f'{ex}/{n}':>8}{ms:>8.1f}")


def _run_cli(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ca_translator",
        description="CA text-translation engine (GUI runs when no subcommand is given)")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("translate", help="guaranteed SOURCE -> TARGET translation")
    t.add_argument("source"); t.add_argument("target")
    t.add_argument("--no-optimize", action="store_true")
    t.add_argument("--json", action="store_true")
    st = sub.add_parser("selftest", help="fuzz-test the guaranteed translator")
    st.add_argument("--rounds", type=int, default=200)
    b = sub.add_parser("bench", help="benchmark guaranteed vs heuristic searches")
    b.add_argument("--n", type=int, default=20)
    ba = sub.add_parser("batch", help="translate a TSV file (source<TAB>target per line)")
    ba.add_argument("infile"); ba.add_argument("outfile")
    args = ap.parse_args(argv)

    if args.cmd == "translate":
        res = translate_text(args.source, args.target, optimize=not args.no_optimize)
        if args.json:
            print(json.dumps({"verified": res.verified, "steps": res.steps,
                              "ms": round(res.elapsed_ms, 2), "path": res.path},
                             ensure_ascii=False))
        else:
            print(f"{'OK  ' if res.verified else 'FAIL'} {res.steps} steps · {res.elapsed_ms:.1f} ms")
            shown = " -> ".join(res.path[:12]) + (" …" if len(res.path) > 12 else "")
            print("path:", shown or "(identity)")
        return 0 if res.verified else 1
    if args.cmd == "selftest":
        return 0 if self_test(args.rounds) else 1
    if args.cmd == "bench":
        benchmark(args.n); return 0
    if args.cmd == "batch":
        print(f"{batch_translate_file(args.infile, args.outfile)} pairs translated")
        return 0
    ap.print_help()
    return 1

# =====================================================================
#  12. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    _CLI_CMDS = {"translate", "selftest", "bench", "batch"}
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_CMDS:
        raise SystemExit(_run_cli())
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())