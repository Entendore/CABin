import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLineEdit, QPushButton, QLabel, 
                               QSpinBox, QTextEdit, QGroupBox, QFormLayout, 
                               QSlider, QComboBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPalette

# --- Core Logic Functions ---

def text_to_bits(text, target_bits=None, pad_char=' '):
    """Converts a string to a list of binary bits (8-bit per char), padding with pad_char if needed."""
    bits = []
    for char in text:
        bits.extend([int(b) for b in format(ord(char), '08b')])
    
    if target_bits:
        if len(bits) < target_bits:
            pad_bits = [int(b) for b in format(ord(pad_char), '08b')]
            while len(bits) < target_bits:
                bits.extend(pad_bits)
            bits = bits[:target_bits]
        else:
            bits = bits[:target_bits]
    return bits

def bits_to_text(bits):
    """Converts a list of binary bits back to a string."""
    chars = []
    valid_bits = bits[:len(bits) - (len(bits) % 8)]
    for i in range(0, len(valid_bits), 8):
        byte_bits = valid_bits[i:i+8]
        byte_val = int("".join(map(str, byte_bits)), 2)
        if 32 <= byte_val <= 126:
            chars.append(chr(byte_val))
        else:
            chars.append('.') # Replace non-printable with dot
    return "".join(chars)

class LogicGateCA2D:
    """Handles the 2D Logic Gate Cellular Automaton logic (Space-Time Diagram)."""
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.grid = []
        self.gate_type = "XOR"
        self.current_row = 0

    def seed_from_bits(self, bits):
        """Sets the top row (Row 0) as the input text."""
        self.grid = [bits]
        self.current_row = 1

    def step(self):
        """Calculates the next row downwards using Logic Gates."""
        if self.current_row >= self.rows:
            return None
        
        prev_row = self.grid[-1]
        new_row = []
        
        for c in range(self.cols):
            L = prev_row[c-1] if c > 0 else 0
            T = prev_row[c]
            R = prev_row[c+1] if c < self.cols - 1 else 0
            
            if self.gate_type == "XOR":
                val = L ^ T ^ R
            elif self.gate_type == "OR":
                val = L | T | R
            elif self.gate_type == "AND":
                val = L & T & R
            elif self.gate_type == "MAJORITY":
                val = 1 if (L + T + R) >= 2 else 0
            elif self.gate_type == "NAND":
                val = 0 if (L & T & R) == 1 else 1
                
            new_row.append(val)
            
        self.grid.append(new_row)
        self.current_row += 1
        return new_row

# --- PySide6 GUI Components ---

class CAView2D(QWidget):
    """Custom widget to paint the top-to-bottom 2D Logic Gate grid."""
    def __init__(self):
        super().__init__()
        self.grid = []
        self.rows = 0
        self.cols = 0
        self.target_rows = 0 
        self.matched_row = -1 # Tracks the specific row where the target was found
        self.setMinimumSize(600, 400)
        self.setAutoFillBackground(True)
        
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#050505"))
        self.setPalette(palette)

    def set_grid(self, grid_2d, target_rows=None, matched_row=-1):
        self.grid = grid_2d
        self.rows = len(grid_2d)
        self.cols = len(grid_2d[0]) if self.rows > 0 else 0
        if target_rows is not None:
            self.target_rows = target_rows
        self.matched_row = matched_row
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        
        if not self.grid or self.rows == 0 or self.cols == 0:
            painter.setPen(QColor("#555"))
            painter.setFont(QFont("Consolas", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Target Search Animation will appear here...")
            return

        cell_w = self.width() / self.cols
        cell_h = self.height() / self.rows
        cell_size = min(cell_w, cell_h)
        
        offset_x = (self.width() - (cell_size * self.cols)) / 2
        offset_y = (self.height() - (cell_size * self.rows)) / 2

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 1:
                    painter.setPen(Qt.NoPen)
                    
                    # Color coding to show Input -> Process -> Matched Output
                    if r == 0:
                        painter.setBrush(QColor("#00aaff")) # Neon Blue for Input
                    elif r == self.matched_row:
                        painter.setBrush(QColor("#ffcc00")) # Neon Gold for Matched Target
                    elif r == self.target_rows - 1 and self.matched_row == -1:
                        painter.setBrush(QColor("#ffcc00")) # Neon Gold for final row if still searching
                    else:
                        painter.setBrush(QColor("#00ff66")) # Neon Green for Intermediate
                        
                    x = int(offset_x + c * cell_size)
                    y = int(offset_y + r * cell_size)
                    size = int(cell_size)
                    painter.drawRect(x, y, size, size)


class TypewriterTextEdit(QTextEdit):
    """Text edit that simulates a typewriter effect, then unlocks for user input."""
    def __init__(self):
        super().__init__()
        self.full_text = ""
        self.current_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._type_next_char)
        
        self.setFont(QFont("Consolas", 14))
        self.setStyleSheet("""
            QTextEdit {
                color: #ffcc00; background-color: #050505; 
                border: 1px solid #333; border-radius: 4px; padding: 10px;
            }
        """)
        self.setReadOnly(False)
        self.setPlaceholderText("Search results and output text will appear here...")

    def start_typing(self, text):
        self.full_text = text
        self.current_index = 0
        self.clear()
        self.setReadOnly(True) # Lock while typing
        self.timer.start(30)

    def _type_next_char(self):
        if self.current_index < len(self.full_text):
            self.insertPlainText(self.full_text[self.current_index])
            self.current_index += 1
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        else:
            self.timer.stop()
            self.setReadOnly(False) # Unlock for user to type/edit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Target-Driven Logic Gate CA Search")
        self.resize(900, 850)
        self.setStyleSheet("background-color: #121212; color: #eee;")
        
        self.ca = None
        self.target_bits = []
        self.search_gates = []
        self.current_gate_idx = 0
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.animation_step)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Input Controls ---
        input_group = QGroupBox("1. Input Text & Target Output Parameters")
        input_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 15px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        form_layout = QFormLayout()
        
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Starting text (e.g., HELLO)...")
        self.input_text.setMaxLength(20)
        self.input_text.setFont(QFont("Consolas", 12))
        self.input_text.setStyleSheet("QLineEdit { padding: 5px; background-color: #2d2d2d; border: 1px solid #555; border-radius: 3px; }")
        
        self.target_text = QLineEdit()
        self.target_text.setPlaceholderText("Target output text to find...")
        self.target_text.setMaxLength(20)
        self.target_text.setFont(QFont("Consolas", 12))
        self.target_text.setStyleSheet("QLineEdit { padding: 5px; background-color: #2d2d2d; border: 1px solid #555; border-radius: 3px; }")

        self.gate_combo = QComboBox()
        self.gate_combo.addItems([
            "Auto-Search (All Gates)", 
            "XOR (Fractals)", 
            "OR (Expanding Waves)", 
            "AND (Shrinking Signals)", 
            "MAJORITY (Organic Flow)", 
            "NAND (Inverted Chaos)"
        ])
        self.gate_combo.setStyleSheet("QComboBox { padding: 5px; background-color: #2d2d2d; border: 1px solid #555; }")

        self.gens_spin = QSpinBox()
        self.gens_spin.setRange(50, 1000)
        self.gens_spin.setValue(300)
        self.gens_spin.setStyleSheet("QSpinBox { padding: 5px; background-color: #2d2d2d; border: 1px solid #555; }")

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 100)
        self.speed_slider.setValue(20) 
        self.speed_slider.setStyleSheet("QSlider::groove:horizontal { background: #444; height: 6px; border-radius: 3px; } QSlider::handle:horizontal { background: #00ff66; width: 16px; margin: -5px 0; border-radius: 8px; }")
        self.speed_label = QLabel("20 ms/row")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} ms/row"))

        self.run_btn = QPushButton("Start Target Search")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #00ff66; color: #111; font-weight: bold; 
                padding: 8px; border-radius: 4px; border: none;
            }
            QPushButton:hover { background-color: #00cc55; }
            QPushButton:pressed { background-color: #009944; }
        """)
        self.run_btn.clicked.connect(self.on_search)

        self.status_label = QLabel("Ready to search.")
        self.status_label.setStyleSheet("color: #00aaff; font-style: italic;")

        form_layout.addRow("Input Text:", self.input_text)
        form_layout.addRow("Target Output:", self.target_text)
        form_layout.addRow("Logic Gate:", self.gate_combo)
        form_layout.addRow("Max Generations:", self.gens_spin)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        form_layout.addRow("Anim Speed:", speed_layout)
        
        form_layout.addRow(self.run_btn)
        form_layout.addRow(self.status_label)
        input_group.setLayout(form_layout)
        main_layout.addWidget(input_group)

        # --- 2D CA Visualization ---
        viz_group = QGroupBox("2. 2D Logic Gate Space-Time Diagram (Searching for Target)")
        viz_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 15px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        viz_layout = QVBoxLayout()
        self.ca_view = CAView2D()
        viz_layout.addWidget(self.ca_view)
        viz_group.setLayout(viz_layout)
        main_layout.addWidget(viz_group)

        # --- Output Text ---
        out_group = QGroupBox("3. Search Results & Output Text")
        out_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 15px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        out_layout = QVBoxLayout()
        self.output_text = TypewriterTextEdit()
        out_layout.addWidget(self.output_text)
        out_group.setLayout(out_layout)
        main_layout.addWidget(out_group)

    def on_search(self):
        input_str = self.input_text.text()
        target_str = self.target_text.text()
        
        if not input_str or not target_str:
            self.output_text.start_typing("Error: Please provide both Input and Target Output texts!")
            return

        self.anim_timer.stop()
        
        # Determine grid width based on the longer text to ensure consistent bit lengths
        max_chars = max(len(input_str), len(target_str), 5)
        cols = max_chars * 8
        rows = self.gens_spin.value()
        
        self.target_bits = text_to_bits(target_str, target_bits=cols)
        
        # Setup gates to search
        gate_map = {
            "XOR (Fractals)": "XOR", 
            "OR (Expanding Waves)": "OR", 
            "AND (Shrinking Signals)": "AND", 
            "MAJORITY (Organic Flow)": "MAJORITY", 
            "NAND (Inverted Chaos)": "NAND"
        }
        
        selected_gate = self.gate_combo.currentText()
        if selected_gate == "Auto-Search (All Gates)":
            self.search_gates = list(gate_map.values())
        else:
            self.search_gates = [gate_map[selected_gate]]
            
        self.current_gate_idx = 0
        self.max_rows = rows
        self.cols = cols
        
        self._init_ca_for_search()
        
        self.anim_timer.start(self.speed_slider.value())
        self.run_btn.setText("Searching...")
        self.run_btn.setEnabled(False)
        self.status_label.setText(f"Initializing search with {self.search_gates[0]} gate...")
        self.output_text.start_typing(f"Starting search for '{target_str}'...")

    def _init_ca_for_search(self):
        input_str = self.input_text.text()
        self.ca = LogicGateCA2D(self.cols, self.max_rows)
        self.ca.gate_type = self.search_gates[self.current_gate_idx]
        
        # Pad input text with spaces to match the grid width
        bits = text_to_bits(input_str, target_bits=self.cols, pad_char=' ')
        self.ca.seed_from_bits(bits)
        self.ca_view.set_grid(self.ca.grid, target_rows=self.max_rows, matched_row=-1)

    def animation_step(self):
        if self.ca.current_row < self.ca.rows:
            self.ca.step()
            self.ca_view.set_grid(self.ca.grid)
            self.status_label.setText(f"Searching {self.ca.gate_type}... Row {self.ca.current_row}/{self.ca.rows}")
            
            # Check if current row matches target
            current_bits = self.ca.grid[-1]
            if current_bits == self.target_bits:
                # MATCH FOUND!
                self.ca_view.set_grid(self.ca.grid, matched_row=self.ca.current_row)
                self.anim_timer.stop()
                self.run_btn.setText("Start Target Search")
                self.run_btn.setEnabled(True)
                self.status_label.setText(f"SUCCESS! Match found at Row {self.ca.current_row}!")
                
                output_str = bits_to_text(current_bits)
                self.output_text.start_typing(f"SUCCESS! Match found using {self.ca.gate_type} gate at Row {self.ca.current_row}.\n\nDecoded Output: {output_str}")
                return
        else:
            # Current gate exhausted, try next gate if in Auto-Search
            self.current_gate_idx += 1
            if self.current_gate_idx < len(self.search_gates):
                self._init_ca_for_search()
                self.status_label.setText(f"Switching to {self.search_gates[self.current_gate_idx]} gate...")
            else:
                # All gates exhausted
                self.anim_timer.stop()
                self.run_btn.setText("Start Target Search")
                self.run_btn.setEnabled(True)
                self.status_label.setText("Search complete. No match found.")
                self.output_text.start_typing("Search complete. The target output text was not found within the maximum generations for the selected gate(s).\n\nTip: Try increasing 'Max Generations' or use 'Auto-Search (All Gates)'.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())