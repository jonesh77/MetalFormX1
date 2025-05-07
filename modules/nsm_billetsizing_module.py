# modules/nsm_billetsizing_module.py (boshlanishi) - Updated with Matplotlib formula rendering
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel,
                            QGridLayout, QFileDialog, QMessageBox, QSpacerItem,
                            QSizePolicy, QFrame, QHBoxLayout, QGroupBox, QTabWidget,
                            QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
                            QFormLayout)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import pyqtSignal, Qt
import os
# Import Matplotlib components
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Dummy classes for testing without scikit-learn and keras
print("Using fallback mode for NSM Billetsizing module")
def load_model(path):
    print(f"Simulating loading model from {path}")
    class DummyModel:
        def predict(self, x, verbose=0):
            print(f"Simulating prediction")
            # Ensure output matches expected structure if model returns list
            return np.array([[0.5]]) # Simplified for testing
    return DummyModel()

class StandardScaler:
    def fit(self, X):
        pass
    def transform(self, X):
        return X

class SelectKBest:
    def __init__(self, score_func=None, k=10):
        self.k = k
    def fit(self, X, y):
        pass
    def transform(self, X):
        return X

def f_regression(X, y):
    return np.ones(X.shape[1]), np.ones(X.shape[1])

class Bounds:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

class NonlinearConstraint:
    def __init__(self, fun, lb, ub):
        self.fun = fun
        self.lb = lb
        self.ub = ub

# Dummy minimize function to simulate optimization result
def minimize(fun, x0, method='trust-constr', bounds=None, constraints=None, tol=1e-1):
    class Result:
        def __init__(self, x):
            # Simulate a plausible result based on x0 and bounds
            simulated_x = []
            for i, val in enumerate(x0):
                low = bounds.lb[i] if bounds and bounds.lb and i < len(bounds.lb) else val - 0.1*abs(val)
                high = bounds.ub[i] if bounds and bounds.ub and i < len(bounds.ub) else val + 0.1*abs(val)
                # Simulate finding a value within bounds, close to mean
                simulated_val = np.clip((low + high) / 2 + np.random.randn()*0.05*(high-low+1e-6), low, high)
                simulated_x.append(simulated_val)
            self.x = np.array(simulated_x)
            print(f"Simulated optimization result: {self.x}")
    return Result(x0)

class NSMBilletsizingModule:
    def __init__(self):
        self.name = "NSM Billetsizing"
        self.model = None
        self.dataset = None
        self.X = None
        # Adjusted column names to match common indices/usage
        self.column_names = ['Feed', 'DepthSchedule', 'NumRotation', 'Pass1', 'Pass2', 'Pass3', 'Pass4', 'Pass5', 'Pass6', 'Pass7', 'ENE']
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_regression, k=10) # k adjusted to match likely features
        self.results_data = {}  # For storing result data

        # Qo'shimcha xususiyatlar
        self.cogging_data = None
        self.bqi_weight_factor = 0.5  # Default BQI vazn koeffitsienti
        self.desired_grain_size = 7.0  # Default istalgan don o'lchami (ASTM E112)
        self.show_forging_details = True  # Bolg'alash detallarini ko'rsatish

        # Define paths
        self.set_default_paths()

    def get_name(self):
        return self.name

    def set_default_paths(self):
        """Set default paths for data files"""
        base_dir = os.getcwd()
        data_dir = os.path.join(base_dir, "data")
        self.cogging_data_path = os.path.join(data_dir, "Cogging data.xlsx")
        # Add path for the sample data Excel file if needed by load_data_file
        self.sample_data_path = os.path.join(data_dir, "SampleNSMData.xlsx") # Example path

    def create_widget(self, parent):
        self.parent = parent  # Store reference to main window
        self.widget = QWidget(parent)

        # Using VBoxLayout as the main layout
        main_layout = QVBoxLayout(self.widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Styles
        button_style = """
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                font-size: 12px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """

        input_style = """
            QLineEdit {
                padding: 6px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                min-height: 25px;
            }
        """

        label_style = """
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #333333;
            }
        """

        group_style = """
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #4a86e8;
            }
        """

        # Create tabs for different functions
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Main tab
        self.main_tab = QWidget()
        self.main_layout = QVBoxLayout(self.main_tab)
        self.tabs.addTab(self.main_tab, "Main")

        # BQI Analysis tab
        self.bqi_tab = QWidget()
        self.bqi_layout = QVBoxLayout(self.bqi_tab)
        self.tabs.addTab(self.bqi_tab, "BQI Analysis")

        # Cogging Simulation tab
        self.cogging_tab = QWidget()
        self.cogging_layout = QVBoxLayout(self.cogging_tab)
        self.tabs.addTab(self.cogging_tab, "Cogging Simulation")

        # ---------- MAIN TAB ----------
        # Top buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.find_min_button = QPushButton("PASS SCHEDULE")
        self.find_min_button.clicked.connect(self.on_find_min)
        self.find_min_button.setStyleSheet(button_style + "background-color: #34a853;")
        buttons_layout.addWidget(self.find_min_button)

        self.load_model_button = QPushButton("Load Model (*.h5)")
        self.load_model_button.clicked.connect(self.load_model_file)
        self.load_model_button.setStyleSheet(button_style)
        buttons_layout.addWidget(self.load_model_button)

        self.load_data_button = QPushButton("Load Data (*.xlsx)")
        self.load_data_button.clicked.connect(self.load_data_file)
        self.load_data_button.setStyleSheet(button_style)
        buttons_layout.addWidget(self.load_data_button)

        self.show_results_button = QPushButton("Show Results")
        self.show_results_button.clicked.connect(self.show_visualization)
        self.show_results_button.setStyleSheet(button_style + "background-color: #e0e0e0; color: #444444;")
        self.show_results_button.setEnabled(False)
        buttons_layout.addWidget(self.show_results_button)

        self.main_layout.addLayout(buttons_layout)

        # Status label (directly below buttons, no overlapping elements)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #4a86e8; font-weight: bold; margin-top: 5px;")
        self.main_layout.addWidget(self.status_label)

        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #cccccc;")
        self.main_layout.addWidget(separator)

        # Content grid for the rest of the interface
        content_layout = QGridLayout()
        content_layout.setVerticalSpacing(15)
        content_layout.setHorizontalSpacing(10)

        current_row = 0

        # Input Parameters section (moved below status)
        input_header = QLabel("Input Parameters:")
        input_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(input_header, current_row, 0, 1, 3)
        current_row += 1

        # Feed, Depth Schedule, Number of Rotation
        content_layout.addWidget(QLabel("Feed"), current_row, 0)
        content_layout.addWidget(QLabel("Depth Schedule"), current_row, 1)
        content_layout.addWidget(QLabel("Number of Rotation"), current_row, 2)
        current_row += 1

        # Using QDoubleSpinBox for Feed and Depth for better control
        self.entry_feed = QDoubleSpinBox()
        self.entry_feed.setDecimals(3)
        self.entry_feed.setRange(0, 1000)
        self.entry_feed.setStyleSheet(input_style)
        content_layout.addWidget(self.entry_feed, current_row, 0)

        self.entry_depth = QDoubleSpinBox()
        self.entry_depth.setDecimals(3)
        self.entry_depth.setRange(0, 100)
        self.entry_depth.setStyleSheet(input_style)
        content_layout.addWidget(self.entry_depth, current_row, 1)

        # Using QSpinBox for Rotation Number
        self.entry_rotation = QSpinBox()
        self.entry_rotation.setRange(1, 10)
        self.entry_rotation.setStyleSheet(input_style)
        content_layout.addWidget(self.entry_rotation, current_row, 2)
        current_row += 1

        # Store references for easier access in on_find_min
        self.input_param_widgets = [self.entry_feed, self.entry_depth, self.entry_rotation]

        # Spacer row
        spacer = QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        content_layout.addItem(spacer, current_row, 0, 1, 3)
        current_row += 1

        # Pass Schedule section
        pass_header = QLabel("Pass Schedule:")
        pass_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(pass_header, current_row, 0, 1, 8)
        current_row += 1

        # Initial Cross-section and Pass labels
        content_layout.addWidget(QLabel("Initial Cross-section [mm]"), current_row, 0)

        self.pass_labels = []
        # Use column names for labels (indices 3 to 9 correspond to Pass1 to Pass7)
        for i in range(3, 10):
            label = QLabel(self.column_names[i])
            label.setStyleSheet(label_style)
            content_layout.addWidget(label, current_row, i-2) # Column index i-2 for grid layout
            self.pass_labels.append(label)
        current_row += 1

        # Initial Cross-section entry and Pass entries
        self.radius_entry = QLineEdit("480") # Initial value example
        self.radius_entry.setStyleSheet(input_style)
        content_layout.addWidget(self.radius_entry, current_row, 0)

        self.pass_entries = []
        for i in range(1, 8): # 7 passes
            entry = QLineEdit()
            entry.setStyleSheet(input_style)
            entry.setReadOnly(True) # Make pass values read-only, calculated by optimization
            content_layout.addWidget(entry, current_row, i)
            self.pass_entries.append(entry)
        current_row += 1

        # Forging Ratios
        content_layout.addWidget(QLabel("Forging Ratios:"), current_row, 0)

        self.forging_labels = [QLabel("") for _ in range(7)]
        for i, label in enumerate(self.forging_labels):
            label.setStyleSheet("border: 1px solid #dddddd; color: red; background-color: #f8f9fa; padding: 5px; font-weight: bold;")
            content_layout.addWidget(label, current_row, i+1)
        current_row += 1

        # Spacer row
        spacer = QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        content_layout.addItem(spacer, current_row, 0, 1, 8)
        current_row += 1

        # Length Changes section
        length_header = QLabel("Length Changes:")
        length_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(length_header, current_row, 0, 1, 8)
        current_row += 1

        # Initial Length and Length Values labels
        content_layout.addWidget(QLabel("Initial Length [mm]"), current_row, 0)
        content_layout.addWidget(QLabel("Length Values [mm]:"), current_row, 1, 1, 7)
        current_row += 1

        # Initial Length entry and Length Values
        self.initial_length_entry = QLineEdit("1500") # Initial value example
        self.initial_length_entry.setStyleSheet(input_style)
        content_layout.addWidget(self.initial_length_entry, current_row, 0)

        self.length_change_labels = [QLabel("") for _ in range(7)]
        for i, label in enumerate(self.length_change_labels):
            label.setStyleSheet("border: 1px solid #dddddd; color: blue; background-color: #f8f9fa; padding: 5px; font-weight: bold;")
            content_layout.addWidget(label, current_row, i+1)
        current_row += 1

        # Spacer row
        spacer = QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        content_layout.addItem(spacer, current_row, 0, 1, 8)
        current_row += 1

        # Cutting Analysis section
        cutting_header = QLabel("Cutting Analysis:")
        cutting_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
        content_layout.addWidget(cutting_header, current_row, 0, 1, 8)
        current_row += 1

        # Cutting Length and Cutted Length Values labels
        content_layout.addWidget(QLabel("Cutting Length [mm]"), current_row, 0)
        content_layout.addWidget(QLabel("Cutted Length (Quantity):"), current_row, 1, 1, 7)
        current_row += 1

        # Cutting Length entry and Cutted Length Values
        self.cutting_length_entry = QLineEdit("3000") # Initial value example
        self.cutting_length_entry.setStyleSheet(input_style)
        content_layout.addWidget(self.cutting_length_entry, current_row, 0)

        self.cutted_length_labels = [QLabel("") for _ in range(7)]
        for i, label in enumerate(self.cutted_length_labels):
            label.setStyleSheet("border: 1px solid #dddddd; color: #555555; background-color: #f8f9fa; padding: 5px; font-weight: bold;")
            content_layout.addWidget(label, current_row, i+1)
        current_row += 1

        # Total cutted counter
        self.yellow_changes_label = QLabel("Total cuts: 0")
        self.yellow_changes_label.setStyleSheet("font-weight: bold; color: #333333;")
        content_layout.addWidget(self.yellow_changes_label, current_row, 7, Qt.AlignmentFlag.AlignRight)

        # Add content layout to main layout
        self.main_layout.addLayout(content_layout)

        # Add stretch at the bottom of main tab
        self.main_layout.addStretch()

       # ---------- BQI ANALYSIS TAB ----------
        # BQI formula info group
        bqi_info_group = QGroupBox("BQI Formula")
        bqi_info_group.setStyleSheet(group_style)
        bqi_info_layout = QVBoxLayout()

        # --- Matplotlib Canvas for BQI Formula ---
        # Use FigureCanvas to render the formula using Matplotlib's MathText
        self.bqi_formula_figure = Figure(figsize=(6, 1), dpi=100) # Adjust figsize as needed
        self.bqi_formula_canvas = FigureCanvas(self.bqi_formula_figure)
        bqi_info_layout.addWidget(self.bqi_formula_canvas)

        # Render the formula onto the canvas
        ax = self.bqi_formula_figure.add_subplot(111)
        # Use raw string (r'') and LaTeX syntax for the formula
        formula_latex = r'$BQI = (\epsilon_s \bar{\epsilon}) \frac{d_s}{d} + w(d_{des} - \bar{d})^2$'
        ax.text(0.5, 0.4, formula_latex, # Adjusted vertical position slightly
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16) # Adjust fontsize as needed
        ax.axis('off') # Hide the axes
        self.bqi_formula_figure.tight_layout(pad=0.1) # Adjust padding
        self.bqi_formula_canvas.draw() # Draw the canvas
        # --- End Matplotlib Canvas ---

        # BQI parameters description (Updated symbols)
        bqi_params_label = QLabel(
            "Where:\n"
            "  ε<sub>s</sub> is the standard deviation effective strain\n"
            "  ε̄ is the average effective strain\n"
            "  d<sub>s</sub> is the standard deviation grain size number\n"
            "  d is the denominator term (verify definition, often related to average grain size)\n"
            "  d̄ is the average grain size number (ASTM E112)\n"
            "  d<sub>des</sub> is the desired grain size number\n"
            "  w is the weight factor"
        )
        bqi_params_label.setTextFormat(Qt.TextFormat.RichText) # Ensure HTML/Rich Text is parsed
        bqi_params_label.setStyleSheet("font-size: 12px; padding: 5px 10px;")
        bqi_params_label.setWordWrap(True)
        bqi_info_layout.addWidget(bqi_params_label)

        bqi_info_group.setLayout(bqi_info_layout)
        self.bqi_layout.addWidget(bqi_info_group)

        # BQI Analysis settings group
        bqi_settings_group = QGroupBox("BQI Analysis Settings")
        bqi_settings_group.setStyleSheet(group_style)
        bqi_settings_layout = QFormLayout()

        # Weight factor input
        self.weight_factor_input = QDoubleSpinBox()
        self.weight_factor_input.setRange(0.0, 5.0) # Increased range
        self.weight_factor_input.setSingleStep(0.05)
        self.weight_factor_input.setValue(self.bqi_weight_factor)
        self.weight_factor_input.valueChanged.connect(self.update_bqi_weight_factor)
        bqi_settings_layout.addRow("Weight Factor (w):", self.weight_factor_input)

        # Desired grain size input
        self.desired_grain_size_input = QDoubleSpinBox()
        self.desired_grain_size_input.setRange(1.0, 14.0)
        self.desired_grain_size_input.setSingleStep(0.5)
        self.desired_grain_size_input.setValue(self.desired_grain_size)
        self.desired_grain_size_input.valueChanged.connect(self.update_desired_grain_size)
        bqi_settings_layout.addRow("Desired Grain Size (d<sub>des</sub>, ASTM E112):", self.desired_grain_size_input)

        # Load Cogging data button
        self.load_cogging_data_button = QPushButton("Load Cogging Data")
        self.load_cogging_data_button.clicked.connect(self.load_cogging_data)
        self.load_cogging_data_button.setStyleSheet(button_style)
        bqi_settings_layout.addRow("Grain/Strain Data:", self.load_cogging_data_button)

        # Analyze button
        self.analyze_bqi_button = QPushButton("Analyze BQI")
        self.analyze_bqi_button.clicked.connect(self.analyze_bqi)
        self.analyze_bqi_button.setStyleSheet(button_style + "background-color: #34a853;")
        bqi_settings_layout.addRow("", self.analyze_bqi_button)

        bqi_settings_group.setLayout(bqi_settings_layout)
        self.bqi_layout.addWidget(bqi_settings_group)

        # BQI Results group
        bqi_results_group = QGroupBox("BQI Analysis Results")
        bqi_results_group.setStyleSheet(group_style)
        bqi_results_layout = QVBoxLayout()

        # BQI results figure - Reuse Matplotlib canvas if possible or create new
        self.bqi_results_figure = Figure(figsize=(8, 4)) # Separate figure for results
        self.bqi_results_canvas = FigureCanvas(self.bqi_results_figure)
        bqi_results_layout.addWidget(self.bqi_results_canvas)
        self.bqi_ax = self.bqi_results_figure.add_subplot(111) # Axes for results plot

        # BQI results label
        self.bqi_results_label = QLabel("No BQI analysis results yet.")
        self.bqi_results_label.setStyleSheet("font-size: 12px; padding: 10px; font-weight: bold;")
        self.bqi_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bqi_results_layout.addWidget(self.bqi_results_label)

        bqi_results_group.setLayout(bqi_results_layout)
        self.bqi_layout.addWidget(bqi_results_group)

        # Add stretch to the bottom of BQI tab
        self.bqi_layout.addStretch()

        # ---------- COGGING SIMULATION TAB ----------
        # Cogging diagram group
        cogging_diagram_group = QGroupBox("Cogging Process Diagram")
        cogging_diagram_group.setStyleSheet(group_style)
        cogging_diagram_layout = QVBoxLayout()

        # Cogging process visualization
        self.cogging_figure, self.cogging_ax = plt.subplots(figsize=(8, 4)) # Smaller figure
        self.cogging_canvas = FigureCanvas(self.cogging_figure)
        cogging_diagram_layout.addWidget(self.cogging_canvas)

        cogging_diagram_group.setLayout(cogging_diagram_layout)
        self.cogging_layout.addWidget(cogging_diagram_group)

        # Cogging Settings group
        cogging_settings_group = QGroupBox("Cogging Settings")
        cogging_settings_group.setStyleSheet(group_style)
        cogging_settings_layout = QFormLayout()

        # Number of rotation selector
        self.rotation_number_input = QSpinBox()
        self.rotation_number_input.setRange(1, 10) # Wider range
        self.rotation_number_input.setValue(3)
        self.rotation_number_input.valueChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("Number of Rotation:", self.rotation_number_input)

        # Feed amount input
        self.feed_amount_input = QDoubleSpinBox()
        self.feed_amount_input.setRange(10.0, 500.0) # Wider range
        self.feed_amount_input.setSingleStep(5.0)
        self.feed_amount_input.setValue(50.0)
        self.feed_amount_input.setSuffix(" mm")
        self.feed_amount_input.valueChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("Feed Amount:", self.feed_amount_input)

        # Depth schedule input
        self.depth_schedule_input = QDoubleSpinBox()
        self.depth_schedule_input.setRange(0.0, 100.0) # Wider range
        self.depth_schedule_input.setSingleStep(1.0)
        self.depth_schedule_input.setValue(20.0)
        self.depth_schedule_input.setSuffix(" mm")
        self.depth_schedule_input.valueChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("Depth Schedule:", self.depth_schedule_input)

        # Overlap option
        self.overlap_checkbox = QCheckBox("Show Overlap/Detail View")
        self.overlap_checkbox.setChecked(True)
        self.overlap_checkbox.stateChanged.connect(self.update_cogging_diagram)
        cogging_settings_layout.addRow("", self.overlap_checkbox)

        # Simulate button
        self.simulate_cogging_button = QPushButton("Apply Settings to Main Tab")
        self.simulate_cogging_button.clicked.connect(self.simulate_cogging)
        self.simulate_cogging_button.setStyleSheet(button_style + "background-color: #ea4335;")
        cogging_settings_layout.addRow("", self.simulate_cogging_button)

        cogging_settings_group.setLayout(cogging_settings_layout)
        self.cogging_layout.addWidget(cogging_settings_group)

        # Initial cogging diagram
        self.update_cogging_diagram()

        # Check if we can load Cogging data automatically
        if os.path.exists(self.cogging_data_path):
            try:
                self.cogging_data = pd.read_excel(self.cogging_data_path)
                self.status_label.setText("Status: Cogging data loaded automatically")
            except Exception as e:
                self.status_label.setText(f"Status: Error auto-loading cogging data - {str(e)}")

        return self.widget

    def update_bqi_weight_factor(self, value):
        """Update BQI weight factor"""
        self.bqi_weight_factor = value
        print(f"BQI Weight Factor updated to: {self.bqi_weight_factor}") # Debug print

    def update_desired_grain_size(self, value):
        """Update desired grain size"""
        self.desired_grain_size = value
        print(f"Desired Grain Size updated to: {self.desired_grain_size}") # Debug print

    def load_cogging_data(self):
        """Load cogging data (containing grain size and strain info) from Excel file"""
        filepath, _ = QFileDialog.getOpenFileName(self.widget, "Open Grain Size/Strain Data File", "", "Excel Files (*.xlsx *.xls)")
        if filepath:
            try:
                # Assuming the excel file has sheets or columns named appropriately
                # Example: Columns like 'Pass', 'AvgStrain', 'StdStrain', 'AvgGrainSize', 'StdGrainSize'
                self.cogging_data = pd.read_excel(filepath)
                # --- Add validation for required columns ---
                required_cols = ['AvgStrain', 'StdStrain', 'AvgGrainSize', 'StdGrainSize'] # Example names
                if not all(col in self.cogging_data.columns for col in required_cols):
                     QMessageBox.warning(self.widget, "Warning", f"Loaded data missing required columns: {required_cols}")
                     self.cogging_data = None # Invalidate data
                     return
                # --- End validation ---

                self.cogging_data_path = filepath
                self.status_label.setText("Status: Grain/Strain data loaded successfully")
                QMessageBox.information(self.widget, "Success", "Grain size and strain data loaded successfully!")
            except Exception as e:
                self.cogging_data = None
                self.status_label.setText(f"Status: Error loading grain/strain data - {str(e)}")
                QMessageBox.critical(self.widget, "Error", f"Error loading grain/strain data: {str(e)}")

    def analyze_bqi(self):
        """Analyze BQI based on the loaded data and settings"""
        if self.cogging_data is None:
            QMessageBox.warning(self.widget, "Warning", "Please load grain size and strain data first!")
            return

        # results_data is populated by on_find_min which might not be run yet.
        # BQI analysis might depend only on cogging_data. Let's proceed assuming that.
        # if self.results_data is None or not self.results_data:
        #     QMessageBox.warning(self.widget, "Warning", "Please calculate pass schedule first if needed for BQI!")
        #     return

        try:
            # Ensure required columns exist (adjust names as per your actual Excel file)
            required_cols = ['AvgStrain', 'StdStrain', 'AvgGrainSize', 'StdGrainSize']
            if not all(col in self.cogging_data.columns for col in required_cols):
                 QMessageBox.critical(self.widget, "Error", f"Grain/Strain data is missing columns: {required_cols}")
                 return

            # Extract data (assuming one row per pass, or aggregate if needed)
            # If multiple rows per pass, you might need to group and aggregate first
            # Example: Assuming data is already aggregated per pass
            avg_strain = self.cogging_data['AvgStrain'].values
            std_strain = self.cogging_data['StdStrain'].values
            avg_grain_size_num = self.cogging_data['AvgGrainSize'].values # This is d_bar
            std_grain_size_num = self.cogging_data['StdGrainSize'].values # This is d_s

            # --- Handle the 'd' term ---
            # Option 1: Assume d = d_bar (average grain size number)
            # d_denominator = avg_grain_size_num
            # Option 2: Assume 'd' is average grain *diameter* if available in data
            # if 'AvgGrainDiameter' in self.cogging_data.columns:
            #    d_denominator = self.cogging_data['AvgGrainDiameter'].values
            # else:
            #    QMessageBox.warning(self.widget, "Warning", "Column for 'd' (e.g., AvgGrainDiameter) not found. Using AvgGrainSize (d_bar) instead.")
            #    d_denominator = avg_grain_size_num
            # Option 3: Following the formula image strictly, assume 'd' is related to grain size but not d_bar.
            # Requires clarification. For now, let's assume d = d_bar based on common practice.
            d_denominator = avg_grain_size_num
            # --- End Handle 'd' ---


            # Number of passes based on data length
            num_passes = len(avg_strain)
            if num_passes == 0:
                QMessageBox.warning(self.widget, "Warning", "No data rows found in the loaded grain/strain file.")
                return
            passes = np.arange(1, num_passes + 1)

            # Avoid division by zero or near-zero
            epsilon = 1e-9
            avg_strain[avg_strain < epsilon] = epsilon
            d_denominator[d_denominator < epsilon] = epsilon


            # Calculate BQI components using the formula:
            # BQI = (εs / ε̄) * (ds / d) + w * (ddes - d̄)²
            strain_term_multiplier = std_strain / avg_strain
            grain_term_multiplier = std_grain_size_num / d_denominator
            strain_component = strain_term_multiplier * grain_term_multiplier

            # Ensure non-negative components if physically required
            strain_component[strain_component < 0] = 0

            grain_size_component = self.bqi_weight_factor * (self.desired_grain_size - avg_grain_size_num)**2

            bqi_values = strain_component + grain_size_component

            # Plotting
            self.bqi_ax.clear() # Clear previous results plot

            # Plot BQI values
            self.bqi_ax.bar(passes, bqi_values, color='#4a86e8', alpha=0.7, label='Total BQI')
            self.bqi_ax.plot(passes, bqi_values, 'ro-', label='_nolegend_') # Line overlay

            # Plot components
            self.bqi_ax.plot(passes, strain_component, 'gs--', label='Strain/Grain Ratio Component')
            self.bqi_ax.plot(passes, grain_size_component, 'bv--', label='Grain Size Deviation Component')

            # Set graph properties
            self.bqi_ax.set_xlabel('Pass Number', fontsize=10)
            self.bqi_ax.set_ylabel('BQI Value (Lower is better)', fontsize=10)
            self.bqi_ax.set_title(f'BQI Analysis (w={self.bqi_weight_factor:.2f}, d_des={self.desired_grain_size:.1f})', fontsize=12)
            self.bqi_ax.grid(True, linestyle='--', alpha=0.7)
            self.bqi_ax.legend(fontsize=9)
            self.bqi_ax.tick_params(axis='both', which='major', labelsize=9)
            self.bqi_ax.set_xticks(passes) # Ensure all pass numbers are shown
            self.bqi_results_figure.tight_layout()


            # Update canvas
            self.bqi_results_canvas.draw()

            # Update results label
            optimal_pass = np.argmin(bqi_values) + 1
            min_bqi = np.min(bqi_values)
            self.bqi_results_label.setText(f"Optimal Pass: {optimal_pass} (Min BQI = {min_bqi:.4f})\n"
                                          f"Final Pass BQI: {bqi_values[-1]:.4f}")

            # Show a message
            QMessageBox.information(self.widget, "Success", f"BQI Analysis completed. Optimal pass: {optimal_pass}")

        except KeyError as e:
             QMessageBox.critical(self.widget, "Error", f"Missing column in grain/strain data: {e}")
        except Exception as e:
            self.status_label.setText(f"Status: Error in BQI analysis - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Error in BQI analysis: {str(e)}")


    def update_cogging_diagram(self):
        """Update cogging process diagram based on settings"""
        try:
            self.cogging_ax.clear()
            num_rotation = self.rotation_number_input.value()
            show_overlap = self.overlap_checkbox.isChecked()
            feed = self.feed_amount_input.value()
            depth = self.depth_schedule_input.value()

            # Simple representation of passes
            initial_width = 100 # Arbitrary starting size for visualization
            current_dim1 = initial_width
            current_dim2 = initial_width * 0.8 # Assume some initial aspect ratio

            positions = []
            dimensions = []
            orientations = [] # 0 for horizontal major axis, 1 for vertical

            positions.append((0, 0))
            dimensions.append((current_dim1, current_dim2))
            orientations.append(0)

            x_offset = current_dim1 * 1.2
            y_center = 0

            for i in range(num_rotation * 2): # Simplified: each step is a press or rotate
                is_rotate_step = (i % 2 != 0)

                if is_rotate_step and i < num_rotation * 2 -1 : # Rotate
                     current_dim1, current_dim2 = current_dim2, current_dim1
                     orientations.append(1 - orientations[-1])
                     positions.append((x_offset, y_center - (current_dim2 / 2) + (dimensions[-1][1]/2) )) # Adjust y based on new height center
                     dimensions.append((current_dim1, current_dim2))
                     label = f"Rotate {i//2 + 1}"
                     x_offset += current_dim1 * 0.6 # Smaller gap for rotation
                elif not is_rotate_step: # Press
                     # Simulate reduction based on depth (very simplified)
                     reduction_factor = 1.0 - (depth / (current_dim2 if orientations[-1] == 0 else current_dim1) ) * 0.5 # Apply half depth effect visually
                     reduction_factor = max(0.5, reduction_factor) # Limit reduction visually
                     if orientations[-1] == 0: # Pressing height
                         current_dim2 *= reduction_factor
                         current_dim1 += feed * 0.1 # Simulate slight spread
                     else: # Pressing width
                         current_dim1 *= reduction_factor
                         current_dim2 += feed * 0.1 # Simulate slight spread

                     orientations.append(orientations[-1]) # Orientation doesn't change on press
                     positions.append((x_offset, positions[-1][1])) # Same y position as previous
                     dimensions.append((current_dim1, current_dim2))
                     label = f"Press {i//2 + 1}"
                     x_offset += current_dim1 * 1.2 # Gap for next step

                # Draw rectangle for the current step
                pos = positions[-1]
                dim = dimensions[-1]
                rect = plt.Rectangle(pos, dim[0], dim[1], fill=False, edgecolor='black', linewidth=1)
                self.cogging_ax.add_patch(rect)
                self.cogging_ax.text(pos[0] + dim[0]/2, pos[1] - 10, label, ha='center', fontsize=8)

                # Draw arrow to next step
                if i > 0:
                     start_pos = (positions[-2][0] + dimensions[-2][0], positions[-2][1] + dimensions[-2][1]/2)
                     end_pos = (positions[-1][0], positions[-1][1] + dimensions[-1][1]/2)
                     self.cogging_ax.annotate("", xy=end_pos, xytext=start_pos,
                                              arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))

            # Overlap/Detail view
            if show_overlap:
                 detail_x = x_offset + 50
                 detail_y = y_center
                 overlap_ratio = feed / initial_width # Example overlap calculation
                 self.cogging_ax.add_patch(plt.Rectangle((detail_x, detail_y), initial_width, initial_width*0.8, fill=False, edgecolor='gray', linewidth=1, linestyle='--'))
                 # Show feed
                 self.cogging_ax.arrow(detail_x, detail_y - 20, feed, 0, head_width=5, head_length=5, fc='red', ec='red', lw=1)
                 self.cogging_ax.text(detail_x + feed/2, detail_y - 30, f"Feed ({feed:.0f} mm)", color='red', ha='center', fontsize=8)
                 # Show depth
                 self.cogging_ax.arrow(detail_x + initial_width + 10, detail_y + initial_width*0.8, 0, -depth, head_width=5, head_length=5, fc='green', ec='green', lw=1)
                 self.cogging_ax.text(detail_x + initial_width + 15, detail_y + initial_width*0.8 - depth/2, f"Depth ({depth:.0f} mm)", color='green', va='center', ha='left', fontsize=8)
                 self.cogging_ax.text(detail_x + initial_width/2, detail_y + initial_width*0.8 + 10, "Detail View", ha='center', fontsize=9)


            self.cogging_ax.set_title(f"Simplified Cogging Steps (Rotations: {num_rotation})", fontsize=10)
            self.cogging_ax.set_aspect('equal', adjustable='box')
            self.cogging_ax.autoscale_view()
            self.cogging_ax.set_xticks([])
            self.cogging_ax.set_yticks([])
            self.cogging_figure.tight_layout()
            self.cogging_canvas.draw()

        except Exception as e:
            self.status_label.setText(f"Status: Error updating cogging diagram - {str(e)}")
            print(f"Cogging diagram error: {e}") # Debug print


    def simulate_cogging(self):
        """Apply cogging settings to the main tab inputs"""
        try:
            # Get input values from Cogging Tab
            feed = self.feed_amount_input.value()
            depth = self.depth_schedule_input.value()
            num_rotation = self.rotation_number_input.value()

            # Update the Main tab's input widgets
            self.input_param_widgets[0].setValue(feed)    # self.entry_feed
            self.input_param_widgets[1].setValue(depth)   # self.entry_depth
            self.input_param_widgets[2].setValue(num_rotation) # self.entry_rotation

            # Show a message
            QMessageBox.information(self.widget, "Settings Applied", "Cogging simulation settings have been applied to the Main tab inputs.")

            # Switch to main tab for user to run Pass Schedule
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            self.status_label.setText(f"Status: Error applying cogging settings - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Error applying cogging simulation settings: {str(e)}")


    def load_model_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self.widget, "Open Model File", "", "H5 Files (*.h5)")
        if filepath:
            try:
                self.model = load_model(filepath) # Use the dummy load_model
                self.status_label.setText("Status: Model loaded successfully (simulated)")
                QMessageBox.information(self.widget, "Success", "Model loaded successfully! (Simulated)")
            except Exception as e:
                self.status_label.setText(f"Status: Error loading model - {str(e)}")
                QMessageBox.critical(self.widget, "Error", f"Error loading model: {str(e)}")

    def load_data_file(self):
        # Check if default path exists, otherwise prompt user
        default_path = self.sample_data_path # Use the defined sample path
        initial_dir = os.path.dirname(default_path) if os.path.exists(default_path) else ""

        filepath, _ = QFileDialog.getOpenFileName(self.widget, "Open Data File", initial_dir, "Excel Files (*.xlsx)")
        if filepath:
            try:
                # Read data using defined column names
                rawdata = pd.read_excel(filepath, sheet_name='Sheet1', names=self.column_names, header=0) # Assume header is row 0
                self.dataset = rawdata.copy()
                self.dataset = self.dataset.round(5)

                # Validate columns
                if 'ENE' not in self.dataset.columns:
                    raise ValueError("Required column 'ENE' not found in the data file.")
                if len(self.dataset.columns) != len(self.column_names):
                     QMessageBox.warning(self.widget, "Warning", f"Number of columns in file ({len(self.dataset.columns)}) doesn't match expected ({len(self.column_names)}). Check file format.")
                     # return # Optional: stop loading if columns mismatch significantly

                self.X = self.dataset.drop('ENE', axis=1)
                # Ensure all columns used for X exist after drop
                expected_x_cols = self.column_names[:-1]
                missing_cols = [col for col in expected_x_cols if col not in self.X.columns]
                if missing_cols:
                     raise ValueError(f"Missing expected data columns after dropping ENE: {missing_cols}")


                y = self.dataset['ENE']

                # Fit scaler and selector (using dummy versions)
                self.scaler.fit(self.X)
                self.selector.fit(self.X, y)

                self.status_label.setText("Status: Data loaded successfully")
                QMessageBox.information(self.widget, "Success", "Data loaded successfully!")

                # Populate input widgets with mean values as starting point
                mean_values = self.X.mean()
                self.input_param_widgets[0].setValue(mean_values.get('Feed', 0))
                self.input_param_widgets[1].setValue(mean_values.get('DepthSchedule', 0))
                self.input_param_widgets[2].setValue(int(mean_values.get('NumRotation', 3))) # Use int for SpinBox

            except FileNotFoundError:
                 QMessageBox.critical(self.widget, "Error", f"Data file not found at: {filepath}")
            except ValueError as e:
                 QMessageBox.critical(self.widget, "Data Error", f"Error in data file structure: {str(e)}")
            except Exception as e:
                self.status_label.setText(f"Status: Error loading data - {str(e)}")
                QMessageBox.critical(self.widget, "Error", f"Error loading data: {str(e)}")


    def predict_y(self, x):
        """Predict ENE using the loaded model (or dummy)."""
        if self.model is None:
             raise ValueError("Model not loaded for prediction.")
        if len(x) != len(self.column_names) - 1:
             raise ValueError(f"Input vector length mismatch. Expected {len(self.column_names) - 1}, got {len(x)}")

        try:
            # Reshape x to 2D array for scaler/selector
            x_df = pd.DataFrame([x], columns=self.column_names[:-1]) # Create DataFrame for transform
            x_scaled = self.scaler.transform(x_df)
            x_selected = self.selector.transform(x_scaled)
            # Ensure input to predict is numpy array
            prediction = self.model.predict(np.array(x_selected), verbose=0)

            # Handle potential list output from dummy model
            if isinstance(prediction, list):
                result = prediction[0][0, 0] # Assuming first element is the one needed
            elif isinstance(prediction, np.ndarray):
                result = prediction[0, 0]
            else:
                result = float(prediction) # Fallback for single value

            return result

        except Exception as e:
            QMessageBox.critical(self.widget, "Prediction Error", f"Prediction error: {str(e)}")
            print(f"Prediction failed for input: {x}. Error: {e}")
            return 0.0 # Return a default value on error


    def minimize_y(self, x):
        """Function to be minimized (predict ENE)."""
        return self.predict_y(x)

    def constraint_fun(self, x):
        """Constraint function (example: ensure predicted ENE is positive)."""
        y = self.predict_y(x)
        return y  # Constraint: y >= 0 (represented as y itself for lb=0)


    def on_find_min(self):
        """Find the optimal pass schedule using optimization."""
        if self.model is None:
            self.status_label.setText("Status: No model loaded")
            QMessageBox.warning(self.widget, "Warning", "Please load model first!")
            return

        if self.X is None or self.dataset is None:
            self.status_label.setText("Status: No data loaded")
            QMessageBox.warning(self.widget, "Warning", "Please load data first!")
            return

        try:
            self.status_label.setText("Status: Calculating optimal pass schedule...")
            self.widget.repaint() # Force UI update

            # Get custom input values directly from widgets
            custom_inputs = [
                self.input_param_widgets[0].value(), # Feed
                self.input_param_widgets[1].value(), # Depth
                self.input_param_widgets[2].value()  # Rotation
            ]

            # Get mean values from loaded data (self.X) as starting point for optimization
            x0 = self.X.mean().values # Start with mean of all features in X

            # Override starting point with user inputs for the first 3 parameters
            for i, val in enumerate(custom_inputs):
                 # Check if the index is within bounds of x0
                 if i < len(x0):
                      x0[i] = val
                 else:
                      print(f"Warning: Index {i} out of bounds for x0 (length {len(x0)}). Cannot set custom input.")


            # --- Define Bounds ---
            # Get min/max bounds from the loaded data (self.X)
            # Ensure bounds cover the range of the loaded data
            lower_bounds = self.X.min().values
            upper_bounds = self.X.max().values

            # Ensure bounds are reasonable, prevent min == max for fixed inputs
            epsilon_bound = 1e-5
            for i in range(len(lower_bounds)):
                if abs(lower_bounds[i] - upper_bounds[i]) < epsilon_bound:
                    # Slightly widen bounds for fixed inputs if needed by optimizer
                    lower_bounds[i] -= epsilon_bound
                    upper_bounds[i] += epsilon_bound

            # Apply user inputs as fixed values by setting tight bounds
            for i, val in enumerate(custom_inputs):
                 if i < len(lower_bounds):
                    lower_bounds[i] = val - epsilon_bound # Tight bounds around user value
                    upper_bounds[i] = val + epsilon_bound
                 else:
                      print(f"Warning: Index {i} out of bounds for bounds arrays. Cannot set custom bounds.")

            bounds = Bounds(lower_bounds, upper_bounds)

            # --- Define Constraints (Example: predicted ENE must be non-negative) ---
            # The constraint function returns the value to be constrained.
            # lb <= constraint_fun(x) <= ub
            constraint = NonlinearConstraint(self.constraint_fun, lb=0.0, ub=np.inf) # ENE >= 0

            # --- Perform Optimization ---
            # Using the dummy 'minimize' function for demonstration
            result = minimize(self.minimize_y, x0, method='trust-constr',
                             bounds=bounds, constraints=[constraint], tol=1e-1) # Pass constraint as a list

            x_min = result.x # The optimized parameters [Feed, Depth, Rot, Pass1, ..., Pass7]

            # --- Update UI with Results ---
            # Update pass values in the read-only LineEdits
            pass_values = x_min[3:] # Passes start from index 3
            for i, value in enumerate(pass_values):
                if i < len(self.pass_entries):
                    self.pass_entries[i].setText(f"{value:.5f}") # Display optimized pass values

            # --- Calculate and Display Forging Ratios ---
            try:
                radius_value = float(self.radius_entry.text())
                if radius_value <= 0: raise ValueError("Initial radius must be positive.")
                initial_area = np.pi * (radius_value**2) / 4 # Assuming circular initial cross-section? Or is it side length? Let's assume side length of square.
                initial_area = radius_value**2 # Assuming square billet side length

                # Pass values represent the dimension after the pass
                pass_dims = x_min[3:10] # Pass1 to Pass7 values

                # Forging ratio calculation needs clarification. Assuming it relates areas.
                # Ratio = sqrt(Area_before / Area_after). If pass_dims are side lengths: Area = dim^2
                # For Pass1: Ratio = sqrt(initial_area / pass_dims[0]^2) = initial_radius / pass_dims[0]
                # For Pass_i: Ratio = sqrt(pass_dims[i-1]^2 / pass_dims[i]^2) = pass_dims[i-1] / pass_dims[i]

                forging_ratios = []
                # Pass 1 ratio
                if pass_dims[0] > 1e-6:
                     ratio1 = radius_value / pass_dims[0]
                     forging_ratios.append(ratio1)
                     self.forging_labels[0].setText(f"{ratio1:.2f} x {ratio1:.2f}") # Assuming square reduction
                else:
                     self.forging_labels[0].setText("Error")

                # Passes 2 to 7 ratios
                for i in range(1, 7):
                    if i < len(self.forging_labels) and pass_dims[i] > 1e-6:
                        ratio_i = pass_dims[i-1] / pass_dims[i]
                        forging_ratios.append(ratio_i)
                        self.forging_labels[i].setText(f"{ratio_i:.2f} x {ratio_i:.2f}")
                    elif i < len(self.forging_labels):
                        self.forging_labels[i].setText("Error")

            except ValueError as e:
                QMessageBox.warning(self.widget, "Input Error", f"Invalid initial radius: {e}")
                for label in self.forging_labels: label.setText("N/A")
            except IndexError:
                 QMessageBox.warning(self.widget, "Calculation Error", "Mismatch in pass data length for forging ratios.")
                 for label in self.forging_labels: label.setText("N/A")


            # --- Calculate and Display Length Changes ---
            try:
                 initial_length = float(self.initial_length_entry.text())
                 if initial_length <= 0: raise ValueError("Initial length must be positive.")
                 pass_dims = x_min[3:10] # Pass1 to Pass7 values

                 # Assuming volume conservation: A_initial * L_initial = A_pass * L_pass
                 # L_pass = L_initial * (A_initial / A_pass)
                 # If pass_dims are side lengths: A_pass = pass_dims^2
                 # L_pass1 = L_initial * (initial_radius^2 / pass_dims[0]^2)
                 # L_pass_i = L_pass_(i-1) * (pass_dims[i-1]^2 / pass_dims[i]^2)

                 length_changes = []
                 current_length = initial_length
                 initial_area = float(self.radius_entry.text())**2 # Area based on initial side length

                 # Pass 1 length
                 if pass_dims[0] > 1e-6:
                      area1 = pass_dims[0]**2
                      length1 = current_length * (initial_area / area1)
                      length_changes.append(length1)
                      self.length_change_labels[0].setText(f"{length1:.0f}")
                      current_length = length1
                 else:
                      self.length_change_labels[0].setText("Error")


                 # Passes 2 to 7 lengths
                 for i in range(1, 7):
                      if i < len(self.length_change_labels) and pass_dims[i] > 1e-6:
                           area_prev = pass_dims[i-1]**2
                           area_curr = pass_dims[i]**2
                           length_i = current_length * (area_prev / area_curr)
                           length_changes.append(length_i)
                           self.length_change_labels[i].setText(f"{length_i:.0f}")
                           current_length = length_i
                      elif i < len(self.length_change_labels):
                           self.length_change_labels[i].setText("Error")

                 # --- Update Cutting Calculation ---
                 if length_changes: # Only update if length calculation was successful
                      self.update_cutted_length_labels(length_changes)
                 else:
                     for label in self.cutted_length_labels: label.setText("N/A")
                     self.yellow_changes_label.setText("Total cuts: N/A")


            except ValueError as e:
                 QMessageBox.warning(self.widget, "Input Error", f"Invalid initial length: {e}")
                 for label in self.length_change_labels: label.setText("N/A")
                 for label in self.cutted_length_labels: label.setText("N/A")
                 self.yellow_changes_label.setText("Total cuts: N/A")
            except IndexError:
                 QMessageBox.warning(self.widget, "Calculation Error", "Mismatch in pass data length for length changes.")
                 for label in self.length_change_labels: label.setText("N/A")
                 for label in self.cutted_length_labels: label.setText("N/A")
                 self.yellow_changes_label.setText("Total cuts: N/A")


            # --- Final Steps ---
            # Save result data for visualization
            self.results_data = {
                'forging_ratios': forging_ratios if 'forging_ratios' in locals() else [],
                'length_changes': length_changes if 'length_changes' in locals() else [],
                'x_min': x_min.tolist(),
                'initial_radius': float(self.radius_entry.text()), # Store actual input
                'initial_length': float(self.initial_length_entry.text()), # Store actual input
                'cutting_length': float(self.cutting_length_entry.text()), # Store actual input
                # Add cut details if calculated
                'cutted_lengths': self.results_data.get('cutted_lengths', []),
                'cutted_quantities': self.results_data.get('cutted_quantities', [])
            }

            # Enable visualization button
            self.show_results_button.setEnabled(True)
            self.show_results_button.setStyleSheet(button_style + "background-color: #ea4335;") # Make it red

            self.status_label.setText("Status: Pass schedule calculation completed successfully")
            QMessageBox.information(self.widget, "Success", "Pass schedule calculated successfully!")

        except ValueError as e:
             # Handle specific errors like invalid inputs or data issues
             self.status_label.setText(f"Status: Calculation Error - {str(e)}")
             QMessageBox.critical(self.widget, "Calculation Error", f"Error during calculation setup: {str(e)}")
        except Exception as e:
            self.status_label.setText(f"Status: Error in calculation - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"An unexpected error occurred during calculation: {str(e)}")
            import traceback
            print(traceback.format_exc()) # Print detailed traceback for debugging


    def update_cutted_length_labels(self, length_changes):
        """Calculate and display how billets are cut after each pass."""
        try:
            cutting_length = float(self.cutting_length_entry.text())
            if cutting_length <= 0:
                raise ValueError("Cutting length must be positive.")

            all_cut_lengths = []
            all_cut_quantities = []
            total_cuts_across_passes = 0
            previous_total_pieces = 1 # Start with 1 initial billet

            for idx, current_pass_length in enumerate(length_changes):
                # Simulate cutting the billet(s) from the *previous* state to achieve the current length
                # This interpretation might be incorrect. A more common scenario is cutting the *final* length.
                # Let's assume we cut the billet *after* it reaches current_pass_length.

                num_pieces = previous_total_pieces # Start with pieces from previous pass
                length_to_cut = current_pass_length
                cuts_this_pass = 0

                # If a single piece is longer than the cutting length, it needs cutting.
                # We simulate cutting all pieces originating from the initial billet.
                temp_pieces_after_cut = []
                final_length_after_cut = length_to_cut # Assume no cutting if short enough
                num_final_pieces = num_pieces

                if length_to_cut > cutting_length:
                     # Calculate how many pieces of 'cutting_length' can be obtained
                     num_cuts_per_piece = int(np.floor(length_to_cut / cutting_length))
                     remainder_length = length_to_cut % cutting_length
                     final_length_after_cut = cutting_length # The standard cut length
                     num_final_pieces = num_pieces * num_cuts_per_piece
                     cuts_this_pass = (num_cuts_per_piece -1) * num_pieces # Cuts needed per original piece

                     # Optional: handle remainder if needed (e.g., add as smaller pieces)
                     if remainder_length > 1e-2: # If remainder is significant
                          num_final_pieces += num_pieces # Add the remainder pieces

                total_cuts_across_passes += cuts_this_pass

                # Store results for this pass
                all_cut_lengths.append(final_length_after_cut)
                all_cut_quantities.append(num_final_pieces)


                # Update label for this pass
                label_text = f"{final_length_after_cut:.0f} ({num_final_pieces})"
                self.cutted_length_labels[idx].setText(label_text)

                # Highlight if the number of pieces increased (indicating cuts happened)
                bg_color = 'background-color: yellow;' if cuts_this_pass > 0 else 'background-color: #f8f9fa;'
                self.cutted_length_labels[idx].setStyleSheet(f"border: 1px solid #dddddd; color: #555555; {bg_color} padding: 5px; font-weight:bold;")

                previous_total_pieces = num_final_pieces # Update for the next pass

            self.yellow_changes_label.setText(f"Total cuts: {total_cuts_across_passes}")

            # Save results for potential visualization
            self.results_data['cutted_lengths'] = all_cut_lengths
            self.results_data['cutted_quantities'] = all_cut_quantities
            self.results_data['total_cuts'] = total_cuts_across_passes


        except ValueError as e:
            self.status_label.setText(f"Status: Error in cutting calculation - {str(e)}")
            QMessageBox.warning(self.widget, "Warning", f"Invalid cutting length: {str(e)}")
            for label in self.cutted_length_labels: label.setText("Error")
            self.yellow_changes_label.setText("Total cuts: Error")
        except Exception as e:
             self.status_label.setText(f"Status: Error in cutting calculation - {str(e)}")
             QMessageBox.warning(self.widget, "Warning", f"Error during cutting calculation: {str(e)}")
             for label in self.cutted_length_labels: label.setText("Error")
             self.yellow_changes_label.setText("Total cuts: Error")


    def show_visualization(self):
        """Show visualization results"""
        if not self.results_data:
            self.status_label.setText("Status: No results to visualize")
            QMessageBox.warning(self.widget, "Warning", "Please perform calculation first!")
            return

        try:
            # Call visualization manager in the parent (main window)
            if hasattr(self.parent, 'visualization_manager') and self.parent.visualization_manager:
                self.parent.visualization_manager.display_nsm_billetsizing_results(self.results_data)
                self.status_label.setText("Status: Results visualized successfully")
            else:
                self.status_label.setText("Status: Visualization manager not found or not ready")
                QMessageBox.warning(self.widget, "Warning", "Visualization manager is not available.")
        except Exception as e:
            self.status_label.setText(f"Status: Error during visualization - {str(e)}")
            QMessageBox.critical(self.widget, "Error", f"Error displaying visualization: {str(e)}")
            import traceback
            print(traceback.format_exc()) # Print detailed traceback for debugging
