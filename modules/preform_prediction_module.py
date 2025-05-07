import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout, 
                             QFileDialog, QMessageBox, QGroupBox, QFormLayout, QProgressBar,
                             QHBoxLayout, QComboBox, QSpinBox, QCheckBox, QTabWidget,
                             QDoubleSpinBox, QRadioButton, QButtonGroup, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

# TensorFlow diagnostics
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Sys.path: {sys.path}")
print("Trying to import tensorflow...")

try:
    import tensorflow as tf
    print(f"SUCCESS: TensorFlow {tf.__version__} successfully imported.")
    print(f"TensorFlow path: {tf.__file__}")
    from tensorflow.keras import backend as K
    print("Keras backend imported")
    tf_available = True
except Exception as e:
    print(f"ERROR: TensorFlow import failed: {e}")
    print(f"Error type: {type(e).__name__}")
    print("Using dummy implementation.")
    tf_available = False
    
    class tf:
        class keras:
            class models:
                @staticmethod
                def load_model(path, custom_objects=None):
                    print(f"Dummy model loaded from {path}")
                    class DummyModel:
                        def predict(self, x, verbose=0):
                            print(f"Dummy prediction with shape {x.shape if hasattr(x, 'shape') else 'unknown'}")
                            # Return a tuple of two arrays (voxel model, folding prediction)
                            dummy_shape = x.shape if hasattr(x, 'shape') else (64, 64, 64)
                            return np.random.random(dummy_shape), np.random.random(dummy_shape)
                    return DummyModel()
            losses = type('obj', (object,), {
                'binary_crossentropy': lambda y_true, y_pred: 0.0
            })
    class K:
        @staticmethod
        def flatten(x): return x.flatten() if hasattr(x, 'flatten') else x
        @staticmethod
        def cast(x, dtype): return x
        @staticmethod
        def sum(x): return np.sum(x) if hasattr(x, 'sum') else x

# Try to import PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvista.plotting.matplotlib_plotting import _qt_figure_to_mpl_figure
    pyvista_available = True
    print("SUCCESS: PyVista successfully imported.")
except Exception as e:
    print(f"ERROR: PyVista import failed: {e}")
    pyvista_available = False

# Try to import Trimesh for STL operations
try:
    import trimesh
    trimesh_available = True
    print("SUCCESS: Trimesh successfully imported.")
except Exception as e:
    print(f"ERROR: Trimesh import failed: {e}")
    trimesh_available = False

# Custom loss functions
def voxel_loss(y_true, y_pred): 
    if tf_available:
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.0

def folding_loss(y_true, y_pred, alpha=0.25, gamma=2.0): 
    if tf_available:
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.0

def weighted_folding_loss(y_true, y_pred): 
    if tf_available:
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.0

def dice_coef(y_true, y_pred):
    if tf_available:
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0

# Worker thread for running background tasks
class Worker(QThread):
    progress_updated = pyqtSignal(int)
    task_completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            result = self.task(*self.args, progress_callback=self.progress_updated, **self.kwargs)
            self.task_completed.emit(result)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

class PreformPredictionModule:
    def __init__(self):
        self.name = "Preform Prediction"
        self.model = None
        self.min_coords = None
        self.max_coords = None
        self.npy_path = 'train_npy'
        self.result_path = './result'
        self.data_path = 'train_data'
        self.model_name = 'unet_model'
        self.worker = None
        self.preview_worker = None
        self.current_voxel_data = None
        self.current_stl_mesh = None
        self.predicted_result = None
        
        # Create necessary directories
        os.makedirs(self.npy_path, exist_ok=True)
        os.makedirs(os.path.join(self.npy_path, 'train_y'), exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

    def get_name(self):
        return self.name

    def create_widget(self, parent):
        # Store reference to parent
        self.parent = parent
        self.widget = QWidget(parent)
        
        # Define styles
        title_style = """
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
            }
        """
        
        button_style = """
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
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
        
        # Main layout
        main_layout = QVBoxLayout(self.widget)
        
        # Status area
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #4a86e8; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.model_status_label = QLabel("Model: Not loaded")
        self.model_status_label.setStyleSheet("color: #666666;")
        status_layout.addWidget(self.model_status_label)
        
        main_layout.addLayout(status_layout)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_model_tab()
        self.create_data_tab()
        self.create_prediction_tab()
        self.create_visualization_tab()
        
        # Refresh NPY files list
        self.refresh_npy_files()
        
        return self.widget
    
    def create_model_tab(self):
        """Create the Model tab for loading and managing the UNet model"""
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        # Model management group
        model_group = QGroupBox("Model Management")
        model_group.setStyleSheet("""
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
        """)
        model_form = QFormLayout(model_group)
        
        # Load model button
        load_model_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Load Model (.h5)")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
        """)
        load_model_layout.addWidget(self.load_model_button)
        
        # Model path display
        self.model_path_label = QLabel("No model loaded")
        load_model_layout.addWidget(self.model_path_label, 1)
        
        model_form.addRow("Load Model:", load_model_layout)
        
        # Model details
        self.model_details_label = QLabel("Model not loaded")
        model_form.addRow("Model Details:", self.model_details_label)
        
        # Model parameters
        parameters_layout = QHBoxLayout()
        
        self.voxel_resolution_spin = QSpinBox()
        self.voxel_resolution_spin.setRange(32, 256)
        self.voxel_resolution_spin.setValue(128)  # Changed default to 128
        self.voxel_resolution_spin.setSingleStep(16)
        self.voxel_resolution_spin.setSuffix(" voxels")
        parameters_layout.addWidget(QLabel("Voxel Resolution:"))
        parameters_layout.addWidget(self.voxel_resolution_spin)
        
        model_form.addRow("Model Parameters:", parameters_layout)
        
        # Add model group to tab
        model_layout.addWidget(model_group)
        
        # Model operations group
        operations_group = QGroupBox("Model Operations")
        operations_group.setStyleSheet("""
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
        """)
        operations_layout = QVBoxLayout(operations_group)
        
        # Operation buttons
        buttons_layout = QHBoxLayout()
        
        self.test_model_button = QPushButton("Test Model")
        self.test_model_button.clicked.connect(self.test_model)
        self.test_model_button.setStyleSheet("""
            QPushButton {
                background-color: #34a853;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2d9248;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.test_model_button.setEnabled(False)
        buttons_layout.addWidget(self.test_model_button)
        
        operations_layout.addLayout(buttons_layout)
        
        # Operation results
        self.operation_results_label = QLabel("No operations performed")
        operations_layout.addWidget(self.operation_results_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        operations_layout.addWidget(self.progress_bar)
        
        # Add operations group to tab
        model_layout.addWidget(operations_group)
        
        # Add spacer at the bottom
        model_layout.addStretch()
        
        # Add the tab
        self.tabs.addTab(model_tab, "Model")
    
    def create_data_tab(self):
        """Create the Data tab for loading and managing input data"""
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # Input data group
        input_group = QGroupBox("Input Data")
        input_group.setStyleSheet("""
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
        """)
        input_form = QFormLayout(input_group)
        
        # Load STL button
        load_stl_layout = QHBoxLayout()
        self.load_stl_button = QPushButton("Load STL File")
        self.load_stl_button.clicked.connect(self.load_stl_file)
        self.load_stl_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
        """)
        load_stl_layout.addWidget(self.load_stl_button)
        
        # STL path display
        self.stl_path_label = QLabel("No STL file loaded")
        load_stl_layout.addWidget(self.stl_path_label, 1)
        
        input_form.addRow("Load STL:", load_stl_layout)
        
        # Load NPY button
        load_npy_layout = QHBoxLayout()
        self.load_npy_button = QPushButton("Load NPY File")
        self.load_npy_button.clicked.connect(self.load_npy_file)
        self.load_npy_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
        """)
        load_npy_layout.addWidget(self.load_npy_button)
        
        # NPY path display
        self.npy_path_label = QLabel("No NPY file loaded")
        load_npy_layout.addWidget(self.npy_path_label, 1)
        
        input_form.addRow("Load NPY:", load_npy_layout)
        
        # Available files selector
        files_layout = QHBoxLayout()
        
        self.available_files_combo = QComboBox()
        self.available_files_combo.setMinimumWidth(300)
        files_layout.addWidget(self.available_files_combo)
        
        self.refresh_files_button = QPushButton("Refresh")
        self.refresh_files_button.clicked.connect(self.refresh_npy_files)
        self.refresh_files_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        files_layout.addWidget(self.refresh_files_button)
        
        input_form.addRow("Available Files:", files_layout)
        
        # Data preview
        self.data_preview_label = QLabel("No data to preview")
        self.data_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_form.addRow("Data Preview:", self.data_preview_label)
        
        # Add input group to tab
        data_layout.addWidget(input_group)
        
        # Data conversion group
        conversion_group = QGroupBox("Data Conversion")
        conversion_group.setStyleSheet("""
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
        """)
        conversion_layout = QVBoxLayout(conversion_group)
        
        # Conversion parameters
        params_layout = QFormLayout()
        
        # Voxel size
        voxel_layout = QHBoxLayout()
        self.voxel_size_spin = QDoubleSpinBox()
        self.voxel_size_spin.setRange(0.1, 10.0)
        self.voxel_size_spin.setValue(1.0)
        self.voxel_size_spin.setSingleStep(0.1)
        self.voxel_size_spin.setSuffix(" mm")
        voxel_layout.addWidget(self.voxel_size_spin)
        
        params_layout.addRow("Voxel Size:", voxel_layout)
        
        # Resolution
        resolution_layout = QHBoxLayout()
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(16, 256)
        self.resolution_spin.setValue(128)  # Changed default to 128
        self.resolution_spin.setSingleStep(8)
        resolution_layout.addWidget(self.resolution_spin)
        
        params_layout.addRow("Resolution:", resolution_layout)
        
        conversion_layout.addLayout(params_layout)
        
        # Conversion buttons
        conversion_buttons = QHBoxLayout()
        
        self.stl_to_npy_button = QPushButton("STL → NPY")
        self.stl_to_npy_button.clicked.connect(self.convert_stl_to_npy)
        self.stl_to_npy_button.setStyleSheet("""
            QPushButton {
                background-color: #34a853;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2d9248;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stl_to_npy_button.setEnabled(False)
        conversion_buttons.addWidget(self.stl_to_npy_button)
        
        self.npy_to_stl_button = QPushButton("NPY → STL")
        self.npy_to_stl_button.clicked.connect(self.convert_npy_to_stl)
        self.npy_to_stl_button.setStyleSheet("""
            QPushButton {
                background-color: #ea4335;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.npy_to_stl_button.setEnabled(False)
        conversion_buttons.addWidget(self.npy_to_stl_button)
        
        conversion_layout.addLayout(conversion_buttons)
        
        # Conversion progress
        self.conversion_progress = QProgressBar()
        self.conversion_progress.setRange(0, 100)
        self.conversion_progress.setValue(0)
        conversion_layout.addWidget(self.conversion_progress)
        
        # Add conversion group to tab
        data_layout.addWidget(conversion_group)
        
        # Add spacer at the bottom
        data_layout.addStretch()
        
        # Add the tab
        self.tabs.addTab(data_tab, "Data")
    
    def create_prediction_tab(self):
        """Create the Prediction tab for running model predictions"""
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)
        
        # Prediction setup group
        setup_group = QGroupBox("Prediction Setup")
        setup_group.setStyleSheet("""
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
        """)
        setup_form = QFormLayout(setup_group)
        
        # Input selection
        input_layout = QHBoxLayout()
        
        # Input selection options
        input_label = QLabel("Input Source:")
        self.input_file_radio = QRadioButton("File")
        self.input_file_radio.setChecked(True)
        self.input_file_radio.toggled.connect(self.update_prediction_inputs)
        self.input_live_radio = QRadioButton("Current Data")
        self.input_live_radio.toggled.connect(self.update_prediction_inputs)
        
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_file_radio)
        input_layout.addWidget(self.input_live_radio)
        input_layout.addStretch()
        
        setup_form.addRow(input_layout)
        
        # Input file selection
        self.input_file_layout = QHBoxLayout()
        self.input_file_combo = QComboBox()
        self.input_file_combo.setMinimumWidth(300)
        self.input_file_layout.addWidget(self.input_file_combo)
        
        self.input_refresh_button = QPushButton("Refresh")
        self.input_refresh_button.clicked.connect(self.refresh_npy_files)
        self.input_refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.input_file_layout.addWidget(self.input_refresh_button)
        
        setup_form.addRow("Input File:", self.input_file_layout)
        
        # Prediction options
        options_layout = QHBoxLayout()
        
        self.save_result_check = QCheckBox("Save Results")
        self.save_result_check.setChecked(True)
        options_layout.addWidget(self.save_result_check)
        
        self.visualize_result_check = QCheckBox("Visualize Results")
        self.visualize_result_check.setChecked(True)
        options_layout.addWidget(self.visualize_result_check)
        
        self.convert_to_stl_check = QCheckBox("Convert to STL")
        self.convert_to_stl_check.setChecked(True)
        options_layout.addWidget(self.convert_to_stl_check)
        
        setup_form.addRow("Options:", options_layout)
        
        # Add setup group to tab
        prediction_layout.addWidget(setup_group)
        
        # Prediction execution group
        execution_group = QGroupBox("Prediction Execution")
        execution_group.setStyleSheet("""
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
        """)
        execution_layout = QVBoxLayout(execution_group)
        
        # Start prediction button
        self.start_prediction_button = QPushButton("Start Prediction")
        self.start_prediction_button.clicked.connect(self.start_prediction)
        self.start_prediction_button.setStyleSheet("""
            QPushButton {
                background-color: #34a853;
                color: white;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #2d9248;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_prediction_button.setEnabled(False)
        execution_layout.addWidget(self.start_prediction_button)
        
        # Prediction progress
        self.prediction_progress = QProgressBar()
        self.prediction_progress.setRange(0, 100)
        self.prediction_progress.setValue(0)
        execution_layout.addWidget(self.prediction_progress)
        
        # Prediction info
        self.prediction_info_label = QLabel("Ready for prediction")
        self.prediction_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        execution_layout.addWidget(self.prediction_info_label)
        
        # Add execution group to tab
        prediction_layout.addWidget(execution_group)
        
        # Result preview group
        result_group = QGroupBox("Result Preview")
        result_group.setStyleSheet("""
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
        """)
        result_layout = QVBoxLayout(result_group)
        
        # Result preview label
        self.result_preview_label = QLabel("No prediction results yet")
        self.result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview_label.setMinimumHeight(200)
        result_layout.addWidget(self.result_preview_label)
        
        # Add result group to tab
        prediction_layout.addWidget(result_group)
        
        # Add spacer at the bottom
        prediction_layout.addStretch()
        
        # Add the tab
        self.tabs.addTab(prediction_tab, "Prediction")
        
    def create_visualization_tab(self):
        """Create the Visualization tab for 3D visualization"""
        visualization_tab = QWidget()
        visualization_layout = QVBoxLayout(visualization_tab)
        
        # Controls group
        controls_group = QGroupBox("Visualization Controls")
        controls_group.setStyleSheet("""
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
        """)
        controls_layout = QHBoxLayout(controls_group)
        
        # Data selection
        data_selection_layout = QFormLayout()
        
        self.visualize_combo = QComboBox()
        self.visualize_combo.addItems(["Input Data", "Prediction Result", "Both (Side by Side)"])
        self.visualize_combo.setCurrentIndex(2)
        data_selection_layout.addRow("View:", self.visualize_combo)
        
        controls_layout.addLayout(data_selection_layout)
        
        # Visualization options
        viz_options_layout = QFormLayout()
        
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Default", "Rainbow", "Metallic", "Heat"])
        viz_options_layout.addRow("Color Scheme:", self.color_combo)
        
        self.background_combo = QComboBox()
        self.background_combo.addItems(["White", "Black", "Gray", "Gradient"])
        viz_options_layout.addRow("Background:", self.background_combo)
        
        controls_layout.addLayout(viz_options_layout)
        
        # Update view button
        self.update_view_button = QPushButton("Update View")
        self.update_view_button.clicked.connect(self.update_visualization)
        self.update_view_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
            QPushButton:disabled {
                background-color: background-color: #cccccc;
                color: #666666;
            }
        """)
        controls_layout.addWidget(self.update_view_button)
        
        # Add controls group to tab
        visualization_layout.addWidget(controls_group)
        
        # Create visualization container
        self.visualization_container = QWidget()
        self.visualization_layout = QVBoxLayout(self.visualization_container)
        
        # Create placeholder for 3D visualization
        self.visualization_placeholder = QLabel("3D visualization will appear here")
        self.visualization_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_placeholder.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 1px dashed #cccccc;
                border-radius: 5px;
                padding: 20px;
                font-size: 16px;
                color: #666666;
            }
        """)
        self.visualization_placeholder.setMinimumHeight(400)
        self.visualization_layout.addWidget(self.visualization_placeholder)
        
        # Add visualization container to tab
        visualization_layout.addWidget(self.visualization_container)
        
        # Export options group
        export_group = QGroupBox("Export Options")
        export_group.setStyleSheet("""
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
        """)
        export_layout = QHBoxLayout(export_group)
        
        # Export buttons
        self.export_stl_button = QPushButton("Export to STL")
        self.export_stl_button.clicked.connect(self.export_to_stl)
        self.export_stl_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.export_stl_button.setEnabled(False)
        export_layout.addWidget(self.export_stl_button)
        
        self.export_screenshot_button = QPushButton("Export Screenshot")
        self.export_screenshot_button.clicked.connect(self.export_screenshot)
        self.export_screenshot_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.export_screenshot_button.setEnabled(False)
        export_layout.addWidget(self.export_screenshot_button)
        
        # Show in visualization manager button
        self.show_manager_button = QPushButton("Show in Visualization Manager")
        self.show_manager_button.clicked.connect(self.show_in_visualization_manager)
        self.show_manager_button.setStyleSheet("""
            QPushButton {
                background-color: #34a853;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2d9248;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.show_manager_button.setEnabled(False)
        export_layout.addWidget(self.show_manager_button)
        
        # Add export group to tab
        visualization_layout.addWidget(export_group)
        
        # Add the tab
        self.tabs.addTab(visualization_tab, "Visualization")
    
    def refresh_npy_files(self):
        """Refresh the list of NPY files"""
        print("Refreshing NPY files list...")
        
        try:
            # Get NPY files from the train_npy directory
            self.npy_files = [f for f in os.listdir(self.npy_path) if f.endswith('.npy')]
            
            # Update combo boxes
            if hasattr(self, 'available_files_combo'):
                self.available_files_combo.clear()
                if self.npy_files:
                    self.available_files_combo.addItems(self.npy_files)
            
            if hasattr(self, 'input_file_combo'):
                self.input_file_combo.clear()
                if self.npy_files:
                    self.input_file_combo.addItems(self.npy_files)
            
            print(f"Found {len(self.npy_files)} NPY files.")
            self.status_label.setText(f"Status: Found {len(self.npy_files)} NPY files")
            
        except Exception as e:
            print(f"Error refreshing NPY files: {str(e)}")
            self.status_label.setText(f"Status: Error refreshing NPY files - {str(e)}")
    
    def load_model(self):
        """Load a Keras model from file"""
        file_path, _ = QFileDialog.getOpenFileName(self.widget, "Select Keras Model File", "", "Keras Model (*.h5)")
        
        if not file_path:
            return
        
        self.status_label.setText("Status: Loading model...")
        
        try:
            # Update UI
            self.model_path_label.setText(os.path.basename(file_path))
            
            # Load the model
            self.model = tf.keras.models.load_model(
                file_path,
                custom_objects={
                    'voxel_loss': voxel_loss,
                    'folding_loss': folding_loss,
                    'weighted_folding_loss': weighted_folding_loss,
                    'dice_coef': dice_coef
                }
            )
            
            # Enable model-dependent buttons
            self.test_model_button.setEnabled(True)
            self.start_prediction_button.setEnabled(True)
            
            # Update status
            self.model_status_label.setText("Model: Loaded")
            self.status_label.setText(f"Status: Model loaded from {os.path.basename(file_path)}")
            
            # Update model details
            if self.model:
                # Get model summary
                input_shape = self.model.input_shape[1:]  # Remove batch dimension
                output_shape = self.model.output_shape[0][1:] if isinstance(self.model.output_shape, list) else self.model.output_shape[1:]
                total_params = self.model.count_params()
                
                self.model_details_label.setText(
                    f"Input Shape: {input_shape}\n"
                    f"Output Shape: {output_shape}\n"
                    f"Total Parameters: {total_params:,}"
                )
                
                # Update resolution control to match model input
                if len(input_shape) > 2:
                    resolution = input_shape[0]
                    self.voxel_resolution_spin.setValue(resolution)
                    self.resolution_spin.setValue(resolution)
            
            print(f"Model loaded successfully from {file_path}")
            
        except Exception as e:
            print(f"Model load failed: {e}")
            QMessageBox.critical(self.widget, "Model Load Error", f"Failed to load model:\n{str(e)}")
            self.model_status_label.setText("Model: Load failed")
            self.status_label.setText("Status: Model load failed")
            self.model = None
    
    def test_model(self):
        """Test the loaded model with a sample input"""
        if not self.model:
            QMessageBox.warning(self.widget, "Test Error", "No model loaded.")
            return
        
        try:
            # Create a simple test input
            resolution = self.voxel_resolution_spin.value()
            test_input = np.zeros((1, resolution, resolution, resolution, 1), dtype=np.float32)
            
            # Add a simple cube in the middle
            center = resolution // 2
            size = resolution // 4
            test_input[0, 
                     center-size:center+size, 
                     center-size:center+size, 
                     center-size:center+size, 0] = 1.0
            
            # Run prediction
            self.operation_results_label.setText("Running model test...")
            self.progress_bar.setValue(10)
            
            # Run prediction in background
            self.worker = Worker(self.run_test_prediction, test_input)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.task_completed.connect(self.test_completed)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self.widget, "Test Error", f"Test failed:\n{str(e)}")
            self.operation_results_label.setText(f"Test failed: {str(e)}")
    
    def run_test_prediction(self, test_input, progress_callback=None):
        """Run a test prediction with the model"""
        if progress_callback:
            progress_callback.emit(25)
        
        # Run prediction
        prediction = self.model.predict(test_input, verbose=1)
        
        if progress_callback:
            progress_callback.emit(75)
        
        # Return result
        return prediction
    
    def test_completed(self, result):
        """Handle completion of the test prediction"""
        # Update progress
        self.progress_bar.setValue(100)
        
        # Update results
        if isinstance(result, list) or isinstance(result, tuple):
            output_shape = [r.shape for r in result]
            self.operation_results_label.setText(f"Test successful. Output shapes: {output_shape}")
        else:
            self.operation_results_label.setText(f"Test successful. Output shape: {result.shape}")
        
        # Show message
        QMessageBox.information(self.widget, "Test Successful", "Model test completed successfully.")
    
    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
        self.conversion_progress.setValue(value)
        self.prediction_progress.setValue(value)
    
    def handle_error(self, error_message):
        """Handle errors from worker threads"""
        QMessageBox.critical(self.widget, "Error", f"Operation failed:\n{error_message}")
        self.status_label.setText(f"Status: Error - {error_message}")
        self.progress_bar.setValue(0)
        self.conversion_progress.setValue(0)
        self.prediction_progress.setValue(0)
    
    def load_stl_file(self):
        """Load an STL file"""
        file_path, _ = QFileDialog.getOpenFileName(self.widget, "Select STL File", "", "STL Files (*.stl)")
        
        if not file_path:
            return
        
        self.status_label.setText(f"Status: Loading STL file {os.path.basename(file_path)}...")
        
        try:
            if not trimesh_available:
                raise ImportError("The trimesh library is not available. Cannot load STL file.")
            
            # Load the STL file using trimesh
            mesh = trimesh.load(file_path)
            
            # Update UI
            self.stl_path_label.setText(os.path.basename(file_path))
            self.data_preview_label.setText(
                f"Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces\n"
                f"Bounds: {mesh.bounds}"
            )
            
            # Store mesh
            self.current_stl_mesh = mesh
            
            # Enable conversion button
            self.stl_to_npy_button.setEnabled(True)
            
            # Show in visualization if available
            if hasattr(self.parent, 'visualization_manager') and pyvista_available:
                # Convert trimesh to pyvista
                pv_mesh = pv.PolyData(mesh.vertices, mesh.faces)
                self.parent.visualization_manager.display_stl_preview(pv_mesh)
            
            self.status_label.setText(f"Status: STL file {os.path.basename(file_path)} loaded")
            
        except Exception as e:
            QMessageBox.critical(self.widget, "STL Load Error", f"Failed to load STL file:\n{str(e)}")
            self.status_label.setText(f"Status: STL load failed - {str(e)}")
    
    def load_npy_file(self):
        """Load an NPY file"""
        file_path, _ = QFileDialog.getOpenFileName(self.widget, "Select NPY File", "", "NPY Files (*.npy)")
        
        if not file_path:
            return
        
        self.status_label.setText(f"Status: Loading NPY file {os.path.basename(file_path)}...")
        
        try:
            # Load the NPY file
            data = np.load(file_path)
            
            # Update UI
            self.npy_path_label.setText(os.path.basename(file_path))
            self.data_preview_label.setText(
                f"Loaded NPY array with shape: {data.shape}\n"
                f"Data range: [{data.min():.3f}, {data.max():.3f}]"
            )
            
            # Store data
            self.current_voxel_data = data
            
            # Enable conversion button
            self.npy_to_stl_button.setEnabled(True)
            self.show_manager_button.setEnabled(True)
            
            self.status_label.setText(f"Status: NPY file {os.path.basename(file_path)} loaded")
            
        except Exception as e:
            QMessageBox.critical(self.widget, "NPY Load Error", f"Failed to load NPY file:\n{str(e)}")
            self.status_label.setText(f"Status: NPY load failed - {str(e)}")
    
    def convert_stl_to_npy(self):
        """Convert STL file to voxel representation (NPY)"""
        if not self.current_stl_mesh:
            QMessageBox.warning(self.widget, "Conversion Error", "No STL mesh loaded.")
            return
        
        try:
            if not trimesh_available:
                raise ImportError("The trimesh library is not available. Cannot convert STL to NPY.")
            
            # Get parameters
            resolution = self.resolution_spin.value()
            
            # Show progress
            self.status_label.setText("Status: Converting STL to NPY...")
            self.conversion_progress.setValue(10)
            
            # Run conversion in background
            self.worker = Worker(self.run_stl_to_npy_conversion, 
                                self.current_stl_mesh, 
                                resolution=resolution)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.task_completed.connect(self.stl_to_npy_completed)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self.widget, "Conversion Error", f"STL to NPY conversion failed:\n{str(e)}")
            self.status_label.setText(f"Status: STL to NPY conversion failed - {str(e)}")
    
    def run_stl_to_npy_conversion(self, mesh, resolution=128, progress_callback=None):
        """Run STL to NPY conversion"""
        if progress_callback:
            progress_callback.emit(20)
        
        # Voxelize the mesh
        voxels = mesh.voxelized(pitch=1.0/resolution)
        
        if progress_callback:
            progress_callback.emit(50)
        
        # Convert to binary volume
        volume = voxels.matrix
        
        # Ensure the volume has the right shape
        if volume.shape[0] != resolution or volume.shape[1] != resolution or volume.shape[2] != resolution:
            # Resize if needed
            volume = zoom(volume, 
                          (resolution/volume.shape[0], 
                           resolution/volume.shape[1], 
                           resolution/volume.shape[2]), 
                          order=0)
        
        if progress_callback:
            progress_callback.emit(70)
        
        # Add channel dimension for Keras model
        volume = volume.astype(np.float32)
        volume = np.expand_dims(volume, axis=-1)
        
        # Generate a unique filename
        filename = f"voxel_{resolution}_{int(np.sum(volume))}.npy"
        filepath = os.path.join(self.npy_path, filename)
        
        # Save the volume
        np.save(filepath, volume)
        
        if progress_callback:
            progress_callback.emit(90)
        
        return {
            'volume': volume,
            'filepath': filepath,
            'filename': filename
        }
    
    def stl_to_npy_completed(self, result):
        """Handle completion of STL to NPY conversion"""
        # Update progress
        self.conversion_progress.setValue(100)
        
        # Update UI
        self.current_voxel_data = result['volume']
        self.npy_path_label.setText(result['filename'])
        
        # Show data info
        self.data_preview_label.setText(
            f"Converted to voxel array with shape: {result['volume'].shape}\n"
            f"Saved to: {result['filename']}"
        )
        
        # Refresh files list
        self.refresh_npy_files()
        
        # Enable NPY to STL conversion
        self.npy_to_stl_button.setEnabled(True)
        
        # Show message
        QMessageBox.information(self.widget, "Conversion Successful", 
                               f"STL to NPY conversion completed successfully.\nSaved to: {result['filename']}")
        
        # Update status
        self.status_label.setText(f"Status: STL to NPY conversion completed - {result['filename']}")
    
    def convert_npy_to_stl(self):
        """Convert voxel representation (NPY) to STL file"""
        if self.current_voxel_data is None:
            QMessageBox.warning(self.widget, "Conversion Error", "No voxel data loaded.")
            return
        
        try:
            if not trimesh_available:
                raise ImportError("The trimesh library is not available. Cannot convert NPY to STL.")
            
            # Show progress
            self.status_label.setText("Status: Converting NPY to STL...")
            self.conversion_progress.setValue(10)
            
            # Run conversion in background
            self.worker = Worker(self.run_npy_to_stl_conversion, self.current_voxel_data)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.task_completed.connect(self.npy_to_stl_completed)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self.widget, "Conversion Error", f"NPY to STL conversion failed:\n{str(e)}")
            self.status_label.setText(f"Status: NPY to STL conversion failed - {str(e)}")
    
    def run_npy_to_stl_conversion(self, volume, progress_callback=None):
        """Run NPY to STL conversion"""
        if progress_callback:
            progress_callback.emit(20)
        
        # Remove the channel dimension if present
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]
        
        # Binarize the volume if needed
        if np.max(volume) > 1.0 or np.min(volume) < 0.0:
            volume = (volume > 0.5).astype(np.bool_)
        else:
            volume = (volume > 0.5)
        
        if progress_callback:
            progress_callback.emit(40)
        
        # Create a trimesh voxel grid
        voxel_grid = trimesh.voxel.VoxelGrid(volume)
        
        if progress_callback:
            progress_callback.emit(60)
        
        # Convert to mesh
        mesh = voxel_grid.as_mesh()
        
        if progress_callback:
            progress_callback.emit(80)
        
        # Generate a unique filename
        filename = f"mesh_{volume.shape[0]}_{int(np.sum(volume))}.stl"
        filepath = os.path.join(self.result_path, filename)
        
        # Export to STL
        mesh.export(filepath)
        
        if progress_callback:
            progress_callback.emit(90)
        
        return {
            'mesh': mesh,
            'filepath': filepath,
            'filename': filename
        }
    
    def npy_to_stl_completed(self, result):
        """Handle completion of NPY to STL conversion"""
        # Update progress
        self.conversion_progress.setValue(100)
        
        # Update UI
        self.current_stl_mesh = result['mesh']
        self.stl_path_label.setText(result['filename'])
        
        # Show data info
        self.data_preview_label.setText(
            f"Converted to mesh with {len(result['mesh'].vertices)} vertices, {len(result['mesh'].faces)} faces\n"
            f"Saved to: {result['filename']}"
        )
        
        # Enable conversion buttons
        self.stl_to_npy_button.setEnabled(True)
        
        # Show message
        QMessageBox.information(self.widget, "Conversion Successful", 
                               f"NPY to STL conversion completed successfully.\nSaved to: {result['filename']}")
        
        # Update status
        self.status_label.setText(f"Status: NPY to STL conversion completed - {result['filename']}")
        
        # Show in visualization if available
        if hasattr(self.parent, 'visualization_manager') and pyvista_available:
            # Convert trimesh to pyvista
            pv_mesh = pv.PolyData(result['mesh'].vertices, result['mesh'].faces)
            self.parent.visualization_manager.display_stl_preview(pv_mesh)
    
    def update_prediction_inputs(self):
        """Update prediction input controls based on selected input type"""
        # Update based on radio button selection
        if self.input_file_radio.isChecked():
            self.input_file_layout.setEnabled(True)
        else:
            self.input_file_layout.setEnabled(False)
    
    def start_prediction(self):
        """Start model prediction"""
        if not self.model:
            QMessageBox.warning(self.widget, "Prediction Error", "No model loaded.")
            return
        
        try:
            # Get input data
            input_data = None
            
            if self.input_file_radio.isChecked():
                # Load from file
                if self.input_file_combo.currentText():
                    filepath = os.path.join(self.npy_path, self.input_file_combo.currentText())
                    input_data = np.load(filepath)
                    
                    # Add batch dimension if needed
                    if input_data.ndim == 3:
                        input_data = np.expand_dims(input_data, axis=0)
                    elif input_data.ndim == 4 and input_data.shape[-1] != 1:
                        # Assume last dim is not channel dim
                        input_data = np.expand_dims(input_data, axis=-1)
                    
                    if input_data.ndim != 5:
                        input_data = np.expand_dims(input_data, axis=0)
                else:
                    QMessageBox.warning(self.widget, "Prediction Error", "No input file selected.")
                    return
            else:
                # Use current data
                if self.current_voxel_data is not None:
                    input_data = self.current_voxel_data
                    
                    # Add batch dimension if needed
                    if input_data.ndim == 3:
                        input_data = np.expand_dims(input_data, axis=0)
                    elif input_data.ndim == 4 and input_data.shape[-1] != 1:
                        # Assume last dim is not channel dim
                        input_data = np.expand_dims(input_data, axis=-1)
                    
                    if input_data.ndim != 5:
                        input_data = np.expand_dims(input_data, axis=0)
                else:
                    QMessageBox.warning(self.widget, "Prediction Error", "No current data available.")
                    return
            
            # Check input shape
            expected_shape = self.model.input_shape[1:]  # Remove batch dimension
            actual_shape = input_data.shape[1:]  # Remove batch dimension
            
            if expected_shape != actual_shape:
                # Try to resize
                print(f"Resizing input from {actual_shape} to {expected_shape}")
                
                # Get the data without batch dimension
                data_to_resize = input_data[0]
                
                # Remove channel dimension if present
                if data_to_resize.ndim == 4:
                    channel_dim = data_to_resize[..., 0]
                else:
                    channel_dim = data_to_resize
                
                # Resize using zoom with nearest neighbor interpolation
                zoom_factors = (expected_shape[0] / channel_dim.shape[0],
                               expected_shape[1] / channel_dim.shape[1],
                               expected_shape[2] / channel_dim.shape[2])
                
                resized_data = zoom(channel_dim, zoom_factors, order=0)
                
                # Add channel dimension back
                resized_data = np.expand_dims(resized_data, axis=-1)
                
                # Add batch dimension back
                input_data = np.expand_dims(resized_data, axis=0)
                
                QMessageBox.information(self.widget, "Shape Adjustment", 
                                      f"Input shape {actual_shape} resized to match model input shape {expected_shape}")
            
            # Show progress
            self.prediction_info_label.setText("Running prediction...")
            self.prediction_progress.setValue(10)
            
            # Run prediction in background
            self.worker = Worker(self.run_prediction, input_data)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.task_completed.connect(self.prediction_completed)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self.widget, "Prediction Error", f"Prediction failed:\n{error_msg}")
            self.prediction_info_label.setText(f"Prediction failed: {str(e)}")
    
    def run_prediction(self, input_data, progress_callback=None):
        """Run model prediction"""
        if progress_callback:
            progress_callback.emit(20)
        
        # Run prediction
        prediction = self.model.predict(input_data, verbose=1)
        
        if progress_callback:
            progress_callback.emit(50)
        
        # Check if we got multiple outputs (voxel and folding)
        if isinstance(prediction, list) or isinstance(prediction, tuple):
            voxel_output = prediction[0]
            folding_output = prediction[1] if len(prediction) > 1 else None
        else:
            voxel_output = prediction
            folding_output = None
        
        # Generate a unique filename
        output_filename = f"pred_{voxel_output.shape[1]}_{int(np.sum(voxel_output > 0.5))}.npy"
        output_path = os.path.join(self.result_path, output_filename)
        
        # Save prediction
        if self.save_result_check.isChecked():
            np.save(output_path, voxel_output)
        
        if progress_callback:
            progress_callback.emit(80)
        
        # Convert to STL if requested
        stl_result = None
        if self.convert_to_stl_check.isChecked() and trimesh_available:
            try:
                # Convert NPY to STL
                volume = voxel_output[0]  # Remove batch dimension
                
                # Remove the channel dimension if present
                if volume.ndim == 4:
                    volume = volume[:, :, :, 0]
                
                # Binarize the volume
                volume = (volume > 0.5)
                
                # Create a trimesh voxel grid
                voxel_grid = trimesh.voxel.VoxelGrid(volume)
                
                # Convert to mesh
                mesh = voxel_grid.as_mesh()
                
                # Generate a unique filename
                stl_filename = f"pred_mesh_{volume.shape[0]}_{int(np.sum(volume))}.stl"
                stl_filepath = os.path.join(self.result_path, stl_filename)
                
                # Export to STL
                mesh.export(stl_filepath)
                
                stl_result = {
                    'mesh': mesh,
                    'filepath': stl_filepath,
                    'filename': stl_filename
                }
            except Exception as e:
                print(f"STL conversion failed: {e}")
                stl_result = None
        
        if progress_callback:
            progress_callback.emit(95)
        
        return {
            'input': input_data,
            'voxel_output': voxel_output,
            'folding_output': folding_output,
            'output_path': output_path,
            'output_filename': output_filename,
            'stl_result': stl_result
        }
    
    def prediction_completed(self, result):
        """Handle completion of model prediction"""
        # Update progress
        self.prediction_progress.setValue(100)
        
        # Store results
        self.predicted_result = result
        
        # Update UI
        self.prediction_info_label.setText(
            f"Prediction completed. Output shape: {result['voxel_output'].shape}\n"
            f"Saved to: {result['output_filename']}"
        )
        
        # Update preview
        self.result_preview_label.setText(
            f"Prediction Output:\n"
            f"Shape: {result['voxel_output'].shape}\n"
            f"Non-zero voxels: {int(np.sum(result['voxel_output'] > 0.5))}\n"
            f"Value range: [{result['voxel_output'].min():.3f}, {result['voxel_output'].max():.3f}]"
        )
        
        # Store current voxel data
        self.current_voxel_data = result['voxel_output'][0]  # Remove batch dimension
        
        # Store STL mesh if available
        if result['stl_result']:
            self.current_stl_mesh = result['stl_result']['mesh']
            self.stl_path_label.setText(result['stl_result']['filename'])
            
            # Enable STL-related buttons
            self.export_stl_button.setEnabled(True)
            self.export_screenshot_button.setEnabled(True)
            self.show_manager_button.setEnabled(True)
        
        # Refresh files list
        self.refresh_npy_files()
        
        # Show visualization if requested
        if self.visualize_result_check.isChecked():
            if pyvista_available and result['stl_result']:
                # Convert trimesh to pyvista
                pv_mesh = pv.PolyData(result['stl_result']['mesh'].vertices, result['stl_result']['mesh'].faces)
                self.parent.visualization_manager.display_stl_preview(pv_mesh)
                
                # Switch to visualization tab
                self.tabs.setCurrentIndex(3)  # Visualization tab
        
        # Show message
        QMessageBox.information(self.widget, "Prediction Successful", 
                              f"Model prediction completed successfully.\nSaved to: {result['output_filename']}")
        
        # Update status
        self.status_label.setText(f"Status: Prediction completed - {result['output_filename']}")
    
    def update_visualization(self):
        """Update the 3D visualization"""
        if not pyvista_available:
            QMessageBox.warning(self.widget, "Visualization Error", 
                              "PyVista library is not available. 3D visualization is not supported.")
            return
        
        # Check if we have data to visualize
        if self.current_stl_mesh is None and self.current_voxel_data is None:
            QMessageBox.warning(self.widget, "Visualization Error", 
                              "No data available for visualization.")
            return
        
        try:
            # Get visualization options
            view_mode = self.visualize_combo.currentText()
            color_scheme = self.color_combo.currentText()
            background = self.background_combo.currentText()
            
            # Create visualization
            if view_mode == "Input Data" and self.current_voxel_data is not None:
                # Create mesh from input data
                if trimesh_available and self.current_stl_mesh is not None:
                    # Use existing mesh
                    mesh = self.current_stl_mesh
                    pv_mesh = pv.PolyData(mesh.vertices, mesh.faces)
                else:
                    # Create from voxel data
                    volume = self.current_voxel_data
                    if volume.ndim == 4:
                        volume = volume[:, :, :, 0]
                    
                    # Binarize the volume
                    volume = (volume > 0.5)
                    
                    # Create a PyVista uniform grid
                    grid = pv.UniformGrid()
                    grid.dimensions = np.array(volume.shape) + 1
                    grid.origin = (0, 0, 0)
                    grid.spacing = (1, 1, 1)
                    
                    # Populate the uniform grid with the voxel data
                    grid.cell_data["values"] = volume.flatten(order='F')
                    
                    # Extract surface
                    pv_mesh = grid.threshold(0.5).extract_surface()
                
                # Set color based on scheme
                if color_scheme == "Rainbow":
                    pv_mesh.compute_normals(inplace=True)
                    scalars = pv_mesh.points[:, 2]  # Color by Z coordinate
                    cmap = 'rainbow'
                elif color_scheme == "Metallic":
                    pv_mesh.compute_normals(inplace=True)
                    scalars = pv_mesh.points[:, 2]  # Color by Z coordinate
                    cmap = 'copper'
                elif color_scheme == "Heat":
                    pv_mesh.compute_normals(inplace=True)
                    scalars = pv_mesh.points[:, 2]  # Color by Z coordinate
                    cmap = 'hot'
                else:  # Default
                    scalars = None
                    cmap = None
                
                # Create plotter
                plotter = pv.Plotter(off_screen=True)
                
                # Set background
                if background == "Black":
                    plotter.set_background('black')
                elif background == "Gray":
                    plotter.set_background('gray')
                elif background == "Gradient":
                    plotter.set_background('white', top='lightblue')
                else:  # White
                    plotter.set_background('white')
                
                # Add mesh to plotter
                plotter.add_mesh(pv_mesh, color='tan' if scalars is None else None, 
                               scalars=scalars, cmap=cmap)
                
                # Set camera position
                plotter.camera_position = 'iso'
                
                # Render figure and get Matplotlib figure
                plotter.show(screenshot=False)
                mpl_fig = _qt_figure_to_mpl_figure(plotter.ren_win)
                
                # Replace placeholder with visualization
                self.visualization_layout.removeWidget(self.visualization_placeholder)
                self.visualization_placeholder.setVisible(False)
                
                # Create FigureCanvas from the matplotlib figure
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
                self.visualization_canvas = FigureCanvas(mpl_fig)
                self.visualization_layout.addWidget(self.visualization_canvas)
                
                # Enable export buttons
                self.export_screenshot_button.setEnabled(True)
                self.show_manager_button.setEnabled(True)
                
            elif view_mode == "Prediction Result" and self.predicted_result is not None:
                # Create mesh from prediction result
                if trimesh_available and self.predicted_result['stl_result'] is not None:
                    # Use existing mesh
                    mesh = self.predicted_result['stl_result']['mesh']
                    pv_mesh = pv.PolyData(mesh.vertices, mesh.faces)
                else:
                    # Create from voxel data
                    volume = self.predicted_result['voxel_output'][0]  # Remove batch dimension
                    if volume.ndim == 4:
                        volume = volume[:, :, :, 0]
                    
                    # Binarize the volume
                    volume = (volume > 0.5)
                    
                    # Create a PyVista uniform grid
                    grid = pv.UniformGrid()
                    grid.dimensions = np.array(volume.shape) + 1
                    grid.origin = (0, 0, 0)
                    grid.spacing = (1, 1, 1)
                    
                    # Populate the uniform grid with the voxel data
                    grid.cell_data["values"] = volume.flatten(order='F')
                    
                    # Extract surface
                    pv_mesh = grid.threshold(0.5).extract_surface()
                
                # Set color based on scheme
                if color_scheme == "Rainbow":
                    pv_mesh.compute_normals(inplace=True)
                    scalars = pv_mesh.points[:, 2]  # Color by Z coordinate
                    cmap = 'rainbow'
                elif color_scheme == "Metallic":
                    pv_mesh.compute_normals(inplace=True)
                    scalars = pv_mesh.points[:, 2]  # Color by Z coordinate
                    cmap = 'copper'
                elif color_scheme == "Heat":
                    pv_mesh.compute_normals(inplace=True)
                    scalars = pv_mesh.points[:, 2]  # Color by Z coordinate
                    cmap = 'hot'
                else:  # Default
                    scalars = None
                    cmap = None
                
                # Create plotter
                plotter = pv.Plotter(off_screen=True)
                
                # Set background
                if background == "Black":
                    plotter.set_background('black')
                elif background == "Gray":
                    plotter.set_background('gray')
                elif background == "Gradient":
                    plotter.set_background('white', top='lightblue')
                else:  # White
                    plotter.set_background('white')
                
                # Add mesh to plotter
                plotter.add_mesh(pv_mesh, color='tan' if scalars is None else None, 
                               scalars=scalars, cmap=cmap)
                
                # Set camera position
                plotter.camera_position = 'iso'
                
                # Render figure and get Matplotlib figure
                plotter.show(screenshot=False)
                mpl_fig = _qt_figure_to_mpl_figure(plotter.ren_win)
                
                # Replace placeholder with visualization
                self.visualization_layout.removeWidget(self.visualization_placeholder)
                self.visualization_placeholder.setVisible(False)
                
                # Create FigureCanvas from the matplotlib figure
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
                self.visualization_canvas = FigureCanvas(mpl_fig)
                self.visualization_layout.addWidget(self.visualization_canvas)
                
                # Enable export buttons
                self.export_screenshot_button.setEnabled(True)
                self.show_manager_button.setEnabled(True)
                
            elif view_mode == "Both (Side by Side)" and self.current_voxel_data is not None and self.predicted_result is not None:
                # Create a side-by-side visualization
                # First plotter for input data
                plotter1 = pv.Plotter(off_screen=True)
                
                # Set background
                if background == "Black":
                    plotter1.set_background('black')
                elif background == "Gray":
                    plotter1.set_background('gray')
                elif background == "Gradient":
                    plotter1.set_background('white', top='lightblue')
                else:  # White
                    plotter1.set_background('white')
                
                # Create mesh from input data
                if trimesh_available and self.current_stl_mesh is not None:
                    # Use existing mesh
                    mesh = self.current_stl_mesh
                    pv_mesh1 = pv.PolyData(mesh.vertices, mesh.faces)
                else:
                    # Create from voxel data
                    volume = self.current_voxel_data
                    if volume.ndim == 4:
                        volume = volume[:, :, :, 0]
                    
                    # Binarize the volume
                    volume = (volume > 0.5)
                    
                    # Create a PyVista uniform grid
                    grid = pv.UniformGrid()
                    grid.dimensions = np.array(volume.shape) + 1
                    grid.origin = (0, 0, 0)
                    grid.spacing = (1, 1, 1)
                    
                    # Populate the uniform grid with the voxel data
                    grid.cell_data["values"] = volume.flatten(order='F')
                    
                    # Extract surface
                    pv_mesh1 = grid.threshold(0.5).extract_surface()
                
                # Set color based on scheme
                if color_scheme == "Rainbow":
                    pv_mesh1.compute_normals(inplace=True)
                    scalars = pv_mesh1.points[:, 2]  # Color by Z coordinate
                    cmap = 'rainbow'
                elif color_scheme == "Metallic":
                    pv_mesh1.compute_normals(inplace=True)
                    scalars = pv_mesh1.points[:, 2]  # Color by Z coordinate
                    cmap = 'copper'
                elif color_scheme == "Heat":
                    pv_mesh1.compute_normals(inplace=True)
                    scalars = pv_mesh1.points[:, 2]  # Color by Z coordinate
                    cmap = 'hot'
                else:  # Default
                    scalars = None
                    cmap = None
                
                # Add mesh to plotter
                plotter1.add_mesh(pv_mesh1, color='tan' if scalars is None else None, 
                                scalars=scalars, cmap=cmap)
                
                # Set camera position
                plotter1.camera_position = 'iso'
                
                # Render figure and get Matplotlib figure
                plotter1.show(screenshot=False)
                mpl_fig1 = _qt_figure_to_mpl_figure(plotter1.ren_win)
                
                # Second plotter for prediction data
                plotter2 = pv.Plotter(off_screen=True)
                
                # Set background
                if background == "Black":
                    plotter2.set_background('black')
                elif background == "Gray":
                    plotter2.set_background('gray')
                elif background == "Gradient":
                    plotter2.set_background('white', top='lightblue')
                else:  # White
                    plotter2.set_background('white')
                
                # Create mesh from prediction result
                if trimesh_available and self.predicted_result['stl_result'] is not None:
                    # Use existing mesh
                    mesh = self.predicted_result['stl_result']['mesh']
                    pv_mesh2 = pv.PolyData(mesh.vertices, mesh.faces)
                else:
                    # Create from voxel data
                    volume = self.predicted_result['voxel_output'][0]  # Remove batch dimension
                    if volume.ndim == 4:
                        volume = volume[:, :, :, 0]
                    
                    # Binarize the volume
                    volume = (volume > 0.5)
                    
                    # Create a PyVista uniform grid
                    grid = pv.UniformGrid()
                    grid.dimensions = np.array(volume.shape) + 1
                    grid.origin = (0, 0, 0)
                    grid.spacing = (1, 1, 1)
                    
                    # Populate the uniform grid with the voxel data
                    grid.cell_data["values"] = volume.flatten(order='F')
                    
                    # Extract surface
                    pv_mesh2 = grid.threshold(0.5).extract_surface()
                
                # Set color based on scheme
                if color_scheme == "Rainbow":
                    pv_mesh2.compute_normals(inplace=True)
                    scalars = pv_mesh2.points[:, 2]  # Color by Z coordinate
                    cmap = 'rainbow'
                elif color_scheme == "Metallic":
                    pv_mesh2.compute_normals(inplace=True)
                    scalars = pv_mesh2.points[:, 2]  # Color by Z coordinate
                    cmap = 'copper'
                elif color_scheme == "Heat":
                    pv_mesh2.compute_normals(inplace=True)
                    scalars = pv_mesh2.points[:, 2]  # Color by Z coordinate
                    cmap = 'hot'
                else:  # Default
                    scalars = None
                    cmap = None
                
                # Add mesh to plotter
                plotter2.add_mesh(pv_mesh2, color='tan' if scalars is None else None, 
                                scalars=scalars, cmap=cmap)
                
                # Set camera position
                plotter2.camera_position = 'iso'
                
                # Render figure and get Matplotlib figure
                plotter2.show(screenshot=False)
                mpl_fig2 = _qt_figure_to_mpl_figure(plotter2.ren_win)
                
                # Replace placeholder with visualization
                self.visualization_layout.removeWidget(self.visualization_placeholder)
                self.visualization_placeholder.setVisible(False)
                
                # Create layout for side-by-side view
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
                
                # Create a new figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Add the first MPL figure to the first subplot
                for ax in mpl_fig1.axes:
                    # Copy the image from the original axes to the first subplot
                    ax1.imshow(ax.images[0].get_array())
                    ax1.set_title("Input")
                    ax1.axis('off')
                
                # Add the second MPL figure to the second subplot
                for ax in mpl_fig2.axes:
                    # Copy the image from the original axes to the second subplot
                    ax2.imshow(ax.images[0].get_array())
                    ax2.set_title("Prediction")
                    ax2.axis('off')
                
                # Create FigureCanvas from the combined matplotlib figure
                self.visualization_canvas = FigureCanvas(fig)
                self.visualization_layout.addWidget(self.visualization_canvas)
                
                # Enable export buttons
                self.export_screenshot_button.setEnabled(True)
                self.show_manager_button.setEnabled(True)
            
            else:
                QMessageBox.warning(self.widget, "Visualization Error", 
                                  f"Cannot visualize: Missing data for {view_mode}")
        
        except Exception as e:
            QMessageBox.critical(self.widget, "Visualization Error", f"Visualization failed:\n{str(e)}")
            self.status_label.setText(f"Status: Visualization failed - {str(e)}")
    
    def export_to_stl(self):
        """Export current visualization to STL file"""
        if not self.current_stl_mesh and not (self.predicted_result and self.predicted_result.get('stl_result')):
            QMessageBox.warning(self.widget, "Export Error", "No mesh available for export.")
            return
        
        try:
            # Get the mesh to export
            mesh = self.current_stl_mesh
            if not mesh and self.predicted_result and self.predicted_result.get('stl_result'):
                mesh = self.predicted_result['stl_result']['mesh']
            
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(self.widget, "Save STL File", "", "STL Files (*.stl)")
            
            if file_path:
                # Export to STL
                mesh.export(file_path)
                
                # Show message
                QMessageBox.information(self.widget, "Export Successful", 
                                      f"Mesh exported successfully to {file_path}")
                
                # Update status
                self.status_label.setText(f"Status: Mesh exported to {os.path.basename(file_path)}")
        
        except Exception as e:
            QMessageBox.critical(self.widget, "Export Error", f"STL export failed:\n{str(e)}")
            self.status_label.setText(f"Status: STL export failed - {str(e)}")
    
    def export_screenshot(self):
        """Export current visualization as a screenshot"""
        if not hasattr(self, 'visualization_canvas'):
            QMessageBox.warning(self.widget, "Export Error", "No visualization available for export.")
            return
        
        try:
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(self.widget, "Save Screenshot", "", 
                                                      "PNG Files (*.png);;JPEG Files (*.jpg)")
            
            if file_path:
                # Export screenshot
                self.visualization_canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                
                # Show message
                QMessageBox.information(self.widget, "Export Successful", 
                                      f"Screenshot exported successfully to {file_path}")
                
                # Update status
                self.status_label.setText(f"Status: Screenshot exported to {os.path.basename(file_path)}")
        
        except Exception as e:
            QMessageBox.critical(self.widget, "Export Error", f"Screenshot export failed:\n{str(e)}")
            self.status_label.setText(f"Status: Screenshot export failed - {str(e)}")
    
    def show_in_visualization_manager(self):
        """Show current visualization in the main visualization manager"""
        if not hasattr(self.parent, 'visualization_manager'):
            QMessageBox.warning(self.widget, "Visualization Error", 
                              "Visualization manager not available.")
            return
        
        try:
            # Check what to visualize
            if pyvista_available:
                # Prepare mesh for visualization
                if self.current_stl_mesh:
                    # Convert trimesh to pyvista
                    pv_mesh = pv.PolyData(self.current_stl_mesh.vertices, self.current_stl_mesh.faces)
                    self.parent.visualization_manager.display_stl_preview(pv_mesh)
                elif self.predicted_result and self.predicted_result.get('stl_result'):
                    # Convert trimesh to pyvista
                    pv_mesh = pv.PolyData(self.predicted_result['stl_result']['mesh'].vertices, 
                                         self.predicted_result['stl_result']['mesh'].faces)
                    self.parent.visualization_manager.display_stl_preview(pv_mesh)
                else:
                    # Try to create mesh from voxel data
                    volume = self.current_voxel_data
                    if volume.ndim == 4:
                        volume = volume[:, :, :, 0]
                    
                    # Binarize the volume
                    volume = (volume > 0.5)
                    
                    # Create a PyVista uniform grid
                    grid = pv.UniformGrid()
                    grid.dimensions = np.array(volume.shape) + 1
                    grid.origin = (0, 0, 0)
                    grid.spacing = (1, 1, 1)
                    
                    # Populate the uniform grid with the voxel data
                    grid.cell_data["values"] = volume.flatten(order='F')
                    
                    # Extract surface
                    pv_mesh = grid.threshold(0.5).extract_surface()
                    
                    self.parent.visualization_manager.display_stl_preview(pv_mesh)
            
            # Update status
            self.status_label.setText("Status: Visualization sent to visualization manager")
            
        except Exception as e:
            QMessageBox.critical(self.widget, "Visualization Error", 
                               f"Failed to show in visualization manager:\n{str(e)}")
            self.status_label.setText(f"Status: Visualization manager display failed - {str(e)}")
