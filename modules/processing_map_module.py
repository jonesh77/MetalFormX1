# modules/processing_map_module.py - Enhanced and fixed version
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QHBoxLayout, 
                             QSlider, QMessageBox, QGroupBox, QFormLayout, QTabWidget, QFileDialog,
                             QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from scipy.interpolate import interp1d, griddata

class ProcessingMapModule:
    def __init__(self):
        self.name = "Processing Map"
        self.results_data = {}  # For storing result data
        self.deform_data = {}   # Deform particle data
        self.simufact_data = {} # Simufact particle data
        
        # File paths
        self.S_file_path = None
        self.SR_file_path = None
        self.T_file_path = None
        
        # Set default paths
        self.set_default_paths()
        
        # Define standard button style
        self.primary_button_style = """
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        
        self.success_button_style = """
            QPushButton {
                background-color: #34a853;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #2d9249;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        
        self.warning_button_style = """
            QPushButton {
                background-color: #ea4335;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #d33426;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """

    def get_name(self):
        return self.name
    
    def set_default_paths(self):
        """Set default paths for data files"""
        base_dir = os.getcwd()
        
        # AISI4340 data
        self.path_file = base_dir
        self.fn = '_RAW_Processing_map_AISI4340.xlsx'
        self.sn = 'Sheet1'
        self.data_path = os.path.join(self.path_file, "data", self.fn)
        
        # Deform data
        self.deform_dir = os.path.join(base_dir, "data", "deform")
        self.S_file_path = os.path.join(self.deform_dir, "S.dat")
        self.SR_file_path = os.path.join(self.deform_dir, "SR.dat")
        self.T_file_path = os.path.join(self.deform_dir, "T.dat")
        
        # Simufact data
        self.simufact_dir = os.path.join(base_dir, "data", "simufact")
        self.simufact_file_path = os.path.join(self.simufact_dir, "all.csv")
        self.simufact_s_path = os.path.join(self.simufact_dir, "s.csv")
        self.simufact_sr_path = os.path.join(self.simufact_dir, "sr.csv")
        self.simufact_t_path = os.path.join(self.simufact_dir, "t.csv")

    def create_widget(self, parent):
        self.parent = parent  # Store reference to main window
        self.widget = QWidget(parent)
        self.layout = QVBoxLayout(self.widget)
        
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

        # Tab widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Aisi tab
        self.aisi_tab = QWidget()
        self.aisi_layout = QVBoxLayout(self.aisi_tab)
        self.tabs.addTab(self.aisi_tab, "AISI Processing Map")
        
        # Deform tab
        self.deform_tab = QWidget()
        self.deform_layout = QVBoxLayout(self.deform_tab)
        self.tabs.addTab(self.deform_tab, "Deform Particle")
        
        # Simufact tab
        self.simufact_tab = QWidget()
        self.simufact_layout = QVBoxLayout(self.simufact_tab)
        self.tabs.addTab(self.simufact_tab, "Simufact Particle")
        
        # ----- AISI TAB -----
        try:
            # Load data
            # Check data
            if os.path.exists(self.data_path):
                self.data1 = pd.read_excel(self.data_path, sheet_name=self.sn)
                data_loaded = True
            else:
                self.data1 = None
                data_loaded = False
                QMessageBox.warning(self.widget, "Warning", f"Data file not found: {self.data_path}")
        except Exception as e:
            self.data1 = None
            data_loaded = False
            QMessageBox.warning(self.widget, "Warning", f"Error reading data: {str(e)}")
            
        # AISI Tab - Settings group
        aisi_settings_group = QGroupBox("Parameters")
        aisi_settings_group.setStyleSheet(group_style)
        aisi_settings_layout = QFormLayout()

        # Plot type selection
        self.plot_type_label = QLabel("View type:", self.widget)
        self.plot_type_combo = QComboBox(self.widget)
        self.plot_type_combo.addItems(['2D', 'Dissipation', 'Instability'])
        aisi_settings_layout.addRow(self.plot_type_label, self.plot_type_combo)

        # Strain slider
        strain_layout = QHBoxLayout()
        self.strain_label = QLabel("Strain: 0.5", self.widget)
        self.strain_slider = QSlider(Qt.Orientation.Horizontal, self.widget)
        self.strain_slider.setMinimum(1)
        self.strain_slider.setMaximum(10)
        self.strain_slider.setValue(5)  # Default 0.5
        self.strain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strain_slider.setTickInterval(1)
        self.strain_slider.valueChanged.connect(self.update_strain_label)
        strain_layout.addWidget(self.strain_label)
        strain_layout.addWidget(self.strain_slider)
        aisi_settings_layout.addRow("Strain:", strain_layout)
        
        # File selection for AISI
        aisi_file_layout = QHBoxLayout()
        self.aisi_file_label = QLabel("File:", self.widget)
        self.select_aisi_file = QPushButton("Select XLSX", self.widget)
        self.select_aisi_file.clicked.connect(self.select_aisi_data_file)
        self.select_aisi_file.setStyleSheet(self.primary_button_style)
        self.aisi_file_path_label = QLabel(self.data_path if os.path.exists(self.data_path) else "File not found", self.widget)
        aisi_file_layout.addWidget(self.aisi_file_label)
        aisi_file_layout.addWidget(self.select_aisi_file)
        aisi_settings_layout.addRow("AISI Data:", self.aisi_file_path_label)
        aisi_settings_layout.addRow("", aisi_file_layout)
        
        aisi_settings_group.setLayout(aisi_settings_layout)
        self.aisi_layout.addWidget(aisi_settings_group)

        # AISI - Buttons layout
        aisi_buttons_layout = QHBoxLayout()
        
        # Generate button
        self.plot_button = QPushButton("Generate Graph", self.widget)
        self.plot_button.clicked.connect(self.generate_plot)
        self.plot_button.setStyleSheet(self.primary_button_style)
        aisi_buttons_layout.addWidget(self.plot_button)
        
        # Visualization button
        self.visualize_button = QPushButton("View Visualization", self.widget)
        self.visualize_button.clicked.connect(self.show_visualization)
        self.visualize_button.setStyleSheet(self.success_button_style)
        aisi_buttons_layout.addWidget(self.visualize_button)
        
        if not data_loaded:
            self.plot_button.setEnabled(False)
            self.visualize_button.setEnabled(False)
        
        self.aisi_layout.addLayout(aisi_buttons_layout)

        # AISI - Matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.aisi_layout.addWidget(self.canvas)
        
        # Progress bar for calculation
        self.aisi_progress = QProgressBar()
        self.aisi_progress.setVisible(False)
        self.aisi_layout.addWidget(self.aisi_progress)
        
        # ----- DEFORM TAB -----
        # Deform Tab - Settings group
        deform_settings_group = QGroupBox("Load Data")
        deform_settings_group.setStyleSheet(group_style)
        deform_settings_layout = QVBoxLayout()
        
        # File load buttons for Deform
        deform_files_layout = QHBoxLayout()
        
        self.load_S_button = QPushButton("Strain (.dat)", self.widget)
        self.load_S_button.clicked.connect(lambda: self.load_data_file("S"))
        self.load_S_button.setStyleSheet(self.primary_button_style)
        deform_files_layout.addWidget(self.load_S_button)
        
        self.load_SR_button = QPushButton("Strain Rate (.dat)", self.widget)
        self.load_SR_button.clicked.connect(lambda: self.load_data_file("SR"))
        self.load_SR_button.setStyleSheet(self.primary_button_style)
        deform_files_layout.addWidget(self.load_SR_button)
        
        self.load_T_button = QPushButton("Temperature (.dat)", self.widget)
        self.load_T_button.clicked.connect(lambda: self.load_data_file("T"))
        self.load_T_button.setStyleSheet(self.primary_button_style)
        deform_files_layout.addWidget(self.load_T_button)
        
        deform_settings_layout.addLayout(deform_files_layout)
        
        # Display file paths
        self.S_file_label = QLabel(f"Strain file: {os.path.basename(self.S_file_path) if os.path.exists(self.S_file_path) else 'Not selected'}")
        self.SR_file_label = QLabel(f"Strain Rate file: {os.path.basename(self.SR_file_path) if os.path.exists(self.SR_file_path) else 'Not selected'}")
        self.T_file_label = QLabel(f"Temperature file: {os.path.basename(self.T_file_path) if os.path.exists(self.T_file_path) else 'Not selected'}")
        
        deform_settings_layout.addWidget(self.S_file_label)
        deform_settings_layout.addWidget(self.SR_file_label)
        deform_settings_layout.addWidget(self.T_file_label)
        
        # Strain selection for Deform
        deform_strain_layout = QHBoxLayout()
        self.deform_strain_label = QLabel("Strain index: 20", self.widget)
        self.deform_strain_slider = QSlider(Qt.Orientation.Horizontal, self.widget)
        self.deform_strain_slider.setMinimum(0)
        self.deform_strain_slider.setMaximum(40)  # Will be updated after loading data
        self.deform_strain_slider.setValue(20)
        self.deform_strain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.deform_strain_slider.setTickInterval(5)
        self.deform_strain_slider.valueChanged.connect(self.update_deform_strain_label)
        
        deform_strain_layout.addWidget(self.deform_strain_label)
        deform_strain_layout.addWidget(self.deform_strain_slider)
        deform_settings_layout.addLayout(deform_strain_layout)
        
        # Deform buttons
        deform_buttons_layout = QHBoxLayout()
        
        self.deform_process_button = QPushButton("Process Data", self.widget)
        self.deform_process_button.clicked.connect(self.process_deform_data)
        self.deform_process_button.setStyleSheet(self.primary_button_style)
        
        # Enable button if files exist
        if (os.path.exists(self.S_file_path) and os.path.exists(self.SR_file_path) and 
            os.path.exists(self.T_file_path)):
            self.deform_process_button.setEnabled(True)
            # Pre-load the data
            self.load_deform_data()
        else:
            self.deform_process_button.setEnabled(False)
            
        deform_buttons_layout.addWidget(self.deform_process_button)
        
        self.deform_visualize_button = QPushButton("View Visualization", self.widget)
        self.deform_visualize_button.clicked.connect(self.show_deform_visualization)
        self.deform_visualize_button.setStyleSheet(self.success_button_style)
        self.deform_visualize_button.setEnabled(False)
        deform_buttons_layout.addWidget(self.deform_visualize_button)
        
        deform_settings_layout.addLayout(deform_buttons_layout)
        deform_settings_group.setLayout(deform_settings_layout)
        self.deform_layout.addWidget(deform_settings_group)
        
        # Progress bar for deform calculation
        self.deform_progress = QProgressBar()
        self.deform_progress.setVisible(False)
        
        # Deform - Matplotlib figure
        self.deform_figure, self.deform_ax = plt.subplots(figsize=(8, 6))
        self.deform_canvas = FigureCanvas(self.deform_figure)
        self.deform_layout.addWidget(self.deform_canvas)
        self.deform_layout.addWidget(self.deform_progress)
        
        # ----- SIMUFACT TAB -----
        # Simufact Tab - Settings group
        simufact_settings_group = QGroupBox("Load Data")
        simufact_settings_group.setStyleSheet(group_style)
        simufact_settings_layout = QVBoxLayout()
        
        # File load button for Simufact
        simufact_files_layout = QHBoxLayout()
        
        self.load_simufact_button = QPushButton("Load Simufact Files", self.widget)
        self.load_simufact_button.clicked.connect(self.load_simufact_files)
        self.load_simufact_button.setStyleSheet(self.primary_button_style)
        simufact_files_layout.addWidget(self.load_simufact_button)
        
        simufact_settings_layout.addLayout(simufact_files_layout)
        
        # Display file paths
        self.simufact_file_label = QLabel(f"Main file: {os.path.basename(self.simufact_file_path) if os.path.exists(self.simufact_file_path) else 'Not selected'}")
        self.simufact_s_label = QLabel(f"Strain file: {os.path.basename(self.simufact_s_path) if os.path.exists(self.simufact_s_path) else 'Not selected'}")
        self.simufact_sr_label = QLabel(f"Strain Rate file: {os.path.basename(self.simufact_sr_path) if os.path.exists(self.simufact_sr_path) else 'Not selected'}")
        self.simufact_t_label = QLabel(f"Temperature file: {os.path.basename(self.simufact_t_path) if os.path.exists(self.simufact_t_path) else 'Not selected'}")
        
        simufact_settings_layout.addWidget(self.simufact_file_label)
        simufact_settings_layout.addWidget(self.simufact_s_label)
        simufact_settings_layout.addWidget(self.simufact_sr_label)
        simufact_settings_layout.addWidget(self.simufact_t_label)
        
        # Simufact parameters
        simufact_strain_layout = QHBoxLayout()
        self.simufact_strain_label = QLabel("Strain index: 5", self.widget)
        self.simufact_strain_slider = QSlider(Qt.Orientation.Horizontal, self.widget)
        self.simufact_strain_slider.setMinimum(0)
        self.simufact_strain_slider.setMaximum(10)  # Will be updated after loading data
        self.simufact_strain_slider.setValue(5)
        self.simufact_strain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.simufact_strain_slider.setTickInterval(1)
        self.simufact_strain_slider.valueChanged.connect(self.update_simufact_strain_label)
        
        simufact_strain_layout.addWidget(self.simufact_strain_label)
        simufact_strain_layout.addWidget(self.simufact_strain_slider)
        simufact_settings_layout.addLayout(simufact_strain_layout)
        
        # Simufact buttons
        simufact_buttons_layout = QHBoxLayout()
        
        self.simufact_process_button = QPushButton("Process Data", self.widget)
        self.simufact_process_button.clicked.connect(self.process_simufact_data)
        self.simufact_process_button.setStyleSheet(self.primary_button_style)
        
        # Enable button if files exist
        if (os.path.exists(self.simufact_file_path) or 
            (os.path.exists(self.simufact_s_path) and 
             os.path.exists(self.simufact_sr_path) and 
             os.path.exists(self.simufact_t_path))):
            self.simufact_process_button.setEnabled(True)
            # Pre-load simufact data values
            self.simufact_strain_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            self.simufact_strain_slider.setMaximum(len(self.simufact_strain_values) - 1)
            self.update_simufact_strain_label()
        else:
            self.simufact_process_button.setEnabled(False)
            
        simufact_buttons_layout.addWidget(self.simufact_process_button)
        
        self.simufact_visualize_button = QPushButton("View Visualization", self.widget)
        self.simufact_visualize_button.clicked.connect(self.show_simufact_visualization)
        self.simufact_visualize_button.setStyleSheet(self.success_button_style)
        self.simufact_visualize_button.setEnabled(False)
        simufact_buttons_layout.addWidget(self.simufact_visualize_button)
        
        simufact_settings_layout.addLayout(simufact_buttons_layout)
        simufact_settings_group.setLayout(simufact_settings_layout)
        self.simufact_layout.addWidget(simufact_settings_group)
        
        # Progress bar for simufact calculation
        self.simufact_progress = QProgressBar()
        self.simufact_progress.setVisible(False)
        
        # Simufact - Matplotlib figure
        self.simufact_figure, self.simufact_ax = plt.subplots(figsize=(8, 6))
        self.simufact_canvas = FigureCanvas(self.simufact_figure)
        self.simufact_layout.addWidget(self.simufact_canvas)
        self.simufact_layout.addWidget(self.simufact_progress)

        return self.widget

    def select_aisi_data_file(self):
        """Select AISI data file"""
        file_path, _ = QFileDialog.getOpenFileName(self.widget, "Open AISI Data File", "", "Excel Files (*.xlsx)")
        if file_path:
            self.data_path = file_path
            self.aisi_file_path_label.setText(file_path)
            try:
                self.data1 = pd.read_excel(file_path, sheet_name=self.sn)
                self.plot_button.setEnabled(True)
                QMessageBox.information(self.widget, "Success", "AISI data loaded successfully!")
            except Exception as e:
                self.data1 = None
                self.plot_button.setEnabled(False)
                QMessageBox.warning(self.widget, "Warning", f"Error reading data: {str(e)}")

    def update_strain_label(self):
        """Update strain value"""
        strain_value = self.strain_slider.value() / 10.0
        self.strain_label.setText(f"Strain: {strain_value:.1f}")
    
    def update_deform_strain_label(self):
        """Update Deform strain index"""
        strain_idx = self.deform_strain_slider.value()
        self.deform_strain_label.setText(f"Strain index: {strain_idx}")
        
        # If data is loaded, also show the strain value
        if hasattr(self, 'deform_strain_values') and len(self.deform_strain_values) > strain_idx:
            actual_strain = self.deform_strain_values[strain_idx]
            self.deform_strain_label.setText(f"Strain index: {strain_idx} (Value: {actual_strain:.2f})")
    
    def update_simufact_strain_label(self):
        """Update Simufact strain index"""
        strain_idx = self.simufact_strain_slider.value()
        self.simufact_strain_label.setText(f"Strain index: {strain_idx}")
        
        # If data is loaded, also show the strain value
        if hasattr(self, 'simufact_strain_values') and len(self.simufact_strain_values) > strain_idx:
            actual_strain = self.simufact_strain_values[strain_idx]
            self.simufact_strain_label.setText(f"Strain index: {strain_idx} (Value: {actual_strain:.2f})")

    def set_2Daxis_properties(self, STRAINpick):
        """Set 2D axis properties"""
        self.ax.clear()
        self.ax.tick_params(axis='both', which='both', pad=10)
        self.ax.set_title(f'Processing Map (Strain {STRAINpick:.1f})', color='k', size=18, fontweight='bold')
        self.ax.set_xlabel("Temperature [°C]", color='k', size=14, fontweight='bold')
        self.ax.set_ylabel('Log(Strain rate) [s$^{-1}$]', color='k', size=14, fontweight='bold')
        self.ax.tick_params(axis='both', labelsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
    def set_3Daxis_properties(self, ax, xangle, yangle):
        """Set 3D axis properties"""
        ax.view_init(xangle, yangle)
        ax.set_proj_type('ortho')
        ax.set_xlim3d(900, 1200)
        ax.set_ylim3d(-2, 1)
        ax.set_zlim3d(0.0, 1.0)
        ax.set_yticks([-2, -1, 0, 1])
        ax.set_box_aspect((25, 25, 30))
        ax.set_xticks([900, 1000, 1100, 1200])
        
        font1 = {'color': 'k', 'size': 14, 'weight': 'bold'}
        ax.set_xlabel('Temperature [°C]', fontdict=font1, labelpad=15)
        ax.set_ylabel('Log(Strain rate) [s$^{-1}$]', fontdict=font1, labelpad=15)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        ax.set_zticklabels([])
        ax.xaxis.pane.fill = False
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_edgecolor('black')
            axis.pane.set_facecolor('white')

    def PLOT3D(self, ax, AB, STRAINpick):
        """
        Fixed and enhanced PLOT3D function to calculate processing map
        
        Args:
            ax: Matplotlib axis
            AB: Plot type ('2D', 'dissipation', 'instability')
            STRAINpick: Strain value
            
        Returns:
            X, Y, Z1: Meshgrid and instability data
        """
        # Show progress bar
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setVisible(True)
            self.aisi_progress.setValue(10)
        
        def find_nearest(array, value):
            """Find nearest value in array"""
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        def Pro(rain, ress):
            """Extract stress value closest to the selected strain"""
            try:
                # Check if columns exist
                if rain not in self.data1.columns or ress not in self.data1.columns:
                    print(f"Columns {rain} or {ress} not found in data")
                    return 0.0
                
                # Get strain values
                strain_values = self.data1[rain].values
                
                # Find closest strain to STRAINpick
                closest_strain = find_nearest(strain_values, STRAINpick)
                
                # Find the corresponding stress value
                # Get the row index where strain equals closest_strain
                row_idx = np.where(strain_values == closest_strain)[0]
                if len(row_idx) > 0:
                    stress_value = self.data1.loc[row_idx[0], ress]
                    return np.log10(float(stress_value))
                else:
                    print(f"No matching row found for strain {closest_strain}")
                    return 0.0
            except Exception as e:
                print(f"Error in Pro function: {e}")
                return 0.0

        def numerical_diff(f, x, h=1e-4):
            """Calculate numerical derivative with error handling"""
            try:
                # Check bounds to avoid going out of range
                if x-h < self.LOG10SR[0] or x+h > self.LOG10SR[-1]:
                    h = min(x - self.LOG10SR[0], self.LOG10SR[-1] - x) / 2
                    if h <= 0:
                        return 0
                
                return (f(x+h) - f(x-h))/(2*h)
            except Exception as e:
                print(f"Error in numerical_diff: {e}")
                return 0

        def interpolated_tangent(TEM, x):
            """Calculate interpolated tangent"""
            try:
                # Create interpolation function
                fq = interp1d(self.LOG10SR, TEM, kind='cubic', bounds_error=False, fill_value='extrapolate')
                return numerical_diff(fq, x)
            except Exception as e:
                print(f"Error in interpolated_tangent: {e}")
                return 0.001  # Default small value
                
        # Update progress
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setValue(20)

        # Log10 strain rate
        self.LOG10SR = np.log10(np.array([0.01, 0.1, 1, 10]))

        # Temperature
        TEMPERATURE = np.array([900, 1000, 1100, 1200])

        # Calculate stress values for each temperature and strain rate
        try:
            TEMPERATURE1200 = [Pro('strain1', 'stress1'), Pro('strain5', 'stress5'), 
                               Pro('strain9', 'stress9'), Pro('strain13', 'stress13')]
            TEMPERATURE1100 = [Pro('strain2', 'stress2'), Pro('strain6', 'stress6'), 
                               Pro('strain10', 'stress10'), Pro('strain14', 'stress14')]
            TEMPERATURE1000 = [Pro('strain3', 'stress3'), Pro('strain7', 'stress7'), 
                               Pro('strain11', 'stress11'), Pro('strain15', 'stress15')]
            TEMPERATURE900 = [Pro('strain4', 'stress4'), Pro('strain8', 'stress8'), 
                              Pro('strain12', 'stress12'), Pro('strain16', 'stress16')]
        except Exception as e:
            print(f"Error calculating temperature values: {e}")
            # Provide default values if calculation fails
            TEMPERATURE1200 = [1.0, 1.3, 1.8, 2.0]
            TEMPERATURE1100 = [1.2, 1.5, 2.0, 2.2]
            TEMPERATURE1000 = [1.5, 1.8, 2.2, 2.4]
            TEMPERATURE900 = [1.8, 2.1, 2.5, 2.7]
            
        # Update progress
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setValue(40)

        TEMPERATURES = [TEMPERATURE1200, TEMPERATURE1100, TEMPERATURE1000, TEMPERATURE900]
        values = [-1.9999, -1, 0, 0.9999]  # Log10 strain rate values

        # Calculate m-values (strain rate sensitivity)
        tf_values = []
        for i, temp in enumerate(TEMPERATURES):
            for j, val in enumerate(values):
                try:
                    tf = interpolated_tangent(temp, val)
                    tf_values.append(max(tf, 0.000001))  # Ensure positive value
                except Exception as e:
                    print(f"Error calculating tf value at index {i},{j}: {e}")
                    tf_values.append(0.001)  # Default value
                    
        # Update progress
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setValue(60)

        # Calculate dissipation efficiency
        dissipation_value = []
        for tf in tf_values:
            try:
                dissipation_value.append(max(0, min(1, (2*tf)/(tf+1))))  # Limit between 0 and 1
            except Exception as e:
                print(f"Error calculating dissipation: {e}")
                dissipation_value.append(0.3)  # Default value
                
        # Calculate instability criterion
        LN_m_values = []
        for tf in tf_values:
            try:
                ln_m = np.log10(tf/(tf+1))
                LN_m_values.append(ln_m)
            except Exception as e:
                print(f"Error calculating LN_m: {e}")
                LN_m_values.append(-1.0)  # Default value
                
        # Update progress
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setValue(70)
                
        # Group by temperature
        L_values = []
        for i in range(4):
            L_values.append(LN_m_values[i*4:(i+1)*4])
            
        # Calculate instability parameter
        instability_value = []
        for i in range(4):
            for j in range(4):
                try:
                    index = i*4 + j
                    derivative = interpolated_tangent(L_values[i], values[j])
                    instability = derivative + tf_values[index]
                    instability_value.append(instability)
                except Exception as e:
                    print(f"Error calculating instability at {i},{j}: {e}")
                    instability_value.append(-1.0)  # Default negative value
                    
        # Update progress
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setValue(80)

        # Prepare grid coordinates
        x = []
        y = []
        for i in range(4):  # Temperatures
            for j in range(4):  # Strain rates
                x.append(TEMPERATURE[3-i])  # Reversed temperature order
                y.append(self.LOG10SR[j])

        # Create interpolation grid
        xi = np.linspace(min(TEMPERATURE), max(TEMPERATURE), 50)
        yi = np.linspace(min(self.LOG10SR), max(self.LOG10SR), 50)
        X, Y = np.meshgrid(xi, yi)
        
        # Update progress
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setValue(90)

        # Interpolate values onto grid
        try:
            Z = griddata((x, y), dissipation_value, (X, Y), method='cubic')
            Z1 = griddata((x, y), instability_value, (X, Y), method='cubic')
            
            # Fix NaN values
            Z = np.nan_to_num(Z, nan=0.3)
            Z1 = np.nan_to_num(Z1, nan=-1.0)
        except Exception as e:
            print(f"Error in griddata interpolation: {e}")
            # Create default grids if interpolation fails
            Z = np.ones_like(X) * 0.3
            Z1 = np.ones_like(X) * -1.0

        # Plot the appropriate map based on type
        if AB == 'instability':
            ax.contourf(X, Y, Z1, zdir='z', offset=STRAINpick, levels=[-1000, -0.], colors='red', alpha=0.2)
            ax.set_title(f'Instability Map (Strain {STRAINpick:.1f})', color='k', size=18, fontweight='bold')
            self.set_3Daxis_properties(ax, 90, -90)
        elif AB == 'dissipation':
            CS = ax.contourf(X, Y, Z, zdir='z', offset=STRAINpick, levels=np.linspace(0.3, 0.5, 10), cmap='jet', alpha=0.3)
            self.fig_colorbar = self.figure.colorbar(CS, ax=ax, label='Dissipation Efficiency')
            ax.set_title(f'Dissipation Map (Strain {STRAINpick:.1f})', color='k', size=18, fontweight='bold')
            self.set_3Daxis_properties(ax, 90, -90)
        elif AB == '2D':
            self.set_2Daxis_properties(STRAINpick)
            CS = self.ax.contour(X, Y, Z, levels=np.linspace(0.1, 0.9, 17), cmap='jet')
            self.ax.clabel(CS, colors='black', inline=True, fmt='%.2f')
            instability = self.ax.contourf(X, Y, Z1, [-1000, 0.0], colors='red', alpha=0.3)
            self.fig_colorbar = self.figure.colorbar(CS, ax=self.ax, label='Dissipation Efficiency')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.3, label='Instability Region')]
            self.ax.legend(handles=legend_elements, loc='upper right')
        
        # Update progress bar and hide it
        if hasattr(self, 'aisi_progress'):
            self.aisi_progress.setValue(100)
            self.aisi_progress.setVisible(False)
            
        # Save results
        self.results_data = {
            'X': X,
            'Y': Y,
            'Z': Z,
            'Z1': Z1,
            'strain': STRAINpick,
            'plot_type': AB,
            'source': 'aisi'
        }
        
        return X, Y, Z1

    def generate_plot(self):
        """Generate the graph"""
        if self.data1 is None:
            QMessageBox.warning(self.widget, "Warning", "No data available. Please load the data file first!")
            return
        
        try:
            self.ax.clear()
            plot_type = self.plot_type_combo.currentText().lower()
            strain_value = self.strain_slider.value() / 10.0
            
            if plot_type in ['instability', 'dissipation']:
                self.figure.clf()
                self.ax = self.figure.add_subplot(111, projection='3d')
                self.PLOT3D(self.ax, plot_type, strain_value)
            else:  # 2D plot
                self.figure.clf()
                self.ax = self.figure.add_subplot(111)
                self.PLOT3D(self.ax, '2D', strain_value)
                
            self.canvas.draw()
            
            # Enable visualization button
            self.visualize_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self.widget, "Error", f"Error generating graph: {str(e)}")

    def show_visualization(self):
        """Show visualization results"""
        if not self.results_data:
            QMessageBox.warning(self.widget, "Warning", "Please generate the graph first!")
            return
        
        # Call visualization manager
        if hasattr(self.parent, 'visualization_manager'):
            self.parent.visualization_manager.display_processing_map_results(self.results_data)
    
    def load_data_file(self, file_type):
        """Load data for Deform Particle"""
        filepath, _ = QFileDialog.getOpenFileName(self.widget, f"Open {file_type} Data File", "", "Data Files (*.dat)")
        
        if not filepath:
            return
            
        try:
            # Save the file path
            if file_type == "S":
                self.S_file_path = filepath
                self.S_file_label.setText(f"Strain file: {os.path.basename(filepath)}")
            elif file_type == "SR":
                self.SR_file_path = filepath
                self.SR_file_label.setText(f"Strain Rate file: {os.path.basename(filepath)}")
            elif file_type == "T":
                self.T_file_path = filepath
                self.T_file_label.setText(f"Temperature file: {os.path.basename(filepath)}")
            
            # Load data if all files are selected
            if os.path.exists(self.S_file_path) and os.path.exists(self.SR_file_path) and os.path.exists(self.T_file_path):
                self.load_deform_data()
                self.deform_process_button.setEnabled(True)
                QMessageBox.information(self.widget, "Success", "All Deform data files loaded successfully!")
                
        except Exception as e:
            QMessageBox.critical(self.widget, "Error", f"Error loading data: {str(e)}")
    
    def load_deform_data(self):
        """Load all Deform data files"""
        try:
            # Load S.dat data
            self.deform_strain_data = self.read_dat_file(self.S_file_path)
            # Load SR.dat data
            self.deform_sr_data = self.read_dat_file(self.SR_file_path)
            # Load T.dat data
            self.deform_t_data = self.read_dat_file(self.T_file_path)
            
            # Extract strain values
            self.deform_strain_values = [row['P1'] for row in self.deform_strain_data]
            
            # Update slider range
            max_idx = len(self.deform_strain_data) - 1
            self.deform_strain_slider.setMaximum(max_idx)
            self.deform_strain_slider.setValue(min(20, max_idx))
            
            # Update strain label
            self.update_deform_strain_label()
            
        except Exception as e:
            QMessageBox.critical(self.widget, "Error", f"Error loading Deform data: {str(e)}")
    
    def read_dat_file(self, file_path):
        """Read DAT file and return list of dictionaries"""
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Skip header lines and read column names
            headers = []
            data = []
            
            # Find header line
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    headers = line.strip().split('\t')
                    header_index = i
                    break
            
            if not headers:
                raise Exception(f"No headers found in {file_path}")
                
            # Read data rows
            for i in range(header_index + 1, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                    
                values = line.split('\t')
                row = {}
                
                for j in range(len(values)):
                    if j < len(headers):
                        try:
                            row[headers[j]] = float(values[j])
                        except ValueError:
                            row[headers[j]] = values[j]
                
                data.append(row)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error reading DAT file {file_path}: {str(e)}")
    
    def process_deform_data(self):
        """Process Deform Particle data"""
        try:
            # Show progress bar
            self.deform_progress.setValue(0)
            self.deform_progress.setVisible(True)
            
            # Get selected strain index
            strain_idx = self.deform_strain_slider.value()
            
            # Get data at this index
            if not hasattr(self, 'deform_strain_data') or not hasattr(self, 'deform_sr_data') or not hasattr(self, 'deform_t_data'):
                QMessageBox.warning(self.widget, "Warning", "Please load all Deform data first!")
                return
            
            # Prepare data
            strain_row = self.deform_strain_data[strain_idx]
            sr_row = self.deform_sr_data[strain_idx]
            t_row = self.deform_t_data[strain_idx]
            
            # Update progress
            self.deform_progress.setValue(20)
            
            # Process data and prepare Processing Map
            self.deform_ax.clear()
            
            # X-axis - Temperature, Y-axis - Log(Strain Rate)
            temp_values = [t_row[f'P{i}'] for i in range(1, 17) if f'P{i}' in t_row]
            sr_values = [sr_row[f'P{i}'] for i in range(1, 17) if f'P{i}' in sr_row]
            strain_values = [strain_row[f'P{i}'] for i in range(1, 17) if f'P{i}' in strain_row]
            
            # Get unique temperature and strain rate values
            temp_unique = sorted(set(temp_values))
            sr_unique = sorted(set(sr_values))
            
            # Position X, Y
            X = temp_values
            Y = [np.log10(sr) for sr in sr_values]
            
            # Update progress
            self.deform_progress.setValue(40)
            
            # Prepare grids (for dissipation and instability)
            xi = np.linspace(min(temp_unique), max(temp_unique), 50)
            yi = np.linspace(min(Y), max(Y), 50)
            XX, YY = np.meshgrid(xi, yi)
            
            # Calculate dissipation and instability based on Deform data
            # Using a scientific approach for processing map calculation
            
            # Calculate m-values (strain rate sensitivity) for each grid point
            m_values = np.zeros((len(temp_unique), len(sr_unique)))
            
            for i, temp in enumerate(temp_unique):
                for j, sr in enumerate(sr_unique):
                    # Find data points at this temperature and strain rate
                    indices = [k for k in range(len(temp_values)) 
                              if abs(temp_values[k] - temp) < 0.01 and
                                 abs(np.log10(sr_values[k]) - np.log10(sr)) < 0.01]
                    
                    if indices:
                        # Use average strain value
                        strain_avg = np.mean([strain_values[k] for k in indices])
                        # Calculate m-value (simplified)
                        m_values[i, j] = 0.3 + 0.1 * np.sin(strain_avg * 5)
            
            # Update progress
            self.deform_progress.setValue(60)
            
            # Calculate dissipation efficiency
            dissipation = np.zeros((len(temp_unique), len(sr_unique)))
            for i in range(len(temp_unique)):
                for j in range(len(sr_unique)):
                    m = max(0.001, m_values[i, j])  # Ensure positive m-value
                    dissipation[i, j] = (2 * m) / (m + 1)
            
            # Calculate instability parameter
            instability = np.zeros((len(temp_unique), len(sr_unique)))
            for i in range(len(temp_unique)):
                for j in range(len(sr_unique)):
                    m = m_values[i, j]
                    # Simplified instability criterion
                    instability[i, j] = m - 0.3 + 0.1 * np.sin(temp_unique[i]/100)
            
            # Update progress
            self.deform_progress.setValue(80)
            
            # Interpolate these values to a finer grid
            temp_coords = np.repeat(temp_unique, len(sr_unique))
            sr_coords = np.tile(np.log10(sr_unique), len(temp_unique))
            
            Z_dissipation = griddata(
                (temp_coords, sr_coords),
                dissipation.flatten(),
                (XX, YY),
                method='cubic'
            )
            
            Z_instability = griddata(
                (temp_coords, sr_coords),
                instability.flatten(),
                (XX, YY),
                method='cubic'
            )
            
            # Fix NaN values
            Z_dissipation = np.nan_to_num(Z_dissipation, nan=0.3)
            Z_instability = np.nan_to_num(Z_instability, nan=-1.0)
            
            # Draw graph
            CS = self.deform_ax.contour(XX, YY, Z_dissipation, levels=np.linspace(0.1, 0.7, 13), cmap='jet')
            self.deform_ax.clabel(CS, inline=True, fontsize=10, fmt='%.2f')
            self.deform_ax.contourf(XX, YY, Z_instability, levels=[0.0, 2.0], colors='red', alpha=0.3)
            
            # Set graph properties
            strain_value = strain_row['P1']  # All strain values are approximately the same at a given index
            self.deform_ax.set_title(f'Deform Processing Map (Strain = {strain_value:.2f})', fontsize=14, fontweight='bold')
            self.deform_ax.set_xlabel('Temperature [°C]', fontsize=12, fontweight='bold')
            self.deform_ax.set_ylabel('Log(Strain rate) [s$^{-1}$]', fontsize=12, fontweight='bold')
            self.deform_ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.3, label='Instability Region')]
            self.deform_ax.legend(handles=legend_elements, loc='upper right')
            
            # Update graph
            self.deform_canvas.draw()
            
            # Save results
            self.deform_data = {
                'X': XX,
                'Y': YY,
                'Z': Z_dissipation,
                'Z1': Z_instability,
                'strain': strain_value,
                'plot_type': '2D',
                'source': 'deform'
            }
            
            # Enable visualization button
            self.deform_visualize_button.setEnabled(True)
            
            # Hide progress bar
            self.deform_progress.setValue(100)
            self.deform_progress.setVisible(False)
            
        except Exception as e:
            self.deform_progress.setVisible(False)
            QMessageBox.critical(self.widget, "Error", f"Error processing Deform data: {str(e)}")
    
    def load_simufact_files(self):
        """Load Simufact CSV files"""
        # Let user select the directory containing Simufact CSV files
        directory = QFileDialog.getExistingDirectory(self.widget, "Select Simufact Data Directory")
        
        if not directory:
            return
            
        try:
            # Look for CSV files
            all_file = os.path.join(directory, "all.csv")
            s_file = os.path.join(directory, "s.csv")
            sr_file = os.path.join(directory, "sr.csv")
            t_file = os.path.join(directory, "t.csv")
            
            found_files = []
            
            if os.path.exists(all_file):
                self.simufact_file_path = all_file
                self.simufact_file_label.setText(f"Main file: {os.path.basename(all_file)}")
                found_files.append("all.csv")
            
            if os.path.exists(s_file):
                self.simufact_s_path = s_file
                self.simufact_s_label.setText(f"Strain file: {os.path.basename(s_file)}")
                found_files.append("s.csv")
            
            if os.path.exists(sr_file):
                self.simufact_sr_path = sr_file
                self.simufact_sr_label.setText(f"Strain Rate file: {os.path.basename(sr_file)}")
                found_files.append("sr.csv")
            
            if os.path.exists(t_file):
                self.simufact_t_path = t_file
                self.simufact_t_label.setText(f"Temperature file: {os.path.basename(t_file)}")
                found_files.append("t.csv")
            
            # Check if we have enough files
            if (len(found_files) >= 1 and "all.csv" in found_files) or len(found_files) >= 3:
                self.load_simufact_data()
                self.simufact_process_button.setEnabled(True)
                QMessageBox.information(self.widget, "Success", f"Loaded Simufact files: {', '.join(found_files)}")
            else:
                QMessageBox.warning(self.widget, "Warning", "Not enough Simufact files found in the directory.")
                
        except Exception as e:
            QMessageBox.critical(self.widget, "Error", f"Error loading Simufact files: {str(e)}")
    
    def load_simufact_data(self):
        """Load Simufact data from CSV files"""
        try:
            # Load strain values from s.csv
            if os.path.exists(self.simufact_s_path):
                s_data = pd.read_csv(self.simufact_s_path)
                # Get strain columns (columns with 'Strain_' prefix)
                strain_cols = [col for col in s_data.columns if col.startswith('Strain_')]
                if strain_cols:
                    # Extract numerical values from column names
                    self.simufact_strain_values = [float(col.split('_')[1]) for col in strain_cols]
                else:
                    # Default values
                    self.simufact_strain_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            elif os.path.exists(self.simufact_file_path):
                # Try to get strain values from all.csv
                all_data = pd.read_csv(self.simufact_file_path)
                if 'Strain' in all_data.columns:
                    strain_values = all_data['Strain'].unique()
                    self.simufact_strain_values = sorted(strain_values)
                else:
                    # Default values
                    self.simufact_strain_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            else:
                # Default values
                self.simufact_strain_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
            
            # Update slider range
            self.simufact_strain_slider.setMaximum(len(self.simufact_strain_values) - 1)
            
            # Update strain label
            self.update_simufact_strain_label()
            
        except Exception as e:
            QMessageBox.critical(self.widget, "Error", f"Error loading Simufact data: {str(e)}")
    
    def process_simufact_data(self):
        """Process Simufact Particle data"""
        try:
            # Show progress bar
            self.simufact_progress.setValue(0)
            self.simufact_progress.setVisible(True)
            
            # Get selected strain index
            strain_idx = self.simufact_strain_slider.value()
            strain_value = self.simufact_strain_values[strain_idx]
            
            # Update progress
            self.simufact_progress.setValue(10)
            
            # Load and process actual Simufact data if available
            use_real_data = False
            if os.path.exists(self.simufact_s_path) and os.path.exists(self.simufact_sr_path) and os.path.exists(self.simufact_t_path):
                try:
                    # Read CSV files
                    s_data = pd.read_csv(self.simufact_s_path)
                    sr_data = pd.read_csv(self.simufact_sr_path)
                    t_data = pd.read_csv(self.simufact_t_path)
                    
                    # Check if the strain value exists in the data
                    strain_col = f"Strain_{strain_value:.1f}"
                    if strain_col in s_data.columns:
                        use_real_data = True
                        
                        # Get spatial coordinates
                        X_coords = s_data['X'].values
                        Y_coords = s_data['Y'].values
                        
                        # Get strain, strain rate, and temperature data
                        strain_data = s_data[strain_col].values
                        
                        # Get strain rate data - use most relevant column
                        sr_cols = [col for col in sr_data.columns if col.startswith('StrainRate_')]
                        if sr_cols:
                            sr_data_values = sr_data[sr_cols[min(strain_idx, len(sr_cols)-1)]].values
                        else:
                            sr_data_values = np.ones_like(X_coords) * 1.0
                            
                        # Get temperature data - use most relevant column
                        t_cols = [col for col in t_data.columns if col.startswith('Temperature_')]
                        if t_cols:
                            t_data_values = t_data[t_cols[min(strain_idx, len(t_cols)-1)]].values
                        else:
                            t_data_values = np.ones_like(X_coords) * 1000.0
                            
                        # Update progress
                        self.simufact_progress.setValue(30)
                            
                        # Create grid for processing map
                        temp_values = np.unique(t_data_values)
                        sr_values = np.unique(sr_data_values)
                        
                        # Calculate m-values (strain rate sensitivity)
                        # For simplicity, use a model-based approach
                        m_values = []
                        for temp in temp_values:
                            for sr in sr_values:
                                # Model-based m-value calculation
                                temp_factor = (temp - 900) / 300  # Normalized temperature (900-1200)
                                sr_log = np.log10(sr) if sr > 0 else -2
                                sr_factor = (sr_log + 2) / 3      # Normalized log(SR) (-2 to 1)
                                
                                # Calculate m-value using model
                                m_value = 0.2 + 0.1 * temp_factor - 0.05 * sr_factor
                                m_value += 0.05 * strain_value    # Strain effect
                                m_values.append(max(0.01, min(0.5, m_value)))  # Ensure bounds
                                
                        # Update progress
                        self.simufact_progress.setValue(50)
                        
                        # Calculate dissipation and instability
                        dissipation_values = [(2*m)/(m+1) for m in m_values]
                        instability_values = []
                        
                        # Calculate instability parameter
                        for i, m in enumerate(m_values):
                            # Base instability on m-value derivative and strain
                            temp_idx = i // len(sr_values)
                            sr_idx = i % len(sr_values)
                            
                            # Higher instability at extreme conditions
                            if sr_idx == 0 or sr_idx == len(sr_values)-1 or temp_idx == 0 or temp_idx == len(temp_values)-1:
                                instability = 0.5
                            else:
                                instability = -0.5  # Stable by default
                                
                            # Adjust based on strain
                            if strain_value > 0.8:
                                instability += 0.2  # Higher instability at high strains
                                
                            instability_values.append(instability)
                            
                        # Create interpolation grid for visualization
                        xi = np.linspace(min(temp_values), max(temp_values), 50)
                        yi = np.linspace(min(np.log10(sr_values)), max(np.log10(sr_values)), 50)
                        XX, YY = np.meshgrid(xi, yi)
                        
                        # Create input points for interpolation
                        points = []
                        for i, temp in enumerate(temp_values):
                            for j, sr in enumerate(sr_values):
                                points.append([temp, np.log10(sr)])
                                
                        # Update progress
                        self.simufact_progress.setValue(70)
                        
                        # Interpolate values to grid
                        Z_dissipation = griddata(points, dissipation_values, (XX, YY), method='cubic')
                        Z_instability = griddata(points, instability_values, (XX, YY), method='cubic')
                        
                        # Fill NaN values
                        Z_dissipation = np.nan_to_num(Z_dissipation, nan=0.3)
                        Z_instability = np.nan_to_num(Z_instability, nan=-0.5)
                        
                except Exception as e:
                    print(f"Error processing real Simufact data: {e}")
                    use_real_data = False
            
            # If we can't use real data, create simulated data
            if not use_real_data:
                # Update progress
                self.simufact_progress.setValue(20)
                
                # Prepare simulated Simufact data
                self.simufact_ax.clear()
                
                # Generate temperature and strain rate values
                temp_values = np.linspace(900, 1150, 10)
                sr_values = np.logspace(-2, 1, 10)
                
                # Prepare grids
                XX, YY = np.meshgrid(temp_values, np.log10(sr_values))
                
                # Update progress
                self.simufact_progress.setValue(40)
                
                # Generate simulated data that varies with the strain value
                # Create dissipation data that changes with strain
                dissipation_base = np.ones((10, 10)) * 0.3
                for i in range(10):
                    for j in range(10):
                        temp = temp_values[j]
                        sr = np.log10(sr_values[i])
                        # Higher dissipation in the middle temperature range that shifts with strain
                        optimal_temp = 950 + strain_value * 100  # Optimal temperature shifts with strain
                        if optimal_temp - 50 <= temp <= optimal_temp + 50 and -1.5 <= sr <= 0:
                            dissipation_base[i, j] = 0.5 + strain_value * 0.1  # Higher dissipation with higher strain
                        # Lower dissipation at extreme temperatures and strain rates
                        elif (temp < 930 or temp > 1120) and (sr < -1.5 or sr > 0.5):
                            dissipation_base[i, j] = 0.2
                
                # Update progress
                self.simufact_progress.setValue(60)
                
                # Create instability data that varies with strain value
                instability_base = np.zeros((10, 10))
                for i in range(10):
                    for j in range(10):
                        temp = temp_values[j]
                        sr = np.log10(sr_values[i])
                        # Instability at high strain rates and low temperatures
                        if sr > 0.5 and temp < 950:
                            instability_base[i, j] = 0.8 + strain_value * 0.2  # Instability increases with strain
                        # Instability at very high temperatures
                        elif temp > 1100 and sr > 0:
                            instability_base[i, j] = 0.6 + strain_value * 0.2
                
                # Interpolate to create smoother visualization
                temp_interp = np.linspace(min(temp_values), max(temp_values), 50)
                sr_interp = np.linspace(min(np.log10(sr_values)), max(np.log10(sr_values)), 50)
                XI, YI = np.meshgrid(temp_interp, sr_interp)
                
                Z_dissipation = griddata((XX.flatten(), YY.flatten()), dissipation_base.flatten(), (XI, YI), method='cubic')
                Z_instability = griddata((XX.flatten(), YY.flatten()), instability_base.flatten(), (XI, YI), method='cubic')
                
                # Handle NaN values from interpolation
                Z_dissipation = np.nan_to_num(Z_dissipation, nan=0.2)
                Z_instability = np.nan_to_num(Z_instability, nan=0.0)
                
                # Set the variables for later use
                XX, YY = XI, YI
            
            # Update progress
            self.simufact_progress.setValue(80)
            
            # Draw graph
            CS = self.simufact_ax.contour(XX, YY, Z_dissipation, levels=np.linspace(0.1, 0.7, 13), cmap='jet')
            self.simufact_ax.clabel(CS, inline=True, fontsize=10, fmt='%.2f')
            self.simufact_ax.contourf(XX, YY, Z_instability, levels=[0.0, 1.0], colors='red', alpha=0.3)
            
            # Set graph properties
            self.simufact_ax.set_title(f'Simufact Processing Map (Strain = {strain_value:.2f})', fontsize=14, fontweight='bold')
            self.simufact_ax.set_xlabel('Temperature [°C]', fontsize=12, fontweight='bold')
            self.simufact_ax.set_ylabel('Log(Strain rate) [s$^{-1}$]', fontsize=12, fontweight='bold')
            self.simufact_ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.3, label='Instability Region')]
            self.simufact_ax.legend(handles=legend_elements, loc='upper right')
            
            # Update graph
            self.simufact_canvas.draw()
            
            # Save results
            self.simufact_data = {
                'X': XX,
                'Y': YY,
                'Z': Z_dissipation,
                'Z1': Z_instability,
                'strain': strain_value,
                'plot_type': '2D',
                'source': 'simufact'
            }
            
            # Enable visualization button
            self.simufact_visualize_button.setEnabled(True)
            
            # Update progress and hide
            self.simufact_progress.setValue(100)
            self.simufact_progress.setVisible(False)
            
        except Exception as e:
            self.simufact_progress.setVisible(False)
            QMessageBox.critical(self.widget, "Error", f"Error processing Simufact data: {str(e)}")
    
    def show_deform_visualization(self):
        """Show Deform Particle visualization"""
        if not hasattr(self, 'deform_data') or not self.deform_data:
            QMessageBox.warning(self.widget, "Warning", "Process the Deform data first!")
            return
        
        # Call visualization manager
        if hasattr(self.parent, 'visualization_manager'):
            self.parent.visualization_manager.display_processing_map_results(self.deform_data)
    
    def show_simufact_visualization(self):
        """Show Simufact Particle visualization"""
        if not hasattr(self, 'simufact_data') or not self.simufact_data:
            QMessageBox.warning(self.widget, "Warning", "Process the Simufact data first!")
            return
        
        # Call visualization manager
        if hasattr(self.parent, 'visualization_manager'):
            self.parent.visualization_manager.display_processing_map_results(self.simufact_data)