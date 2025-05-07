# main.py
import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QMessageBox, QSplitter, 
                            QTabWidget, QVBoxLayout, QWidget, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from ui.main_window import Ui_MainWindow
from core.module_manager import ModuleManager
from core.parameter_manager import ParameterManager
from core.project import Project
from core.visualization_manager import VisualizationManager
from modules.processing_map_module import ProcessingMapModule
from modules.nsm_billetsizing_module import NSMBilletsizingModule
from modules.preform_prediction_module import PreformPredictionModule

class MainApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = None
        self.settings = QSettings("MetalCastingLab", "MetalCastingApplication")
        
        # Set application style
        self.app.setStyle('Fusion')  # More modern style
        
        # General style
        self.app.setStyleSheet("""
    QMainWindow {
        background-color: #f8f9fa;
    }

    QTabWidget::pane {
        border: 1px solid #cccccc;
        background-color: white;
        border-radius: 3px;
    }

    QTabBar::tab {
        background-color: #007bff;
        color: white;
        padding: 8px 15px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
        font-weight: bold;
    }

    QTabBar::tab:selected {
        background-color: #0056b3;
        border-bottom: 3px solid #003f7f;
    }

    QTabBar::tab:disabled {
        background-color: #cccccc;
        color: #666666;
    }

    QTabBar::tab:hover {
        background-color: #339aff;
    }

    QLabel {
        font-size: 12px;
    }

    QLineEdit {
        padding: 5px;
        border: 1px solid #cccccc;
        border-radius: 3px;
    }

    QPushButton {
        padding: 6px 12px;
        border-radius: 3px;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #0056b3;
    }
""")


    
    def init(self):
        try:
            print("Application is starting...")
            self.main_window = MainWindow(self.settings)
            print("Loading modules...")
            self.main_window.load_modules()
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error initializing application: {str(e)}")
            print(f"Error: {str(e)}")
            return False
    
    def run(self):
        print("Displaying application window...")
        self.main_window.show()
        return self.app.exec()

class MainWindow(QMainWindow):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        print("Creating MainWindow...")
        self.settings = settings
        self.ui = Ui_MainWindow()
        print("Calling setupUi...")
        self.ui.setupUi(self)
        print("setupUi completed")
        
        # Create splitter for visualization and modules
        self.setup_splitter()
        
        # Create main components
        self._init_components()
        
        # Connect signals and slots
        self._connect_signals()
        
        # Set up status bar
        self._setup_status_bar()
        
        print("MainWindow created")
    
    def setup_splitter(self):
        """Create splitter for visualization and modules"""
        print("Creating splitter...")
        # Create splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Add module tabs to splitter
        self.splitter.addWidget(self.ui.module_tabs)
        
        # Separate widget for visualization
        self.ui.visualization_frame.setMinimumWidth(500)
        self.splitter.addWidget(self.ui.visualization_frame)
        
        # Set splitter proportions
        self.splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        self.ui.centralwidget.layout().addWidget(self.splitter)
        print("Splitter created")
    
    def _init_components(self):
        """Create main components"""
        print("Creating module manager...")
        self.module_manager = ModuleManager()
        print("Creating parameter manager...")
        self.parameter_manager = ParameterManager()
        print("Creating visualization manager...")
        self.visualization_manager = VisualizationManager(self.ui.visualization_frame)
        print("Creating project...")
        self.current_project = Project(self.parameter_manager)
    
    def _connect_signals(self):
        """Connect signals and slots"""
        # Connect menu signals
        if hasattr(self.ui, 'action_exit'):
            self.ui.action_exit.triggered.connect(self.close)
        
        if hasattr(self.ui, 'action_about'):
            self.ui.action_about.triggered.connect(self._show_about)
    
    def _setup_status_bar(self):
        """Set up status bar"""
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        # Style for statusbar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #f0f0f0; 
                color: #333333;
                border-top: 1px solid #cccccc;
            }
        """)
    
    def _show_about(self):
        """Show information about the application"""
        QMessageBox.about(self, "About Application", 
                         "MetalCastingApplication\n"
                         "Version: 1.0\n"
                         "© 2025 MetalCastingLab")
    
    def load_modules(self):
        print("Loading modules...")
        
        try:
            print("Creating Processing Map module...")
            processing_map_module = ProcessingMapModule()
            self.module_manager.register_module(processing_map_module)
            
            print("Creating NSM Billetsizing module...")
            nsm_billetsizing_module = NSMBilletsizingModule()
            self.module_manager.register_module(nsm_billetsizing_module)
            
            print("Creating Preform Prediction module...")
            preform_prediction_module = PreformPredictionModule()
            self.module_manager.register_module(preform_prediction_module)
            
            print("Creating module interfaces...")
            self._create_module_ui()
            
            print(f"Total {len(self.module_manager.get_module_names())} modules loaded")
            
            self.status_label.setText(f"{len(self.module_manager.get_module_names())} modules loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading modules: {str(e)}")
            print(f"Module error: {str(e)}")
    
    def _create_module_ui(self):
        print("Creating module interfaces...")
        
        # Display modules
        for module_name in self.module_manager.get_module_names():
            module = self.module_manager.get_module(module_name)
            if module:
                print(f"Creating interface for module {module_name}...")
                module_widget = module.create_widget(self)
                
                # Create tab for module
                module_index = self.ui.module_tabs.addTab(module_widget, module.get_name())
                print(f"Module interface created: {module_name}, index: {module_index}")

if __name__ == "__main__":
    print("Application started...")
    app = MainApp()
    print("MainApp created...")
    if app.init():
        print("init() executed successfully...")
        sys.exit(app.run())
    else:
        print("init() failed...")