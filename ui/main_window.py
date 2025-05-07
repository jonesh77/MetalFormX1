# ui/main_window.py
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
                            QMenuBar, QMenu, QStatusBar, QFrame)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QSize

class Ui_MainWindow:
    def setupUi(self, MainWindow):
        # Set window properties
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        MainWindow.setMinimumSize(QSize(800, 600))
        MainWindow.setWindowTitle("MetalFormX")
        
        # Create central widget
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        
        # Create module tabs widget
        self.module_tabs = QTabWidget(self.centralwidget)
        self.module_tabs.setObjectName("module_tabs")
        
        # Create visualization frame
        self.visualization_frame = QFrame(self.centralwidget)
        self.visualization_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.visualization_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.visualization_frame.setObjectName("visualization_frame")
        
        # Set central widget
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Create menu bar
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(0, 0, 1200, 22)
        self.menubar.setObjectName("menubar")
        
        # Create File menu
        self.menu_file = QMenu(self.menubar)
        self.menu_file.setObjectName("menu_file")
        self.menu_file.setTitle("File")
        
        # Create Help menu
        self.menu_help = QMenu(self.menubar)
        self.menu_help.setObjectName("menu_help")
        self.menu_help.setTitle("Help")
        
        # Set menu bar
        MainWindow.setMenuBar(self.menubar)
        
        # Create status bar
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        # Create actions
        self.action_exit = QAction(MainWindow)
        self.action_exit.setObjectName("action_exit")
        self.action_exit.setText("Exit")
        
        self.action_about = QAction(MainWindow)
        self.action_about.setObjectName("action_about")
        self.action_about.setText("About")
        
        # Add actions to menus
        self.menu_file.addAction(self.action_exit)
        self.menu_help.addAction(self.action_about)
        
        # Add menus to menu bar
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_help.menuAction())