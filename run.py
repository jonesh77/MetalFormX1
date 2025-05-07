#!/usr/bin/env python
# run.py - MetalFormX Launcher

import os
import sys
import importlib.util
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'PyQt6',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy'
    ]
    
    optional_packages = [
        'tensorflow',
        'pyvista',
        'trimesh',
        'pymeshlab'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_required.append(package)
    
    for package in optional_packages:
        if importlib.util.find_spec(package) is None:
            missing_optional.append(package)
    
    return missing_required, missing_optional

def install_packages(packages):
    """Install packages using pip"""
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_environment():
    """Set up environment before running the application"""
    # Check dependencies
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"Missing required packages: {', '.join(missing_required)}")
        response = input("Do you want to install them now? (y/n): ")
        if response.lower() == 'y':
            install_packages(missing_required)
        else:
            print("Required packages must be installed to run the application.")
            return False
    
    if missing_optional:
        print(f"Missing optional packages: {', '.join(missing_optional)}")
        print("These packages are not required but will enhance functionality.")
        response = input("Do you want to install them now? (y/n): ")
        if response.lower() == 'y':
            install_packages(missing_optional)
    
    # Setup data directories
    from setup_data import setup_data_directories
    setup_data_directories()
    
    return True

def run_application():
    """Run the MetalFormX application"""
    print("Starting MetalFormX application...")
    from main import MainApp
    app = MainApp()
    if app.init():
        print("init() successful. Launching GUI...")
        sys.exit(app.run())
    else:
        print("init() failed.")

if __name__ == "__main__":
    # Print welcome banner
    print("=" * 60)
    print("MetalFormX - Forging Process Simulation Application")
    print("=" * 60)
    
    # Setup environment and run GUI
    if setup_environment():
        run_application()
    else:
        print("Failed to set up environment. Exiting.")
        sys.exit(1)
