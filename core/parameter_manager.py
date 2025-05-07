# core/parameter_manager.py
import json
import os

class ParameterManager:
    """Class for managing project parameters"""
    
    def __init__(self):
        """Initialize parameter manager"""
        self.parameters = {}
    
    def set_parameter(self, module_name, param_name, value):
        """Set parameter value"""
        if module_name not in self.parameters:
            self.parameters[module_name] = {}
        
        self.parameters[module_name][param_name] = value
    
    def get_parameter(self, module_name, param_name, default=None):
        """Get parameter value"""
        if module_name in self.parameters and param_name in self.parameters[module_name]:
            return self.parameters[module_name][param_name]
        return default
    
    def get_module_parameters(self, module_name):
        """Get all parameters for a module"""
        if module_name in self.parameters:
            return self.parameters[module_name].copy()
        return {}
    
    def clear_parameters(self):
        """Clear all parameters"""
        self.parameters = {}
    
    def save_parameters(self, filepath):
        """Save parameters to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save parameters to JSON file
            with open(filepath, 'w') as f:
                json.dump(self.parameters, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving parameters: {str(e)}")
            return False
    
    def load_parameters(self, filepath):
        """Load parameters from file"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return False
            
            # Load parameters from JSON file
            with open(filepath, 'r') as f:
                self.parameters = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading parameters: {str(e)}")
            return False