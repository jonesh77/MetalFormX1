# core/module_manager.py
class ModuleManager:
    """Class for managing application modules"""
    
    def __init__(self):
        """Initialize module manager"""
        self.modules = {}
    
    def register_module(self, module):
        """Register a module"""
        module_name = module.get_name()
        self.modules[module_name] = module
    
    def unregister_module(self, module_name):
        """Unregister a module"""
        if module_name in self.modules:
            del self.modules[module_name]
    
    def get_module(self, module_name):
        """Get module by name"""
        if module_name in self.modules:
            return self.modules[module_name]
        return None
    
    def get_module_names(self):
        """Get list of all module names"""
        return list(self.modules.keys())
    
    def get_all_modules(self):
        """Get all modules"""
        return self.modules.values()