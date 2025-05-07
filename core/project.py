# core/project.py
class Project:
    """Class for managing project data"""
    
    def __init__(self, parameter_manager):
        """Initialize project"""
        self.parameter_manager = parameter_manager
        self.project_name = "New Project"
        self.project_path = None
        self.modified = False
    
    def new_project(self):
        """Create new project"""
        self.project_name = "New Project"
        self.project_path = None
        self.modified = False
        self.parameter_manager.clear_parameters()
        
    def open_project(self, path):
        """Open project from file"""
        self.project_path = path
        self.project_name = os.path.basename(path)
        # Load parameters
        self.parameter_manager.load_parameters(path)
        self.modified = False
        
    def save_project(self, path=None):
        """Save project to file"""
        if path:
            self.project_path = path
            self.project_name = os.path.basename(path)
        
        if self.project_path:
            # Save parameters
            self.parameter_manager.save_parameters(self.project_path)
            self.modified = False
            return True
        return False
    
    def is_modified(self):
        """Check if project is modified"""
        return self.modified
    
    def set_modified(self, modified=True):
        """Set project modified flag"""
        self.modified = modified
    
    def get_name(self):
        """Get project name"""
        return self.project_name
    
    def get_path(self):
        """Get project path"""
        return self.project_path