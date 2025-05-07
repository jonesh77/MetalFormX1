# core/visualization_manager.py - Updated with STL Preview functionality
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget, QHBoxLayout, QPushButton, QSplitter
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

class VisualizationManager:
    def __init__(self, parent_frame):
        """
        Create visualization manager
        
        Args:
            parent_frame: Main frame where visualization will be placed
        """
        self.parent_frame = parent_frame
        self.layout = QVBoxLayout(self.parent_frame)
        
        # Create tab widget for visualization
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Tab for AISI Processing Map
        self.aisi_widget = QWidget()
        self.aisi_layout = QVBoxLayout(self.aisi_widget)
        self.aisi_figure = Figure(figsize=(10, 8), dpi=100)
        self.aisi_canvas = FigureCanvas(self.aisi_figure)
        self.aisi_layout.addWidget(self.aisi_canvas)
        self.tab_widget.addTab(self.aisi_widget, "AISI Processing Map")
        
        # Tab for Deform Particle
        self.deform_widget = QWidget()
        self.deform_layout = QVBoxLayout(self.deform_widget)
        self.deform_figure = Figure(figsize=(10, 8), dpi=100)
        self.deform_canvas = FigureCanvas(self.deform_figure)
        self.deform_layout.addWidget(self.deform_canvas)
        self.tab_widget.addTab(self.deform_widget, "Deform Particle")
        
        # Tab for Simufact Particle
        self.simufact_widget = QWidget()
        self.simufact_layout = QVBoxLayout(self.simufact_widget)
        self.simufact_figure = Figure(figsize=(10, 8), dpi=100)
        self.simufact_canvas = FigureCanvas(self.simufact_figure)
        self.simufact_layout.addWidget(self.simufact_canvas)
        self.tab_widget.addTab(self.simufact_widget, "Simufact Particle")
        
        # Tab for NSM Billetsizing
        self.nsm_widget = QWidget()
        self.nsm_layout = QVBoxLayout(self.nsm_widget)
        self.nsm_figure = Figure(figsize=(10, 8), dpi=100)
        self.nsm_canvas = FigureCanvas(self.nsm_figure)
        self.nsm_layout.addWidget(self.nsm_canvas)
        self.tab_widget.addTab(self.nsm_widget, "NSM Billetsizing")
        
        # Tab for STL Preview
        self.stl_preview_widget = QWidget()
        self.stl_preview_layout = QVBoxLayout(self.stl_preview_widget)
        self.stl_preview_figure = Figure(figsize=(10, 8), dpi=100)
        self.stl_preview_canvas = FigureCanvas(self.stl_preview_figure)
        self.stl_preview_layout.addWidget(self.stl_preview_canvas)
        self.tab_widget.addTab(self.stl_preview_widget, "STL Preview")
        
        # Layout for control buttons
        control_layout = QHBoxLayout()
        
        # Save button
        self.save_button = QPushButton("Save Graph")
        self.save_button.clicked.connect(self.save_current_figure)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3b73d1;
            }
        """)
        control_layout.addWidget(self.save_button)
        
        # Results status
        self.status_label = QLabel("Visualization ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: #4a86e8; font-size: 12px;")
        control_layout.addWidget(self.status_label)
        
        self.layout.addLayout(control_layout)
        
        # Custom color schemes
        self.create_custom_colormaps()
        
    def create_custom_colormaps(self):
        """Create custom color scales"""
        # Custom color scheme for dissipation
        colors_dissipation = [(0, '#2c3e50'), (0.3, '#2980b9'), 
                             (0.5, '#27ae60'), (0.7, '#f39c12'), 
                             (1.0, '#e74c3c')]
        self.cmap_dissipation = LinearSegmentedColormap.from_list('custom_dissipation', colors_dissipation)
        
        # Custom color scheme for instability
        colors_instability = [(0, '#880000'), (0.5, '#d35400'), (1.0, '#c0392b')]
        self.cmap_instability = LinearSegmentedColormap.from_list('custom_instability', colors_instability)
    
    def save_current_figure(self):
        """Save current graph"""
        current_index = self.tab_widget.currentIndex()
        current_tab = self.tab_widget.tabText(current_index)
        
        if current_tab == "AISI Processing Map":
            self.aisi_figure.savefig('aisi_processing_map.png', dpi=300, bbox_inches='tight')
            self.status_label.setText("AISI Processing Map saved: aisi_processing_map.png")
        elif current_tab == "Deform Particle":
            self.deform_figure.savefig('deform_processing_map.png', dpi=300, bbox_inches='tight')
            self.status_label.setText("Deform Processing Map saved: deform_processing_map.png")
        elif current_tab == "Simufact Particle":
            self.simufact_figure.savefig('simufact_processing_map.png', dpi=300, bbox_inches='tight')
            self.status_label.setText("Simufact Processing Map saved: simufact_processing_map.png")
        elif current_tab == "NSM Billetsizing":
            self.nsm_figure.savefig('nsm_billetsizing.png', dpi=300, bbox_inches='tight')
            self.status_label.setText("NSM Billetsizing saved: nsm_billetsizing.png")
        elif current_tab == "STL Preview":
            self.stl_preview_figure.savefig('stl_preview.png', dpi=300, bbox_inches='tight')
            self.status_label.setText("STL Preview saved: stl_preview.png")
    
    def display_processing_map_results(self, data_dict):
        """
        Display Processing Map results
        
        Args:
            data_dict: Results from Processing Map module
        """
        # Select appropriate tab based on data source
        source = data_dict.get('source', 'aisi')
        
        if source == 'deform':
            target_figure = self.deform_figure
            target_canvas = self.deform_canvas
            target_widget = self.deform_widget
            title_prefix = "Deform Particle"
        elif source == 'simufact':
            target_figure = self.simufact_figure
            target_canvas = self.simufact_canvas
            target_widget = self.simufact_widget
            title_prefix = "Simufact Particle"
        else:  # Default AISI
            target_figure = self.aisi_figure
            target_canvas = self.aisi_canvas
            target_widget = self.aisi_widget
            title_prefix = "AISI"
        
        # Clear previous graphs
        target_figure.clear()
        
        # Draw graph using 2D mode
        ax = target_figure.add_subplot(111)
        
        # Data dict contains:
        # - X, Y: meshgrid data
        # - Z: dissipation values
        # - Z1: instability values
        if 'X' in data_dict and 'Y' in data_dict and 'Z' in data_dict and 'Z1' in data_dict:
            X = data_dict['X']
            Y = data_dict['Y']
            Z = data_dict['Z']
            Z1 = data_dict['Z1']
            strain = data_dict.get('strain', 0.5)  # Default value
            plot_type = data_dict.get('plot_type', '2D')
            
            if plot_type == '2D':
                # Draw graph
                # Draw dissipation contours
                levels = np.linspace(0.1, 0.9, 17)  # More contour lines for detail
                CS = ax.contour(X, Y, Z, levels=levels, cmap=self.cmap_dissipation, linewidths=1.2)
                ax.clabel(CS, inline=True, fontsize=10, fmt='%.2f')
                
                # Colored dissipation map
                filled_contour = ax.contourf(X, Y, Z, levels=levels, cmap=self.cmap_dissipation, alpha=0.7)
                cbar = target_figure.colorbar(filled_contour, ax=ax)
                cbar.set_label('Dissipation Efficiency', fontsize=14, fontweight='bold')
                cbar.ax.tick_params(labelsize=12)
                
                # Draw instability zones
                instability = ax.contourf(X, Y, Z1, [-1000, 0.0], colors='red', alpha=0.4)
                
                # Set graph properties
                ax.set_title(f'{title_prefix} Processing Map (Strain = {strain:.2f})', 
                             fontsize=18, fontweight='bold', pad=15, color='#0066cc')  # Enhanced title with blue color
                ax.set_xlabel('Temperature [°C]', fontsize=16, fontweight='bold', labelpad=10)
                ax.set_ylabel('Log(Strain rate) [s$^{-1}$]', fontsize=16, fontweight='bold', labelpad=10)
                ax.tick_params(axis='both', labelsize=14)
                ax.grid(True, linestyle='--', alpha=0.5, color='gray')
                
                # Set axis limits (if needed)
                if source == 'aisi':
                    ax.set_xlim([900, 1200])
                    ax.set_ylim([-2, 1])
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='red', alpha=0.4, label='Instability Region')]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
            
            elif plot_type == '3D' or plot_type.lower() in ['instability', 'dissipation']:
                # Draw 3D graph
                ax = target_figure.add_subplot(111, projection='3d')
                
                if plot_type.lower() == 'instability':
                    surf = ax.contourf(X, Y, Z1, zdir='z', offset=strain, levels=[-1000, -0.], 
                                      colors='red', alpha=0.5)
                    ax.set_title(f'{title_prefix} Instability Map (Strain = {strain:.2f})', 
                                 fontsize=18, fontweight='bold', pad=15, color='#0066cc')  # Enhanced title
                else:  # Dissipation or default 3D
                    levels = np.linspace(0.3, 0.9, 15)  # Contour lines
                    surf = ax.contourf(X, Y, Z, zdir='z', offset=strain, levels=levels, 
                                       cmap=self.cmap_dissipation, alpha=0.8)
                    cbar = target_figure.colorbar(surf, ax=ax, shrink=0.8)
                    cbar.set_label('Dissipation Efficiency', fontsize=14, fontweight='bold')
                    cbar.ax.tick_params(labelsize=12)
                    ax.set_title(f'{title_prefix} Dissipation Map (Strain = {strain:.2f})', 
                                fontsize=18, fontweight='bold', pad=15, color='#0066cc')  # Enhanced title
                
                # Set 3D graph properties
                ax.set_xlabel('Temperature [°C]', fontsize=16, fontweight='bold', labelpad=15)
                ax.set_ylabel('Log(Strain rate) [s$^{-1}$]', fontsize=16, fontweight='bold', labelpad=15)
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
                ax.tick_params(axis='z', labelsize=14)
                
                # Configure 3D view
                ax.view_init(elev=35, azim=-40)
                
                # Set axis limits (if needed)
                if source == 'aisi':
                    ax.set_xlim([900, 1200])
                    ax.set_ylim([-2, 1])
                    ax.set_zlim([0, 1])
        else:
            ax.text(0.5, 0.5, "Data not found", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=18,
                    color='red')
        
        # Update graph
        target_figure.tight_layout()
        target_canvas.draw()
        
        # Activate tab
        self.tab_widget.setCurrentWidget(target_widget)
        
        # Update status
        self.status_label.setText(f"{title_prefix} Processing Map (strain={strain:.2f}) results displayed")
        
    def display_nsm_billetsizing_results(self, results_data):
        """
        Display NSM Billetsizing results
        
        Args:
            results_data: Dictionary with forging ratios, length changes, etc.
        """
        # Clear previous graphs
        self.nsm_figure.clear()
        
        # Create a 2x1 subplot layout
        gs = self.nsm_figure.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # Forging ratios plot
        ax1 = self.nsm_figure.add_subplot(gs[0])
        
        if 'forging_ratios' in results_data:
            forging_ratios = results_data['forging_ratios']
            x = range(1, len(forging_ratios) + 1)
            
            # Plot forging ratios
            ax1.bar(x, forging_ratios, color='#4a86e8', alpha=0.7, width=0.6)
            ax1.set_title('Forging Ratios by Pass', fontsize=16, fontweight='bold', pad=15)
            ax1.set_xlabel('Pass Number', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Forging Ratio', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Add value labels
            for i, v in enumerate(forging_ratios):
                ax1.text(i+1, v + 0.5, f"{v:.1f}", 
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
        else:
            ax1.text(0.5, 0.5, "Forging ratio data not found", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax1.transAxes,
                    fontsize=14,
                    color='red')
        
        # Length changes plot
        ax2 = self.nsm_figure.add_subplot(gs[1])
        
        if 'length_changes' in results_data:
            length_changes = results_data['length_changes']
            x = range(1, len(length_changes) + 1)
            
            # Plot length changes
            ax2.plot(x, length_changes, 'o-', linewidth=2, markersize=8, color='#34a853')
            ax2.set_title('Length Changes by Pass', fontsize=16, fontweight='bold', pad=15)
            ax2.set_xlabel('Pass Number', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Length (mm)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # Add value labels
            for i, v in enumerate(length_changes):
                ax2.text(i+1, v + (max(length_changes) * 0.03), f"{v:.0f}", 
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
        else:
            ax2.text(0.5, 0.5, "Length change data not found", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes,
                    fontsize=14,
                    color='red')
        
        # Apply tight layout
        self.nsm_figure.tight_layout()
        
        # Draw the figure
        self.nsm_canvas.draw()
        
        # Activate tab
        self.tab_widget.setCurrentWidget(self.nsm_widget)
        
        # Update status
        self.status_label.setText("NSM Billetsizing results displayed")
    
    def display_stl_preview(self, stl_mesh):
        """
        Display STL preview using PyVista mesh
        
        Args:
            stl_mesh: PyVista mesh object
        """
        try:
            # Clear previous figure
            self.stl_preview_figure.clear()
            
            # Check if PyVista is available
            import pyvista as pv
            from pyvista.plotting.matplotlib_plotting import _qt_figure_to_mpl_figure
            
            # Create PyVista plotter
            p = pv.Plotter(off_screen=True, notebook=False)
            
            # Add mesh to plotter
            p.add_mesh(stl_mesh, color='tan', show_edges=False)
            
            # Set camera position
            p.camera_position = 'iso'
            p.set_background('white')
            
            # Render figure and get Matplotlib figure
            p.show(screenshot=False)
            mpl_fig = _qt_figure_to_mpl_figure(p.ren_win)
            
            # Copy the content to our figure
            for ax in mpl_fig.axes:
                self.stl_preview_figure.axes.append(ax)
            
            # Draw figure
            self.stl_preview_canvas.draw()
            
            # Activate tab
            self.tab_widget.setCurrentWidget(self.stl_preview_widget)
            
            # Update status
            self.status_label.setText("STL preview displayed")
            
        except ImportError:
            # Alternative: use matplotlib to display simple representation
            try:
                ax = self.stl_preview_figure.add_subplot(111, projection='3d')
                
                # Convert PyVista mesh to simple points
                points = stl_mesh.points
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]
                
                # Plot a subsample of points for faster rendering
                subsample = min(5000, len(x))
                indices = np.linspace(0, len(x)-1, subsample).astype(int)
                
                ax.scatter(x[indices], y[indices], z[indices], s=1, alpha=0.5)
                ax.set_title("STL Preview (Simplified)", fontsize=16)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
                
                # Draw figure
                self.stl_preview_figure.tight_layout()
                self.stl_preview_canvas.draw()
                
                # Activate tab
                self.tab_widget.setCurrentWidget(self.stl_preview_widget)
                
                # Update status
                self.status_label.setText("STL preview displayed (simplified view)")
                
            except Exception as e:
                # Show error message
                self.stl_preview_figure.clear()
                ax = self.stl_preview_figure.add_subplot(111)
                ax.text(0.5, 0.5, f"Error displaying STL: {str(e)}", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        fontsize=14,
                        color='red')
                self.stl_preview_figure.tight_layout()
                self.stl_preview_canvas.draw()
                
                # Activate tab
                self.tab_widget.setCurrentWidget(self.stl_preview_widget)
                
                # Update status
                self.status_label.setText("Error displaying STL preview")