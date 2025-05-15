# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import LogLocator, FuncFormatter
import time
from typing import List, Tuple
from scipy import stats
import os

class FractalVisualizer:
    """Visualization tools for fractal analysis."""
    
    def __init__(self, fractal_type=None, base=None):
        """Initialize the visualizer."""
        self.fractal_type = fractal_type
        self.base = base  # Reference to the FractalBase object
    
    def plot_fractal_curve(self, segments, bounding_box=None, plot_boxes=False, 
                          box_sizes=None, box_counts=None, custom_filename=None, level=None):
        """Plot the fractal curve with optional box overlay."""
        import matplotlib
        
        # Get segment count
        segment_count = len(segments)
        
        # Determine if this is a file-based curve with unknown fractal type
        is_file_based = self.fractal_type is None or (custom_filename and not level)
        
        # Set parameters based on fractal type, level, and segment count
        if is_file_based and segment_count > 20000:
            # Large file-based dataset
            matplotlib.rcParams['agg.path.chunksize'] = 50000
            plot_dpi = 200
            fig_size = (12, 10)
            use_rasterized = True
        else:
            # Default settings
            matplotlib.rcParams['agg.path.chunksize'] = 20000
            plot_dpi = 300
            fig_size = (10, 8)
            use_rasterized = False
        
        # Set simplification threshold
        matplotlib.rcParams['path.simplify_threshold'] = 0.1
        
        # Create figure
        plt.figure(figsize=fig_size)
        
        # Convert segments to plotting format
        start_time = time.time()
        print("Plotting curve segments...")
        
        # For large datasets, use a simplified plotting method
        if segment_count > 10000:
            print(f"Large dataset ({segment_count} segments), using simplified plotting...")
            step = max(1, segment_count // 20000)
            sampled_segments = segments[::step]
            print(f"Sampled down to {len(sampled_segments)} segments for visualization")
            
            x_points = []
            y_points = []
            for (x1, y1), (x2, y2) in sampled_segments:
                x_points.extend([x1, x2, None])  # None creates a break in the line
                y_points.extend([y1, y2, None])
            
            x_points = x_points[:-1]
            y_points = y_points[:-1]
            
            plt.plot(x_points, y_points, 'k-', linewidth=1, rasterized=use_rasterized)
        else:
            # Normal plotting for smaller datasets
            x_points = []
            y_points = []
            for (x1, y1), (x2, y2) in segments:
                x_points.extend([x1, x2, None])
                y_points.extend([y1, y2, None])
            
            x_points = x_points[:-1]
            y_points = y_points[:-1]
            
            plt.plot(x_points, y_points, 'k-', linewidth=1, rasterized=use_rasterized)
        
        print(f"Curve plotting completed in {time.time() - start_time:.2f} seconds")
        
        # Calculate bounding box if not provided
        if bounding_box is None:
            min_x = min(min(s[0][0], s[1][0]) for s in segments)
            max_x = max(max(s[0][0], s[1][0]) for s in segments)
            min_y = min(min(s[0][1], s[1][1]) for s in segments)
            max_y = max(max(s[0][1], s[1][1]) for s in segments)
            bounding_box = (min_x, min_y, max_x, max_y)
        
        # Unpack the bounding box
        min_x, min_y, max_x, max_y = bounding_box
        
        # Set a slightly larger margin for the plot view
        view_margin = max(max_x - min_x, max_y - min_y) * 0.05
        plt.xlim(min_x - view_margin, max_x + view_margin)
        plt.ylim(min_y - view_margin, max_y + view_margin)
        
        # If requested, plot boxes at a specific scale
        if plot_boxes and box_sizes is not None and box_counts is not None:
            # Choose the smallest box size to visualize
            smallest_idx = len(box_sizes) - 1
            box_size = box_sizes[smallest_idx]
            self.plot_box_overlay(segments, box_size, min_x, min_y, max_x, max_y, bounding_box)
        
        # Title and labels
        title = f'{self.fractal_type.capitalize() if self.fractal_type else "Fractal"} Curve'
        if level is not None:
            title += f' (Level {level})'
        if plot_boxes and box_sizes is not None:
            smallest_idx = len(box_sizes) - 1
            box_size = box_sizes[smallest_idx]
            title += f'\nwith Box Counting Overlay (Box Size: {box_size:.6f})'
        
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Use the provided custom_filename if available, otherwise create default filename
        if custom_filename:
            curve_filename = custom_filename
        else:
            curve_filename = f'{self.fractal_type if self.fractal_type else "fractal"}_curve'
            if level is not None:
                curve_filename += f'_Level_{level}'
            curve_filename += '.png'
        
        # Save the plot
        plt.savefig(curve_filename, dpi=plot_dpi)
        plt.close()
        
        return curve_filename
    
    def plot_loglog(self, box_sizes, box_counts, fractal_dimension, error, intercept=None, 
                   custom_filename=None):
        """Plot ln-ln analysis."""
        plt.figure(figsize=(10, 8))
        
        plt.loglog(box_sizes, box_counts, 'bo-', label='Data points')
        
        # Perform linear regression for plotting the fit line
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # If intercept wasn't provided, calculate it
        if intercept is None:
            _, intercept, _, _, _ = stats.linregress(log_sizes, log_counts)
        
        # Plot the linear regression line
        fit_counts = np.exp(intercept + (-fractal_dimension) * log_sizes)
        plt.loglog(box_sizes, fit_counts, 'r-', 
                   label=f'Fit: D = {fractal_dimension:.4f} ± {error:.4f}')
        
        # Custom formatter for scientific notation
        def scientific_formatter(x, pos):
            if x == 0:
                return '0'
            
            exponent = int(np.log10(x))
            coef = x / 10**exponent
            
            if abs(coef - 1.0) < 0.01:
                return r'$10^{%d}$' % exponent
            elif abs(coef - 3.0) < 0.01:
                return r'$3{\times}10^{%d}$' % exponent
            else:
                return r'${%.1f}{\times}10^{%d}$' % (coef, exponent)
        
        # Set axis properties with FuncFormatter
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        
        plt.title('Box Counting: ln(N) vs ln(1/r)')
        plt.xlabel('Box Size (r)')
        plt.ylabel('Number of Boxes (N)')
        plt.legend()
        plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
        
        # Use provided filename or generate default
        if custom_filename:
            filename = custom_filename
        else:
            # Use fractal type in filename if available
            filename = 'box_counting_loglog'
            if self.fractal_type:
                filename = f'{self.fractal_type}_box_counting_loglog'
            filename += '.png'
        
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return filename

    def plot_dimension_vs_level(self, levels, dimensions, errors, r_squared, 
                               theoretical_dimension, fractal_type):
        """Plot dimension vs. iteration level."""
        plt.figure(figsize=(10, 6))
        plt.errorbar(levels, dimensions, yerr=errors, fmt='o-', capsize=5, 
                     label='Calculated Dimension')
        
        if theoretical_dimension is not None:
            plt.axhline(y=theoretical_dimension, color='r', linestyle='--', 
                        label=f'Theoretical Dimension ({theoretical_dimension:.6f})')
        
        plt.xlabel(f'{fractal_type.capitalize()} Curve Iteration Level')
        plt.ylabel('Fractal Dimension')
        plt.title(f'Fractal Dimension vs. {fractal_type.capitalize()} Curve Iteration Level')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add a second y-axis for R-squared values
        ax2 = plt.gca().twinx()
        ax2.plot(levels, r_squared, 'g--', marker='s', label='R-squared')
        ax2.set_ylabel('R-squared', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim([0.9, 1.01])
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        
        filename = f'{fractal_type}_dimension_vs_level.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return filename

    def plot_box_overlay(self, segments, box_size, min_x, min_y, max_x, max_y, bounding_box, 
                         segment_grid=None, grid_size=None):
        """
        Plot box overlay for visual verification - optimized version.
    
        Args:
            segments: List of line segments representing the fractal
            box_size: Size of boxes to draw
            min_x, min_y, max_x, max_y: Bounding coordinates
            bounding_box: Bounding box tuple (min_x, min_y, max_x, max_y)
            segment_grid: Optional pre-computed spatial index
            grid_size: Optional grid size for spatial index
        """
        box_time = time.time()
        print("Generating box overlay (optimized)...")
    
        # Calculate box coordinates
        num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
        print(f"Grid dimensions: {num_boxes_x} x {num_boxes_y} boxes (size: {box_size})")
    
        # Create spatial index for efficient intersection tests if not provided
        if segment_grid is None or grid_size is None:
            from .analysis import BoxCounter
            box_counter = BoxCounter(self.base)
            grid_size = box_size * 2
            segment_grid, _, _ = box_counter.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, grid_size)
    
        print(f"Spatial index created in {time.time() - box_time:.2f} seconds")
        box_time = time.time()
    
        # Performance optimization: Use lists for vertex arrays
        box_vertices_x = []
        box_vertices_y = []
    
        # For very large grids, use sampling to speed up visualization
        sampling_factor = 1
        if num_boxes_x * num_boxes_y > 1000000:
            sampling_factor = max(1, int(np.sqrt(num_boxes_x * num_boxes_y) / 1000))
            print(f"Using sampling factor {sampling_factor} for large grid")
    
        boxes_checked = 0
        boxes_occupied = 0
    
        # Instead of drawing individual rectangles, build lists of vertices
        for i in range(0, num_boxes_x, sampling_factor):
            for j in range(0, num_boxes_y, sampling_factor):
                box_xmin = min_x + i * box_size
                box_ymin = min_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size
            
                # Find which grid cell this box belongs to
                cell_x = int((box_xmin - min_x) / grid_size)
                cell_y = int((box_ymin - min_y) / grid_size)
            
                # Get segments that might intersect this box
                segments_to_check = set()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        adjacent_key = (cell_x + dx, cell_y + dy)
                        segments_to_check.update(segment_grid.get(adjacent_key, []))
            
                boxes_checked += 1
            
                # Check for intersection with the candidate segments
                box_occupied = False
                for seg_idx in segments_to_check:
                    (x1, y1), (x2, y2) = segments[seg_idx]
                    if self.base.liang_barsky_line_box_intersection(
                            x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                        box_occupied = True
                        boxes_occupied += 1
                        break
            
                # If box is occupied, add lines for its edges
                if box_occupied:
                    # Add vertices for the box outline (in the order to form lines)
                    box_vertices_x.extend([box_xmin, box_xmax, box_xmax, box_xmin, box_xmin])
                    box_vertices_y.extend([box_ymin, box_ymin, box_ymax, box_ymax, box_ymin])
                    # Add None to separate boxes
                    box_vertices_x.append(None)
                    box_vertices_y.append(None)
    
        print(f"Box intersection tests: checked {boxes_checked}, found {boxes_occupied} occupied")
        print(f"Intersection tests completed in {time.time() - box_time:.2f} seconds")
        box_time = time.time()
    
        # Plot all box outlines at once as a single line collection
        if box_vertices_x:
            plt.plot(box_vertices_x, box_vertices_y, 'r-', linewidth=0.8, alpha=0.7)
    
        print(f"Box rendering completed in {time.time() - box_time:.2f} seconds")
        print(f"Total boxes drawn: {boxes_occupied}")

    def create_quick_box_visualization(self, fractal_type, output_dir, level=6):
        """
        Create a quick visualization with box overlay for a specific fractal.
        This function bypasses the complex spatial indexing for speed.
    
        Args:
            fractal_type: Type of fractal to visualize
            output_dir: Directory to save visualizations
            level: Iteration level for the fractal
        
        Returns:
            tuple: (curve_file_path, box_overlay_file_path)
        """
        print(f"Creating quick visualization for {fractal_type} (level {level})...")
    
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
    
        # Initialize analyzer with the correct fractal type
        from .main import FractalAnalyzer
        analyzer = FractalAnalyzer(fractal_type)
    
        # Generate fractal
        _, segments = analyzer.generate_fractal(fractal_type, level=level)
        print(f"Generated {fractal_type} with {len(segments)} segments")
    
        # Calculate bounding box
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
    
        # Add margin
        margin = max(max_x - min_x, max_y - min_y) * 0.05
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
    
        # Figure for plain fractal
        plt.figure(figsize=(10, 10))
    
        # Draw segments
        for (x1, y1), (x2, y2) in segments:
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.7)
    
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.axis('equal')
        plt.grid(False)
        plt.title(f'{fractal_type.capitalize()} Curve (Level {level})')
    
        # Save plain curve
        plain_file = os.path.join(output_dir, f"{fractal_type}_curve.png")
        plt.savefig(plain_file, dpi=300)
        plt.close()
    
        # Choose a good box size for visualization
        extent = max(max_x - min_x, max_y - min_y)
        box_size = extent / 40  # Reasonably sized boxes
    
        # Figure for box overlay
        plt.figure(figsize=(10, 10))
    
        # Draw segments
        for (x1, y1), (x2, y2) in segments:
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.7)
    
        # Calculate and draw boxes directly
        box_vertices_x = []
        box_vertices_y = []
    
        num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
        print(f"Grid size: {num_boxes_x}x{num_boxes_y}")
    
        # Find and draw occupied boxes
        boxes_occupied = 0
        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                box_xmin = min_x + i * box_size
                box_ymin = min_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size
            
                # Check if any segment intersects with this box
                for (x1, y1), (x2, y2) in segments:
                    if self.base.liang_barsky_line_box_intersection(
                            x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                        # Add vertices for the box outline
                        box_vertices_x.extend([box_xmin, box_xmax, box_xmax, box_xmin, box_xmin, None])
                        box_vertices_y.extend([box_ymin, box_ymin, box_ymax, box_ymax, box_ymin, None])
                        boxes_occupied += 1
                        break
    
        # Plot all box outlines at once
        plt.plot(box_vertices_x, box_vertices_y, 'r-', linewidth=0.8, alpha=0.7)
    
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.axis('equal')
        plt.grid(False)
        plt.title(f'{fractal_type.capitalize()} with Box Counting (Size: {box_size:.6f})')
    
        # Save box overlay
        box_file = os.path.join(output_dir, f"{fractal_type}_box_overlay.png")
        plt.savefig(box_file, dpi=300)
        plt.close()
    
        print(f"Created quick visualizations for {fractal_type} ({boxes_occupied} boxes drawn)")
        return plain_file, box_file

