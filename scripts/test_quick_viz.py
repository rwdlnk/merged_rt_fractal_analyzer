#!/usr/bin/env python3
"""
Test script for quick box visualization.
"""

from fractal_analyzer import FractalAnalyzer
import os

def main():
    """Run quick visualizations for all fractal types."""
    output_dir = "./quick_viz_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # For each fractal
    for fractal_type in ['koch', 'sierpinski', 'dragon', 'hilbert', 'minkowski']:
        try:
            # Set level based on complexity
            level = 5 if fractal_type == 'hilbert' else 6
            
            # Create visualizer and run quick visualization
            analyzer = FractalAnalyzer(fractal_type)
            
            fractal_dir = os.path.join(output_dir, fractal_type)
            curve_file, box_file = analyzer.visualizer.create_quick_box_visualization(
                fractal_type, 
                fractal_dir, 
                level=level
            )
            
            print(f"Created visualizations for {fractal_type}:")
            print(f"  Curve: {curve_file}")
            print(f"  Box overlay: {box_file}")
            
        except Exception as e:
            print(f"Error creating visualization for {fractal_type}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
