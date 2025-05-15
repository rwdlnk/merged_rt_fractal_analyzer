#!/usr/bin/env python3
print("Testing fractal generation...")

from fractal_analyzer import FractalAnalyzer

try:
    # Create analyzer
    analyzer = FractalAnalyzer(fractal_type='koch')
    print("✓ Successfully created FractalAnalyzer instance")
    
    # Generate a Koch curve
    print("Generating Koch curve...")
    curve, segments = analyzer.generate_fractal('koch', level=3)
    print(f"✓ Successfully generated Koch curve with {len(segments)} segments")
    
    # Calculate fractal dimension
    print("Calculating fractal dimension...")
    fd, error, box_sizes, box_counts, bounding_box, intercept = analyzer.calculate_fractal_dimension(segments)
    print(f"✓ Calculated fractal dimension: {fd:.6f} ± {error:.6f}")
    
    # Save visualization
    print("Creating visualization...")
    plot_file = analyzer.visualizer.plot_fractal_curve(segments, bounding_box, plot_boxes=True, box_sizes=box_sizes, box_counts=box_counts, level=3)
    print(f"✓ Created visualization: {plot_file}")
    
    print("Fractal generation test completed successfully!")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
