Yes, here's the final README.md for the examples directory:

# examples

Example applications and demonstrations of the fractal analysis tools.

## Overview

This directory contains example applications that showcase how to use the `fractal_analyzer` and `rt_analyzer` packages for various analysis tasks. These examples serve as practical demonstrations and starting points for your own analysis.

## Directory Structure

- **fractal_examples/**: Examples for basic fractal analysis
- **rt_examples/**: Examples for RT-specific analysis

## Fractal Examples

The `fractal_examples/` directory includes:

### koch_analysis.py

Demonstrates how to generate and analyze a Koch curve:

```python
from fractal_analyzer import FractalAnalyzer

# Create analyzer for Koch curve
analyzer = FractalAnalyzer('koch')

# Generate Koch curves at different levels
for level in range(1, 6):
    curve, segments = analyzer.generate_fractal('koch', level=level)
    
    # Calculate fractal dimension
    fd, error, box_sizes, box_counts, bbox, intercept = analyzer.calculate_fractal_dimension(segments)
    print(f"Level {level}: D = {fd:.6f} ± {error:.6f}")
    
    # Create visualization
    analyzer.visualizer.plot_fractal_curve(segments, bbox, level=level)
```

### linear_region_study.py

Shows how to study the effect of linear region selection on dimension calculation:

```python
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

analyzer = FractalAnalyzer('koch')
tools = FractalAnalysisTools(analyzer)

# Generate a Koch curve
curve, segments = analyzer.generate_fractal('koch', level=4)

# Analyze linear region selection
windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = tools.analyze_linear_region(
    segments, fractal_type='koch', plot_results=True
)
```

### custom_curve_analysis.py

Demonstrates how to analyze a custom curve from a file:

```python
from fractal_analyzer import FractalAnalyzer

# Create generic analyzer
analyzer = FractalAnalyzer()

# Read curve from file
segments = analyzer.read_line_segments("custom_curve.dat")

# Analyze dimension
fd, error, box_sizes, box_counts, bbox, intercept = analyzer.calculate_fractal_dimension(segments)
print(f"Custom curve dimension: {fd:.6f} ± {error:.6f}")
```

## RT Examples

The `rt_examples/` directory includes:

### temporal_analysis.py

Shows how to analyze the temporal evolution of RT interfaces:

```python
from rt_analyzer import RTAnalyzer
import glob

# Create analyzer
analyzer = RTAnalyzer("./temporal_output")

# Get all VTK files in time order
vtk_files = sorted(glob.glob("RT800x800-*.vtk"))

# Process all files
results = analyzer.process_vtk_series(vtk_files, resolution=800)

# Plot evolution over time
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.errorbar(results['time'], results['fractal_dim'], yerr=results['fd_error'],
            fmt='o-', capsize=3)
plt.xlabel('Time')
plt.ylabel('Fractal Dimension')
plt.grid(True)
plt.savefig("dimension_evolution.png")
```

### multifractal_demo.py

Demonstrates how to perform multifractal analysis:

```python
from rt_analyzer import RTAnalyzer
import numpy as np

# Create analyzer
analyzer = RTAnalyzer("./multifractal_output")

# Read VTK file
data = analyzer.read_vtk_file("RT800x800-9000.vtk")

# Define q-values
q_values = np.arange(-5, 5.1, 0.5)

# Perform multifractal analysis
result = analyzer.compute_multifractal_spectrum(
    data, q_values=q_values, output_dir="./multifractal_output"
)

print(f"D(0) = {result['D0']:.4f}")
print(f"D(1) = {result['D1']:.4f}")
print(f"D(2) = {result['D2']:.4f}")
```

### resolution_convergence.py

Shows how to analyze resolution convergence:

```python
from rt_analyzer import RTAnalyzer

# Create analyzer
analyzer = RTAnalyzer("./convergence_output")

# Analyze files at different resolutions
resolutions = [100, 200, 400, 800]
vtk_files = [f"RT{res}x{res}-9000.vtk" for res in resolutions]

# Run convergence analysis
results = analyzer.analyze_resolution_convergence(vtk_files, resolutions)

# Extrapolate to infinite resolution
fd_values = [row['fractal_dim'] for _, row in results.iterrows()]
extrapolation = analyzer.extrapolate_to_infinite_resolution(
    resolutions, fd_values, name="Fractal Dimension"
)

print(f"Extrapolated dimension at infinite resolution: {extrapolation['value']:.6f} ± {extrapolation['error']:.6f}")
```

## Running the Examples

Most examples can be run directly with Python:

```bash
python examples/fractal_examples/koch_analysis.py
python examples/rt_examples/temporal_analysis.py
```

## Adapting the Examples

These examples are designed to be easily adapted for your specific research needs. They demonstrate the core workflows and can be modified to suit different fractal types, simulation data, or analysis parameters.
