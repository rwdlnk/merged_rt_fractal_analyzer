#!/usr/bin/env python3
print("Testing imports...")

try:
    from fractal_analyzer import FractalAnalyzer, FractalBase, BoxCounter, FractalVisualizer
    print("✓ Successfully imported fractal_analyzer core modules")
except ImportError as e:
    print(f"✗ Failed to import fractal_analyzer modules: {e}")

try:
    from fractal_analyzer.analysis_tools import FractalAnalysisTools
    print("✓ Successfully imported FractalAnalysisTools")
except ImportError as e:
    print(f"✗ Failed to import FractalAnalysisTools: {e}")

try:
    from rt_analyzer import RTAnalyzer
    print("✓ Successfully imported RTAnalyzer")
except ImportError as e:
    print(f"✗ Failed to import RTAnalyzer: {e}")

print("Import test completed.")
