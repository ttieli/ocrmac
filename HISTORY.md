# History

## 1.3.0 (2026-01-19)
* Added **Smart OCR** feature: Automatic strategy selection for optimal OCR results.
* Added **Adaptive Slicing**: Intelligent image slicing at safe boundaries for long screenshots.
* Added **Text Cleaning**: Automatic duplicate removal, broken word merging, and format normalization.
* Reorganized module structure:
  - `ocrmac.processing`: SmartOCR, AdaptiveOCR, SmartSlicer
  - `ocrmac.analysis`: ImageAnalyzer, LayoutAnalyzer, RegionDetector, TableDetector
  - `ocrmac.preprocessing`: ImagePreprocessor
  - `ocrmac.postprocessing`: TextCleaner
* Added `smart_ocr()` convenience function for one-line intelligent OCR.
* Improved handling of Chinese text (headers, lists, broken characters).

## 1.1.0 (2026-01-19)
* Added **Table Recovery** feature: Detects and reconstructs table structures from OCR results.
* Added **Layout Analysis** feature: Detects paragraphs, headings, and lists.
* Improved `TableDetector` with configurable tolerances for rows and columns.
* Added `LayoutAnalyzer` for better document structure understanding.
* Export capabilities to Markdown for both tables and document layouts.

## 1.0.1 (2026-01-08)
* Added GitHub Actions workflow for PyPI releases
* Fixed build configuration for proper package distribution
* Updated test image comparison threshold for cross-system compatibility
* Added notebook for regenerating test reference images

## 1.0.0 (2024)
* Added LiveText framework support via `framework='livetext'`
* Added output granularity options for LiveText
* Fixed language code documentation
* Updated test infrastructure

## 0.1.0 (2022-12-30)
* First release on PyPI.
* Basic functionality for PIL and matplotlib
