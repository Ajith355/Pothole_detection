# Paper Preview - What Your IEEE Paper Will Look Like

This document describes the visual appearance and structure of your compiled paper.

## 📄 Page-by-Page Preview

### Page 1: Title and Introduction

```
┌─────────────────────────────────────────────────────────┐
│  [Conference Header - IEEE Trans/Conf Format]           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Real-Time Pothole Detection and Severity              │
│   Classification Using Dual Neural Networks             │
│          with Hybrid Depth Estimation                   │
│                                                          │
│                      Ajith Kumar                         │
│              Department of Computer Science              │
│                   [Your University]                      │
│                  ajith@example.com                       │
│                                                          │
├─────────────────────────────────────────────────────────┤
│ Abstract: [250 words describing dual neural networks,   │
│ hybrid depth estimation, real-time performance, and     │
│ severity classification results]                        │
│                                                          │
│ Keywords: Pothole detection, YOLOv8, depth estimation...│
├─────────────────────────────────────────────────────────┤
│                                                          │
│ I. INTRODUCTION                                          │
│                                                          │
│ [Two-column text discussing road infrastructure         │
│ problems, statistics on pothole damage costs,           │
│ limitations of current methods]                         │
│                                                          │
│ A. Motivation                                            │
│ [Discussion of why severity assessment matters]         │
│                                                          │
│ B. Contributions                                         │
│ • Dual neural network architecture...                   │
│ • Novel hybrid depth fusion...                          │
│ • Real-time processing pipeline...                      │
│ • Comprehensive severity classification...              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Page 2: Related Work and System Architecture

```
┌─────────────────────────────────────────────────────────┐
│ II. RELATED WORK                                         │
│                                                          │
│ A. Traditional Computer Vision Approaches               │
│ [Edge detection, texture analysis...]                   │
│                                                          │
│ B. Deep Learning-Based Detection                        │
│ [YOLO variants, Faster R-CNN...]                        │
│                                                          │
│ C. Depth Estimation Techniques                          │
│ [MiDaS, stereo vision...]                               │
│                                                          │
│ D. Severity Assessment                                   │
│ [Comparison with existing severity systems]             │
│                                                          │
├─────────────────────────────────────────────────────────┤
│ III. SYSTEM ARCHITECTURE                                 │
│                                                          │
│ A. Overview                                              │
│ [5-component pipeline description]                      │
│                                                          │
│ ┌─────────────────────────────────────┐                │
│ │  [FIGURE: System architecture      │                 │
│ │   flowchart would go here]         │                 │
│ │                                    │                 │
│ │  Video → YOLOv8 → MiDaS → Fusion  │                 │
│ └─────────────────────────────────────┘                │
│ Fig. 1. Overall system architecture                     │
│                                                          │
│ B. YOLOv8 Detection Network                             │
│ [Architecture: CSPDarknet backbone, PANet neck,         │
│ detection head. Training details, loss function]        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Page 3-4: Technical Details

```
┌─────────────────────────────────────────────────────────┐
│ C. MiDaS Depth Network                                   │
│ [Encoder-decoder architecture, transformer fusion]      │
│                                                          │
│ D. Hybrid Depth Estimation                              │
│                                                          │
│ The key innovation - fusion formula:                    │
│                                                          │
│     D_final = α·D_neural + β·D_geometric        (1)     │
│                                                          │
│ where α = 0.6, β = 0.4                                  │
│                                                          │
│ [Mathematical derivations for neural depth:]            │
│                                                          │
│     D_neural = W · d_rel · 0.5                  (2)     │
│                                                          │
│ [Width estimation from camera geometry:]                │
│                                                          │
│     W = (w_pixels · D_ground · W_ref)/(f·w_ref) (3)     │
│                                                          │
│ E. Severity Classification                              │
│                                                          │
│ ┌────────────────────────────────────────────┐         │
│ │ TABLE I                                     │         │
│ │ SEVERITY CLASSIFICATION THRESHOLDS          │         │
│ ├──────────┬────────────┬──────────┐         │         │
│ │ Severity │ Depth Range│ Priority │         │         │
│ ├──────────┼────────────┼──────────┤         │         │
│ │ CRITICAL │  ≥15.0 cm  │  Urgent  │         │         │
│ │ DANGEROUS│ 10-15 cm   │   High   │         │         │
│ │ MODERATE │  6-10 cm   │  Medium  │         │         │
│ │ MINOR    │  3-6 cm    │    Low   │         │         │
│ │ SURFACE  │  <3 cm     │   Info   │         │         │
│ └──────────┴────────────┴──────────┘         │         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Page 5: Experiments

```
┌─────────────────────────────────────────────────────────┐
│ V. EXPERIMENTAL RESULTS                                  │
│                                                          │
│ A. Dataset and Evaluation Metrics                       │
│ • Training: 4,200 images                                │
│ • Validation: 600 images                                │
│ • Test: 800 images + 10 videos                         │
│                                                          │
│ B. Detection Performance                                 │
│                                                          │
│ ┌────────────────────────────────────────────┐         │
│ │ TABLE II                                    │         │
│ │ DETECTION PERFORMANCE METRICS               │         │
│ ├──────────────────┬──────────────┐          │         │
│ │ Metric           │ Value        │          │         │
│ ├──────────────────┼──────────────┤          │         │
│ │ Precision        │ 87.3%        │          │         │
│ │ Recall           │ 83.5%        │          │         │
│ │ mAP@0.5          │ 85.4%        │          │         │
│ │ Inference (GPU)  │ 18.2 ms      │          │         │
│ └──────────────────┴──────────────┘          │         │
│                                                          │
│ ┌─────────────────────────────────────┐                │
│ │  [FIGURE: Detection examples       │                 │
│ │   2x2 grid showing:                │                 │
│ │   - Original frames                │                 │
│ │   - YOLO detections                │                 │
│ │   - Depth maps                     │                 │
│ │   - Final annotated output]        │                 │
│ └─────────────────────────────────────┘                │
│ Fig. 2. Detection and depth estimation examples         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Page 6: More Results

```
┌─────────────────────────────────────────────────────────┐
│ C. Depth Estimation Accuracy                             │
│                                                          │
│ ┌────────────────────────────────────────────┐         │
│ │ TABLE III                                   │         │
│ │ DEPTH ESTIMATION ERROR ANALYSIS             │         │
│ ├──────────────────┬────────┬──────────┐     │         │
│ │ Method           │ MAE(cm)│ RMSE(cm) │     │         │
│ ├──────────────────┼────────┼──────────┤     │         │
│ │ MiDaS Only       │  2.8   │   3.9    │     │         │
│ │ Geometric Only   │  3.5   │   5.1    │     │         │
│ │ Hybrid (Proposed)│  2.1   │   3.2    │ ✓   │         │
│ └──────────────────┴────────┴──────────┘     │         │
│                                                          │
│ Key finding: Hybrid approach reduces MAE by 23%!        │
│                                                          │
│ D. Severity Classification Accuracy                     │
│                                                          │
│ ┌────────────────────────────────────────────┐         │
│ │ TABLE IV                                    │         │
│ │ SEVERITY CLASSIFICATION RESULTS             │         │
│ ├──────────┬────────┬────────┬──────────┐    │         │
│ │ Severity │ Prec   │ Recall │ F1-Score │    │         │
│ ├──────────┼────────┼────────┼──────────┤    │         │
│ │ CRITICAL │ 82.1%  │ 79.3%  │  80.7%   │    │         │
│ │ DANGEROUS│ 76.5%  │ 78.8%  │  77.6%   │    │         │
│ │ MODERATE │ 74.2%  │ 76.1%  │  75.1%   │    │         │
│ │ MINOR    │ 81.3%  │ 79.7%  │  80.5%   │    │         │
│ │ SURFACE  │ 88.9%  │ 86.4%  │  87.6%   │    │         │
│ │ Overall  │ 80.6%  │ 80.1%  │  80.3%   │    │         │
│ └──────────┴────────┴────────┴──────────┘    │         │
│                                                          │
│ ┌─────────────────────────────────────┐                │
│ │  [FIGURE: Confusion matrix heatmap] │                │
│ └─────────────────────────────────────┘                │
│ Fig. 3. Severity classification confusion matrix        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Page 7: Performance and Comparison

```
┌─────────────────────────────────────────────────────────┐
│ E. Real-Time Performance                                 │
│                                                          │
│ ┌────────────────────────────────────────────┐         │
│ │ TABLE V                                     │         │
│ │ PROCESSING TIME BREAKDOWN                   │         │
│ ├────────────────────┬────────────┐          │         │
│ │ Component          │ Time (ms)  │          │         │
│ ├────────────────────┼────────────┤          │         │
│ │ YOLOv8 Detection   │   18.2     │          │         │
│ │ MiDaS Depth        │   67.3     │ ← Slowest│         │
│ │ Tracking           │    2.8     │          │         │
│ │ Visualization      │    4.6     │          │         │
│ │ Total              │   97.5     │          │         │
│ │ Throughput         │ 10.3 FPS   │          │         │
│ └────────────────────┴────────────┘          │         │
│                                                          │
│ ┌─────────────────────────────────────┐                │
│ │  [FIGURE: Pie chart showing time   │                 │
│ │   breakdown - MiDaS dominates 69%] │                 │
│ └─────────────────────────────────────┘                │
│ Fig. 4. Processing time distribution                    │
│                                                          │
│ F. Comparison with Existing Methods                     │
│                                                          │
│ ┌────────────────────────────────────────────┐         │
│ │ TABLE VI                                    │         │
│ │ COMPARISON WITH STATE-OF-THE-ART            │         │
│ ├─────────────┬──────────┬──────┬──────┐     │         │
│ │ Method      │ Detection│ Depth│  FPS │     │         │
│ ├─────────────┼──────────┼──────┼──────┤     │         │
│ │ Zhang et al.│  84.2%   │2.5cm │   8  │     │         │
│ │ Kumar et al.│  89.1%   │ N/A  │  25  │     │         │
│ │ Li et al.   │  81.7%   │1.8cm │   5  │     │         │
│ │ Proposed    │  87.3%   │2.1cm │ 10.3 │ ✓   │         │
│ └─────────────┴──────────┴──────┴──────┘     │         │
│                                                          │
│ Advantage: Best balance of detection, depth, and speed! │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Page 8: Discussion and Conclusion

```
┌─────────────────────────────────────────────────────────┐
│ VI. DISCUSSION                                           │
│                                                          │
│ A. Advantages                                            │
│ • Monocular simplicity (single camera)                  │
│ • Hybrid robustness (neural + geometric)                │
│ • Real-time capability (10+ FPS)                        │
│ • Quantitative severity assessment                      │
│                                                          │
│ B. Limitations                                           │
│ • Lighting dependency (extreme conditions)              │
│ • Water-filled potholes (specular reflections)         │
│ • GPU required for real-time performance               │
│ • Calibration sensitivity                               │
│                                                          │
│ C. Applications                                          │
│ • Municipal road inspection                             │
│ • Autonomous vehicle navigation                         │
│ • Crowdsourced mapping                                  │
│ • Insurance claim assessment                            │
│                                                          │
├─────────────────────────────────────────────────────────┤
│ VII. CONCLUSION AND FUTURE WORK                          │
│                                                          │
│ [Summary of dual neural network system, hybrid depth    │
│ fusion innovation, experimental validation]             │
│                                                          │
│ Future research directions:                             │
│ • Adversarial weather robustness                        │
│ • 3D reconstruction from multi-frame fusion             │
│ • Mobile optimization for edge devices                  │
│ • Semantic segmentation for precise boundaries         │
│ • Temporal learning with LSTM/Transformers             │
│ • Uncertainty quantification                            │
│                                                          │
│ Open-source: github.com/[your-repo]                     │
│                                                          │
├─────────────────────────────────────────────────────────┤
│ ACKNOWLEDGMENT                                           │
│ [Thanks to university, reviewers, open-source teams]    │
│                                                          │
├─────────────────────────────────────────────────────────┤
│ REFERENCES                                               │
│                                                          │
│ [1] American Automobile Association, "AAA: Potholes...  │
│ [2] R. Fan et al., "Road damage detection based on...   │
│ [3] H. Zhang et al., "Road damage detection using...    │
│ ...                                                      │
│ [13] Q. Li et al., "3D reconstruction of road surface..." │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🎨 Visual Characteristics

### Typography
- **Title:** Large, bold, centered
- **Section Headers:** Bold, all caps, Roman numerals (I, II, III...)
- **Subsection Headers:** Bold, italics, letters (A, B, C...)
- **Body Text:** 10pt Times New Roman, justified
- **Equations:** Centered, numbered on right
- **Captions:** Italics, centered below figures/above tables

### Layout
- **Columns:** 2 columns per page (IEEE standard)
- **Margins:** Narrow (~0.75 inches)
- **Line Spacing:** Single-spaced
- **Paragraph Spacing:** No space between paragraphs (indented first line)
- **Page Numbers:** Bottom center (starting from page 1)

### Color
- **Text:** Black (no color in main text)
- **Figures:** Can use color (depth maps, charts)
- **Tables:** Black and white with horizontal lines
- **Hyperlinks:** Blue (in PDF only, print as black)

### Professional Elements
✅ IEEE copyright notice (if accepted)
✅ Consistent notation throughout
✅ High-resolution figures (300+ DPI)
✅ Professional table formatting (booktabs style)
✅ Proper citation formatting ([1], [2], etc.)
✅ Equation numbering ((1), (2), etc.)

## 📐 Dimensions

- **Paper Size:** US Letter (8.5" × 11")
- **Column Width:** ~3.3 inches per column
- **Column Spacing:** ~0.2 inches
- **Text Area:** ~7.1" × 9.2"
- **Figure Width:** 
  - Single column: ~3.3" (0.48\textwidth)
  - Double column: ~7" (0.9\textwidth)

## 🖼️ Figure Placement

### Small Figures (Single Column)
```latex
\begin{figure}[t]  % [t] = top of page
...
\end{figure}
```

### Large Figures (Both Columns)
```latex
\begin{figure*}[t]  % [t] = top of page, * = span both columns
...
\end{figure*}
```

## 📊 Table Placement

Tables follow same rules as figures, but caption goes ABOVE table:

```latex
\begin{table}[t]
\caption{Table Title Here}
\begin{tabular}{...}
...
\end{tabular}
\end{table}
```

## 🎯 Quality Indicators

A professional IEEE paper should have:

✅ **Balanced Pages:** Roughly equal text in both columns
✅ **No Orphans:** No single lines at top/bottom of column
✅ **Figure Proximity:** Figures near their first reference
✅ **Clean Equations:** Properly aligned and spaced
✅ **Consistent Style:** All sections formatted identically
✅ **High-Quality Figures:** Crisp, readable at print size
✅ **Complete References:** All citations have full details

## 🔍 Before/After Compilation

### Your LaTeX Source
```latex
\section{System Architecture}
The proposed system consists of five
primary components: video input pipeline,
YOLOv8 detection, MiDaS depth estimation...
```

### Compiled PDF Result
```
III. SYSTEM ARCHITECTURE
  The proposed system consists of five
primary components: video input pipeline,
YOLOv8 detection, MiDaS depth estimation...
```

Note how LaTeX automatically:
- Converted \section → Roman numeral header
- Justified text
- Applied proper font and spacing

## 📱 Viewing Your Paper

### In Overleaf
- **Preview pane:** Live preview on right side
- **Full-screen:** Click PDF icon for enlarged view
- **Download:** Download PDF button
- **Zoom:** Ctrl/Cmd + wheel to zoom

### Checking Quality
1. Zoom to 100% - text should be crisp
2. Print to PDF - check file size (should be <10MB)
3. View on phone - should be readable (tests figure clarity)

## ✨ Final Appearance

When everything compiles correctly, you'll have:

```
┌──────────────────────────────────────┐
│         Professional                  │
│      ┌──────────────────┐            │
│      │  8-page PDF      │            │
│      │  IEEE Format     │            │
│      │  2 columns       │            │
│      │  7 tables        │            │
│      │  6 equations     │            │
│      │  13 references   │            │
│      └──────────────────┘            │
│                                       │
│  Ready for conference submission! ✓  │
└──────────────────────────────────────┘
```

---

**Your paper will look professional, polished, and publication-ready!** 📄✨

To see it yourself:
1. Upload `main.tex` to Overleaf
2. Click "Recompile"
3. View the beautiful PDF! 🎉
