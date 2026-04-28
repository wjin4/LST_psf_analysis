# LST PSF Image Analysis GUI

A standalone Python GUI for viewing FITS images (e.g., from ZWO ASI cameras) and performing point spread function (PSF) analysis on selected stars.

---

## 📌 Overview

This tool allows you to:

* Load 2D FITS images (8-bit or 16-bit)
* Interactively select a star using a mouse drag
* Compute PSF metrics including:

  * Centroid
  * Background level
  * Flux
  * FWHM (multiple methods)
  * Radial intensity profile
  * Encircled energy
* Visualize results in real time

---



## 📦 Requirements

* Conda (Anaconda or Miniconda recommended)

Dependencies (installed via environment file):

* Python 3.10
* numpy
* matplotlib
* astropy
* scipy (optional but recommended)
* tkinter (via `tk`)

---

## ⚙️ Installation

### 1. Get the code

Clone or download:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

### 2. Create the conda environment

```bash
conda env create -f environment.yml
```

---

### 3. Activate the environment

```bash
conda activate lst-psf-env
```

---

## ▶️ Running the Application

```bash
python lst_psf_image_analysis_gui.py
```

A GUI window will open.

---

## 🧭 Basic Workflow

### Step 1 — Load an Image

* Click **"Open FITS"**
* Select a `.fits`, `.fit`, or `.fts` file

You will see:

* Image displayed in grayscale
* FITS header on the left
* Image info (size, bit depth)

---

### Step 2 — Enter PSF Mode

* Click **"Analyze PSF"**
* Status will change to indicate selection mode

---

### Step 3 — Select a Star

* Click and drag a region around a star
* The tool will:

  * Convert your selection into a circular region
  * Automatically analyze the PSF

---

### Step 4 — View Results

#### On the Image:

* ❌ Red X → centroid
* 🔴 Dashed red circle → 80% encircled-energy radius
* ⚪ Red circles → rejected outlier pixels

#### On the Right Panel:

* Radial profile plot
* Encircled energy curve

#### Metrics Panel:

Displays numerical results (see below)

---

## 📊 Output Metrics Explained

### Centroid

```
Centroid: x, y
```

* Subpixel position of the star center

---

### Background

```
Background: value
```

* Estimated sky/background level from an annulus

---

### Peak Pixel

```
Peak pixel: value
```

* Brightest pixel in the ROI (not background-subtracted)

---

### Total Flux

```
Total flux above background
```

* Sum of positive signal inside the star boundary

---

### FWHM (Full Width at Half Maximum)

#### X / Y (Moment-based)

```
Approx FWHM X / Y
```

* Computed from second moments
* Sensitive to asymmetry

#### Radial FWHM

```
Radial FWHM
```

* Based on radial profile
* Often more stable than X/Y

#### Gaussian Fit (if SciPy available)

```
Gaussian-fit FWHM
```

* Fit of:
  y(r) = A * exp(-r² / 2σ²) + background

---

### PSF Radius (Important)

```
80% encircled-energy radius
```

* Radius containing 80% of total flux
* Used as **practical PSF size**

⚠️ Not a physical edge — just a useful metric

---

### Star Boundary Radius

```
Estimated star boundary radius
```

* Where the radial profile merges into noise
* Used to define the aperture for flux calculation

---

## 📈 Plots Explained

### Radial Profile

* Intensity vs radius
* Shows how brightness falls off from the center

### Encircled Energy

* Fraction of total flux vs radius
* Used to determine PSF size



* The dashed red circle is **NOT a physical boundary**
* It represents the **80% encircled energy radius**
* Results depend on:

  * ROI selection
  * image quality
  * background noise

---

## 🧪 Tips for Best Results

* Select a **single isolated star**
* Avoid:

  * saturated stars
  * overlapping stars
  * edges of the image
* Use a region slightly larger than the star
* Ensure background is included in selection

---

## 🛠 Troubleshooting



