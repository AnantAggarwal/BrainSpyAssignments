# MRI Data Visualization — Report

## Overview

This report outlines my approach to understanding and visualizing 3D medical imaging data stored in **NIfTI** (`.nii`) and **DICOM** (`.dcm`) formats. The task was performed in a Kaggle notebook environment, using Python libraries like `nibabel`, `pydicom`, `numpy`, and `matplotlib`. 
## Data Sources

* **NIfTI (.nii.gz)**: Sample Data uploaded in the assignments repo
* **DICOM (.dcm)**: The OSIC Chest fibrosis progression competition on Kaggle.

---

## Tools Used

* `nibabel`: For loading and exploring NIfTI files.
* `pydicom`: For parsing individual DICOM slices and metadata.
* `numpy`: For stacking and manipulating 3D volumes.
* `matplotlib`: For static 2D visualizations of axial, sagittal, and coronal views.

---

## Workflow Summary

### 1. Loading Files

* NIfTI file loaded in one step using `nibabel.load()`, returning both the voxel data and affine matrix.
* DICOM files sloaded in slices array using `pydicom.dcmread`

### 2. Metadata Extraction

* NIfTI: Inspected shape, voxel spacing, data type, and header fields.
* DICOM: Printed all meta-data the dcm file had. Pixel spacing seems similar to NIFTII

### 3. Stacking DICOM Slices

```python
volume = np.stack([s.pixel_array for s in sorted_slices], axis=-1)
```

Slices were sorted and then stacked into a 3D array to match the structure of the NIfTI volume.

### 4. Visualization

I created slice views from all three anatomical planes:

* **Axial (top-down)**
* **Coronal (front-back)**
* **Sagittal (side-to-side)**

Each plot was labeled and rendered using `matplotlib`. Here's an example of an axial slice:

![Axial Screenshot](axial_view.png)

---

## Image Orientation

### NIfTI Affine Matrix

```text
[[ 6.29074156e-01 -1.33072212e-02 -3.11387163e-02 -7.36054077e+01]
 [ 1.63442213e-02  6.21981978e-01  5.91654330e-02 -1.00073349e+02]
 [ 2.99664345e-02 -5.98863661e-02  6.21413589e-01 -1.19990189e+02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```

**Interpretation:**

* The top-left 3×3 submatrix defines the voxel axes’ scaling and orientation in 3D space.
* Each row corresponds to the direction of an axis in real-world coordinates.
* The last column (`[-73.61, -100.07, -119.99]`) is the world-space position of the first voxel.
* Voxel sizes are roughly `0.63 mm`, with slight shearing due to small off-diagonal elements.

### DICOM Orientation Tag

```text
[1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
```

**Interpretation:**

* This tag consists of two direction vectors:

  * The **row vector** `[1, 0, 0]` indicates the image rows go left to right in patient space.
  * The **column vector** `[0, 1, 0]` indicates image columns go from posterior to anterior.
* Since the third component is `0` in both vectors, this slice is aligned with the axial plane.
* DICOM orientation is simpler to read but less descriptive than the affine in NIfTI.

### Summary

| Feature              | NIfTI                         | DICOM                                   |
| -------------------- | ----------------------------- | --------------------------------------- |
| Orientation Storage  | Affine matrix (4×4)           | 6-element direction cosine tag          |
| Contains Origin Info | Yes                           | Requires separate tag (`ImagePosition`) |
| Handles Shear/Skew   | Yes                           | No                                      |
| Interpretation Ease  | Medium (requires matrix math) | High (visually intuitive)               |

---

## NIfTI vs DICOM: Comparison

| Feature             | NIfTI                     | DICOM                                 |
| ------------------- | ------------------------- | ------------------------------------- |
| Typical Use         | Research                  | Clinical diagnosis                    |
| File Structure      | Single file per scan      | Multiple files (one per slice)        |
| Metadata Storage    | Compact header            | Rich tag-based metadata               |
| Spatial Orientation | Affine matrix             | Orientation and position tags         |
| Ease of Loading     | Straightforward in Python | Requires sorting and metadata parsing |
| Tooling             | `nibabel`, `SimpleITK`    | `pydicom`, `SimpleITK`, `dcmtk`       |

---

## ⚙️ Observations & Challenges

### Obsercations

* Loading the DICOM slices took a lot more time than the NIFTII one
* It is ideal to have a large number of slices to make sense of axial and sagittal in DICOM
* DICOM consists a lot of metadata apart the from the image itself. However it may not be that useful for our model
* DICOM slices needed to be sorted carefully to avoid volume distortion.
* Interpreting the affine matrix required understanding voxel orientation and world coordinates. However it yields more info
* DICOM metadata was sometimes inconsistent between slices(In some other datasets I tried).

### Assumption

* All DICOM slices belong to a single scan with uniform spacing and orientation.
* The NIfTI affine matrix correctly represents voxel-to-world transformation.
