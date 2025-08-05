# Ayna-ML-Assignment

## Polygon Colorization via Robust Conditional UNet

### 🚀 Overview

This project implements a Conditional UNet capable of filling polygons with any requested color using deep, consistent color conditioning. The model learns to map a polygon outline and a color condition to a correctly filled, colored output image—handling multiple shapes and color classes robustly.

---

### 🧹 Methodology & Model Evolution

The model evolved through several major iterations, integrating best practices from deep learning research in conditional generation:

#### 1. Baseline UNet (No Conditioning)

* **Approach**: Classic UNet with only the polygon outline as input.
* **Limitation**: Always produced the most common or average color; could not handle conditional coloring.

#### 2. Color Embedding + FiLM at Bottleneck

* **Approach**: Learnable color embedding via `nn.Embedding`, injected only at the bottleneck using FiLM (scale and bias).
* **Outcome**: Partial response to color condition; results were inconsistent with frequent color mistakes.

#### 3. Unified color2idx Mapping

* **Improvement**: Enforced a single, unified `color2idx` mapping across train/val/test and code.
* **Outcome**: Significantly reduced color swapping and confusion.

#### 4. (Final) Embedding Concatenation at Input + FiLM at Bottleneck

* **Approach**:

  * Concatenated color embedding to input image as extra channels (early conditioning).
  * Added FiLM conditioning at the bottleneck (deep conditioning).
  * Balanced dataset across all shape and color classes.
* **Outcome**: Robust color filling with high accuracy for all shapes and colors.

#### Summary Table of Techniques and Improvements

| Iteration | Technique                                    | Problem Addressed                | Outcome                          |
| --------- | -------------------------------------------- | -------------------------------- | -------------------------------- |
| 1         | Baseline UNet, no color input                | No way to pick output color      | Model outputs average/color bias |
| 2         | Color embedding + FiLM at bottleneck         | Weak/late conditioning           | Some color control, not reliable |
| 3         | Unified color2idx mapping, balanced datasets | Split-wise label confusion       | Stable, fewer mistakes           |
| 4 (Final) | Embedding concat + FiLM at bottleneck        | Mode collapse, weak conditioning | Robust, accurate color control   |

---

### 🛠️ Final Model Configuration

* **Architecture**: Custom UNet with skip connections.
* **Input**: \[Batch, 3, H, W] polygon image concatenated with color embedding \[Batch, ColorEmbedDim, H, W]
* **Color Conditioning**:

  * `nn.Embedding` per color label → expanded and concatenated at input
  * FiLM scale/bias projection added at bottleneck via MLPs
* **Loss Function**: L1 loss (Mean Absolute Error) for sharper boundaries and perceptually clean fills.

#### Key Hyperparameters

* Learning Rate: 1e-3
* Batch Size: 16
* Epochs: 200
* Color Embedding Size: 64

#### Metrics

* L1 validation loss
* SSIM (Structural Similarity Index)
* Visual validation for shape-color pairs
* Experiment tracking via Weights & Biases (\[add your link here])

---

### 📈 Results & Key Learnings

* **Early and deep conditioning** (input channel concat + FiLM) is critical for robust conditional generation.
* **Consistent label mapping** is necessary to avoid label confusion.
* **L1 loss** provides sharper, more perceptually accurate outputs than MSE.

#### Final Model Achievements

* Accurately fills any supported shape with specified color.
* No longer confuses or ignores conditioning color.
* Robust across all shape and color variations.

---

### ⚡ Future Work

* Try perceptual or GAN-based loss for improved realism.
* Support arbitrary RGB color inputs (beyond fixed palette).
* Optimize inference for deployment or real-time usage.

---

### 💡 How to Use

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Prepare Dataset

* Organize images and mappings (`data.json`) as per project structure.
* Use consistent `color2idx` mapping across all data splits.

#### 3. Train the Model

```bash
python train.py
```

#### 4. Run Inference or Visualization

```bash
python inference.py
```

Or use the Jupyter notebook for demo/EDA.

#### 5. Track Experiments

* Review learning curves and outputs on Weights & Biases.

---

### 📚 Lessons & Recommendations

* **Condition early, condition deep**: Combine input concatenation + bottleneck FiLM.
* **Avoid mapping mismatches**: Always reuse the same `color2idx` dictionary.
* **Balance your dataset**: Include all class combinations in every split.
* **Visual validation**: Regularly inspect edge-case color-shape combinations.

---
