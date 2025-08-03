# Ayna-ML-Assignment

## Polygon Colorization via Conditional UNet

### Methodology

* **Architecture**: Custom UNet enhanced with FiLM (Feature-wise Linear Modulation) for conditional color control.
* **Conditioning Mechanism**: Multi-scale FiLM injection using learned embeddings for each color name.
* **Loss Function**: Composite loss combining L1 and MSE: `0.7 * L1 + 0.3 * MSE`, promoting both boundary sharpness and color fidelity.

### Hyperparameters

* **Learning Rate**: 1e-3
* **Batch Size**: 16
* **Epochs**: 50
* **Color Embedding Size**: 64

### Experiments & Results

* Tracked using **Weights & Biases** (WandB): \[Add your WandB project link here]
* Evaluated using **SSIM** (Structural Similarity Index) and **PSNR** (Peak Signal-to-Noise Ratio)
* **Qualitative Gains**: FiLM conditioning yielded more accurate and vivid outputs compared to naive concatenation.

### Key Learnings

* FiLM-based conditioning significantly improves output quality.
* Data augmentation enhances model generalization.

### Future Work

* Explore **perceptual** and **adversarial** loss components.
* Optimize **runtime performance**.
* Enable **arbitrary custom color inputs**.

---

## How to Use

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download & organize dataset**
   Place the dataset in a `data/` directory following the structure outlined in the project guide.

3. **Train the model**

   ```bash
   python train.py
   ```

4. **Run inference and demo**
   Use the notebook:

   ```
   inference.ipynb
   ```

5. **Track experiments**
   Connect to your WandB project by editing the script/report with your WandB link.

6. **Package & submit**
   Review all outputs and prepare final deliverables for submission.

---


