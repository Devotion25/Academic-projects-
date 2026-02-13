# CelebA Facial Attribute Classification

Multi-output deep learning model for gender and facial attribute classification using transfer learning with VGG16 on the CelebA dataset.

## 🎯 Project Overview

This project implements a convolutional neural network (CNN) using VGG16 transfer learning to classify facial attributes from celebrity images. The model performs multi-task learning to simultaneously predict gender and eyebrow characteristics from over 200,000 celebrity images.

## 📊 Dataset

- **Source**: CelebA (Celebrity Faces Attributes Dataset)
- **Size**: 202,599 celebrity images
- **Image Resolution**: 218 x 178 pixels
- **Attributes Used**: 
  - Male/Female (Gender)
  - Bushy Eyebrows
  - Smiling (for meta-data analysis)
- **Data Split**:
  - Training: 162,770 images (10% sampled: 16,277)
  - Validation: 19,962 images (1,000 sampled)
  - Test: 19,867 images (1,000 sampled)

## 🏗️ Model Architecture

### R1: Single-Output Gender Classification
```
Input (218 x 178 x 3)
    ↓
VGG16 (Pre-trained, frozen)
    ↓
Flatten
    ↓
Dense(1024, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(1, Sigmoid)
```

**Performance**:
- Validation Accuracy: 87.9%
- Test Accuracy: 88.7%
- Training Time: ~2796s

### R2: Multi-Output Classification (Gender + Eyebrows)
```
Input (128 x 96 x 3)  # Reduced image size
    ↓
VGG16 (Pre-trained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(2, Sigmoid)  # Two outputs
```

**Performance**:
- Gender Precision: 0.8606
- Eyebrow Precision: 0.2857
- Training Time: ~932s (3x faster)

## 🔍 Key Features

1. **Transfer Learning**: Leveraged pre-trained VGG16 on ImageNet
2. **Custom Data Generator**: Efficient batch processing with memory optimization
3. **Multi-Task Learning**: Simultaneous prediction of multiple attributes
4. **Landmark Analysis**: Investigated mouth width correlation with smiling
5. **Performance Optimization**:
   - Image size reduction (218x178 → 128x96)
   - Global Average Pooling instead of Flatten
   - Adam optimizer (faster than SGD)
   - Early stopping to prevent overfitting

## 📈 Results

### Gender Classification
| Metric | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Female | 0.94 | 0.89 | 0.91 | 571 |
| Male | 0.86 | 0.92 | 0.89 | 429 |
| **Overall** | **0.90** | **0.90** | **0.90** | **1000** |

### Eyebrow Detection
| Metric | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| No Bushy Eyebrows | 0.88 | 0.98 | 0.92 | 871 |
| Bushy Eyebrows | 0.29 | 0.06 | 0.10 | 129 |
| **Overall** | **0.86** | **0.86** | **0.82** | **1000** |

### Meta-Data Analysis
- **Mouth Width Q1 vs Non-Q1**: M1 = 0.6161
- **Mouth Width Q3 vs Non-Q3**: M2 = 0.6816
- **Conclusion**: Mouth width is NOT a strong predictor of smiling (|M1-M2| < 0.1)

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **OpenCV** - Image processing
- **Scikit-learn** - Metrics and evaluation
- **Kaggle API** - Dataset access

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Devotion25/CelebA-Facial-Classification.git

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d jessicali9530/celeba-dataset
```

## 🚀 Usage

```python
# Open the Jupyter notebook
jupyter notebook celeba_classification.ipynb

# Or run in Google Colab
# Upload the notebook and run all cells
```

## 📝 Key Insights

1. **Transfer Learning Effectiveness**: Pre-trained VGG16 achieved high accuracy with minimal training
2. **Gender vs Eyebrow Detection**: Gender classification significantly outperformed eyebrow detection due to class imbalance
3. **Image Resolution Trade-off**: Reducing image size by 40% maintained accuracy while reducing training time by 3x
4. **Facial Landmarks**: Geometric features (mouth width) provided limited predictive value for expression classification

## 🎓 Course Information

- **Course**: CSCE 5215 - Machine Learning
- **Institution**: University of North Texas
- **Semester**: Spring 2025

## 👨‍💻 Author

**Devotion Ekueku**
- GitHub: [@Devotion25](https://github.com/Devotion25)
- LinkedIn: [devotionekueku](https://www.linkedin.com/in/devotionekueku/)

## 📄 License

This project is for educational purposes as part of coursework at the University of North Texas.

## 🙏 Acknowledgments

- CelebA Dataset creators
- VGG16 architecture by Visual Geometry Group, Oxford
- Kaggle for dataset hosting
- Course instructors and TAs at UNT

## 📚 References

1. Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. ICCV.
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.
3. Transfer Learning Tutorial - TensorFlow Documentation
