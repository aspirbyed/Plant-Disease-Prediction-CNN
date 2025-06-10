# 🌿 Plant Disease Classification using CNN

## 📌 Introduction

Plant diseases pose a major threat to global food security. Early detection and treatment are crucial to minimize crop loss. Traditional identification methods rely on manual inspection, which can be error-prone and time-consuming.

This project leverages **Convolutional Neural Networks (CNN)** and the **PlantVillage dataset** to automatically classify various plant leaf diseases from images. The model is implemented in **TensorFlow** and evaluated on a held-out test set.

---

## 📚 Literature Review

Several studies have demonstrated the effectiveness of deep learning in plant disease detection:

- **Mohanty et al. (2016)** applied CNNs to the PlantVillage dataset and achieved ~99% accuracy.
- **Fuentes et al.** used region-based CNNs for real-time tomato disease detection.
- **Transfer learning** approaches using pre-trained models like **ResNet** and **Inception** have also shown impressive results.

These studies confirm that CNNs often outperform traditional machine learning techniques for image classification tasks.

---

## 🎯 Problem Statement & Theory

### Problem Statement:
To build and evaluate a CNN-based image classification model that accurately identifies plant leaf diseases using the PlantVillage dataset.

### Theory:
Convolutional Neural Networks (CNNs) are specialized neural networks for image data. They extract spatial features using convolutional layers followed by pooling layers. CNNs learn filters to detect edges, textures, and abstract patterns—making them highly effective for disease detection in leaf images.

---

## 📂 Dataset Description

### Source:
- [PlantVillage Dataset via TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)

### Details:
- **Classes**: 38 plant-disease categories  
- **Format**: RGB leaf images with labels  
- **Split**: 80% training, 20% testing  

### Preprocessing:
- Resized images to **128×128**
- Augmentations: **horizontal flip**, **brightness adjustment**
- Normalized pixel values to the **[0, 1]** range

### Visualizations:
- 📸 Sample Images (include in notebook/report)
- 📊 Class Distribution (include plot)

---

## 🧠 Model Architecture & Results

### Architecture:
- Stacked `Conv2D` + `BatchNorm` + `ReLU` + `MaxPooling` layers
- `Dropout` layers to reduce overfitting
- `GlobalAveragePooling2D` before the dense output
- `Dense` layer with **Softmax** activation for multi-class classification

### Performance:
- ✅ **Training Accuracy**: ~99%
- ✅ **Validation Accuracy**: ~98.7%

### Metrics:
| Metric       | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| **Macro Avg**    | 0.99      | 0.99   | 0.99     |
| **Weighted Avg** | 0.99      | 0.99   | 0.99     |

### Visual Results:
- 📈 Training vs Validation Accuracy/Loss Curves
- 🔀 Confusion Matrix
- 📄 Classification Report

---

## ✅ Conclusion

This project successfully demonstrates the potential of CNNs in accurately classifying plant diseases using image data. With proper preprocessing and data augmentation, the model achieved outstanding accuracy.

### Future Work:
- Use **transfer learning** with architectures like EfficientNet
- Deploy the model on **mobile/web platforms** for farmers
- Adapt for **real-world detection** in natural conditions

---

## 🚀 Getting Started

### Clone this repository

```bash
git clone https://github.com/your-username/plant-disease-cnn.git
cd plant-disease-cnn
