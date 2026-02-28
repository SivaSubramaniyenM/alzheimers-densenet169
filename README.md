# 🧠 Alzheimer's Disease Classification using DenseNet169

A deep learning model that classifies MRI brain scans into 4 stages of Alzheimer's disease using DenseNet169 transfer learning, developed as an individual project after completing the Deep Learning course.

---

## 🔬 About the Project

Alzheimer's disease is a progressive neurological disorder that affects millions worldwide. Early detection through MRI scans can significantly improve patient outcomes. This project uses transfer learning with **DenseNet169** pretrained on ImageNet to classify brain MRI scans into 4 dementia categories with an **AUC of ~90%**.

---

## 🗂️ Categories

| Class | Description |
|---|---|
| `NonDemented` | No signs of Alzheimer's disease |
| `VeryMildDemented` | Very early traces of Alzheimer's |
| `MildDemented` | Mild signs of Alzheimer's disease |
| `ModerateDemented` | Moderate progression of Alzheimer's |

---

## 🧱 Model Architecture

- **Base Model:** DenseNet169 (pretrained on ImageNet, all layers frozen)
- **Classification Head:**
  - Dropout (0.5)
  - Flatten
  - BatchNormalization
  - Dense (2048) + BatchNorm + ReLU + Dropout (0.5)
  - Dense (1024) + BatchNorm + ReLU + Dropout (0.5)
  - Dense (4) + Softmax

---

## 📊 Results

| Metric | Score |
|---|---|
| AUC | ~90% |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Epochs | 5 (with Early Stopping) |

---

## 🗂️ Project Structure

```
alzheimers_prediction/
│
├── alzheimers-densenet169.ipynb    # Complete notebook (data download → training → evaluation)
└── README.md
```

---

## ⚙️ Setup & Usage

This project runs entirely on **Google Colab**. No local setup needed.

### Step 1 — Open in Colab
Upload `alzheimers-densenet169.ipynb` to [Google Colab](https://colab.research.google.com)

### Step 2 — Get Kaggle API Key
Go to **kaggle.com → Profile → Settings → API → Create New Token**
This downloads `kaggle.json` to your machine.

### Step 3 — Run All Cells
The notebook is fully self-contained and will:
1. Mount your Google Drive
2. Prompt you to upload `kaggle.json`
3. Download and extract the dataset automatically
4. Auto-detect train/test paths
5. Train the DenseNet169 model
6. Plot Loss, AUC, and Accuracy curves
7. Run predictions on all 4 test cases
8. Display side-by-side MRI scan comparison

---

## 🛠️ Tech Stack

| Library | Use |
|---|---|
| `TensorFlow / Keras` | Model building and training |
| `DenseNet169` | Transfer learning base model |
| `ImageDataGenerator` | Data augmentation |
| `NumPy` | Numerical operations |
| `Matplotlib` | Visualisation |
| `OpenCV (cv2)` | Image loading and colour conversion |
| `scikit-learn` | Data utilities |

---

## 📦 Dataset

**Alzheimer's Dataset (4 Class of Images)** from Kaggle
- 🔗 [tourist55/alzheimers-dataset-4-class-of-images](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
- ~6,400 MRI scan images across train and test sets
- 4 classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented

---

## 👤 Author

**M. Siva Subramaniyen**

---

## 📝 Notes

- The dataset is not included in this repository — it is downloaded automatically via the Kaggle API inside the notebook
- Model weights are saved to Google Drive during training as `best_weights.keras`
- The notebook uses `val_auc` as the monitor metric for both EarlyStopping and ModelCheckpoint
