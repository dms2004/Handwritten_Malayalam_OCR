# Handwritten Malayalam OCR

![Malayalam OCR Banner](https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Malayalam_Alphabet_Chart.svg/1200px-Malayalam_Alphabet_Chart.svg.png)

##  Overview

This project presents a complete pipeline for Optical Character Recognition (OCR) of **handwritten Malayalam text** using deep learning. From scanned PDFs to editable Unicode text, this system performs PDF-to-image conversion, line/word/character segmentation, character recognition using CNNs, and structure-preserving text reconstruction.

---

## 📁 Repository Structure

```bash
Handwritten_Malayalam_OCR/
├── Code/                       # All notebooks related to the project
├── Datasets/                   # Raw and preprocessed datasets
├── Model/                      # Saved models 
├── Testing/                    # Model evaluation and test pipeline
└── README.md                   # This file
```

---

## 📄 Dataset

- Based on a labeled handwritten Malayalam character dataset.
- CSV files contain:
  - **Label** column (class index 1-85)
  - **Flattened pixel values** of 32x32 binarized character images.

Sample visualization:

```
+--------+----------+----------+ ...
| Label  | Pixel_0  | Pixel_1  | ...
+--------+----------+----------+ ...
|   20   |    0     |   255    | ...  # -> 'ക'
```

---

##  Model Architecture

We use a **CNN-based classifier** with the following architecture:

```
Input (32x32x1)
└── Conv2D (32 filters) ➔ ReLU ➔ MaxPooling
└── Conv2D (64 filters) ➔ ReLU ➔ MaxPooling
└── Conv2D (128 filters) ➔ ReLU ➔ MaxPooling
└── Flatten ➔ Dense(256) ➔ Dropout(0.5)
└── Output Layer (Softmax over 85 classes)
```

Trained using:
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Epochs: 35
- Accuracy achieved: **~92% on validation**

![Model Accuracy](https://user-images.githubusercontent.com/placeholder/train-val-accuracy.png)

---

## 📝 Preprocessing Pipeline

### 1. PDF to Image
- Uses `pdf2image` to extract pages.

### 2. Line Segmentation
- Horizontal projection + thresholding
- Saves each line as: `line_01.jpg`, `line_02.jpg`, ...

### 3. Word Segmentation
- Morphological closing + contour detection
- Saved as: `Segmented_Words/line1/word1.jpg`...

### 4. Character Segmentation
- Uses Active Contour Models (ACM-FGM) + bounding box filtering
- Output structure:

```
Segmented_Characters/
├── line1_word1/
│   └── char_1.jpg
└── line1_word2/
     └── char_2.jpg
```

---

## 🔢 Text Reconstruction

###  Logic:
- Flatten predictions
- Use `document_structure.json` to restore original order
- Fix pre-base vowels (e.g., `െ`, `േ`, `ൈ`) using smart shifting:

> **Pre-base vowels are placed *after* the consonant they belong to,**
> not before (e.g., `['െ', 'സ']` becomes `['സ', 'െ']`)

### Normalization:
- Unicode `NFC` normalization is used to render complex characters properly

---

## 📈 Sample Results

### Sample Prediction Flow
```
Predicted Labels:
[8, 20, 73, 2] ➔ ['െ', 'ക', 'ന്ത', 'ാ']
Reordered: ['ക', 'െ', 'ന്ത', 'ാ'] ➔ "കെന്താ"
```

### Output Text File (`output.txt`):
```
കെന്താ നിങ്ങൾ പറഞ്ഞത്?
അവൻ പൊയിരിക്കുന്നു.
```

---

## 📅 Visual Summary

```mermaid
graph TD
A[PDF Input] --> B[Page Images]
B --> C[Line Segmentation]
C --> D[Word Segmentation]
D --> E[Character Segmentation]
E --> F[Image Normalization]
F --> G[Model Prediction]
G --> H[Reconstruction (via JSON)]
H --> I[Unicode Text Output (.txt)]
```

---

##  Getting Started

### Clone the Repo
```bash
git clone https://github.com/dms2004/Handwritten_Malayalam_OCR.git
cd Handwritten_Malayalam_OCR
```

### Install Requirements
```bash
pip install -r requirements.txt
```

### Run Pipeline
```bash
python Preprocessing/pdf_to_image.py
python Preprocessing/segment_lines.py
python Preprocessing/segment_words.py
python Preprocessing/segment_characters.py
python Model/train_model.py
python Testing/test_model.py
python Output/reconstruct_text.py
```

---

## 📊 Future Improvements
- Transformer-based sequence recognition (CRNN or ViT)
- Language model integration for context-aware correction
- Better handling of conjunct clusters and halant logic

---

##  Acknowledgements
- Malayalam handwritten datasets
- TensorFlow/Keras
- OpenCV
- pdf2image

---

## ⚖️ License
This project is licensed under the MIT License.

