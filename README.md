# 🤖 ChromaSort: CNN-Based Color Detection & Conveyor Belt Routing

A ChromaSort is a smart object identification deep learning project that uses a **Convolutional Neural Network (CNN)** to detect the color of an object from its image and automatically decide which conveyor belt it should be sent to.

---

### 🎯 Color-to-Belt Mapping

| Detected Color | Assigned Conveyor Belt |
|:---:|:---:|
| 🔵 Blue | Belt A |
| 🟡 Yellow | Belt B |
| 🟣 Purple | Belt C |

---

## ✨ Highlights

- 🧠 Custom CNN **built and trained from scratch**, with no pretrained weights and no transfer learning
- 📸 **Hand-made dataset** of 165 original photographs captured with phone cameras, with no images taken from the internet
- 🎯 **96.30% test accuracy** on 27 unseen images (26 / 27 correct)
- ⚖️ **Well-fitted model**: training, validation and test accuracy are all close to each other and the loss curves fall together, so there is no sign of significant overfitting or underfitting
- 📂 Lightweight: **60,675 parameters**, about 760 KB on disk
- 🔄 Automatic handling of mixed image formats (JPG, PNG, iPhone HEIC)
- 📤 Built-in upload-and-predict demo with confidence scores

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Data & Evaluation | NumPy, Scikit-learn |
| Visualization | Matplotlib |
| Image Handling | Pillow, Pillow-HEIF |
| Environment | Google Colab, Google Drive |

---

## 🧠 Model Architecture

The CNN is built with Keras `Sequential` and works in three stages:

| Stage | Layers | Purpose |
|---|---|---|
| **Preprocessing** | Input (128 × 128 × 3), Rescaling (1/255) | Scales pixel values to 0–1 so training is stable |
| **Feature Extraction** | Conv2D + MaxPooling2D | Learns edges and color patches first, then more complex patterns, while each pooling step halves the image size |
| **Classification** | GlobalAveragePooling2D, Dropout (0.4), Dense (64, ReLU), Dense (3, Softmax) | Compresses the features into a 64-value summary and outputs probabilities for blue, purple and yellow |

**Total trainable parameters:** 60,675

### 💡 Why This Design Works

- **Global average pooling instead of Flatten:** passes 64 values to the dense layer instead of 16,384, so there are far fewer weights and less chance of memorizing a small dataset.
- **Dropout (0.4):** randomly switches off 40% of the values during training so the network cannot rely on a single feature, which reduces overfitting.
- **128 × 128 input:** color is a large-scale property, so high resolution is not needed. This keeps training fast while still showing shape and shading.

### 🛠️ Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam (learning rate 0.001) |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 16 |
| Maximum Epochs | 60 |
| Early Stopping | Monitor `val_loss`, patience 10, best weights restored |

---

## 📂 Dataset

The dataset was created manually by photographing real objects and household items with Android and iPhone cameras, mostly on plain light surfaces or striped fabric.

| Class | Images |
|---|---:|
| 🔵 Blue | 55 |
| 🟣 Purple | 55 |
| 🟡 Yellow | 55 |
| **Total** | **165** |

**Split (per class, 70 / 15 / 15):**

| Set | Per Class | Total |
|---|---:|---:|
| Training | 38 | 114 |
| Validation | 8 | 24 |
| Testing | 9 | 27 |

---

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Validation Accuracy | 100% |
| **Test Accuracy** | **96.30%** |
| Test Images | 27 |
| Correct Predictions | 26 |
| Incorrect Predictions | 1 |

**Classification report (test set):**

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| Blue | 1.00 | 1.00 | 1.00 | 9 |
| Purple | 1.00 | 0.89 | 0.94 | 9 |
| Yellow | 0.90 | 1.00 | 0.95 | 9 |
| **Macro Avg** | 0.97 | 0.96 | 0.96 | 27 |

**Confusion matrix** (rows = true class, columns = predicted class):

| | Pred. Blue | Pred. Purple | Pred. Yellow |
|---|:---:|:---:|:---:|
| **True Blue** | 9 | 0 | 0 |
| **True Purple** | 0 | 8 | 1 |
| **True Yellow** | 0 | 0 | 9 |

The only mistake was one purple object classified as yellow.

**Fit check:** the model reached about 98% training accuracy and 100% validation accuracy at its best epoch, and 96.30% on the held-out test set. These numbers are close to each other, which is what a well-fitted model looks like. A large gap between training and test accuracy would signal overfitting, and low accuracy everywhere would signal underfitting. Neither appears here.

---

## 🔄 Prediction Workflow

1. Upload an image of an object.
2. Resize it to 128 × 128 pixels and convert it to an RGB array.
3. Pass it through the trained CNN.
4. Get probabilities for blue, purple and yellow.
5. Select the highest probability with `np.argmax()`.
6. Use that index to look up the color name and conveyor belt from lists.
7. Display the predicted color, confidence and belt.

---

## 🚀 How to Run

This project is designed to run in **Google Colab**.

**1. Clone the repository**

```bash
git clone https://github.com/HuzaifaNawaid/chromasort-cnn.git
```

**2. Open the notebook**

Open `ChromaSort_CNN.ipynb` in Google Colab.

**3. Prepare the dataset**

Place the dataset in Google Drive with this structure (55 images per folder):

```text
COLOR_DATASET/
├── blue/
├── purple/
└── yellow/
```

If your dataset is in a different location, change the `DATA` path in Step 1.

**4. Connect Google Drive**

Run the first cell and authorize Drive access when prompted.

**5. Run all cells**

Run the notebook from Step 1 to Step 13 in one session. It preprocesses the images, splits the data, trains the CNN, evaluates it, and saves the model.

**6. Test your own images**

In the final step, click **Choose Files**, upload any image, and the notebook shows the predicted color, confidence and assigned belt.

### Using the Saved Model

```python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("chromasort_cnn.keras")
class_names = ["blue", "purple", "yellow"]   # alphabetical = label order
belt_for_color = {"blue": "Belt A", "yellow": "Belt B", "purple": "Belt C"}
belts = [belt_for_color[n] for n in class_names]

img = tf.keras.utils.load_img("your_image.jpg", target_size=(128, 128), interpolation="bilinear")
arr = tf.keras.utils.img_to_array(img)[np.newaxis, ...]
probs = model.predict(arr, verbose=0)[0]
idx = int(np.argmax(probs))
print(f"{class_names[idx]} ({probs[idx]*100:.1f}%) -> {belts[idx]}")
```

Install dependencies with `pip install -r requirements.txt`.

---

## 🗂️ Repository Structure

```text
.
├── ChromaSort_CNN.ipynb       # Full pipeline: data prep, training, evaluation, demo
├── COLOR_DATASET/             # Hand-made dataset (blue / purple / yellow)
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🔮 Future Improvements

- Expand the dataset with more objects, backgrounds and lighting conditions
- Add data augmentation (brightness, flips, small rotations)
- Introduce an "unknown color / no object" class
- Integrate a live camera for real-time classification

---

## 👨‍💻 Created by [HuzaifaNawaid](https://github.com/HuzaifaNawaid) (AI/ML Engineer)
