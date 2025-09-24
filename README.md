---

# 🏞️ Natural Scene Image Classifier
---
This project is a deep learning–based image classification system that identifies natural scenes such as **buildings, forests, glaciers, mountains, seas, and streets**. It uses **MobileNetV2** with transfer learning for model training and a **Streamlit web app** for interactive predictions.

---

## 📌 Features

* ✅ Train a CNN using **MobileNetV2** on Intel’s Natural Scene dataset.
* ✅ Use **data augmentation** to improve generalization.
* ✅ Save and load the trained model (`scene_classifier_model.h5`).
* ✅ Deploy a **Streamlit app** to classify uploaded images.
* ✅ Outputs the predicted class along with confidence percentage.

---

## 📂 Project Structure

```
.
├── app.py                   # Streamlit web application
├── train_model.py           # Model training script
├── scene_classifier_model.h5 # Saved trained model (generated after training)
├── data/                    # Dataset folder (download from Kaggle)
└── README.md                # Project documentation
```

---

## 📊 Dataset

The model is trained on the **Intel Image Classification Dataset** from Kaggle:

🔗 [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

* **Classes (6 total):**

  * 🏙️ Buildings
  * 🌲 Forest
  * 🧊 Glacier
  * ⛰️ Mountain
  * 🌊 Sea
  * 🚦 Street

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/mithun-27/natural-scene-classifier.git
   cd natural-scene-classifier
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   streamlit
   tensorflow
   pillow
   numpy
   ```

3. **Download the dataset**

   * Download from Kaggle: [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
   * Extract it into the `data/` folder.

---

## 🏋️‍♂️ Train the Model

Run the training script:

```bash
python train_model.py
```

This will:

* Load and preprocess the dataset.
* Train a MobileNetV2-based classifier.
* Save the trained model as `scene_classifier_model.h5`.

---

## 🚀 Run the Streamlit App

Start the app:

```bash
streamlit run app.py
```

Then, open the provided **local URL** in your browser.

* Upload an image (`jpg`, `jpeg`, or `png`).
* Click **Classify Image**.
* See the predicted class and confidence score.

---

## 🖼️ Screenshots

Below are example screenshots of the app in action:

### 🔹 Upload Screen

<img width="1919" height="877" alt="image" src="https://github.com/user-attachments/assets/b73521df-eae1-4a82-ae61-605da6cba65e" />


### 🔹 Prediction Screen

<img width="1919" height="880" alt="image" src="https://github.com/user-attachments/assets/f20aaa4a-8758-416e-ab3d-44b430532821" />

---

## 📈 Model Details

* **Base Model:** MobileNetV2 (pretrained on ImageNet).
* **Input Size:** 150×150 RGB images.
* **Optimizer:** Adam (lr = 0.001).
* **Loss Function:** Categorical Crossentropy.
* **Training Epochs:** 5 (can be increased for better accuracy).

---

## 🛠️ Future Improvements

* [ ] Increase training epochs for higher accuracy.
* [ ] Fine-tune MobileNetV2 layers.
* [ ] Deploy as a web service (Flask/FastAPI backend).
* [ ] Add support for more scene categories.

---

## 👨‍💻 Author

Developed by **Mithun** 🚀

* 📧 Email: [kvl202014@gmail.com](mailto:kvl202014@gmail.com)
* 💼 [LinkedIn](https://www.linkedin.com/in/mithun-s-732939280)
* 💻 [GitHub](https://github.com/mithun-27)

---
