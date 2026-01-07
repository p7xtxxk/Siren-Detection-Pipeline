
# SirenGuardian

A deep learning–based pipeline for detecting emergency sirens from audio signals using TensorFlow and Keras.
This model was developed by **Prateek Priyanshu** as part of a **Minor Project** in **ECSc.** Course at **KIIT University** and which focused on audio preprocessing, model training, and deployment-ready inference formats.

---

## Project Overview

Emergency siren detection is a critical component in intelligent transportation systems and smart city applications.
This project explores multiple deep learning approaches to accurately classify siren sounds from environmental audio.

The pipeline includes:

* Audio preprocessing
* Model training and evaluation
* Model export to multiple deployment formats (Keras, H5, TensorFlow Lite)

---
## Model Versions
The project was developed iteratively. Each version is documented in a Jupyter Notebook:

| Notebook         | Description                             |
| ---------------- | --------------------------------------- |
| `version1.ipynb` | Initial model and baseline experiments  |
| `version2.ipynb` | Improved preprocessing and architecture |
| `version3.ipynb` | Final optimized model                   |

---
## Repository Structure

```
Siren-Detection-Pipeline/
│
├── version1.ipynb          # Initial notebook
├── version2.ipynb          # Improved version
├── version3.ipynb          # Final version
│
├── siren_model.keras       # Final Keras model
├── siren_model.h5          # H5 model format
├── best_siren_model.h5     # Best-performing model
├── siren_model.tflite      # TensorFlow Lite model
│
├── siren_savedmodel/       # TensorFlow SavedModel directory
│
├── datasets/               # Datasets after cleaning as per need
├── data/                   # Datasets - Siren and Non Siren files
├── LICENSE                 # GPL-3.0 License
└── README.md               # Project documentation
```

---

## Dataset Information

The datasets used in this project are **large (~10 GB)** and cannot be hosted directly on GitHub. Instead I have provided the links to the RAW and cleaned up versions

### 🔗 Dataset Download Links
* **Raw Dataset:**
  * [ESC-50](https://github.com/karolpiczak/ESC-50)
  * [sireNNet](https://data.mendeley.com/datasets/j4ydzzv4kb/1)
  * [UrbanSound8K - Urban Sound Datasets](https://urbansounddataset.weebly.com/urbansound8k.html)
  * [Siren-Sound-Dataset](https://github.com/vishnu-u/Siren-Sound-Dataset)
* **Processed Data:**
  * [Drive Link](https://drive.google.com/drive/folders/1zhE1oUzgD_GW2fLH3NrJfP4Be6zQp2NX?usp=sharing) 

### Data Directory Structure

After downloading and extracting the datasets, place them as follows:

```
Siren-Detection-Pipeline/
├── datasets/
├── data/
```

⚠️ These folders are intentionally excluded from GitHub using `.gitignore`.

---
## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Librosa
* Jupyter Notebook

---

## Model Metrics

#### **Non-Siren**
- **Precision:** 0.99
- **Recall:** 1.00
- **F1-Score:** 1.00
- **Support:** 7731
#### **Siren**
- **Precision:** 0.98
- **Recall:** 0.97
- **F1-Score:** 0.98
- **Support:** 1518

At a threshold of 0.215, the siren detection model achieved 99.13% accuracy
## Model Deployment

The trained model is available in multiple formats:

* **`.keras` / `.h5`** → Training & inference
* **`.tflite`** → Edge & mobile deployment (Specifically Raspberry Pi 4B)

---
## License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for details.

---
## Author

**Prateek Priyanshu**

---

## References

* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Librosa Audio Processing: [https://librosa.org/](https://librosa.org/)
* Emergency Sound Detection Research Papers
