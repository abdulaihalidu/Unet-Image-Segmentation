
# Brain Tumor and Neuronal Cells Segmentation using U-Net

## 📌 Project Overview
This project implements a **Deep Learning model for Brain Tumor and Neuronal Cells Segmentation** using the **U-Net architecture**. The model is trained to automatically segment brain tumors from MRI scans using a dataset of medical images.

## 📂 Project Structure
```
Final Project/
│── data/                     # Dataset scripts and preprocessing
│   ├── dataset.py            # Data loading and processing functions
│   ├── __init__.py           # Module initialization
│── models/                   # Model architecture and training utilities
│   ├── model.py              # U-Net model implementation
│   ├── brain_tumor_trainer.py # Training script for the brain tumor dataset
│   ├── train_utils.py        # Training script for the the Neuronal Cells dataset 
│   ├── __init__.py           # Module initialization
│── notebooks/                # Jupyter Notebooks for model development
│   ├── Segmentation_tasks_using_UNet.ipynb  # Notebook for segmentation with U-Net
│── predicted_masks/          # Model-generated masks for brain tumors/neuronal cells
│── docs/                     # Research paper and references
│── README.md                 # Project documentation
```

## 🛠️ Installation and Setup
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/abdulaihalidu/Unet-Image-Segmentation.git
   cd Unet-Image-Segmentation

   ```

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Download and Extract Data**:
   The dataset should be downloaded and placed in the data directory
   ```sh
   unzip data.zip -d data/
   ```

## 🚀 Training the Model
To train the **U-Net model** on the brain tumor dataset, check:
```sh
the notebook file in the notebooks directory
```

## 📊 Evaluating the Model
Check the notebook in the notebooks directory

## 📈 Results
The predicted brain tumor masks are saved in:
```
predicted_masks/
```

## 🏗️ Model Architecture
The model is based on **U-Net**, a powerful segmentation network:
- **Contracting path**: Extracts features using convolutional layers.
- **Expanding path**: Restores spatial resolution for precise segmentation.
- **Skip connections**: Preserve fine-grained details.

## 📚 References
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)
- Research papers are available in the `docs/` directory.

## ✨ Contributors
- **Halidu Abdulai** 

## 📢 License
This project is open-source under the **MIT License**.

