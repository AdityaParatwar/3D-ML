# 🧠 Point Cloud Classification (3D ML)

This project focuses on processing 3D mesh data and converting it into point clouds for object classification using deep learning techniques.

---

## 📌 Overview

* Convert `.off` mesh files → point clouds
* Visualize 3D data using Open3D
* Build a dataset pipeline for training
* Train a deep learning model (PointNet)

### 🪑 Supported Classes

* Chair
* Table
* Sofa
* Bed

---

## ⚙️ Tech Stack

* Python 3.10
* PyTorch
* Open3D
* Trimesh
* NumPy
* Matplotlib

---

## 📂 Project Structure

```
Point_Cloud/
│
├── data/                # Dataset (ignored in Git)
├── images/              # Output images
├── scripts/             # Conversion + dataset loader
├── requirements.txt
└── README.md
```

---

## 🔄 Mesh to Point Cloud

This project converts 3D mesh files (`.off`) into point clouds using sampling techniques.

---

## 📊 Output Visualization

### 🔹 Chair Example

![Chair Example](images/chair.png)

---

## 🚀 How to Run

### 1️⃣ Create Virtual Environment

```
py -3.10 -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Convert Dataset

```
python scripts/convert_to_pointcloud.py
```

### 4️⃣ Visualize

```
python scripts/visualize.py
```

---

## 🧪 Dataset

* ModelNet10 Dataset
* Contains 3D object categories like chairs, tables, etc.
* Mesh format: `.off`
* Converted to point clouds (`.npy`)

---

## 🎯 Features

* ✅ Mesh → Point Cloud conversion
* ✅ 3D visualization using Open3D
* ✅ Custom PyTorch Dataset Loader
* 🚧 PointNet Model (in progress)

---

## 👨‍💻 Author

**Aditya Paratwar**

---