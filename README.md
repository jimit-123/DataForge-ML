# DataForge ML

DataForge ML is an AI-powered data preprocessing platform that automatically transforms raw datasets into machine learning-ready data.

The project focuses on solving one of the most time-consuming parts of machine learning workflows: cleaning, preprocessing, and preparing messy real-world datasets efficiently.

---

# 📁 Project Structure

```bash
DataForge-ML/
│
├── app.py
├── preprocessing.py
├── utils.py
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## 1 — Clone the Repository

```bash
git clone [YOUR_GITHUB_REPO_LINK]
cd DataForge-ML
```

---

## 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4 — Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 🗂️ Supported File Formats

| Format | Extension        | Max Size |
| ------ | ---------------- | -------- |
| CSV    | `.csv`           | 20 MB    |
| Excel  | `.xlsx` / `.xls` | 20 MB    |
| JSON   | `.json`          | 20 MB    |

---

# ⚙️ Features

## 🧹 Data Cleaning

* Duplicate row removal
* Missing value handling
* Automatic datatype fixing
* Text normalization
* Useless column detection

---

## 📊 Outlier Detection & Handling

* IQR-based outlier detection
* Z-score outlier detection
* Outlier capping
* Outlier removal

---

## 🧠 Feature Engineering

* Feature scaling
* Label encoding
* One-hot encoding
* Correlation-based feature reduction
* Low variance feature removal
* PCA dimensionality reduction

---

## 📈 Dataset Preparation

* Train / Validation / Test splitting
* Model-ready dataset generation
* Export cleaned datasets

---

## 🎛️ Interactive Dashboard

* Dataset preview
* Cleaning reports
* Outlier analytics
* Feature distribution visualization
* Processing summaries

---

# 🧩 How This Works

The preprocessing pipeline follows a step-by-step workflow to convert raw datasets into model-trainable data.

## Step 1 — Dataset Upload

The user uploads a CSV, Excel, or JSON dataset through the Streamlit interface.

The system:

* validates file size
* checks file format
* loads the dataset into a Pandas DataFrame

---

## Step 2 — Duplicate Removal

Exact duplicate rows are automatically detected and removed.

This helps reduce redundant information and improves dataset quality.

---

## Step 3 — Datatype Correction

The system automatically detects incorrect datatypes.

Examples:

* numeric values stored as text → converted to numbers
* date strings → converted to datetime objects

---

## Step 4 — Missing Value Handling

Missing values are handled intelligently:

* Numerical columns → median or mean imputation
* Categorical columns → mode imputation

---

## Step 5 — Text Normalization

Text columns are standardized by:

* converting text to lowercase
* removing extra spaces
* cleaning inconsistent formatting

---

## Step 6 — Outlier Detection

The platform detects abnormal values using:

* IQR method
* Z-score method

Outliers can either:

* be capped safely
* or removed completely

---

## Step 7 — Feature Encoding

Categorical columns are converted into numerical form using:

* Label Encoding
* One-Hot Encoding

This makes the dataset compatible with machine learning models.

---

## Step 8 — Feature Scaling

Numerical features are normalized using:

* StandardScaler
* MinMaxScaler

This prevents large-value columns from dominating the model.

---

## Step 9 — Overfitting Prevention

To reduce overfitting, the system includes:

* correlated feature removal
* low variance feature removal
* PCA dimensionality reduction
* train / validation / test splitting

---

## Step 10 — Dataset Splitting

The final dataset is automatically split into:

* Training set
* Validation set
* Test set

The processed datasets can then be downloaded directly.

---

# 💡 How To Use

## 1. Upload Dataset

Upload a CSV, Excel, or JSON dataset through the dashboard.

---

## 2. Configure Preprocessing Settings

Choose preprocessing options such as:

* missing value strategy
* outlier handling method
* encoding type
* scaling method
* PCA settings

---

## 3. Run Pipeline

Click the preprocessing button to run the complete automated pipeline.

The system will:

* clean the dataset
* handle outliers
* encode features
* scale data
* prepare trainable datasets

---

## 4. Analyze Results

Review:

* cleaning reports
* outlier summaries
* feature distributions
* dataset statistics

---

## 5. Download Processed Data

Download:

* cleaned dataset
* training dataset
* validation dataset
* testing dataset

---

# 🛡️ Overfitting Prevention

The pipeline includes several mechanisms to reduce overfitting:

1. Feature scaling
2. Correlated feature removal
3. Low variance feature removal
4. PCA dimensionality reduction
5. Train / Validation / Test splitting

---

# 📦 Dependencies

```bash
streamlit
pandas
numpy
scikit-learn
scipy
plotly
openpyxl
xlrd
matplotlib
seaborn
```

---

# 💻 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly

---

# 📝 License

This project is open-source and available under the MIT License.
