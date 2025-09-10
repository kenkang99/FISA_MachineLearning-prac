# Machine Learning Project for Dacon Basic Competition

## Project Overview

This project tackles the Freebies' MLDL Mini Project [Competition on Dacon](https://dacon.io/competitions/official/236590/overview/description), a machine learning competition focused on a multi-class classification task. The goal is to design a model that classifies normal/abnormal types from anonymized, black-box data features (`X_01`, `X_02`, etc.).

- **Problem Type**: Multi-class classification (21 classes, 0-20)
- **Evaluation Metric**: Macro-F1 Score
- **Data Size**: Train (21,693 samples), Test (15,004 samples)

The repository includes scripts for exploratory data analysis and the implementation of several models, from classical ML approaches to a deep learning-based Transformer architecture.

## Repository Structure

```
FISA_MachineLearning-prac/
├───.gitignore
├───README.md
├───requirements.txt
├───data/
│   ├───sample_submission.csv
│   ├───test.csv
│   └───train.csv
└───src/
    ├───eda.py
    ├───gyemoo_code.ipynb
    ├───knn.py
    ├───lightgbm.py
    ├───poly.py
    ├───tabnet.ipynb
    └───xgb.py
```

- **`data/`**: Contains the training data (`train.csv`), test data (`test.csv`), and a sample submission file.
- **`src/`**: Contains all the Python scripts and notebooks for analysis and modeling.

## Common Strategies

Across the different models, several common strategies were employed to ensure robustness and performance:

- **Outlier Handling**: Quantile clipping (winsorizing) at 0.5% and 99.5% was applied to stabilize the training of tree-based models.
- **Cross-Validation**: A 5-fold stratified cross-validation strategy was used to make efficient use of the data and build a more generalizable model.
- **Ensembling**: For the GBDT and Transformer models, predictions from multiple runs with different random seeds were averaged (Out-of-Fold) to create the final submission.

## Model Performance

| Model | Macro-F1 Score |
| :--- | :--- |
| KNN | 0.5569 |
| TabNet | 0.7007 |
| LightGBM | 0.7445 |
| XGBoost | 0.7594 |
| **FT-Transformer** | **0.8205** |

*Scores are based on the project presentation document.*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd FISA_MachineLearning-prac
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Each script in the `src/` directory can be run independently to train a specific model and generate a submission file. The scripts assume they are run from the root directory of the project.

**Example:**
```bash
# Run the LightGBM script
python src/lightgbm.py

# Run the XGBoost script
python src/xgb.py
```
The generated submission files will be saved in the `data/` directory.

## Models

### 1. Exploratory Data Analysis (`eda.py`)
- Performs analysis on the training data to select features with high information value and low redundancy.
- Includes checks for multicollinearity (VIF), correlation, and distributional similarity (Kullback-Leibler Divergence). 23 features were selected based on this analysis.

### 2. K-Nearest Neighbors (`knn.py`)
- A non-parametric classifier that uses proximity to make classifications.
- Uses the subset of 23 features identified during EDA.
- **Note**: Suffers from the curse of dimensionality and is sensitive to noisy features, which explains its lower performance in this context.

### 3. Polynomial Regression (`poly.py`)
- A linear model that captures non-linear relationships by adding polynomial features.
- Also uses the pre-selected subset of features.

### 4. TabNet (`tabnet.ipynb`)
- A deep learning model that uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and more efficient learning.
- **Implementation**: Uses a train-validation split and a custom Macro-F1 metric for evaluation.

### 5. LightGBM (`lightgbm.py`)
- A high-performance gradient boosting framework that uses a leaf-wise tree growth strategy.
- **Implementation**: Uses GPU acceleration (`device: 'gpu'`), 5-fold stratified CV, and averages results across 3 random seeds.
- Generates `data/lgbm_oof_seedavg_submit.csv`.

### 6. XGBoost (`xgb.py`)
- A robust gradient boosting model known for its performance and regularity.
- **Implementation**: Uses GPU acceleration (`tree_method: 'gpu_hist'`), 5-fold stratified CV, and averages results across 3 random seeds.
- Generates `data/xgb_oof_seedavg_submit.csv`.

### 7. FT-Transformer (`gyemoo_code.ipynb`)
- A deep learning architecture that applies the Transformer's self-attention mechanism to tabular data.
- **Core Idea**: Each feature is treated as a "token" and embedded. A CLS token is added to learn a representation of the entire sample, which is then used for the final classification. This allows the model to learn complex feature interactions automatically.
- **Implementation**: Trained with 5-fold CV, AdamW optimizer, and uses Early Stopping with a ReduceLROnPlateau scheduler to prevent overfitting.
- Generates `data/gyemoo_fttransformer_5fold_submit.csv`.
