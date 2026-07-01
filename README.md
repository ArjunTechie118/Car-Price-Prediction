# 🚗 Car Price Prediction

A machine learning project that predicts the resale price of used cars based on features like brand, model, manufacturing year, kilometers driven, and fuel type. The dataset was scraped from Quikr and required extensive cleaning before model training.

## 📌 Overview

This project walks through a complete ML pipeline:
1. **Data Cleaning** — handling messy, inconsistent scraped data
2. **Exploratory Data Analysis** — understanding feature distributions and outliers
3. **Model Building** — Linear Regression with One-Hot Encoding for categorical features
4. **Model Evaluation** — R² score across multiple train/test splits
5. **Hyperparameter Tuning** — Ridge regression with GridSearchCV

## 🛠️ Technologies Used

- **Language:**
Python 3

- **Data Manipulation:**

pandas - reading, cleaning, and transforming the dataset

NumPy - numerical operations

- **Machine Learning:**

scikit-learn -

LinearRegression - core prediction model

Ridge - regularized regression for hyperparameter tuning

OneHotEncoder - encoding categorical features (name, company, fuel_type)

ColumnTransformer - applying encoding only to categorical columns

Pipeline / make_pipeline - chaining preprocessing and model into one object

train_test_split - splitting data for training/evaluation

GridSearchCV - tuning the Ridge regularization strength

r2_score - model evaluation metric

- **Visualization:**

Matplotlib - scatter plot of actual vs. predicted prices

- **Environment:**

Google Colab

## 📂 Dataset

The raw dataset (`quikr_car.csv`) contains 892 rows with the following columns:

| Column | Description |
|---|---|
| `name` | Full car name/listing title |
| `company` | Car manufacturer |
| `year` | Manufacturing year |
| `Price` | Listed price (target variable) |
| `kms_driven` | Kilometers driven |
| `fuel_type` | Petrol / Diesel / LPG |

The raw data was messy, non-numeric years, string-formatted prices, missing values, and inconsistent naming, so significant cleaning was required (see below).

## 🧹 Data Cleaning Steps

- **`year`**: Removed non-numeric entries (e.g. `'zest'`, `'150k'`) and converted to `int`
- **`Price`**: Removed `"Ask For Price"` rows, stripped commas, and converted to `int`
- **`kms_driven`**: Stripped the `"kms"` suffix and commas, removed non-numeric rows, converted to `int`
- **`fuel_type`**: Dropped rows with missing fuel type
- **`name`**: Truncated to the first two words (brand + model) to reduce cardinality
- **Outliers**: Removed a car priced above ₹60,00,000 as an outlier
- Final cleaned dataset (815 rows) exported to `Cleaned car.csv`

## 🤖 Model

- **Preprocessing**: `OneHotEncoder` applied to `name`, `company`, and `fuel_type`; numeric columns (`year`, `kms_driven`) passed through unchanged, using `ColumnTransformer`
- **Pipeline**: `make_pipeline(ColumnTransformer, LinearRegression)`
- **Train/Test Split**: 80/20

### Performance

- R² score varies significantly depending on the random seed used for the train/test split (ranging from ~0.49 to ~0.79 across 10 different splits), reflecting the small dataset size and high cardinality of categorical features.
- Average R² across 10 runs:
  - **Train R²**: ~0.756
  - **Test R²**: ~0.644
- Ridge regression with `GridSearchCV` was also tested for regularization tuning.

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn (`LinearRegression`, `Ridge`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `GridSearchCV`)
- matplotlib

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Usage

1. Clone this repository
2. Place `quikr_car.csv` in the project directory
3. Run the notebook (`car_price_prediction.ipynb`) cell by cell
4. The cleaned dataset will be saved as `Cleaned car.csv`

### Example Prediction

```python
import pandas as pd

sample = pd.DataFrame(
    [['Maruti Suzuki', 'Maruti', 2019, 100, 'Petrol']],
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
)

predicted_price = pipe.predict(sample)
print(predicted_price)
```

## 📊 Results

A scatter plot of Actual vs Predicted prices is included in the notebook to visually assess model fit.

## 🔮 Future Improvements

- Try more robust models (Random Forest, Gradient Boosting) to improve R² and reduce variance across splits
- Better handling of high-cardinality categorical features (e.g. target encoding)
- More thorough outlier detection
- Cross-validation instead of single train/test splits for more reliable performance estimates

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
