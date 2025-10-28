import pandas as pd
import pytest
import os

@pytest.fixture
def wine_data():
    return pd.read_csv("wine.csv", sep=";")

def test_dataset_exists():
    assert os.path.exists("wine.csv"), "wine.csv file not found"

def test_csv_can_be_parsed():
    try:
        df = pd.read_csv("wine.csv", sep=";")
        assert df is not None
    except Exception as e:
        pytest.fail(f"Failed to parse CSV: {str(e)}")

def test_required_columns_present(wine_data):
    required_columns = [
        'type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
        'alcohol', 'quality'
    ]
    missing_columns = set(required_columns) - set(wine_data.columns)
    assert not missing_columns, f"Missing required columns: {missing_columns}"

def test_no_extra_columns(wine_data):
    expected_columns = [
        'type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
        'alcohol', 'quality'
    ]
    extra_columns = set(wine_data.columns) - set(expected_columns)
    assert not extra_columns, f"Unexpected columns found: {extra_columns}"

def test_minimum_rows(wine_data):
    min_rows = 100
    assert len(wine_data) >= min_rows, f"Dataset has only {len(wine_data)} rows, minimum is {min_rows}"

def test_no_empty_rows(wine_data):
    empty_rows = wine_data.isna().all(axis=1).sum()
    assert empty_rows == 0, f"Found {empty_rows} completely empty rows"

def test_numeric_columns_are_numeric(wine_data):
    numeric_columns = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
        'alcohol', 'quality'
    ]
    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(wine_data[col]), f"Column '{col}' is not numeric"

def test_type_column_values(wine_data):
    valid_types = ['red', 'white']
    invalid_types = wine_data[~wine_data['type'].isin(valid_types)]['type'].unique()
    assert len(invalid_types) == 0, f"Invalid wine types found: {invalid_types}"

def test_no_missing_values_critical_columns(wine_data):
    critical_columns = ['type', 'quality']
    for col in critical_columns:
        missing_count = wine_data[col].isna().sum()
        assert missing_count == 0, f"Critical column '{col}' has {missing_count} missing values"

def test_no_excessive_duplicates(wine_data):
    duplicates = wine_data.duplicated().sum()
    assert duplicates < len(wine_data) * 0.1, f"Too many duplicate rows: {duplicates} ({duplicates/len(wine_data)*100:.1f}%)"

def test_no_negative_values_where_inappropriate(wine_data):
    non_negative_columns = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'sulphates', 'alcohol'
    ]
    for col in non_negative_columns:
        negative_count = (wine_data[col] < 0).sum()
        assert negative_count == 0, f"Column '{col}' has {negative_count} negative values"

def test_quality_range(wine_data):
    assert wine_data['quality'].min() >= 0, "Quality scores below 0 found"
    assert wine_data['quality'].max() <= 10, "Quality scores above 10 found"

def test_ph_range(wine_data):
    assert wine_data['pH'].min() >= 2.5, "pH values too low (< 2.5)"
    assert wine_data['pH'].max() <= 4.5, "pH values too high (> 4.5)"

def test_alcohol_range(wine_data):
    assert wine_data['alcohol'].min() >= 5, "Alcohol content too low (< 5%)"
    assert wine_data['alcohol'].max() <= 20, "Alcohol content too high (> 20%)"

def test_balanced_dataset(wine_data):
    type_counts = wine_data['type'].value_counts()
    ratio = type_counts.max() / type_counts.min()
    assert ratio <= 10, f"Dataset very imbalanced: {dict(type_counts)}, ratio={ratio:.2f}:1"

def test_quality_distribution(wine_data):
    mid_range = wine_data['quality'].between(5, 7).sum()
    mid_range_percent = mid_range / len(wine_data) * 100
    assert mid_range_percent >= 50, f"Only {mid_range_percent:.1f}% of wines in 5-7 quality range (expected >= 50%)"

def test_no_extreme_outliers(wine_data):
    numeric_cols = wine_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col == 'quality':
            continue
        Q1 = wine_data[col].quantile(0.25)
        Q3 = wine_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = wine_data[(wine_data[col] < lower_bound) | (wine_data[col] > upper_bound)]
        outlier_percent = len(outliers) / len(wine_data) * 100
        assert outlier_percent < 5, f"Column '{col}' has {outlier_percent:.1f}% extreme outliers (> 5%)"