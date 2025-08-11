"""
Data cleaning and preprocessing utilities for the data analysis project.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple, Union

class DataCleaner:
    """Class to handle data cleaning and preprocessing tasks."""
    
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize the DataCleaner.
        
        Args:
            dataset (pd.DataFrame): Dataset to clean
        """
        self.original_dataset = dataset.copy()
        self.dataset = dataset.copy()
        self.cleaning_log = []
    
    def handle_missing_values(self, strategy: str = 'auto', columns: List[str] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values ('auto', 'drop', 'mean', 'median', 'mode', 'forward', 'backward')
            columns (List[str]): Specific columns to handle (if None, handle all)
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        if columns is None:
            columns = self.dataset.columns.tolist()
        
        for col in columns:
            if col not in self.dataset.columns:
                continue
                
            missing_count = self.dataset[col].isnull().sum()
            if missing_count == 0:
                continue
            
            self.cleaning_log.append(f"Handling {missing_count} missing values in column '{col}'")
            
            if strategy == 'drop':
                self.dataset = self.dataset.dropna(subset=[col])
                self.cleaning_log.append(f"Dropped rows with missing values in '{col}'")
                
            elif strategy == 'mean' and self.dataset[col].dtype in ['int64', 'float64']:
                mean_val = self.dataset[col].mean()
                self.dataset[col].fillna(mean_val, inplace=True)
                self.cleaning_log.append(f"Filled missing values in '{col}' with mean: {mean_val}")
                
            elif strategy == 'median' and self.dataset[col].dtype in ['int64', 'float64']:
                median_val = self.dataset[col].median()
                self.dataset[col].fillna(median_val, inplace=True)
                self.cleaning_log.append(f"Filled missing values in '{col}' with median: {median_val}")
                
            elif strategy == 'mode':
                mode_val = self.dataset[col].mode()[0] if not self.dataset[col].mode().empty else 0
                self.dataset[col].fillna(mode_val, inplace=True)
                self.cleaning_log.append(f"Filled missing values in '{col}' with mode: {mode_val}")
                
            elif strategy == 'forward':
                self.dataset[col].fillna(method='ffill', inplace=True)
                self.cleaning_log.append(f"Filled missing values in '{col}' with forward fill")
                
            elif strategy == 'backward':
                self.dataset[col].fillna(method='bfill', inplace=True)
                self.cleaning_log.append(f"Filled missing values in '{col}' with backward fill")
                
            elif strategy == 'auto':
                # Auto strategy based on data type
                if self.dataset[col].dtype in ['int64', 'float64']:
                    if missing_count / len(self.dataset) < 0.05:
                        # Less than 5% missing, use median
                        median_val = self.dataset[col].median()
                        self.dataset[col].fillna(median_val, inplace=True)
                        self.cleaning_log.append(f"Auto-filled missing values in '{col}' with median: {median_val}")
                    else:
                        # More than 5% missing, drop rows
                        self.dataset = self.dataset.dropna(subset=[col])
                        self.cleaning_log.append(f"Auto-dropped rows with missing values in '{col}' (>5% missing)")
                else:
                    # Categorical data, use mode
                    mode_val = self.dataset[col].mode()[0] if not self.dataset[col].mode().empty else 'Unknown'
                    self.dataset[col].fillna(mode_val, inplace=True)
                    self.cleaning_log.append(f"Auto-filled missing values in '{col}' with mode: {mode_val}")
        
        return self.dataset
    
    def remove_duplicates(self, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset (List[str]): Columns to consider for duplicate detection
            keep (str): Which duplicate to keep ('first', 'last', False)
            
        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        initial_rows = len(self.dataset)
        self.dataset = self.dataset.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(self.dataset)
        
        if removed_rows > 0:
            self.cleaning_log.append(f"Removed {removed_rows} duplicate rows")
        
        return self.dataset
    
    def encode_categorical_variables(self, method: str = 'label', columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            method (str): Encoding method ('label', 'onehot', 'frequency')
            columns (List[str]): Specific columns to encode (if None, encode all categorical)
            
        Returns:
            pd.DataFrame: Dataset with encoded variables
        """
        if columns is None:
            columns = self.dataset.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col not in self.dataset.columns:
                continue
            
            unique_count = self.dataset[col].nunique()
            
            if method == 'label':
                le = LabelEncoder()
                self.dataset[col] = le.fit_transform(self.dataset[col].astype(str))
                self.cleaning_log.append(f"Label encoded column '{col}' ({unique_count} unique values)")
                
            elif method == 'frequency':
                freq_map = self.dataset[col].value_counts().to_dict()
                self.dataset[col] = self.dataset[col].map(freq_map)
                self.cleaning_log.append(f"Frequency encoded column '{col}' ({unique_count} unique values)")
                
            elif method == 'onehot' and unique_count <= 10:  # Only for low cardinality
                dummies = pd.get_dummies(self.dataset[col], prefix=col)
                self.dataset = pd.concat([self.dataset.drop(col, axis=1), dummies], axis=1)
                self.cleaning_log.append(f"One-hot encoded column '{col}' ({unique_count} unique values)")
        
        return self.dataset
    
    def scale_numerical_variables(self, method: str = 'standard', columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical variables.
        
        Args:
            method (str): Scaling method ('standard', 'minmax')
            columns (List[str]): Specific columns to scale (if None, scale all numerical)
            
        Returns:
            pd.DataFrame: Dataset with scaled variables
        """
        if columns is None:
            columns = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
            self.dataset[columns] = scaler.fit_transform(self.dataset[columns])
            self.cleaning_log.append(f"Standard scaled columns: {columns}")
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.dataset[columns] = scaler.fit_transform(self.dataset[columns])
            self.cleaning_log.append(f"Min-Max scaled columns: {columns}")
        
        return self.dataset
    
    def remove_outliers(self, columns: List[str] = None, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numerical columns.
        
        Args:
            columns (List[str]): Specific columns to process (if None, process all numerical)
            method (str): Method for outlier detection ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        if columns is None:
            columns = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_rows = len(self.dataset)
        
        for col in columns:
            if col not in self.dataset.columns:
                continue
            
            if method == 'iqr':
                Q1 = self.dataset[col].quantile(0.25)
                Q3 = self.dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (self.dataset[col] >= lower_bound) & (self.dataset[col] <= upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.dataset[col] - self.dataset[col].mean()) / self.dataset[col].std())
                mask = z_scores <= threshold
            
            self.dataset = self.dataset[mask]
            removed_outliers = initial_rows - len(self.dataset)
            
            if removed_outliers > 0:
                self.cleaning_log.append(f"Removed {removed_outliers} outliers from column '{col}' using {method} method")
        
        return self.dataset
    
    def get_cleaning_report(self) -> str:
        """
        Get a report of all cleaning operations performed.
        
        Returns:
            str: Cleaning report
        """
        if not self.cleaning_log:
            return "No cleaning operations performed."
        
        report = "Data Cleaning Report:\n"
        report += "=" * 50 + "\n"
        for i, operation in enumerate(self.cleaning_log, 1):
            report += f"{i}. {operation}\n"
        
        report += "\nDataset Changes:\n"
        report += f"Original rows: {len(self.original_dataset)}\n"
        report += f"Final rows: {len(self.dataset)}\n"
        report += f"Rows removed: {len(self.original_dataset) - len(self.dataset)}\n"
        
        return report

# Feature engineering utilities
class FeatureEngineer:
    """Class to handle feature engineering tasks."""
    
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize the FeatureEngineer.
        
        Args:
            dataset (pd.DataFrame): Dataset to engineer features for
        """
        self.dataset = dataset.copy()
        self.feature_log = []
    
    def create_polynomial_features(self, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            columns (List[str]): Columns to create polynomial features for
            degree (int): Degree of polynomial features
            
        Returns:
            pd.DataFrame: Dataset with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.dataset[columns])
        feature_names = poly.get_feature_names_out(columns)
        
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=self.dataset.index)
        
        # Remove original columns and add polynomial features
        self.dataset = self.dataset.drop(columns=columns)
        self.dataset = pd.concat([self.dataset, poly_df], axis=1)
        
        self.feature_log.append(f"Created polynomial features for columns: {columns} (degree {degree})")
        
        return self.dataset
    
    def create_interaction_features(self, column_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features (multiplication of two columns).
        
        Args:
            column_pairs (List[Tuple[str, str]]): Pairs of columns to create interactions for
            
        Returns:
            pd.DataFrame: Dataset with interaction features
        """
        for col1, col2 in column_pairs:
            if col1 in self.dataset.columns and col2 in self.dataset.columns:
                interaction_name = f"{col1}_x_{col2}"
                self.dataset[interaction_name] = self.dataset[col1] * self.dataset[col2]
                self.feature_log.append(f"Created interaction feature: {interaction_name}")
        
        return self.dataset
    
    def create_binned_features(self, columns: List[str], bins: int = 5, labels: List[str] = None) -> pd.DataFrame:
        """
        Create binned features from continuous variables.
        
        Args:
            columns (List[str]): Columns to bin
            bins (int): Number of bins
            labels (List[str]): Labels for bins
            
        Returns:
            pd.DataFrame: Dataset with binned features
        """
        for col in columns:
            if col in self.dataset.columns:
                binned_name = f"{col}_binned"
                self.dataset[binned_name] = pd.cut(self.dataset[col], bins=bins, labels=labels)
                self.feature_log.append(f"Created binned feature: {binned_name} ({bins} bins)")
        
        return self.dataset
    
    def get_feature_report(self) -> str:
        """
        Get a report of all feature engineering operations performed.
        
        Returns:
            str: Feature engineering report
        """
        if not self.feature_log:
            return "No feature engineering operations performed."
        
        report = "Feature Engineering Report:\n"
        report += "=" * 50 + "\n"
        for i, operation in enumerate(self.feature_log, 1):
            report += f"{i}. {operation}\n"
        
        return report

# Example usage function
def clean_and_engineer_data(dataset: pd.DataFrame, 
                          missing_strategy: str = 'auto',
                          encode_method: str = 'label',
                          scale_method: str = 'standard') -> Tuple[DataCleaner, FeatureEngineer, pd.DataFrame]:
    """
    Perform comprehensive data cleaning and feature engineering.
    
    Args:
        dataset (pd.DataFrame): Dataset to process
        missing_strategy (str): Strategy for handling missing values
        encode_method (str): Method for encoding categorical variables
        scale_method (str): Method for scaling numerical variables
        
    Returns:
        Tuple[DataCleaner, FeatureEngineer, pd.DataFrame]: Cleaner, Engineer instances and processed dataset
    """
    # Data cleaning
    cleaner = DataCleaner(dataset)
    cleaner.handle_missing_values(strategy=missing_strategy)
    cleaner.remove_duplicates()
    cleaner.encode_categorical_variables(method=encode_method)
    cleaner.scale_numerical_variables(method=scale_method)
    
    # Feature engineering (basic example)
    engineer = FeatureEngineer(cleaner.dataset)
    
    return cleaner, engineer, cleaner.dataset
