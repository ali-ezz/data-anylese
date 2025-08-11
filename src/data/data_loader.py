"""
Data loading utilities for the data analysis project.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os

class DataLoader:
    """Class to handle data loading and initial dataset exploration."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = data_path or 'data/raw'
        self.dataset = None
        self.dataset_info = {}
    
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load dataset from various file formats.
        
        Args:
            filename (str): Name of the file to load
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        if filename.endswith('.csv'):
            self.dataset = pd.read_csv(file_path)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            self.dataset = pd.read_excel(file_path)
        elif filename.endswith('.json'):
            self.dataset = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV, Excel, or JSON files.")
        
        self._extract_dataset_info()
        return self.dataset
    
    def _extract_dataset_info(self):
        """Extract basic information about the dataset."""
        if self.dataset is not None:
            self.dataset_info = {
                'shape': self.dataset.shape,
                'columns': list(self.dataset.columns),
                'dtypes': self.dataset.dtypes.to_dict(),
                'missing_values': self.dataset.isnull().sum().to_dict(),
                'memory_usage': self.dataset.memory_usage(deep=True).sum(),
                'numeric_columns': self.dataset.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': self.dataset.select_dtypes(include=['object']).columns.tolist()
            }
    
    def get_dataset_summary(self) -> dict:
        """
        Get comprehensive dataset summary.
        
        Returns:
            dict: Dataset summary information
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        
        summary = {
            'dataset_size': f"{self.dataset_info['shape'][0]} rows × {self.dataset_info['shape'][1]} columns",
            'memory_usage': f"{self.dataset_info['memory_usage'] / 1024**2:.2f} MB",
            'numeric_variables': len(self.dataset_info['numeric_columns']),
            'categorical_variables': len(self.dataset_info['categorical_columns']),
            'total_missing_values': sum(self.dataset_info['missing_values'].values()),
            'columns_info': self._get_columns_info()
        }
        
        return summary
    
    def _get_columns_info(self) -> list:
        """Get detailed information about each column."""
        columns_info = []
        for col in self.dataset.columns:
            col_info = {
                'name': col,
                'dtype': str(self.dataset[col].dtype),
                'missing_count': int(self.dataset_info['missing_values'][col]),
                'missing_percentage': round((self.dataset_info['missing_values'][col] / len(self.dataset)) * 100, 2)
            }
            
            # Add additional info based on data type
            if self.dataset[col].dtype in ['int64', 'float64']:
                col_info['min'] = float(self.dataset[col].min())
                col_info['max'] = float(self.dataset[col].max())
                col_info['mean'] = float(self.dataset[col].mean())
                col_info['std'] = float(self.dataset[col].std())
            else:
                col_info['unique_values'] = int(self.dataset[col].nunique())
                if self.dataset[col].nunique() <= 10:
                    col_info['top_values'] = self.dataset[col].value_counts().head().to_dict()
            
            columns_info.append(col_info)
        
        return columns_info
    
    def identify_target_variables(self) -> list:
        """
        Identify potential target variables in the dataset.
        
        Returns:
            list: List of potential target variables
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        
        target_candidates = []
        
        # Look for columns that might be targets based on naming conventions
        target_indicators = ['target', 'label', 'class', 'outcome', 'result', 'score', 'value']
        
        for col in self.dataset.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in target_indicators):
                target_candidates.append(col)
        
        # Also consider columns with few unique values (potential classification targets)
        for col in self.dataset.columns:
            if col not in target_candidates and self.dataset[col].nunique() <= 10:
                target_candidates.append(col)
        
        return target_candidates

# Example usage function
def load_and_explore_dataset(filename: str, data_path: str = None) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Load and explore a dataset.
    
    Args:
        filename (str): Name of the file to load
        data_path (str): Path to the data directory
        
    Returns:
        Tuple[DataLoader, pd.DataFrame]: DataLoader instance and loaded dataset
    """
    loader = DataLoader(data_path)
    dataset = loader.load_dataset(filename)
    return loader, dataset
