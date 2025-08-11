"""
Exploratory Data Analysis utilities for the data analysis project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ExploratoryDataAnalysis:
    """Class to perform comprehensive exploratory data analysis."""
    
    def __init__(self, dataset: pd.DataFrame, target_column: str = None):
        """
        Initialize the EDA class.
        
        Args:
            dataset (pd.DataFrame): Dataset to analyze
            target_column (str): Target variable for analysis
        """
        self.dataset = dataset
        self.target_column = target_column
        self.numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from feature lists if it exists
        if target_column:
            if target_column in self.numeric_columns:
                self.numeric_columns.remove(target_column)
            if target_column in self.categorical_columns:
                self.categorical_columns.remove(target_column)
    
    def get_basic_statistics(self) -> pd.DataFrame:
        """
        Get basic statistical summary of the dataset.
        
        Returns:
            pd.DataFrame: Statistical summary
        """
        return self.dataset.describe()
    
    def analyze_missing_data(self) -> pd.DataFrame:
        """
        Analyze missing data patterns in the dataset.
        
        Returns:
            pd.DataFrame: Missing data analysis
        """
        missing_data = pd.DataFrame({
            'Missing Count': self.dataset.isnull().sum(),
            'Missing Percentage': (self.dataset.isnull().sum() / len(self.dataset)) * 100
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)
        return missing_data
    
    def analyze_data_types(self) -> pd.DataFrame:
        """
        Analyze data types and memory usage.
        
        Returns:
            pd.DataFrame: Data type analysis
        """
        dtypes_analysis = pd.DataFrame({
            'Data Type': self.dataset.dtypes,
            'Non-Null Count': self.dataset.count(),
            'Null Count': self.dataset.isnull().sum(),
            'Unique Values': [self.dataset[col].nunique() for col in self.dataset.columns]
        })
        return dtypes_analysis
    
    def plot_distribution(self, columns: List[str] = None, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot distribution of numerical variables.
        
        Args:
            columns (List[str]): Specific columns to plot (if None, plot all numerical)
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Distribution plot
        """
        if columns is None:
            columns = self.numeric_columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                self.dataset[col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot correlation matrix of numerical variables.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Correlation matrix plot
        """
        # Include target column in correlation analysis if specified
        corr_columns = self.numeric_columns.copy()
        if self.target_column and self.target_column in self.dataset.columns:
            if self.target_column not in corr_columns:
                corr_columns.append(self.target_column)
        
        corr_matrix = self.dataset[corr_columns].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix')
        
        return fig
    
    def plot_categorical_distributions(self, columns: List[str] = None, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot distributions of categorical variables.
        
        Args:
            columns (List[str]): Specific columns to plot (if None, plot all categorical)
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Categorical distribution plot
        """
        if columns is None:
            columns = self.categorical_columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                value_counts = self.dataset[col].value_counts().head(10)  # Top 10 categories
                value_counts.plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def analyze_target_relationships(self, top_n: int = 10) -> Dict:
        """
        Analyze relationships between features and target variable.
        
        Args:
            top_n (int): Number of top correlations to return
            
        Returns:
            Dict: Target relationship analysis
        """
        if not self.target_column or self.target_column not in self.dataset.columns:
            return {"error": "Target column not specified or not found in dataset"}
        
        target = self.dataset[self.target_column]
        relationships = {}
        
        # Numerical feature correlations
        if self.numeric_columns:
            correlations = {}
            for col in self.numeric_columns:
                if col != self.target_column:
                    corr, p_value = stats.pearsonr(self.dataset[col].dropna(), target.dropna())
                    correlations[col] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            # Sort by absolute correlation
            sorted_correlations = dict(sorted(correlations.items(), 
                                            key=lambda x: abs(x[1]['correlation']), 
                                            reverse=True)[:top_n])
            relationships['numerical_correlations'] = sorted_correlations
        
        # Categorical feature relationships (using ANOVA for numerical targets)
        if self.categorical_columns and target.dtype in ['int64', 'float64']:
            anova_results = {}
            for col in self.categorical_columns:
                groups = [group[target.name].dropna() for name, group in self.dataset.groupby(col)]
                if len(groups) > 1 and all(len(group) > 0 for group in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results[col] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            # Sort by F-statistic
            sorted_anova = dict(sorted(anova_results.items(), 
                                     key=lambda x: x[1]['f_statistic'], 
                                     reverse=True)[:top_n])
            relationships['categorical_anova'] = sorted_anova
        
        return relationships
    
    def detect_outliers(self, columns: List[str] = None, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers in numerical columns.
        
        Args:
            columns (List[str]): Specific columns to check (if None, check all numerical)
            method (str): Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: Outlier detection results
        """
        if columns is None:
            columns = self.numeric_columns
        
        outlier_results = {}
        
        for col in columns:
            if col not in self.dataset.columns:
                continue
            
            if method == 'iqr':
                Q1 = self.dataset[col].quantile(0.25)
                Q3 = self.dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.dataset[(self.dataset[col] < lower_bound) | (self.dataset[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.dataset[col].dropna()))
                outliers = self.dataset.iloc[z_scores.index[z_scores > 3]]
            
            outlier_results[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(self.dataset)) * 100,
                'lower_bound': lower_bound if method == 'iqr' else None,
                'upper_bound': upper_bound if method == 'iqr' else None
            }
        
        return pd.DataFrame(outlier_results).T
    
    def generate_eda_report(self) -> str:
        """
        Generate a comprehensive EDA report.
        
        Returns:
            str: EDA report
        """
        report = "Exploratory Data Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Dataset overview
        report += "1. DATASET OVERVIEW\n"
        report += "-" * 20 + "\n"
        report += f"Dataset Shape: {self.dataset.shape[0]} rows × {self.dataset.shape[1]} columns\n"
        report += f"Memory Usage: {self.dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
        report += f"Numeric Columns: {len(self.numeric_columns)}\n"
        report += f"Categorical Columns: {len(self.categorical_columns)}\n\n"
        
        # Missing data
        missing_data = self.analyze_missing_data()
        if not missing_data.empty:
            report += "2. MISSING DATA ANALYSIS\n"
            report += "-" * 25 + "\n"
            for col, row in missing_data.head().iterrows():
                report += f"{col}: {row['Missing Count']} ({row['Missing Percentage']:.2f}%)\n"
            report += "\n"
        
        # Data types
        report += "3. DATA TYPES\n"
        report += "-" * 15 + "\n"
        dtypes_analysis = self.analyze_data_types()
        for col, row in dtypes_analysis.head(10).iterrows():
            report += f"{col}: {row['Data Type']} ({row['Unique Values']} unique values)\n"
        report += "\n"
        
        # Basic statistics
        report += "4. BASIC STATISTICS\n"
        report += "-" * 20 + "\n"
        basic_stats = self.get_basic_statistics()
        report += f"Mean of numerical columns range from {basic_stats.loc['mean'].min():.2f} to {basic_stats.loc['mean'].max():.2f}\n"
        report += f"Standard deviation ranges from {basic_stats.loc['std'].min():.2f} to {basic_stats.loc['std'].max():.2f}\n\n"
        
        # Target relationships (if available)
        if self.target_column:
            report += "5. TARGET RELATIONSHIPS\n"
            report += "-" * 22 + "\n"
            relationships = self.analyze_target_relationships()
            if 'numerical_correlations' in relationships:
                report += "Top numerical feature correlations:\n"
                for col, stats in list(relationships['numerical_correlations'].items())[:5]:
                    significance = "significant" if stats['significant'] else "not significant"
                    report += f"  {col}: {stats['correlation']:.3f} ({significance})\n"
            report += "\n"
        
        return report

# Advanced visualization utilities
class AdvancedVisualizations:
    """Class for advanced data visualizations."""
    
    @staticmethod
    def plot_boxplots_by_target(dataset: pd.DataFrame, target_column: str, 
                              numeric_columns: List[str] = None, 
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot boxplots of numerical features grouped by target variable.
        
        Args:
            dataset (pd.DataFrame): Dataset to visualize
            target_column (str): Target variable for grouping
            numeric_columns (List[str]): Numerical columns to plot
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Boxplot figure
        """
        if numeric_columns is None:
            numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
        
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                ax = axes[i]
                dataset.boxplot(column=col, by=target_column, ax=ax)
                ax.set_title(f'{col} by {target_column}')
                ax.set_xlabel(target_column)
                ax.set_ylabel(col)
        
        # Hide empty subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_scatter_matrix(dataset: pd.DataFrame, columns: List[str] = None,
                          figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """
        Plot scatter matrix for numerical variables.
        
        Args:
            dataset (pd.DataFrame): Dataset to visualize
            columns (List[str]): Columns to include in scatter matrix
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Scatter matrix figure
        """
        if columns is None:
            columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        fig, ax = plt.subplots(figsize=figsize)
        pd.plotting.scatter_matrix(dataset[columns], ax=ax, alpha=0.6, 
                                 figsize=figsize, diagonal='hist')
        plt.tight_layout()
        return fig

# Example usage function
def perform_comprehensive_eda(dataset: pd.DataFrame, target_column: str = None) -> Tuple[ExploratoryDataAnalysis, Dict]:
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        dataset (pd.DataFrame): Dataset to analyze
        target_column (str): Target variable for analysis
        
    Returns:
        Tuple[ExploratoryDataAnalysis, Dict]: EDA instance and analysis results
    """
    eda = ExploratoryDataAnalysis(dataset, target_column)
    
    results = {
        'basic_statistics': eda.get_basic_statistics(),
        'missing_data': eda.analyze_missing_data(),
        'data_types': eda.analyze_data_types(),
        'outliers': eda.detect_outliers(),
        'target_relationships': eda.analyze_target_relationships() if target_column else {}
    }
    
    return eda, results
