"""
Statistical analysis and hypothesis testing utilities for the data analysis project.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class HypothesisTester:
    """Class to perform various statistical hypothesis tests."""
    
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize the HypothesisTester.
        
        Args:
            dataset (pd.DataFrame): Dataset to test hypotheses on
        """
        self.dataset = dataset
        self.test_results = []
    
    def t_test_independent(self, group1_column: str, group2_column: str, 
                          equal_var: bool = True) -> Dict[str, Any]:
        """
        Perform independent t-test between two groups.
        
        Args:
            group1_column (str): Column name for first group
            group2_column (str): Column name for second group
            equal_var (bool): Assume equal variances
            
        Returns:
            Dict: Test results
        """
        group1 = self.dataset[group1_column].dropna()
        group2 = self.dataset[group2_column].dropna()
        
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        result = {
            'test_type': 'Independent T-Test',
            'group1': group1_column,
            'group2': group2_column,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'alpha': 0.05,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            'group1_std': group1.std(),
            'group2_std': group2.std()
        }
        
        self.test_results.append(result)
        return result
    
    def t_test_paired(self, before_column: str, after_column: str) -> Dict[str, Any]:
        """
        Perform paired t-test.
        
        Args:
            before_column (str): Column name for before measurements
            after_column (str): Column name for after measurements
            
        Returns:
            Dict: Test results
        """
        before = self.dataset[before_column].dropna()
        after = self.dataset[after_column].dropna()
        
        # Ensure same length
        min_length = min(len(before), len(after))
        before = before[:min_length]
        after = after[:min_length]
        
        t_stat, p_value = stats.ttest_rel(before, after)
        
        result = {
            'test_type': 'Paired T-Test',
            'before_column': before_column,
            'after_column': after_column,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'alpha': 0.05,
            'before_mean': before.mean(),
            'after_mean': after.mean(),
            'mean_difference': (after - before).mean()
        }
        
        self.test_results.append(result)
        return result
    
    def anova_test(self, dependent_variable: str, independent_variable: str) -> Dict[str, Any]:
        """
        Perform one-way ANOVA test.
        
        Args:
            dependent_variable (str): Dependent variable column
            independent_variable (str): Independent variable column (categorical)
            
        Returns:
            Dict: Test results
        """
        groups = [group[dependent_variable].dropna() 
                 for name, group in self.dataset.groupby(independent_variable)]
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        result = {
            'test_type': 'One-Way ANOVA',
            'dependent_variable': dependent_variable,
            'independent_variable': independent_variable,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'alpha': 0.05,
            'num_groups': len(groups)
        }
        
        self.test_results.append(result)
        return result
    
    def chi_square_test(self, column1: str, column2: str) -> Dict[str, Any]:
        """
        Perform chi-square test of independence.
        
        Args:
            column1 (str): First categorical column
            column2 (str): Second categorical column
            
        Returns:
            Dict: Test results
        """
        contingency_table = pd.crosstab(self.dataset[column1], self.dataset[column2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        result = {
            'test_type': 'Chi-Square Test',
            'variable1': column1,
            'variable2': column2,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'alpha': 0.05
        }
        
        self.test_results.append(result)
        return result
    
    def correlation_test(self, column1: str, column2: str, method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation test.
        
        Args:
            column1 (str): First numerical column
            column2 (str): Second numerical column
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict: Test results
        """
        x = self.dataset[column1].dropna()
        y = self.dataset[column2].dropna()
        
        # Ensure same length
        min_length = min(len(x), len(y))
        x = x[:min_length]
        y = y[:min_length]
        
        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        result = {
            'test_type': f'{method.capitalize()} Correlation',
            'variable1': column1,
            'variable2': column2,
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'alpha': 0.05
        }
        
        self.test_results.append(result)
        return result
    
    def mann_whitney_u_test(self, group1_column: str, group2_column: str) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            group1_column (str): Column name for first group
            group2_column (str): Column name for second group
            
        Returns:
            Dict: Test results
        """
        group1 = self.dataset[group1_column].dropna()
        group2 = self.dataset[group2_column].dropna()
        
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        result = {
            'test_type': 'Mann-Whitney U Test',
            'group1': group1_column,
            'group2': group2_column,
            'u_statistic': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'alpha': 0.05,
            'group1_median': group1.median(),
            'group2_median': group2.median()
        }
        
        self.test_results.append(result)
        return result
    
    def shapiro_wilk_test(self, column: str) -> Dict[str, Any]:
        """
        Perform Shapiro-Wilk test for normality.
        
        Args:
            column (str): Column to test for normality
            
        Returns:
            Dict: Test results
        """
        data = self.dataset[column].dropna()
        
        # Shapiro-Wilk test requires sample size between 3 and 5000
        if len(data) < 3 or len(data) > 5000:
            raise ValueError("Shapiro-Wilk test requires sample size between 3 and 5000")
        
        stat, p_value = stats.shapiro(data)
        
        result = {
            'test_type': 'Shapiro-Wilk Normality Test',
            'column': column,
            'w_statistic': stat,
            'p_value': p_value,
            'normal': p_value > 0.05,
            'alpha': 0.05
        }
        
        self.test_results.append(result)
        return result
    
    def get_test_results(self) -> List[Dict[str, Any]]:
        """
        Get all test results.
        
        Returns:
            List[Dict]: List of all test results
        """
        return self.test_results
    
    def generate_hypothesis_report(self) -> str:
        """
        Generate a comprehensive hypothesis testing report.
        
        Returns:
            str: Hypothesis testing report
        """
        if not self.test_results:
            return "No hypothesis tests have been performed."
        
        report = "Hypothesis Testing Report\n"
        report += "=" * 40 + "\n\n"
        
        for i, test in enumerate(self.test_results, 1):
            report += f"{i}. {test['test_type']}\n"
            report += "-" * 30 + "\n"
            
            # Add test-specific details
            if 'p_value' in test:
                significance = "SIGNIFICANT" if test['significant'] else "NOT SIGNIFICANT"
                report += f"   Result: {significance} (p = {test['p_value']:.6f})\n"
            
            # Add variable information
            if 'group1' in test and 'group2' in test:
                report += f"   Groups: {test['group1']} vs {test['group2']}\n"
            elif 'variable1' in test and 'variable2' in test:
                report += f"   Variables: {test['variable1']} vs {test['variable2']}\n"
            
            # Add statistics
            if 't_statistic' in test:
                report += f"   T-statistic: {test['t_statistic']:.4f}\n"
            elif 'f_statistic' in test:
                report += f"   F-statistic: {test['f_statistic']:.4f}\n"
            elif 'chi2_statistic' in test:
                report += f"   Chi-square: {test['chi2_statistic']:.4f}\n"
            elif 'correlation' in test:
                report += f"   Correlation: {test['correlation']:.4f}\n"
            
            report += "\n"
        
        return report

class StatisticalSummary:
    """Class to generate comprehensive statistical summaries."""
    
    @staticmethod
    def descriptive_statistics(dataset: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Generate descriptive statistics for numerical columns.
        
        Args:
            dataset (pd.DataFrame): Dataset to analyze
            columns (List[str]): Specific columns to analyze (if None, all numerical)
            
        Returns:
            pd.DataFrame: Descriptive statistics
        """
        if columns is None:
            columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_df = dataset[columns].describe()
        
        # Add additional statistics
        additional_stats = pd.DataFrame(index=['skewness', 'kurtosis'], columns=columns)
        for col in columns:
            additional_stats.loc['skewness', col] = stats.skew(dataset[col].dropna())
            additional_stats.loc['kurtosis', col] = stats.kurtosis(dataset[col].dropna())
        
        return pd.concat([stats_df, additional_stats])
    
    @staticmethod
    def categorical_summary(dataset: pd.DataFrame, columns: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate summary statistics for categorical columns.
        
        Args:
            dataset (pd.DataFrame): Dataset to analyze
            columns (List[str]): Specific columns to analyze (if None, all categorical)
            
        Returns:
            Dict: Dictionary of summary statistics for each categorical column
        """
        if columns is None:
            columns = dataset.select_dtypes(include=['object']).columns.tolist()
        
        summaries = {}
        for col in columns:
            value_counts = dataset[col].value_counts()
            percentages = dataset[col].value_counts(normalize=True) * 100
            
            summary = pd.DataFrame({
                'Count': value_counts,
                'Percentage': percentages
            })
            summary.index.name = 'Category'
            
            summaries[col] = summary
        
        return summaries
    
    @staticmethod
    def correlation_analysis(dataset: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Perform comprehensive correlation analysis.
        
        Args:
            dataset (pd.DataFrame): Dataset to analyze
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            pd.DataFrame: Correlation matrix with significance tests
        """
        numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        # Correlation matrix
        corr_matrix = dataset[numeric_columns].corr(method=method)
        
        # P-values matrix
        p_values = pd.DataFrame(np.zeros((len(numeric_columns), len(numeric_columns))), 
                               index=numeric_columns, columns=numeric_columns)
        
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i != j:
                    x = dataset[col1].dropna()
                    y = dataset[col2].dropna()
                    min_length = min(len(x), len(y))
                    x = x[:min_length]
                    y = y[:min_length]
                    
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(x, y)
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(x, y)
                    elif method == 'kendall':
                        _, p_val = stats.kendalltau(x, y)
                    
                    p_values.loc[col1, col2] = p_val
        
        return corr_matrix, p_values

# Example usage function
def perform_statistical_analysis(dataset: pd.DataFrame, 
                               target_column: str = None,
                               test_hypotheses: List[Dict] = None) -> Tuple[HypothesisTester, Dict]:
    """
    Perform comprehensive statistical analysis.
    
    Args:
        dataset (pd.DataFrame): Dataset to analyze
        target_column (str): Target variable for analysis
        test_hypotheses (List[Dict]): List of hypotheses to test
        
    Returns:
        Tuple[HypothesisTester, Dict]: Tester instance and analysis results
    """
    tester = HypothesisTester(dataset)
    
    # Perform specified hypothesis tests
    if test_hypotheses:
        for hypothesis in test_hypotheses:
            test_type = hypothesis.get('test_type')
            if test_type == 't_test_independent':
                tester.t_test_independent(hypothesis['group1'], hypothesis['group2'])
            elif test_type == 'anova':
                tester.anova_test(hypothesis['dependent'], hypothesis['independent'])
            elif test_type == 'correlation':
                tester.correlation_test(hypothesis['var1'], hypothesis['var2'])
            # Add more test types as needed
    
    # Generate statistical summaries
    summary = StatisticalSummary()
    results = {
        'descriptive_stats': summary.descriptive_statistics(dataset),
        'categorical_summaries': summary.categorical_summary(dataset),
        'correlation_analysis': summary.correlation_analysis(dataset)
    }
    
    return tester, results
