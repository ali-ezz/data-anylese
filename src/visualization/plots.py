"""
Data visualization utilities for the data analysis project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class DataVisualizer:
    """Class to create various data visualizations."""
    
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize the DataVisualizer.
        
        Args:
            dataset (pd.DataFrame): Dataset to visualize
        """
        self.dataset = dataset
        self.numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
    
    def plot_histogram(self, column: str, bins: int = 30, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot histogram for a numerical column.
        
        Args:
            column (str): Column to plot
            bins (int): Number of bins
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Histogram figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.dataset[column].hist(bins=bins, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        return fig
    
    def plot_boxplot(self, column: str, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot boxplot for a numerical column.
        
        Args:
            column (str): Column to plot
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Boxplot figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.dataset.boxplot(column=column, ax=ax)
        ax.set_title(f'Boxplot of {column}')
        ax.set_ylabel(column)
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, x_column: str, y_column: str, 
                    color_column: str = None, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot scatter plot between two numerical columns.
        
        Args:
            x_column (str): X-axis column
            y_column (str): Y-axis column
            color_column (str): Column to color by (optional)
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Scatter plot figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_column:
            scatter = ax.scatter(self.dataset[x_column], self.dataset[y_column], 
                               c=self.dataset[color_column], alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(self.dataset[x_column], self.dataset[y_column], alpha=0.6, color='blue')
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f'{y_column} vs {x_column}')
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Correlation heatmap figure
        """
        corr_matrix = self.dataset[self.numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        plt.tight_layout()
        return fig
    
    def plot_categorical_counts(self, column: str, top_n: int = 10, 
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot bar chart of categorical value counts.
        
        Args:
            column (str): Categorical column to plot
            top_n (int): Number of top categories to show
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Bar chart figure
        """
        value_counts = self.dataset[column].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title(f'Top {top_n} Categories in {column}')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, date_column: str, value_column: str, 
                        aggregation: str = 'mean', figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot time series data.
        
        Args:
            date_column (str): Date column
            value_column (str): Value column to plot
            aggregation (str): Aggregation method ('mean', 'sum', 'count')
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Time series plot figure
        """
        # Convert date column to datetime
        df_ts = self.dataset.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        
        # Aggregate data
        if aggregation == 'mean':
            ts_data = df_ts.groupby(date_column)[value_column].mean()
        elif aggregation == 'sum':
            ts_data = df_ts.groupby(date_column)[value_column].sum()
        elif aggregation == 'count':
            ts_data = df_ts.groupby(date_column)[value_column].count()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ts_data.index, ts_data.values, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{aggregation.title()} {value_column}')
        ax.set_title(f'Time Series: {aggregation.title()} {value_column} over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

class InteractiveVisualizer:
    """Class to create interactive visualizations using Plotly."""
    
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize the InteractiveVisualizer.
        
        Args:
            dataset (pd.DataFrame): Dataset to visualize
        """
        self.dataset = dataset
    
    def interactive_scatter(self, x_column: str, y_column: str, 
                           color_column: str = None, size_column: str = None) -> go.Figure:
        """
        Create interactive scatter plot.
        
        Args:
            x_column (str): X-axis column
            y_column (str): Y-axis column
            color_column (str): Column to color by (optional)
            size_column (str): Column to size by (optional)
            
        Returns:
            go.Figure: Interactive scatter plot
        """
        fig = px.scatter(self.dataset, x=x_column, y=y_column, 
                        color=color_column, size=size_column,
                        title=f'{y_column} vs {x_column}',
                        hover_data=self.dataset.columns.tolist())
        return fig
    
    def interactive_histogram(self, column: str, nbins: int = 30) -> go.Figure:
        """
        Create interactive histogram.
        
        Args:
            column (str): Column to plot
            nbins (int): Number of bins
            
        Returns:
            go.Figure: Interactive histogram
        """
        fig = px.histogram(self.dataset, x=column, nbins=nbins,
                          title=f'Distribution of {column}')
        return fig
    
    def interactive_boxplot(self, y_column: str, x_column: str = None) -> go.Figure:
        """
        Create interactive boxplot.
        
        Args:
            y_column (str): Y-axis column
            x_column (str): X-axis column for grouping (optional)
            
        Returns:
            go.Figure: Interactive boxplot
        """
        if x_column:
            fig = px.box(self.dataset, x=x_column, y=y_column,
                        title=f'{y_column} by {x_column}')
        else:
            fig = px.box(self.dataset, y=y_column,
                        title=f'Boxplot of {y_column}')
        return fig
    
    def interactive_correlation_matrix(self) -> go.Figure:
        """
        Create interactive correlation matrix heatmap.
        
        Returns:
            go.Figure: Interactive correlation heatmap
        """
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = self.dataset[numeric_columns].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title='Correlation Matrix')
        return fig
    
    def interactive_time_series(self, date_column: str, value_column: str,
                               aggregation: str = 'mean') -> go.Figure:
        """
        Create interactive time series plot.
        
        Args:
            date_column (str): Date column
            value_column (str): Value column to plot
            aggregation (str): Aggregation method
            
        Returns:
            go.Figure: Interactive time series plot
        """
        # Convert date column to datetime
        df_ts = self.dataset.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        
        # Aggregate data
        if aggregation == 'mean':
            ts_data = df_ts.groupby(date_column)[value_column].mean().reset_index()
        elif aggregation == 'sum':
            ts_data = df_ts.groupby(date_column)[value_column].sum().reset_index()
        elif aggregation == 'count':
            ts_data = df_ts.groupby(date_column)[value_column].count().reset_index()
        
        fig = px.line(ts_data, x=date_column, y=value_column,
                     title=f'{aggregation.title()} {value_column} over Time')
        return fig

class DashboardCreator:
    """Class to create comprehensive dashboards."""
    
    @staticmethod
    def create_eda_dashboard(dataset: pd.DataFrame, target_column: str = None) -> go.Figure:
        """
        Create a comprehensive EDA dashboard.
        
        Args:
            dataset (pd.DataFrame): Dataset to visualize
            target_column (str): Target variable (optional)
            
        Returns:
            go.Figure: Dashboard figure
        """
        numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature lists
        if target_column:
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
        
        # Determine subplot layout
        n_plots = min(6, len(numeric_columns) + len(categorical_columns))
        n_rows = min(3, (n_plots + 1) // 2)
        n_cols = min(2, n_plots)
        
        fig = make_subplots(rows=n_rows, cols=n_cols,
                           subplot_titles=['Dataset Overview', 'Missing Data', 
                                         'Numeric Distributions', 'Categorical Distributions',
                                         'Correlation Matrix', 'Target Analysis'][:n_plots],
                           specs=[[{"type": "table"}, {"type": "bar"}],
                                 [{"type": "histogram"}, {"type": "bar"}],
                                 [{"type": "heatmap"}, {"type": "scatter"}]][:n_rows])
        
        # Add dataset overview table
        overview_data = [
            ['Metric', 'Value'],
            ['Rows', str(len(dataset))],
            ['Columns', str(len(dataset.columns))],
            ['Numeric Columns', str(len(numeric_columns))],
            ['Categorical Columns', str(len(categorical_columns))],
            ['Missing Values', str(dataset.isnull().sum().sum())]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=overview_data[0],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[row[0] for row in overview_data[1:]],
                          fill_color='lavender',
                          align='left')
            ),
            row=1, col=1
        )
        
        # Add missing data bar chart
        missing_data = dataset.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig.add_trace(
                go.Bar(x=missing_data.index, y=missing_data.values,
                      name='Missing Data'),
                row=1, col=2
            )
        
        return fig

# Utility functions for common visualization tasks
def save_plot(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure.
    
    Args:
        fig (plt.Figure): Figure to save
        filename (str): Filename to save as
        dpi (int): DPI for the saved image
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def create_visualization_report(dataset: pd.DataFrame, output_dir: str = 'reports/figures') -> List[str]:
    """
    Create a comprehensive visualization report.
    
    Args:
        dataset (pd.DataFrame): Dataset to visualize
        output_dir (str): Directory to save figures
        
    Returns:
        List[str]: List of generated figure filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DataVisualizer(dataset)
    generated_figures = []
    
    # Create basic plots for numerical columns
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_columns[:5]:  # Limit to first 5 columns
        try:
            # Histogram
            fig = visualizer.plot_histogram(col)
            filename = f"{output_dir}/{col}_histogram.png"
            save_plot(fig, filename)
            generated_figures.append(filename)
            
            # Boxplot
            fig = visualizer.plot_boxplot(col)
            filename = f"{output_dir}/{col}_boxplot.png"
            save_plot(fig, filename)
            generated_figures.append(filename)
        except Exception as e:
            print(f"Could not create plots for {col}: {e}")
    
    # Create correlation heatmap if we have numerical columns
    if len(numeric_columns) > 1:
        try:
            fig = visualizer.plot_correlation_heatmap()
            filename = f"{output_dir}/correlation_heatmap.png"
            save_plot(fig, filename)
            generated_figures.append(filename)
        except Exception as e:
            print(f"Could not create correlation heatmap: {e}")
    
    return generated_figures

# Example usage function
def create_comprehensive_visualizations(dataset: pd.DataFrame, 
                                     target_column: str = None) -> Tuple[DataVisualizer, List]:
    """
    Create comprehensive visualizations for the dataset.
    
    Args:
        dataset (pd.DataFrame): Dataset to visualize
        target_column (str): Target variable for analysis
        
    Returns:
        Tuple[DataVisualizer, List]: Visualizer instance and list of generated figures
    """
    visualizer = DataVisualizer(dataset)
    figures = []
    
    # Generate basic plots
    if visualizer.numeric_columns:
        # Histogram for first numerical column
        fig1 = visualizer.plot_histogram(visualizer.numeric_columns[0])
        figures.append(('histogram', fig1))
        
        # Correlation heatmap
        if len(visualizer.numeric_columns) > 1:
            fig2 = visualizer.plot_correlation_heatmap()
            figures.append(('correlation_heatmap', fig2))
    
    if visualizer.categorical_columns:
        # Bar chart for first categorical column
        fig3 = visualizer.plot_categorical_counts(visualizer.categorical_columns[0])
        figures.append(('categorical_counts', fig3))
    
    return visualizer, figures
