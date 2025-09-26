"""
CrewAI - Data Analysis Crew Implementation
Features:
- YAML-based configuration
- Hierarchical processes
- Memory management
- Human-in-the-loop integration
- Advanced error handling
- Real-time monitoring
- Structured outputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# CrewAI imports
from crewai import Agent, Task, Crew, LLM, Process
from crewai.memory import LongTermMemory
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


load_dotenv()

# ============================================================================
# BACKEND: DATA MODELS AND SCHEMAS
# ============================================================================

class DataAnalysisToolInput(BaseModel):
    """
    Input schema for data analysis tools

    Defines the structure and validation for data analysis parameters
    received from the frontend interface.
    """
    data: str = Field(..., description="Data to analyze")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default={}, description="Additional parameters")

# ============================================================================
# BACKEND: CUSTOM TOOLS FOR DATA ANALYSIS
# ============================================================================

class StatisticalAnalysisTool(BaseTool):
    """
    Statistical Analysis Tool

    Performs comprehensive statistical analysis on datasets including
    descriptive statistics, correlations, and statistical tests.
    """
    name: str = "statistical_analysis_tool"
    description: str = "Perform comprehensive statistical analysis on datasets"
    args_schema: type[BaseModel] = DataAnalysisToolInput

    def _run(self, data: str, analysis_type: str, parameters: Dict[str, Any] = {}) -> str:
        """
        Execute statistical analysis on provided data

        Args:
            data (str): Data to analyze (CSV format or file path)
            analysis_type (str): Type of analysis to perform
            parameters (Dict[str, Any]): Additional analysis parameters

        Returns:
            str: JSON formatted statistical analysis results
        """
        try:
            # Parse data (assuming CSV format)
            df = pd.read_csv(data) if os.path.isfile(data) else pd.read_csv(pd.io.common.StringIO(data))

            results = {
                "summary_statistics": df.describe().to_dict(),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "correlation_matrix": df.select_dtypes(include=[np.number]).corr().to_dict()
            }

            if analysis_type == "comprehensive":
                # Add advanced analytics
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                results["advanced_metrics"] = {
                    "skewness": df[numeric_cols].skew().to_dict(),
                    "kurtosis": df[numeric_cols].kurtosis().to_dict(),
                    "outliers": self._detect_outliers(df[numeric_cols])
                }

            return json.dumps(results, indent=2)

        except Exception as e:
            return f"Error in statistical analysis: {str(e)}"

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers using IQR method"""
        outliers = {}
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        return outliers

class BusinessInsightsTool(BaseTool):
    """
    Business Insights Generation Tool

    Analyzes data to generate actionable business insights, trends,
    and strategic recommendations based on data patterns.
    """
    name: str = "business_insights_tool"
    description: str = "Generate business insights and strategic recommendations from data analysis"
    args_schema: type[BaseModel] = DataAnalysisToolInput

    def _run(self, data: str, analysis_type: str, parameters: Dict[str, Any] = {}) -> str:
        """
        Generate business insights from data analysis

        Args:
            data (str): Data to analyze (CSV format or file path)
            analysis_type (str): Type of insights to generate
            parameters (Dict[str, Any]): Additional parameters

        Returns:
            str: JSON formatted business insights and recommendations
        """
        try:
            # Parse data
            df = pd.read_csv(data) if os.path.isfile(data) else pd.read_csv(pd.io.common.StringIO(data))

            insights = {
                "executive_summary": self._generate_executive_summary(df),
                "key_insights": self._extract_key_insights(df),
                "business_recommendations": self._generate_recommendations(df),
                "risk_factors": self._identify_risks(df),
                "growth_opportunities": self._identify_opportunities(df),
                "performance_metrics": self._calculate_performance_metrics(df)
            }

            return json.dumps(insights, indent=2)

        except Exception as e:
            return f"Error generating business insights: {str(e)}"

    def _generate_executive_summary(self, df: pd.DataFrame) -> str:
        """Generate executive summary of the dataset"""
        total_records = len(df)
        total_columns = len(df.columns)
        numeric_columns = len(df.select_dtypes(include=[np.number]).columns)
        missing_data_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)

        return f"Dataset contains {total_records:,} records across {total_columns} variables ({numeric_columns} numeric). Data completeness: {100-missing_data_pct:.1f}%. Analysis reveals key patterns in performance trends and operational metrics suitable for strategic decision-making."

    def _extract_key_insights(self, df: pd.DataFrame) -> List[str]:
        """Extract key business insights from data"""
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            # Growth trends
            for col in numeric_df.columns:
                if 'revenue' in col.lower() or 'sales' in col.lower() or 'income' in col.lower():
                    growth_rate = ((numeric_df[col].iloc[-1] - numeric_df[col].iloc[0]) / numeric_df[col].iloc[0] * 100) if len(numeric_df) > 1 and numeric_df[col].iloc[0] != 0 else 0
                    insights.append(f"{col.title()} shows {growth_rate:.1f}% change from first to last record")

            # Volatility analysis
            for col in numeric_df.columns:
                cv = (numeric_df[col].std() / numeric_df[col].mean() * 100) if numeric_df[col].mean() != 0 else 0
                if cv > 50:
                    insights.append(f"{col.title()} shows high volatility (CV: {cv:.1f}%), indicating potential risk or opportunity")
                elif cv < 10:
                    insights.append(f"{col.title()} demonstrates stable performance (CV: {cv:.1f}%), suggesting predictable outcomes")

            # Correlation insights
            corr_matrix = numeric_df.corr()
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and abs(corr_matrix.iloc[i, j]) > 0.7:
                        relationship = "strong positive" if corr_matrix.iloc[i, j] > 0 else "strong negative"
                        insights.append(f"{relationship.title()} correlation between {col1} and {col2} ({corr_matrix.iloc[i, j]:.2f})")

        return insights[:10]  # Limit to top 10 insights

    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable business recommendations"""
        recommendations = []
        numeric_df = df.select_dtypes(include=[np.number])

        # Data quality recommendations
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 20]
        if not high_missing.empty:
            recommendations.append(f"Improve data collection for {', '.join(high_missing.index)} (>20% missing values)")

        # Performance optimization
        if not numeric_df.empty:
            for col in numeric_df.columns:
                if 'cost' in col.lower() or 'expense' in col.lower():
                    if numeric_df[col].std() > numeric_df[col].mean():
                        recommendations.append(f"Investigate {col} variability to identify cost optimization opportunities")

                if 'efficiency' in col.lower() or 'productivity' in col.lower():
                    low_performers = numeric_df[numeric_df[col] < numeric_df[col].quantile(0.25)]
                    if len(low_performers) > 0:
                        recommendations.append(f"Focus improvement efforts on bottom 25% performers in {col}")

        # Strategic recommendations
        recommendations.extend([
            "Implement automated monitoring dashboards for key performance indicators",
            "Establish data-driven decision protocols for operational improvements",
            "Consider predictive analytics for proactive issue identification"
        ])

        return recommendations[:8]  # Limit to top 8 recommendations

    def _identify_risks(self, df: pd.DataFrame) -> List[str]:
        """Identify potential business risks from data patterns"""
        risks = []
        numeric_df = df.select_dtypes(include=[np.number])

        # Outlier risks
        for col in numeric_df.columns:
            Q1, Q3 = numeric_df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = numeric_df[(numeric_df[col] < Q1 - 1.5*IQR) | (numeric_df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > len(numeric_df) * 0.05:  # >5% outliers
                risks.append(f"High outlier rate in {col} ({len(outliers)}/{len(numeric_df)}) may indicate data quality or process issues")

        # Trend risks
        for col in numeric_df.columns:
            if len(numeric_df) > 5:
                recent_trend = numeric_df[col].tail(5).mean() - numeric_df[col].head(5).mean()
                if 'revenue' in col.lower() and recent_trend < 0:
                    risks.append(f"Declining trend in {col} requires immediate attention")
                elif 'cost' in col.lower() and recent_trend > 0:
                    risks.append(f"Rising {col} trend may impact profitability")

        return risks[:5]  # Limit to top 5 risks

    def _identify_opportunities(self, df: pd.DataFrame) -> List[str]:
        """Identify growth and improvement opportunities"""
        opportunities = []
        numeric_df = df.select_dtypes(include=[np.number])

        # Performance gaps
        for col in numeric_df.columns:
            if 'performance' in col.lower() or 'efficiency' in col.lower():
                gap = numeric_df[col].max() - numeric_df[col].mean()
                if gap > 0:
                    opportunities.append(f"Potential {gap:.1f}% improvement opportunity in {col} by reaching top performance levels")

        # Underutilized resources
        for col in numeric_df.columns:
            if 'utilization' in col.lower() or 'capacity' in col.lower():
                if numeric_df[col].mean() < 80:
                    opportunities.append(f"Increase {col} from {numeric_df[col].mean():.1f}% to optimize resource usage")

        return opportunities[:5]  # Limit to top 5 opportunities

    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key performance metrics"""
        metrics = {}
        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            metrics["data_completeness"] = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            metrics["average_numeric_variance"] = numeric_df.var().mean()
            metrics["correlation_strength"] = abs(numeric_df.corr()).mean().mean()

            # Business-specific metrics
            for col in numeric_df.columns:
                if 'revenue' in col.lower():
                    metrics[f"{col}_growth_rate"] = ((numeric_df[col].iloc[-1] - numeric_df[col].iloc[0]) / numeric_df[col].iloc[0] * 100) if len(numeric_df) > 1 and numeric_df[col].iloc[0] != 0 else 0
                if 'efficiency' in col.lower():
                    metrics[f"{col}_average"] = numeric_df[col].mean()

        return metrics

class DataVisualizationTool(BaseTool):
    """
    Data Visualization and Chart Generation Tool

    Creates comprehensive visualizations and charts for data analysis
    including statistical plots, business dashboards, and trend analysis.
    """
    name: str = "data_visualization_tool"
    description: str = "Generate comprehensive data visualizations and charts"
    args_schema: type[BaseModel] = DataAnalysisToolInput

    def _run(self, data: str, analysis_type: str, parameters: Dict[str, Any] = {}) -> str:
        """
        Generate visualization recommendations and create chart configurations

        Args:
            data (str): Data to visualize (CSV format or file path)
            analysis_type (str): Type of visualizations to create
            parameters (Dict[str, Any]): Visualization parameters

        Returns:
            str: JSON formatted visualization specifications and recommendations
        """
        try:
            # Parse data
            df = pd.read_csv(data) if os.path.isfile(data) else pd.read_csv(pd.io.common.StringIO(data))

            visualizations = {
                "recommended_charts": self._recommend_charts(df),
                "statistical_plots": self._generate_statistical_plots(df),
                "business_dashboards": self._create_dashboard_layout(df),
                "trend_analysis": self._analyze_trends(df),
                "interactive_features": self._suggest_interactive_features(df)
            }

            return json.dumps(visualizations, indent=2)

        except Exception as e:
            return f"Error generating visualizations: {str(e)}"

    def _recommend_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recommend appropriate chart types based on data characteristics"""
        recommendations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Distribution charts
        for col in numeric_cols:
            recommendations.append({
                "chart_type": "histogram",
                "title": f"Distribution of {col.title()}",
                "data_column": col,
                "description": f"Shows the frequency distribution of {col} values",
                "business_value": "Understand data spread and identify patterns"
            })

            recommendations.append({
                "chart_type": "box_plot",
                "title": f"Box Plot: {col.title()}",
                "data_column": col,
                "description": f"Displays quartiles and outliers for {col}",
                "business_value": "Identify outliers and data quality issues"
            })

        # Correlation analysis
        if len(numeric_cols) > 1:
            recommendations.append({
                "chart_type": "correlation_heatmap",
                "title": "Correlation Matrix",
                "data_columns": list(numeric_cols),
                "description": "Shows relationships between numeric variables",
                "business_value": "Identify which factors influence each other"
            })

            # Scatter plots for strong correlations
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and abs(corr_matrix.iloc[i, j]) > 0.5:
                        recommendations.append({
                            "chart_type": "scatter_plot",
                            "title": f"{col1.title()} vs {col2.title()}",
                            "x_column": col1,
                            "y_column": col2,
                            "description": f"Relationship between {col1} and {col2}",
                            "business_value": f"Understand how {col1} affects {col2}"
                        })

        # Categorical analysis
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Reasonable number of categories
                recommendations.append({
                    "chart_type": "bar_chart",
                    "title": f"Count by {col.title()}",
                    "data_column": col,
                    "description": f"Shows frequency of different {col} values",
                    "business_value": f"Understand distribution of {col} categories"
                })

        # Time series (if date columns detected)
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            for date_col in date_cols:
                for num_col in numeric_cols:
                    recommendations.append({
                        "chart_type": "line_chart",
                        "title": f"{num_col.title()} Over Time",
                        "x_column": date_col,
                        "y_column": num_col,
                        "description": f"Trend of {num_col} over {date_col}",
                        "business_value": "Track performance trends and seasonality"
                    })

        return recommendations[:12]  # Limit to top 12 recommendations

    def _generate_statistical_plots(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate statistical plot specifications"""
        plots = []
        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            # Q-Q plots for normality testing
            for col in numeric_df.columns:
                plots.append({
                    "plot_type": "qq_plot",
                    "title": f"Q-Q Plot: {col.title()}",
                    "data_column": col,
                    "purpose": "Test normality assumption",
                    "interpretation": "Points closer to diagonal line indicate normal distribution"
                })

            # Residual plots if there are relationships
            corr_matrix = numeric_df.corr()
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and abs(corr_matrix.iloc[i, j]) > 0.7:
                        plots.append({
                            "plot_type": "residual_plot",
                            "title": f"Residuals: {col1} vs {col2}",
                            "x_column": col1,
                            "y_column": col2,
                            "purpose": "Validate linear relationship assumption",
                            "interpretation": "Random scatter indicates good linear fit"
                        })

        return plots[:6]  # Limit to top 6 statistical plots

    def _create_dashboard_layout(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create business dashboard layout recommendations"""
        dashboard = {
            "layout": "grid",
            "sections": [],
            "kpi_cards": [],
            "filters": []
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # KPI Cards
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'profit', 'count', 'total']):
                dashboard["kpi_cards"].append({
                    "title": col.title(),
                    "value": float(df[col].sum()),
                    "format": "currency" if any(money in col.lower() for money in ['revenue', 'sales', 'profit', 'cost']) else "number",
                    "trend": "up" if len(df) > 1 and df[col].iloc[-1] > df[col].iloc[0] else "down"
                })

        # Dashboard sections
        dashboard["sections"] = [
            {
                "title": "Overview",
                "charts": ["kpi_cards", "summary_table"],
                "position": {"row": 1, "col": 1, "width": 12}
            },
            {
                "title": "Performance Analysis",
                "charts": ["correlation_heatmap", "trend_lines"],
                "position": {"row": 2, "col": 1, "width": 8}
            },
            {
                "title": "Distribution Analysis",
                "charts": ["histograms", "box_plots"],
                "position": {"row": 2, "col": 9, "width": 4}
            }
        ]

        # Interactive filters
        for col in categorical_cols:
            if df[col].nunique() <= 50:  # Reasonable filter options
                dashboard["filters"].append({
                    "column": col,
                    "type": "multiselect",
                    "default": "all"
                })

        return dashboard

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in the data"""
        trends = {
            "linear_trends": [],
            "seasonal_patterns": [],
            "anomalies": []
        }

        numeric_df = df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            if len(df) > 3:
                # Simple linear trend
                x = np.arange(len(df))
                slope = np.polyfit(x, numeric_df[col], 1)[0]
                trend_direction = "increasing" if slope > 0 else "decreasing"
                trends["linear_trends"].append({
                    "column": col,
                    "direction": trend_direction,
                    "slope": float(slope),
                    "strength": "strong" if abs(slope) > numeric_df[col].std() / len(df) else "weak"
                })

        return trends

    def _suggest_interactive_features(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Suggest interactive features for visualizations"""
        features = [
            {
                "feature": "drill_down",
                "description": "Click on chart elements to see detailed breakdowns",
                "use_case": "Explore data at different granularity levels"
            },
            {
                "feature": "dynamic_filtering",
                "description": "Filter charts in real-time using dropdown menus",
                "use_case": "Focus on specific data segments"
            },
            {
                "feature": "tooltip_details",
                "description": "Hover over data points for additional context",
                "use_case": "Get precise values and metadata"
            },
            {
                "feature": "cross_filtering",
                "description": "Selection in one chart filters others automatically",
                "use_case": "Explore relationships between different views"
            }
        ]

        return features

class ChartGeneratorTool(BaseTool):
    """
    Real Chart Generation Tool

    Creates actual interactive charts and visualizations using Plotly
    for display in Streamlit interface.
    """
    name: str = "chart_generator_tool"
    description: str = "Generate real interactive charts and visualizations"
    args_schema: type[BaseModel] = DataAnalysisToolInput

    def _run(self, data: str, analysis_type: str, parameters: Dict[str, Any] = {}) -> str:
        """
        Generate real charts and return configuration for display

        Args:
            data (str): Data to visualize (CSV format or file path)
            analysis_type (str): Type of charts to create
            parameters (Dict[str, Any]): Chart parameters

        Returns:
            str: JSON formatted chart configurations
        """
        try:
            # Parse data
            df = pd.read_csv(data) if os.path.isfile(data) else pd.read_csv(pd.io.common.StringIO(data))

            charts_config = {
                "charts_created": [],
                "data_summary": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                    "categorical_columns": list(df.select_dtypes(include=['object']).columns)
                }
            }

            return json.dumps(charts_config, indent=2)

        except Exception as e:
            return f"Error generating charts: {str(e)}"

# ============================================================================
# BACKEND: CHART GENERATION FUNCTIONS
# ============================================================================

def generate_distribution_charts(df: pd.DataFrame) -> List[go.Figure]:
    """Generate distribution charts for numeric columns"""
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols[:4]:  # Limit to first 4 columns
        # Histogram
        fig = px.histogram(
            df,
            x=col,
            title=f'Distribution of {col.title()}',
            nbins=30,
            marginal="box"
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title=col.title(),
            yaxis_title="Frequency"
        )
        charts.append(fig)

    return charts

def generate_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Generate correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        return None

    corr_matrix = numeric_df.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        height=600,
        width=600
    )

    return fig

def generate_scatter_plots(df: pd.DataFrame) -> List[go.Figure]:
    """Generate scatter plots for strong correlations"""
    charts = []
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        return charts

    corr_matrix = numeric_df.corr()

    # Find strong correlations
    strong_corrs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j and abs(corr_matrix.iloc[i, j]) > 0.5:
                strong_corrs.append((col1, col2, corr_matrix.iloc[i, j]))

    # Create scatter plots for top 3 correlations
    for col1, col2, corr_val in strong_corrs[:3]:
        fig = px.scatter(
            df,
            x=col1,
            y=col2,
            title=f'{col1.title()} vs {col2.title()} (r={corr_val:.2f})',
            trendline="ols"
        )

        fig.update_layout(
            height=400,
            xaxis_title=col1.title(),
            yaxis_title=col2.title()
        )

        charts.append(fig)

    return charts

def generate_categorical_charts(df: pd.DataFrame) -> List[go.Figure]:
    """Generate charts for categorical data"""
    charts = []
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols[:3]:  # Limit to first 3 columns
        if df[col].nunique() <= 20:  # Reasonable number of categories
            value_counts = df[col].value_counts().head(10)  # Top 10 categories

            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {col.title()}',
                labels={'x': col.title(), 'y': 'Count'}
            )

            fig.update_layout(
                height=400,
                xaxis_title=col.title(),
                yaxis_title="Count",
                xaxis_tickangle=-45
            )

            charts.append(fig)

    return charts

def generate_box_plots(df: pd.DataFrame) -> List[go.Figure]:
    """Generate box plots for outlier detection"""
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols[:4]:  # Limit to first 4 columns
        fig = px.box(
            df,
            y=col,
            title=f'Box Plot: {col.title()} (Outlier Detection)',
            points="outliers"
        )

        fig.update_layout(
            height=400,
            yaxis_title=col.title()
        )

        charts.append(fig)

    return charts

def generate_time_series_charts(df: pd.DataFrame) -> List[go.Figure]:
    """Generate time series charts if date columns exist"""
    charts = []

    # Try to detect date columns
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                continue

    if not date_cols:
        return charts

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for date_col in date_cols[:1]:  # Use first date column
        for num_col in numeric_cols[:3]:  # First 3 numeric columns
            fig = px.line(
                df.sort_values(date_col),
                x=date_col,
                y=num_col,
                title=f'{num_col.title()} Over Time'
            )

            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title=num_col.title()
            )

            charts.append(fig)

    return charts

def generate_summary_statistics_chart(df: pd.DataFrame) -> go.Figure:
    """Generate summary statistics visualization"""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return None

    summary_stats = numeric_df.describe()

    # Create subplots for different statistics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Mean Values', 'Standard Deviation', 'Min Values', 'Max Values'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # Mean values
    fig.add_trace(
        go.Bar(x=summary_stats.columns, y=summary_stats.loc['mean'], name='Mean'),
        row=1, col=1
    )

    # Standard deviation
    fig.add_trace(
        go.Bar(x=summary_stats.columns, y=summary_stats.loc['std'], name='Std Dev'),
        row=1, col=2
    )

    # Min values
    fig.add_trace(
        go.Bar(x=summary_stats.columns, y=summary_stats.loc['min'], name='Min'),
        row=2, col=1
    )

    # Max values
    fig.add_trace(
        go.Bar(x=summary_stats.columns, y=summary_stats.loc['max'], name='Max'),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        title_text="Summary Statistics Overview",
        showlegend=False
    )

    return fig

def create_kpi_dashboard(df: pd.DataFrame) -> Dict[str, Any]:
    """Create KPI dashboard data"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    kpis = {}

    # Basic KPIs
    kpis['total_records'] = len(df)
    kpis['total_columns'] = len(df.columns)
    kpis['missing_data_pct'] = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)

    # Numeric KPIs
    for col in numeric_cols:
        col_name = col.replace('_', ' ').title()
        kpis[f'{col_name} Average'] = df[col].mean()
        kpis[f'{col_name} Total'] = df[col].sum()

        # Growth rate if enough data points
        if len(df) > 1:
            growth_rate = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0] * 100) if df[col].iloc[0] != 0 else 0
            kpis[f'{col_name} Growth Rate'] = growth_rate

    return kpis

# ============================================================================
# BACKEND: YAML CONFIGURATION VALIDATION AND UTILITIES
# ============================================================================

def validate_data_analysis_yaml_configs(config_path: Path) -> bool:
    """
    Validate that all required YAML configuration files exist

    Args:
        config_path (Path): Path to configuration directory

    Returns:
        bool: True if all configs are valid, False otherwise
    """
    required_files = ['agents.yaml', 'tasks.yaml', 'crew.yaml']

    for file_name in required_files:
        file_path = config_path / file_name
        if not file_path.exists():
            st.error(f"Required configuration file missing: {file_name}")
            return False

    return True

def get_available_data_analysis_tool_names() -> List[str]:
    """
    Get list of available tool names for YAML configuration

    Returns:
        List[str]: Available tool names
    """
    return ['statistical_analysis_tool', 'business_insights_tool', 'data_visualization_tool', 'chart_generator_tool']

def display_data_analysis_yaml_configuration_help():
    """
    Display help information about YAML configuration structure
    """
    st.info("""
    **Data Analysis Crew YAML Configuration:**

    This application uses YAML files for complete configuration:

    📄 **agents.yaml** - Define data analysis agents with roles, goals, and tools
    📄 **tasks.yaml** - Define analysis workflow tasks with descriptions and dependencies
    📄 **crew.yaml** - Configure crew behavior, process type, and settings

    **Key Features:**
    - ✅ **Pure YAML Configuration** - No hardcoded agents or tasks
    - ✅ **Dynamic Tool Assignment** - Data analysis tools assigned based on YAML config
    - ✅ **Flexible Analysis Workflows** - Define complex data analysis dependencies
    - ✅ **Multiple Process Types** - Sequential or hierarchical execution
    - ✅ **Template Variables** - Dynamic data analysis parameter substitution
    - ✅ **Memory Management** - Long-term memory for analysis context

    **Available Data Analysis Tools:**
    - `statistical_analysis_tool` - Comprehensive statistical analysis
    - `business_insights_tool` - Business intelligence and strategic recommendations
    - `data_visualization_tool` - Chart generation and visualization strategies
    - `chart_generator_tool` - Real interactive chart creation with Plotly

    **Customization:**
    Modify the YAML files in the `/config` directory to customize:
    - Data analysis agent specializations and capabilities
    - Analysis task workflows and dependencies
    - Crew execution parameters and analysis methodologies
    - Tool assignments per analysis agent type
    """)

# ============================================================================
# BACKEND: CORE DATA ANALYSIS CREW MANAGEMENT CLASS
# ============================================================================

class DataAnalysisCrew:
    """
    CrewAI Implementation with YAML Configuration for Data Analysis

    Manages comprehensive data analysis workflows using multiple AI analysis agents.
    Loads configurations from YAML files for agents, tasks, and crew settings.
    Includes memory management and advanced data analysis capabilities.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the data analysis crew manager

        Args:
            config_path (str, optional): Path to configuration directory
        """
        self.config_path = config_path or Path(__file__).parent / "config"

        # Validate configuration files exist
        if not validate_data_analysis_yaml_configs(self.config_path):
            raise FileNotFoundError("Required YAML configuration files are missing")

        self.config = self._load_configurations()
        # Memory disabled for stability
        self.memory = False
        self.tools = self._setup_tools()

    def _load_configurations(self) -> Dict[str, Any]:
        """
        Load YAML configuration files for agents, tasks, and crew settings

        Returns:
            Dict[str, Any]: Loaded configuration data
        """
        config = {}
        config_files = ['agents.yaml', 'tasks.yaml', 'crew.yaml']

        for file_name in config_files:
            file_path = self.config_path / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config[file_name.split('.')[0]] = yaml.safe_load(f)
            else:
                st.warning(f"Configuration file {file_name} not found")

        return config


    def _setup_memory(self) -> Optional[LongTermMemory]:
        """Setup long-term memory"""
        memory_config = self.config.get('crew', {}).get('memory_config', {})
        if memory_config.get('provider') == 'long_term_memory':
            return LongTermMemory(
                storage=memory_config.get('storage', {})
            )
        return None

    def _setup_tools(self) -> List[BaseTool]:
        """Setup analysis tools"""
        return [
            StatisticalAnalysisTool(),
            BusinessInsightsTool(),
            DataVisualizationTool(),
            ChartGeneratorTool()
        ]

    def _create_llm(self, provider: str, model: str, **kwargs) -> LLM:
        """Create LLM instance with proper configuration"""
        provider_keys = {
            "Gemini": "GEMINI_API_KEY",
            "Groq": "GROQ_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "OpenAI": "OPENAI_API_KEY"
        }

        key_name = provider_keys.get(provider)
        if key_name:
            os.environ[key_name] = os.getenv(key_name)

        # Format model name with provider prefix for proper LiteLLM routing in CrewAI based implementation
        if provider.lower() == 'ollama':
            # CrewAI's built-in Ollama support via LiteLLM
            formatted_model = f"ollama/{model}"
        elif provider.lower() == 'gemini':
            formatted_model = f"gemini/{model}"
        elif provider.lower() == 'groq':
            formatted_model = f"groq/{model}"
        elif provider.lower() == 'anthropic':
            formatted_model = f"anthropic/{model}"
        elif provider.lower() == 'openai':
            formatted_model = f"openai/{model}"
        else:
            formatted_model = model

        # Create LLM parameters
        llm_params = {
            'model': formatted_model,
            'temperature': kwargs.get('temperature', 0.7)
        }

        # Add Ollama-specific parameters
        if provider.lower() == 'ollama' and 'base_url' in kwargs:
            llm_params['base_url'] = kwargs['base_url']

        # Add any additional kwargs that don't conflict
        for key, value in kwargs.items():
            if key not in ['temperature', 'model', 'base_url']:
                llm_params[key] = value

        return LLM(**llm_params)

    def _create_agents(self, llm_provider: str, model_name: str, **llm_kwargs) -> tuple[List[Agent], dict]:
        """Create agents from YAML configuration and return both list and mapping"""
        agents = []
        agent_mapping = {}
        agents_config = self.config.get('agents', {})

        llm = self._create_llm(llm_provider, model_name, **llm_kwargs)

        for agent_name, agent_config in agents_config.items():
            agent = Agent(
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                tools=self.tools if agent_config.get('tools') else [],
                llm=llm,
                verbose=agent_config.get('verbose', True),
                allow_delegation=agent_config.get('allow_delegation', False),
                max_iter=agent_config.get('max_iter', 5),
                max_rpm=agent_config.get('max_rpm', 10),
                memory=False  # Disable individual agent memory to avoid validation errors
            )
            agents.append(agent)
            agent_mapping[agent_name] = agent  # Map YAML key to agent object

        return agents, agent_mapping

    def _create_tasks(self, agent_mapping: dict, data_context: str, human_in_loop: bool = False) -> List[Task]:
        """Create tasks from YAML configuration using agent mapping"""
        tasks = []
        tasks_config = self.config.get('tasks', {})

        for task_name, task_config in tasks_config.items():
            # Replace placeholders in description
            description = task_config['description'].format(
                data_context=data_context
            )

            # Get agent for this task using the YAML key
            agent_key = task_config['agent']
            agent = agent_mapping.get(agent_key)

            if not agent:
                st.warning(f"Agent not found for task {task_name}")
                continue

            # Get context tasks
            context_tasks = []
            if task_config.get('context'):
                for context_task_name in task_config['context']:
                    for existing_task in tasks:
                        if context_task_name in existing_task.description or hasattr(existing_task, 'name') and existing_task.name == context_task_name:
                            context_tasks.append(existing_task)
                            break

            # Override human_input based on configuration
            task_human_input = task_config.get('human_input', False) and human_in_loop

            task = Task(
                description=description,
                expected_output=task_config['expected_output'],
                agent=agent,
                tools=self.tools if task_config.get('tools') else [],
                async_execution=task_config.get('async_execution', False),
                output_file=task_config.get('output_file'),
                human_input=task_human_input,  # Use configuration override
                context=context_tasks
            )
            tasks.append(task)

        return tasks

    def _step_callback(self, step_output):
        """Callback for step-by-step execution monitoring"""
        # Update UI if in Streamlit context
        if hasattr(st, 'session_state'):
            st.write(f"🔄 Step completed: {step_output}")

    def create_crew(self, llm_provider: str, model_name: str, data_context: str, human_in_loop: bool = False, **llm_kwargs) -> Crew:
        """Create the complete crew with hierarchical process"""
        agents, agent_mapping = self._create_agents(llm_provider, model_name, **llm_kwargs)
        tasks = self._create_tasks(agent_mapping, data_context, human_in_loop)

        crew_config = self.config.get('crew', {}).get('crew_config', {})

        # Setup manager LLM for hierarchical process
        manager_llm_config = crew_config.get('manager_llm', {})
        # Remove 'model' from kwargs to avoid conflict
        manager_kwargs = {k: v for k, v in manager_llm_config.items() if k != 'model'}
        # Merge manager-specific kwargs with LLM kwargs (manager_kwargs takes precedence)
        merged_kwargs = {**llm_kwargs, **manager_kwargs}
        manager_llm = self._create_llm(
            llm_provider,
            manager_llm_config.get('model', model_name),
            **merged_kwargs
        )

        # Create crew with hierarchical process
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical if crew_config.get('process') == 'hierarchical' else Process.sequential,
            memory=False,  # Disable memory to avoid validation error
            cache=crew_config.get('cache', True),
            max_rpm=crew_config.get('max_rpm', 100),
            manager_llm=manager_llm,
            planning=crew_config.get('planning', True),
            verbose=crew_config.get('verbose', True)
        )

        return crew

    def get_data_context(self, data_path: str) -> str:
        """Generate comprehensive data context"""
        try:
            df = pd.read_csv(data_path)

            context = f"""
            Dataset Analysis Context:

            📊 Basic Information:
            - Total Records: {len(df):,}
            - Total Columns: {len(df.columns)}
            - Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB
            - Data Types: {dict(df.dtypes.value_counts())}

            📈 Data Quality Metrics:
            - Missing Values: {df.isnull().sum().sum():,} ({(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%)
            - Duplicate Rows: {df.duplicated().sum():,}
            - Complete Records: {len(df.dropna()):,}

            🔍 Column Details:
            """

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    context += f"""
                    {col} (Numeric):
                    - Range: {df[col].min():.2f} to {df[col].max():.2f}
                    - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}
                    - Std Dev: {df[col].std():.2f}
                    - Missing: {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df)*100):.1f}%)
                    """
                else:
                    context += f"""
                    {col} (Categorical):
                    - Unique Values: {df[col].nunique():,}
                    - Most Common: {df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A'}
                    - Missing: {df[col].isnull().sum()} ({(df[col].isnull().sum()/len(df)*100):.1f}%)
                    """

            return context

        except Exception as e:
            return f"Error generating data context: {str(e)}"

# ============================================================================
# FRONTEND: STREAMLIT USER INTERFACE
# ============================================================================

def render_crew_interface():
    """
    Render the main Streamlit user interface for Data Analysis Crew

    Provides:
    - LLM provider and model selection
    - Data analysis input form
    - Advanced analysis configuration options
    - Results display with multiple tabs
    """
    st.header("📊 Data Analysis Crew")

    # ========================================
    # UI SECTION: Sidebar Configuration Panel
    # ========================================
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            [ "Gemini", "Ollama", "Groq", "Anthropic", "OpenAI"],
            key='data_llm_provider',
            help="Choose your preferred AI model provider"
        )

        model_options = {
            # Ollama: Free Open Models, runs on your local system (no API key required)
            "Ollama": ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "gemma2:2b", "gemma2:9b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "codestral:22b", "deepseek-coder:1.3b"],
            # Gemini: Google's Gemini models (requires API key)
            "Gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
            # Groq: Open Models (requires API key)
            "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
            # Anthropic: Claude AI models, good at reasoning & Coding (requires API key)
            "Anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"],
            # OpenAI: ChatGPT and GPT models, good at reasoning(requires API key)
            "OpenAI": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
        }

        # Model Selection based on provider
        model_name = st.selectbox(
            "Model",
            model_options[llm_provider],
            key='data_model_name',
            help="Select the specific model variant"
        )

        # Ollama-specific configuration
        ollama_base_url = None
        if llm_provider == "Ollama":
            st.markdown("**🔧 Ollama Configuration**")
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                key='data_ollama_url',
                help="URL where Ollama server is running"
            )

            # Check Ollama status
            try:
                import requests
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    st.success("✅ Ollama server is running")
                else:
                    st.error("❌ Ollama server not accessible")
            except Exception as e:
                st.error("❌ Cannot connect to Ollama server")
                st.markdown("**Setup Instructions:**")
                st.code(f"1. Install Ollama from https://ollama.com\n2. Run: ollama serve\n3. Pull model: ollama pull {model_name}")

        # Crew Process Configuration
        process_type = st.selectbox(
            "Crew Process",
            ["Sequential", "Hierarchical"],
            key='data_process',
            help="Sequential: agents work one after another. Hierarchical: manager coordinates agents."
        )

        # Configuration help
        if st.button("ℹ️ YAML Config Help"):
            display_data_analysis_yaml_configuration_help()

    # ========================================
    # UI SECTION: Main Analysis Interface
    # ========================================
    # Advanced Configuration Section
    with st.expander("🔧 Advanced Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎛️ Model Settings**")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                key='data_temperature',
                help="Controls response variability"
            )

        with col2:
            st.markdown("**📊 Analysis Settings**")
            analysis_depth = st.selectbox(
                "Analysis Depth",
                ["Basic", "Standard", "Comprehensive"],
                index=1,
                key='analysis_depth',
                help="Level of detail in analysis"
            )

            enable_visualization = st.checkbox(
                "Generate Visualizations",
                value=True,
                key='enable_viz',
                help="Generate charts and graphs in analysis"
            )

            include_recommendations = st.checkbox(
                "Include Recommendations",
                value=True,
                key='include_rec',
                help="Generate actionable business recommendations"
            )

        # Add human-in-the-loop configuration
        st.markdown("**🤝 Human Interaction**")
        human_in_loop = st.checkbox(
            "Human-in-the-Loop",
            value=False,
            key='data_human_in_loop',
            help="Enable human review and approval at key stages"
        )

    # Configuration display
    with st.expander("⚙️ Crew Configuration & Features", expanded=False):
        st.markdown("""
        **Data Analysis Crew Features:**

        🧠 **Intelligent Data Processing**
        - Automated data quality assessment
        - Statistical analysis and modeling
        - Pattern recognition and anomaly detection

        📊 **Advanced Analytics**
        - Comprehensive statistical summaries
        - Correlation analysis and insights
        - Predictive modeling capabilities

        📈 **Visualization & Reporting**
        - Interactive charts and graphs
        - Executive summary generation
        - Actionable business recommendations

        🔧 **Technical Features**
        - YAML-based configuration for flexibility
        - Multi-agent collaboration workflow
        - Error handling and data validation
        """)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=['csv'],
        key='data_file_upload'
    )

    if uploaded_file:
        # Initialize the crew
        crew_manager = DataAnalysisCrew()

        # Display file info
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

        # Show data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(), width="stretch")

        # Analysis execution
        if st.button("Start Analysis", type="primary", key='start_data_analysis'):
            try:
                with st.spinner("🔄 Initializing advanced crew..."):
                    # Setup proper file management with input folder
                    input_dir = Path(__file__).parent / "input"
                    input_dir.mkdir(exist_ok=True)

                    # Save uploaded file in the input directory
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    temp_filename = f"data_analysis_{timestamp}.csv"
                    temp_path = input_dir / temp_filename
                    df.to_csv(temp_path, index=False)

                    # Clean up old files (keep only last 5 files)
                    cleanup_old_files(input_dir)

                    # Get data context
                    data_context = crew_manager.get_data_context(temp_path)

                    # Create and execute crew with advanced configuration
                    llm_kwargs = {
                        'temperature': temperature
                    }
                    if llm_provider == "Ollama" and ollama_base_url:
                        llm_kwargs['base_url'] = ollama_base_url

                    # Add analysis preferences to data context
                    enhanced_data_context = f"{data_context}\n\nAnalysis Configuration:\n- Depth: {analysis_depth}\n- Include Visualizations: {enable_visualization}\n- Include Recommendations: {include_recommendations}"

                    crew = crew_manager.create_crew(llm_provider, model_name, enhanced_data_context, human_in_loop, **llm_kwargs)

                with st.spinner("🔍 Executing hierarchical analysis workflow..."):
                    # Execute crew analysis
                    result = crew.kickoff()

                # Display results
                st.success("✅ Analysis completed successfully!")

                # Results tabs
                tabs = st.tabs([
                    "📋 Executive Summary",
                    "📊 Detailed Analysis",
                    "💡 Business Insights",
                    "📈 Visualizations",
                ])

                with tabs[0]:
                    st.markdown("### Executive Summary")
                    st.markdown(str(result))

                with tabs[1]:
                    st.markdown("### Detailed Analysis Results")
                    if hasattr(result, 'tasks_outputs'):
                        for i, task_output in enumerate(result.tasks_outputs):
                            st.markdown(f"#### Task {i+1}: {task_output.agent}")
                            st.markdown(task_output.raw)

                with tabs[2]:
                    st.markdown("### Business Insights")
                    business_insights = extract_business_insights_from_result(result, str(temp_path))
                    if business_insights:
                        display_business_insights(business_insights)
                    else:
                        st.info("Generating business insights... Please check the detailed analysis tab for comprehensive results.")

                with tabs[3]:
                    st.markdown("### Data Visualizations")

                    # Generate and display real charts
                    try:
                        display_real_charts(df)
                    except Exception as e:
                        st.error(f"Error generating charts: {str(e)}")

                        # Fallback to text recommendations
                        viz_recommendations = extract_visualization_recommendations_from_result(result, str(temp_path))
                        if viz_recommendations:
                            display_visualization_recommendations(viz_recommendations)
                        else:
                            st.info("Chart generation failed. Please check your data format.")

                # Download options
                st.markdown("### 📥 Download Reports")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button(
                        "Download Full Report",
                        str(result),
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

                with col2:
                    st.download_button(
                        "Download JSON",
                        json.dumps({"result": str(result), "timestamp": datetime.now().isoformat()}, indent=2),
                        file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                # Cleanup temporary file after analysis
                if temp_path.exists():
                    temp_path.unlink()

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

# ============================================================================
# BACKEND: UTILITY FUNCTIONS FOR FILE MANAGEMENT AND RESULT PROCESSING
# ============================================================================

def cleanup_old_files(input_dir: Path, max_files: int = 5):
    """
    Clean up old data files, keeping only the most recent ones

    Args:
        input_dir (Path): Directory containing data files
        max_files (int): Maximum number of files to keep
    """
    try:
        # Get all CSV files in the input directory
        csv_files = list(input_dir.glob("data_analysis_*.csv"))

        if len(csv_files) > max_files:
            # Sort by modification time (newest first)
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove older files
            for old_file in csv_files[max_files:]:
                old_file.unlink()

    except Exception as e:
        st.warning(f"Could not clean up old files: {str(e)}")

def extract_business_insights_from_result(result, data_path: str) -> Dict[str, Any]:
    """
    Extract business insights from crew result or generate them directly

    Args:
        result: CrewAI result object
        data_path (str): Path to the data file

    Returns:
        Dict[str, Any]: Business insights data
    """
    try:
        # Try to extract from result first
        result_text = str(result).lower()

        # If the result contains business insights, try to parse them
        if 'business' in result_text or 'insight' in result_text or 'recommendation' in result_text:
            # Look for structured insights in the result
            if hasattr(result, 'tasks_outputs'):
                for task_output in result.tasks_outputs:
                    if hasattr(task_output, 'raw') and ('business' in task_output.raw.lower() or 'insight' in task_output.raw.lower()):
                        # Try to parse JSON if available
                        try:
                            import re
                            json_match = re.search(r'\{.*\}', task_output.raw, re.DOTALL)
                            if json_match:
                                return json.loads(json_match.group())
                        except:
                            pass

        # If no insights found in result, generate them directly
        insights_tool = BusinessInsightsTool()
        insights_result = insights_tool._run(data_path, "comprehensive", {})

        # Parse the JSON result
        try:
            return json.loads(insights_result)
        except json.JSONDecodeError:
            return {"error": "Could not parse business insights", "raw_result": insights_result}

    except Exception as e:
        return {"error": f"Error extracting business insights: {str(e)}"}

def extract_visualization_recommendations_from_result(result, data_path: str) -> Dict[str, Any]:
    """
    Extract visualization recommendations from crew result or generate them directly

    Args:
        result: CrewAI result object
        data_path (str): Path to the data file

    Returns:
        Dict[str, Any]: Visualization recommendations data
    """
    try:
        # Try to extract from result first
        result_text = str(result).lower()

        # If the result contains visualization info, try to parse them
        if 'visualization' in result_text or 'chart' in result_text or 'graph' in result_text:
            # Look for structured viz recommendations in the result
            if hasattr(result, 'tasks_outputs'):
                for task_output in result.tasks_outputs:
                    if hasattr(task_output, 'raw') and ('visualization' in task_output.raw.lower() or 'chart' in task_output.raw.lower()):
                        # Try to parse JSON if available
                        try:
                            import re
                            json_match = re.search(r'\{.*\}', task_output.raw, re.DOTALL)
                            if json_match:
                                return json.loads(json_match.group())
                        except:
                            pass

        # If no visualizations found in result, generate them directly
        viz_tool = DataVisualizationTool()
        viz_result = viz_tool._run(data_path, "comprehensive", {})

        # Parse the JSON result
        try:
            return json.loads(viz_result)
        except json.JSONDecodeError:
            return {"error": "Could not parse visualization recommendations", "raw_result": viz_result}

    except Exception as e:
        return {"error": f"Error extracting visualization recommendations: {str(e)}"}

def display_business_insights(insights: Dict[str, Any]):
    """
    Display business insights in a structured format

    Args:
        insights (Dict[str, Any]): Business insights data
    """
    if "error" in insights:
        st.error(f"Error generating insights: {insights['error']}")
        return

    # Executive Summary
    if "executive_summary" in insights:
        st.markdown("#### 📊 Executive Summary")
        st.info(insights["executive_summary"])

    # Key Insights
    if "key_insights" in insights and insights["key_insights"]:
        st.markdown("#### 🔍 Key Insights")
        for i, insight in enumerate(insights["key_insights"], 1):
            st.markdown(f"**{i}.** {insight}")

    # Business Recommendations
    if "business_recommendations" in insights and insights["business_recommendations"]:
        st.markdown("#### 💡 Strategic Recommendations")
        for i, rec in enumerate(insights["business_recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")

    # Risk Factors
    if "risk_factors" in insights and insights["risk_factors"]:
        st.markdown("#### ⚠️ Risk Factors")
        for i, risk in enumerate(insights["risk_factors"], 1):
            st.markdown(f"**{i}.** {risk}")

    # Growth Opportunities
    if "growth_opportunities" in insights and insights["growth_opportunities"]:
        st.markdown("#### 🚀 Growth Opportunities")
        for i, opp in enumerate(insights["growth_opportunities"], 1):
            st.markdown(f"**{i}.** {opp}")

    # Performance Metrics
    if "performance_metrics" in insights and insights["performance_metrics"]:
        st.markdown("#### 📈 Performance Metrics")
        metrics = insights["performance_metrics"]

        # Display as columns for key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            if "data_completeness" in metrics:
                st.metric("Data Completeness", f"{metrics['data_completeness']:.1f}%")

        with col2:
            if "correlation_strength" in metrics:
                st.metric("Avg Correlation", f"{metrics['correlation_strength']:.2f}")

        with col3:
            # Show first business metric found
            for key, value in metrics.items():
                if key not in ["data_completeness", "correlation_strength", "average_numeric_variance"]:
                    st.metric(key.replace('_', ' ').title(), f"{value:.1f}" if isinstance(value, float) else str(value))
                    break

def display_visualization_recommendations(viz_data: Dict[str, Any]):
    """
    Display visualization recommendations in a structured format

    Args:
        viz_data (Dict[str, Any]): Visualization recommendations data
    """
    if "error" in viz_data:
        st.error(f"Error generating visualizations: {viz_data['error']}")
        return

    # Recommended Charts
    if "recommended_charts" in viz_data and viz_data["recommended_charts"]:
        st.markdown("#### 📊 Recommended Charts")

        # Group charts by type
        chart_types = {}
        for chart in viz_data["recommended_charts"]:
            chart_type = chart.get("chart_type", "unknown")
            if chart_type not in chart_types:
                chart_types[chart_type] = []
            chart_types[chart_type].append(chart)

        # Display by chart type
        for chart_type, charts in chart_types.items():
            with st.expander(f"📈 {chart_type.replace('_', ' ').title()}", expanded=False):
                for chart in charts:
                    st.markdown(f"**{chart.get('title', 'Chart')}**")
                    st.markdown(f"*{chart.get('description', 'No description')}*")
                    st.markdown(f"**Business Value:** {chart.get('business_value', 'General analysis')}")
                    if chart.get('data_column'):
                        st.markdown(f"**Data Column:** `{chart['data_column']}`")
                    st.markdown("---")

    # Dashboard Layout
    if "business_dashboards" in viz_data:
        dashboard = viz_data["business_dashboards"]
        st.markdown("#### 📊 Dashboard Layout")

        # KPI Cards
        if "kpi_cards" in dashboard and dashboard["kpi_cards"]:
            st.markdown("**Key Performance Indicators:**")
            kpi_cols = st.columns(min(3, len(dashboard["kpi_cards"])))
            for i, kpi in enumerate(dashboard["kpi_cards"][:3]):  # Show first 3 KPIs
                with kpi_cols[i % 3]:
                    value = kpi.get("value", 0)
                    if kpi.get("format") == "currency":
                        st.metric(kpi.get("title", "KPI"), f"${value:,.2f}")
                    else:
                        st.metric(kpi.get("title", "KPI"), f"{value:,.0f}")

        # Dashboard Sections
        if "sections" in dashboard and dashboard["sections"]:
            st.markdown("**Dashboard Sections:**")
            for section in dashboard["sections"]:
                st.markdown(f"- **{section.get('title', 'Section')}:** {', '.join(section.get('charts', []))}")

    # Trend Analysis
    if "trend_analysis" in viz_data:
        trends = viz_data["trend_analysis"]
        if "linear_trends" in trends and trends["linear_trends"]:
            st.markdown("#### 📈 Trend Analysis")
            for trend in trends["linear_trends"]:
                direction_emoji = "📈" if trend["direction"] == "increasing" else "📉"
                st.markdown(f"{direction_emoji} **{trend['column']}** - {trend['direction']} trend ({trend['strength']} strength)")

    # Interactive Features
    if "interactive_features" in viz_data and viz_data["interactive_features"]:
        st.markdown("#### 🎛️ Recommended Interactive Features")
        for feature in viz_data["interactive_features"]:
            st.markdown(f"**{feature.get('feature', 'Feature').replace('_', ' ').title()}:** {feature.get('description', 'No description')}")

def display_real_charts(df: pd.DataFrame):
    """
    Display real interactive charts using the uploaded data

    Args:
        df (pd.DataFrame): The dataset to visualize
    """
    # Create KPI Dashboard first
    st.markdown("#### 📊 Key Performance Indicators")
    kpis = create_kpi_dashboard(df)

    # Display top KPIs in columns
    kpi_keys = list(kpis.keys())
    if len(kpi_keys) >= 3:
        col1, col2, col3 = st.columns(3)

        with col1:
            key = kpi_keys[0]
            value = kpis[key]
            if isinstance(value, (int, float)):
                st.metric(key, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
            else:
                st.metric(key, str(value))

        with col2:
            key = kpi_keys[1]
            value = kpis[key]
            if isinstance(value, (int, float)):
                st.metric(key, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
            else:
                st.metric(key, str(value))

        with col3:
            key = kpi_keys[2]
            value = kpis[key]
            if isinstance(value, (int, float)):
                st.metric(key, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
            else:
                st.metric(key, str(value))

    st.markdown("---")

    # Distribution Charts
    st.markdown("#### 📈 Distribution Analysis")
    distribution_charts = generate_distribution_charts(df)
    if distribution_charts:
        # Display in 2 columns
        if len(distribution_charts) >= 2:
            col1, col2 = st.columns(2)
            for i, chart in enumerate(distribution_charts[:4]):  # Show up to 4 charts
                with col1 if i % 2 == 0 else col2:
                    st.plotly_chart(chart, use_container_width=True)
        else:
            for chart in distribution_charts:
                st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No numeric columns found for distribution analysis")

    st.markdown("---")

    # Correlation Analysis
    st.markdown("#### 🔗 Correlation Analysis")
    corr_chart = generate_correlation_heatmap(df)
    if corr_chart:
        st.plotly_chart(corr_chart, use_container_width=True)

        # Scatter plots for strong correlations
        scatter_charts = generate_scatter_plots(df)
        if scatter_charts:
            st.markdown("**Strong Correlations:**")
            for i, chart in enumerate(scatter_charts):
                st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for correlation analysis")

    st.markdown("---")

    # Categorical Analysis
    st.markdown("#### 📊 Categorical Analysis")
    categorical_charts = generate_categorical_charts(df)
    if categorical_charts:
        for chart in categorical_charts:
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No suitable categorical columns found for analysis")

    st.markdown("---")

    # Outlier Detection
    st.markdown("#### 🎯 Outlier Detection")
    box_charts = generate_box_plots(df)
    if box_charts:
        # Display in 2 columns
        if len(box_charts) >= 2:
            col1, col2 = st.columns(2)
            for i, chart in enumerate(box_charts[:4]):  # Show up to 4 charts
                with col1 if i % 2 == 0 else col2:
                    st.plotly_chart(chart, use_container_width=True)
        else:
            for chart in box_charts:
                st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No numeric columns found for outlier detection")

    st.markdown("---")

    # Time Series Analysis (if applicable)
    st.markdown("#### ⏰ Time Series Analysis")
    time_series_charts = generate_time_series_charts(df)
    if time_series_charts:
        for chart in time_series_charts:
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No date/time columns detected for time series analysis")

    st.markdown("---")

    # Summary Statistics
    st.markdown("#### 📋 Summary Statistics Overview")
    summary_chart = generate_summary_statistics_chart(df)
    if summary_chart:
        st.plotly_chart(summary_chart, use_container_width=True)
    else:
        st.info("No numeric data available for summary statistics")

    # Additional Insights
    st.markdown("#### 💡 Chart Insights")
    st.info("""
    **How to interpret these charts:**
    - **Distribution Charts**: Show how your data is spread - look for normal distributions, skewness, or multiple peaks
    - **Correlation Matrix**: Red = negative correlation, Blue = positive correlation, White = no correlation
    - **Scatter Plots**: Points along a line indicate strong correlation, scattered points indicate weak correlation
    - **Box Plots**: The box shows the middle 50% of data, dots outside are outliers
    - **Bar Charts**: Compare frequencies or counts across categories
    - **Time Series**: Track changes over time - look for trends, seasonality, or sudden changes
    """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="CrewAI - Advanced Data Analysis",
        page_icon="📊",
        layout="wide"
    )

    render_crew_interface()