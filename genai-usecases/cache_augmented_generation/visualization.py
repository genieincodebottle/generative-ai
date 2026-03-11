import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from models import TestResults

def display_metrics(results: TestResults):
    """Display metrics in the Streamlit UI."""
    cols = st.columns(4)
    metrics = [
        ("Average Similarity", results.avg_similarity, ""),
        ("Average Cache Time", results.avg_cache_time, "s"),
        ("Average Generation Time", results.avg_generate_time, "s"),
        ("Preparation Time", results.prepare_time, "s")
    ]

    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.metric(label, f"{value:.4f}{unit}")

def create_results_dataframe(results: TestResults) -> pd.DataFrame:
    """Create a DataFrame from test results."""
    return pd.DataFrame({
        'Timestamp': results.timestamps,
        'Question': results.prompts,
        'Response': results.responses,
        'Ground Truth': results.ground_truths,
        'Similarity': results.similarity,
        'Cache Time (s)': results.cache_time,
        'Generation Time (s)': results.generate_time
    })

def plot_performance_metrics(df: pd.DataFrame):
    """Create performance metrics plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Cache Time (s)'],
        name='Cache Time',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Generation Time (s)'],
        name='Generation Time',
        mode='lines+markers'
    ))
    fig.update_layout(
        title='Performance Metrics Over Time',
        xaxis_title='Question Index',
        yaxis_title='Time (seconds)'
    )
    return fig