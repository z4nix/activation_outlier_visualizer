# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import pickle

def load_stats_file(file):
    """Load processed statistics from an uploaded pickle file"""
    return pickle.load(file)

def get_readable_name(metric):
    """Convert metric name to readable format"""
    name = metric.replace('_std_per_channel', '')
    name = name.replace('_recorder', '')
    name = name.replace('_', ' ')
    return name.title()

def create_visualization(processed_data, component, metric, std_threshold, subsample_factor):
    """Create the 3D visualization"""
    metric_data = processed_data[component][metric]
    data = metric_data['data']
    valid_layers = metric_data['valid_layers']

    # Subsample data
    output_size = data.shape[1]
    x_data = np.arange(0, output_size, subsample_factor)
    z_data = data[:, ::subsample_factor]

    # Create meshgrid
    X, Y = np.meshgrid(x_data, valid_layers)

    # Create the 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=z_data,
            colorscale=[
                [0, 'rgb(220,220,220)'],      # Light gray for low values
                [0.5, 'rgb(65,105,225)'],     # Royal blue for medium values
                [1, 'rgb(139,0,0)']           # Dark red for high values
            ],
            cmin=0,
            cmax=std_threshold,
            colorbar=dict(
                title='σ',
                titleside='right',
                ticksuffix='σ'
            )
        )
    ])

    # Update layout
    metric_name = get_readable_name(metric)
    fig.update_layout(
        title=dict(
            text=f'{metric_name} Standard Deviation',
            x=0.5,
            y=0.95,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Channel Index',
            yaxis_title='Layer Index',
            zaxis_title='Standard Deviation (σ)',
            camera=dict(
                eye=dict(x=2, y=2, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            aspectratio=dict(x=1.5, y=1, z=0.7)
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

def main():
    st.set_page_config(page_title="Model Statistics Visualizer", layout="wide")
    
    st.title("Model Statistics Visualizer")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a statistics file", type=['pkl'])
    
    if uploaded_file is not None:
        # Load the statistics
        processed_data = load_stats_file(uploaded_file)
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Component selection
        component = st.sidebar.selectbox(
            "Select Component",
            options=list(processed_data.keys())
        )
        
        # Metric selection
        metrics = list(processed_data[component].keys())
        metric = st.sidebar.selectbox(
            "Select Metric",
            options=metrics,
            format_func=get_readable_name
        )
        
        # Standard deviation threshold
        std_threshold = st.sidebar.slider(
            "Standard Deviation Threshold (σ)",
            min_value=0.0,
            max_value=12.0,
            value=6.0,
            step=0.1
        )
        
        # Subsample factor
        subsample_factor = st.sidebar.slider(
            "Subsample Factor",
            min_value=1,
            max_value=32,
            value=8,
            step=1
        )
        
        # Create and display visualization
        fig = create_visualization(
            processed_data,
            component,
            metric,
            std_threshold,
            subsample_factor
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
