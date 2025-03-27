# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pickle
import os
from pathlib import Path

st.set_page_config(page_title="Model Statistics Visualizer", layout="wide")

def get_example_files():
    """Get list of example pickle files from the examples directory"""
    examples_dir = Path("example")
    if not examples_dir.exists():
        return []
    return sorted([f for f in examples_dir.glob("*.pkl")])

def load_stats_file(file_path):
    """Load processed statistics from a pickle file path or uploaded file"""
    try:
        if isinstance(file_path, (str, Path)):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = pickle.load(file_path)
        
        # Ensure all data is in numpy array format
        for component in data:
            for metric in data[component]:
                if 'data' in data[component][metric]:
                    metric_data = data[component][metric]['data']
                    
                    # If it's a list, try to convert to numpy array
                    if isinstance(metric_data, list):
                        try:
                            # Convert the list to a numpy array
                            data[component][metric]['data'] = np.array(metric_data)
                        except Exception as e:
                            st.warning(f"Couldn't convert {component}.{metric} to numpy array: {str(e)}")
                            # If conversion fails, ensure each element is at least a numpy array
                            for i, item in enumerate(metric_data):
                                if isinstance(item, list):
                                    metric_data[i] = np.array(item)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_readable_name(metric):
    """Convert metric name to readable format"""
    name = metric.replace('_std_per_channel', '')
    name = name.replace('_recorder', '')
    name = name.replace('_', ' ')
    return name.title()

def create_visualization(processed_data, component, metric, std_threshold, subsample_factor):
    """Create the 3D visualization"""
    try:
        if component not in processed_data or metric not in processed_data[component]:
            st.error(f"Component '{component}' or metric '{metric}' not found in the data")
            return None
            
        metric_data = processed_data[component][metric]
        if 'data' not in metric_data or 'valid_layers' not in metric_data:
            st.error(f"Invalid data structure for {component}.{metric}")
            return None
            
        data = metric_data['data']
        valid_layers = metric_data['valid_layers']
        
        # Convert to numpy array if still a list
        if isinstance(data, list):
            try:
                data = np.array(data)
            except Exception as e:
                st.error(f"Cannot convert data to numpy array: {e}")
                return None
        
        # Debug information
        st.info(f"Data shape: {data.shape}, Valid layers: {len(valid_layers)}")
        
        # Make sure data is 2D (layers x channels)
        if len(data.shape) != 2:
            st.error(f"Data shape is {data.shape}, expected 2D array (layers x channels)")
            return None
        
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
                    title=dict(
                        text='σ',
                        side='right'
                    ),
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
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.error(f"Error details: {type(e).__name__} at line {e.__traceback__.tb_lineno}")
        return None

def main():
    st.title("Model Statistics Visualizer")
    
    # Add description
    st.write("""
    This tool visualizes neural network model statistics in 3D. 
    Either select an example file from the examples directory or upload your own statistics file.
    """)
    
    # Get example files
    example_files = get_example_files()
    
    # Create two columns for file selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Example File")
        if example_files:
            selected_example = st.selectbox(
                "Choose an example file",
                options=example_files,
                format_func=lambda x: x.name
            )
        else:
            st.warning("No example files found in the 'examples' directory")
            selected_example = None
    
    with col2:
        st.subheader("Or Upload Your Own File")
        uploaded_file = st.file_uploader("Choose a statistics file", type=['pkl'])
    
    # Use either the selected example or uploaded file
    file_to_process = uploaded_file if uploaded_file is not None else selected_example
    
    if file_to_process is not None:
        try:
            # Load the statistics
            processed_data = load_stats_file(file_to_process)
            
            if processed_data is None:
                st.error("Failed to load statistics from the file")
                return
                
            # Display available components
            components = list(processed_data.keys())
            if not components:
                st.error("No components found in the data")
                return
                
            # Sidebar controls
            with st.sidebar:
                st.header("Visualization Controls")
                
                # Component selection
                component = st.selectbox(
                    "Select Component",
                    options=components
                )
                
                # Metric selection
                metrics = list(processed_data[component].keys())
                if not metrics:
                    st.error(f"No metrics found for component '{component}'")
                    return
                    
                metric = st.selectbox(
                    "Select Metric",
                    options=metrics,
                    format_func=get_readable_name
                )
                
                # Standard deviation threshold
                std_threshold = st.slider(
                    "Standard Deviation Threshold (σ)",
                    min_value=0.0,
                    max_value=12.0,
                    value=6.0,
                    step=0.1
                )
                
                # Subsample factor
                subsample_factor = st.slider(
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
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to create visualization")
        
        except Exception as e:
            st.error(f"""
            Error processing the file: {str(e)}
            
            Make sure your pickle file contains statistics in the correct format:
            ```python
            {{
                'component_name': {{
                    'metric_name': {{
                        'data': numpy.ndarray,  # Shape: [layers, channels]
                        'valid_layers': list    # List of layer indices
                    }}
                }}
            }}
            ```
            """)

if __name__ == "__main__":
    main()
