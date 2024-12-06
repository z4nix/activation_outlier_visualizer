import streamlit as st
import os

# Override any external config by setting environment variables
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_THEME_BASE'] = 'light'

def main():
    # Configure the page
    st.set_page_config(
        page_title="Test App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Test App")
    st.write("Hello World!")

if __name__ == "__main__":
    main()
