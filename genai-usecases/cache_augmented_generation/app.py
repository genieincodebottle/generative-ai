import os
import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image

from models import CAGModel, TestResults
from visualization import display_metrics, create_results_dataframe, plot_performance_metrics

def setup_environment():
    """Initialize environment variables."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise Exception("HF_TOKEN is not set in .env file")
    return token

def render_help_section():
    """Render help documentation."""
    with st.expander("💡 Need Help Getting Started?"):
        st.markdown("""
        ### Initial Setup
        1. Rename `.env.example` to `.env`
        2. Get your Hugging Face token from [Hugging Face Tokens Page](https://huggingface.co/settings/tokens)
        3. Copy the token to `HF_TOKEN` in your .env file
        
        ### Key Features
        * Cache-Augmented Generation (CAG)
        * Model Management
        * Data Processing Pipeline
        
        ### Troubleshooting
        * Check HF_TOKEN setup
        * Verify internet connection
        * Clear cache if needed
        """)

def render_workflow_diagram():
    """Render system workflow diagram."""
    with st.expander("📖 Overview & System Workflow", expanded=False):
        st.markdown("""
        #### Overview
                    
        Cache Augmented Generation (CAG) is an new paradigm for enhancing language model performance 
        that eliminates the need for real-time retrieval by preloading relevant documents into the model's 
        extended context and precomputing key-value (KV) caches. Unlike traditional Retrieval Augmented Generation (RAG), 
        CAG streamlines the system architecture by encoding documents into a KV cache that captures the model's inference 
        state and storing it for reuse, enabling retrieval-free question answering.
        
        The key characteristics that define CAG include:

        * Preloading and caching of context rather than real-time retrieval
        * Use of precomputed KV caches that store the model's inference state
        * A simplified architecture that removes the separate retrieval pipeline
        * A cache reset mechanism that truncates appended tokens to maintain performance

        This approach offers significant advantages over traditional RAG systems, including reduced inference time 
        (from 94.35 seconds to 2.33 seconds on large HotPotQA datasets), improved accuracy (achieving higher BERT scores), 
        and simplified system architecture. CAG is particularly effective when working with manageable knowledge bases 
        that can fit within the model's extended context window and in scenarios where low latency is critical.
        
        #### Advantages
        * Reduced Latency
        * Improved Reliability
        * Simplified Design
        
        #### Limitations
        * Knowledge Size Limits
        * Context Length Issues
                    
        #### References
        * [Research Paper](https://arxiv.org/abs/2412.15605)
        * https://github.com/hhhuang/CAG
        """)
        
        image_path = Path(__file__).parent / 'images'
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(image_path / 'cag_diagram.png'), 'High Level Architecture')
        with col2:
            st.image(Image.open(image_path / 'cag_sequence_diagram.png'), 'Sequence Diagram')

def initialize_model(model_name: str, quantized: bool) -> CAGModel:
    """Initialize the model with given configuration."""
    token = setup_environment()
    model = CAGModel(token)
    return model if model.load_model(model_name, quantized) else None

def process_cache(model: CAGModel, documents: str) -> tuple:
    """Process and save KV cache."""
    cache_progress = st.progress(0, text="Preparing KV Cache...")
    try:
        for i in range(100):
            cache_progress.progress(i/100)
            if i == 50:
                cache, prep_time = model.prepare_cache(documents)
                cache_path = Path("./data_cache/cache_knowledges.pt")
                torch.save(cache, cache_path)
                st.session_state.prep_time = prep_time
        cache_progress.progress(1.0, text="KV Cache Ready!")
        return True, prep_time
    except Exception as e:
        return False, str(e)

def process_questions(model: CAGModel, dataset: list, documents: str, use_cache: bool) -> TestResults:
    """Process questions and return results."""
    prep_time = st.session_state.get('prep_time', 0.0) if use_cache else 0.0
    
    def update_progress(progress):
        st.session_state.progress.progress(progress)
    
    st.session_state.progress = st.progress(0, text="Processing questions...")
    results = model.process_questions(dataset, documents, use_cache, update_progress)
    results.prepare_time = prep_time
    st.session_state.progress.empty()
    
    return results

def main():
    """Main application."""
    st.set_page_config(page_title="CAG", page_icon="🗂️", layout="wide")
    st.subheader("🗂️ Cache-Augmented Generation (CAG)")
    
    render_workflow_diagram()
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Model Configuration")
        model_name = st.selectbox(
                        "Model Name",
                        ["meta-llama/Llama-3.2-1B-Instruct"],
                        key='model_name'
                     )
        use_cache = st.checkbox("Use CAG (KV Cache)", value=True)
        
        quantized = False
        if torch.cuda.is_available():
            quantized = st.checkbox("Enable Quantization", value=True)
            st.markdown("✅ GPU Available", unsafe_allow_html=True)
        else:
            st.markdown("❌ Running on CPU", unsafe_allow_html=True)
            
        initialize_button = st.button("Initialize Model", type="primary")
        render_help_section()
    
    # Initialize model
    if initialize_button:
        with st.spinner("Initializing model..."):
            model = initialize_model(model_name, quantized)
            if model:
                st.session_state.model = model
                st.success("✅ Model initialized!")
            else:
                st.error("❌ Model initialization failed")
                return
    
    if 'model' not in st.session_state:
        st.info("👈 Please initialize the model")
        return
    
    # Load and process data
    try:
        # Load dataset
        df = pd.read_csv("./datasets/sample_qa_dataset.csv")
        with st.expander("📄 Preview Data"):
            st.dataframe(df.head(10))
        
        # Configure processing
        max_questions = st.number_input("Questions to Process", min_value=1, value=5)
        dataset = list(zip(df['sample_question'].iloc[:max_questions],
                         df['sample_ground_truth'].iloc[:max_questions]))
        documents = '\n\n'.join(df["text"].tolist())
        
        # Processing controls
        col1, col2 = st.columns(2)
        with col1:
            prepare_cache = st.button("1️⃣ Prepare Cache", disabled=not use_cache)
        with col2:
            process_button = st.button("2️⃣ Process Questions")
        
        # Prepare cache if requested
        if prepare_cache and use_cache:
            with st.spinner("Preparing cache..."):
                success, result = process_cache(st.session_state.model, documents)
                if success:
                    st.success(f"✅ Cache prepared in {result:.2f} seconds")
                else:
                    st.error(f"❌ Cache preparation failed: {result}")
        
        # Process questions
        if process_button:
            try:
                results = process_questions(
                    st.session_state.model,
                    dataset,
                    documents,
                    use_cache
                )
                
                # Display results
                display_metrics(results)
                df_results = create_results_dataframe(results)
                
                tab1, tab2 = st.tabs(["📊 Results Table", "📈 Performance Metrics"])
                with tab1:
                    st.dataframe(df_results, use_container_width=True)
                with tab2:
                    st.plotly_chart(plot_performance_metrics(df_results), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error processing questions: {str(e)}")
                st.exception(e)
                
    except Exception as e:
        st.error(f"❌ Error loading or processing data: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()