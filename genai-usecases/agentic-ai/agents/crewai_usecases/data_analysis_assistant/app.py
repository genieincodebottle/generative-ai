import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from crewai import Agent, Task, Crew, LLM
from typing import Dict, List
import tempfile
import json
import os
from datetime import datetime
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Configuration
LLM_CONFIGS = {
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini"]
    },
    "Anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", 
                  "claude-3-opus-20240229"]
    },
    "Gemini": {
        "models": ["gemini-2.0-flash-exp", "gemini-1.5-flash", 
                  "gemini-1.5-flash-8b", "gemini-1.5-pro"]
    },
    "Groq": {
        "models": ["groq/deepseek-r1-distill-llama-70b", "groq/llama3-70b-8192", "groq/llama-3.1-8b-instant", 
                  "groq/llama-3.3-70b-versatile", "groq/gemma2-9b-it", "groq/mixtral-8x7b-32768"]
    }
}

class DataAnalyzer:
    def __init__(self, llm_provider: str, model_name: str):
        self.llm_provider = llm_provider
        self.model_name = model_name

    def get_data_context(self, data_path: str) -> str:
        """Generate data context from the CSV file."""
        df = pd.read_csv(data_path)
        
        context = f"""
        Dataset Summary:
        - Total Records: {len(df)}
        - Columns: {', '.join(df.columns.tolist())}
        
        Column Information:
        """
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                context += f"""
                {col}:
                - Type: Numeric
                - Range: {df[col].min()} to {df[col].max()}
                - Average: {df[col].mean():.2f}
                - Missing Values: {df[col].isnull().sum()}
                """
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                context += f"""
                {col}:
                - Type: Date/Time
                - Range: {df[col].min()} to {df[col].max()}
                - Missing Values: {df[col].isnull().sum()}
                """
            else:
                context += f"""
                {col}:
                - Type: Categorical
                - Unique Values: {df[col].nunique()}
                - Top Values: {', '.join(df[col].value_counts().nlargest(3).index.astype(str))}
                - Missing Values: {df[col].isnull().sum()}
                """
        
        return context

    def initialize_llm(self, llm_provider: str, model_name: str) -> LLM:
        """Initialize the language model"""
        provider_keys = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Gemini": "GOOGLE_API_KEY",
            "Groq": "GROQ_API_KEY"
        }
        
        key_name = provider_keys.get(llm_provider)
        if key_name:
            os.environ[key_name] = os.getenv(key_name)
        
        return LLM(
            model=model_name,
            temperature=0.7,
            timeout=120,
            max_tokens=4000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

    def create_crew(self, data_path: str) -> Crew:
        """Create a crew of agents for data analysis."""
        
        llm = self.initialize_llm(self.llm_provider, self.model_name)
        # Get data context
        data_context = self.get_data_context(data_path)
        
        # Create agents
        data_analyst = Agent(
            role='Data Analyst',
            goal='Provide comprehensive statistical analysis with actionable insights',
            backstory="""Expert data analyst with strong statistical background and 
            business intelligence expertise. Specialized in identifying patterns and 
            deriving meaningful insights from data.""",
            llm=llm,
            verbose=True
        )
        
        insights_generator = Agent(
            role='Business Intelligence Specialist',
            goal='Generate strategic business insights and recommendations',
            backstory="""Senior business analyst with expertise in converting data 
            patterns into actionable business strategies. Experienced in multiple 
            industries and business contexts.""",
            llm=llm,
            verbose=True
        )
        
        # Create tasks with specific data context
        analysis_task = Task(
            description=f"""
            Analyze the provided dataset with the following context:
            {data_context}
            
            Provide a comprehensive analysis including:
            1. Key Statistical Findings:
               - Important trends in each numeric column
               - Significant correlations between variables
               - Notable patterns in categorical data
               - Time-based patterns if temporal data exists
            
            2. Data Distribution Analysis:
               - Distribution patterns of numeric variables
               - Frequency analysis of categorical variables
               - Identification of any anomalies or outliers
            
            3. Relationship Analysis:
               - Dependencies between variables
               - Cause-effect relationships if apparent
               - Segment analysis if applicable
            
            Format the analysis in a clear, structured manner using actual column names 
            and specific data points. Include numerical values and percentages where relevant.
            """,
            expected_output="""A detailed analysis report containing:
            1. Statistical summary with specific numbers and trends
            2. Pattern analysis with concrete examples
            3. Relationship insights with supporting data
            4. Anomaly detection with specific instances""",
            agent=data_analyst
        )
        
        insights_task = Task(
            description=f"""
            Based on the analysis and the following data context:
            {data_context}
            
            Generate actionable business insights including:
            1. Key Business Findings:
               - Critical patterns affecting business performance
               - Customer/market behavior insights
               - Operational efficiency indicators
               - Risk factors and opportunities
            
            2. Strategic Recommendations:
               - Specific, actionable suggestions
               - Priority areas for improvement
               - Growth opportunities
               - Risk mitigation strategies
            
            3. Implementation Guidelines:
               - Short-term actions
               - Long-term strategies
               - Success metrics
               - Potential challenges
            
            Use actual column names and specific data points. Include concrete examples 
            and quantifiable metrics where possible.
            """,
            expected_output="""A comprehensive insights report containing:
            1. Business insights backed by specific data points
            2. Actionable recommendations with clear rationale
            3. Implementation strategy with measurable goals
            4. Risk assessment with mitigation plans""",
            agent=insights_generator
        )
        
        # Create crew
        crew = Crew(
            agents=[data_analyst, insights_generator],
            tasks=[analysis_task, insights_task],
            verbose=True
        )
        
        return crew

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess DataFrame to handle data type issues."""
    # Convert Int64 to standard int64
    for col in df.select_dtypes(include=['Int64']).columns:
        df[col] = df[col].astype('int64')
    
    # Handle datetime columns
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    return df

def create_visualizations(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create basic visualizations for the dataset."""
    visualizations = {}
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    try:
        # Create histograms for numeric columns
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            visualizations[f'{col}_hist'] = fig
        
        # Create bar charts for categorical columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            fig = px.bar(x=value_counts.index, 
                        y=value_counts.values,
                        title=f'Distribution of {col}')
            visualizations[f'{col}_bar'] = fig
        
        # Create correlation heatmap for numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title='Correlation Heatmap',
                           color_continuous_scale='RdBu')
            visualizations['correlation'] = fig
    except Exception as e:
        st.warning(f"Some visualizations could not be created: {str(e)}")
    
    return visualizations

def display_dataframe_info(df: pd.DataFrame):
    """Display DataFrame in a table format with comprehensive information and data quality metrics."""
    
    # Create expandable section for data preview
    with st.expander("📊 Data Preview", expanded=True):
        # Add column configuration with tooltips
        column_config = {}
        for col in df.columns:
            # Format numeric data in the dataframe itself
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
            
            column_config[col] = st.column_config.Column(
                width="auto",
                help=f"Type: {df[col].dtype}"
            )
        
        st.dataframe(
            df.head(10),
            use_container_width=True,
            column_config=column_config,
            height=400
        )
    
    # Dataset Overview section
    st.subheader("📈 Dataset Overview")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**Dataset Dimensions**")
        st.info(f"""
        • Total Rows: {df.shape[0]:,}
        • Total Columns: {df.shape[1]}
        • Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB
        """)
    
    with col2:
        st.markdown("**Data Quality**")
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
        duplicates = df.duplicated().sum()
        
        st.info(f"""
        • Missing Values: {total_missing:,} ({missing_percentage:.2f}%)
        • Duplicate Rows: {duplicates:,}
        • Complete Rows: {df.dropna().shape[0]:,}
        """)
    
    with col3:
        st.markdown("**Column Types**")
        type_counts = df.dtypes.value_counts()
        st.info('\n'.join([f"• {k}: {v}" for k, v in type_counts.items()]))
    
    # Detailed Column Information
    with st.expander("🔍 Detailed Column Information", expanded=False):
        col_data = []
        for col in df.columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            col_info = {
                "Column": col,
                "Type": str(df[col].dtype),
                "Unique Values": f"{unique_count:,}",
                "Missing": f"{missing_count:,} ({missing_pct:.1f}%)"
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "Min": f"{df[col].min():.2f}",
                    "Max": f"{df[col].max():.2f}",
                    "Mean": f"{df[col].mean():.2f}"
                })
            
            col_data.append(col_info)
        
        st.dataframe(
            pd.DataFrame(col_data),
            use_container_width=True,
            hide_index=True
        )
    
    # Numerical Summary Statistics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        with st.expander("📊 Numerical Statistics", expanded=False):
            st.dataframe(
                df[numeric_cols].describe().round(2),
                use_container_width=True
            )

def render_workflow_diagram():
    """Render the Data Analysis Assistant workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'data_analysis_architecture.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'data_analysis_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def main():
    st.set_page_config(
        page_title="Data Analysis Assistant",  
        page_icon="📊",
        layout="wide")
    
    st.header("📊 Data Analysis Assistant")
    
    # Sidebar configuration
    with st.sidebar:
        # LLM Configuration
        st.header("⚙️ LLM Configuration")
        llm_provider = st.selectbox(
            "Select LLM Provider",
            options=list(LLM_CONFIGS.keys()),
            key='selected_llm_provider',
            help="Choose the AI model provider"
        )
        
        selected_model = st.selectbox(
            "Select Model",
            options=LLM_CONFIGS[llm_provider]["models"],
            key='selected_llm_model',
            help=f"Choose the specific {llm_provider} model"
        )
    
    # Show Architecture and Sequence Diagram
    render_workflow_diagram()

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", 
                                   type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            # Read the data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Preprocess the dataframe
            df = preprocess_dataframe(df)
            
            # Display data preview and information
            display_dataframe_info(df)
            
            # Analysis button
            if st.button("Start Analysis"):
                with st.spinner("Analyzing data..."):
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, 'temp_data.csv')
                    
                    try:
                        # Save data to temporary directory
                        df.to_csv(temp_file_path, index=False)
                        
                        # Initialize analyzer and create crew
                        analyzer = DataAnalyzer(llm_provider, selected_model)
                        crew = analyzer.create_crew(temp_file_path)
                        
                        # Run analysis
                        result = crew.kickoff()
                        
                    finally:
                        # Cleanup: Close any open file handles and remove temporary files
                        try:
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
                            if os.path.exists(temp_dir):
                                os.rmdir(temp_dir)
                        except Exception as cleanup_error:
                            st.warning(f"Warning: Could not clean up temporary files: {cleanup_error}")
                
                # Display results
                st.success("Analysis completed!")
                
                # Convert CrewOutput to string for JSON serialization
                result_str = str(result)  # Convert CrewOutput to string
                
                # Create tabs for results
                tabs = st.tabs(["📈 Visualizations", "📊 Analysis"])
                
                # Visualizations tab
                with tabs[0]:
                    st.subheader("Data Visualizations")
                    visualizations = create_visualizations(df)
                    
                    if visualizations:
                        for name, fig in visualizations.items():
                            try:
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not display visualization {name}: {str(e)}")
                    else:
                        st.warning("No visualizations could be created for this dataset.")
                
                # Analysis tab
                with tabs[1]:
                    st.subheader("Analysis Results")
                    st.markdown(result_str)
                
                
                # Create report data with string result
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': result_str,  # Use string version
                    'dataset_info': {
                        'rows': df.shape[0],
                        'columns': df.shape[1],
                        'column_types': {
                            col: str(dtype)  # Convert dtype to string
                            for col, dtype in df.dtypes.items()
                        }
                    }
                }
                
                # Add download button with properly serializable data
                try:
                    json_report = json.dumps(report_data, indent=2)
                    st.download_button(
                        label="Download Analysis Report",
                        data=json_report,
                        file_name="analysis_report.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.warning(f"Could not create download file: {str(e)}")
                    st.info("Analysis results are still available in the tabs above.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your data format and try again.")

if __name__ == "__main__":
    main()