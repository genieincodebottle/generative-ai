import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Default Research Values
DEFAULT_RESEARCH_TOPIC = "Cryptocurrency and DeFi Trends in Emerging Markets"
DEFAULT_REQUIREMENTS = """1. Adoption rates in different regions
2. Regulatory landscape
3. Popular platforms and services
4. Financial inclusion impact
5. Market growth projections"""

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

def initialize_llm(llm_provider: str, model_name: str) -> LLM:
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

def create_research_crew(llm_provider: str, model_name: str):
    """Create the research crew with agents"""
    
    llm = initialize_llm(llm_provider, model_name)
    search_tool = SerperDevTool(
        search_url="https://google.serper.dev/search",
        n_results=2,
    )
    
    researcher = Agent(
        role='Research Analyst',
        goal='Conduct thorough research on topics',
        backstory='Expert researcher with years of experience',
        verbose=True,
        tools=[search_tool],
        llm=llm,
        allow_delegation=True
    )

    analyst = Agent(
        role='Data Analyst',
        goal='Analyze findings and extract insights',
        backstory='Experienced data analyst and interpreter',
        verbose=True,
        llm=llm,
        allow_delegation=True
    )

    writer = Agent(
        role='Content Writer',
        goal='Create clear, engaging content',
        backstory='Professional technical writer',
        verbose=True,
        llm=llm,
        allow_delegation=True
    )

    return researcher, analyst, writer

def create_tasks(researcher, analyst, writer, research_topic, research_type, 
                depth_level, requirements):
    """Create research tasks"""
    tasks = [
        Task(
            description=f"""
            Research the topic: {research_topic}
            Type: {research_type}
            Depth: {depth_level}
            Requirements: {requirements}
            """,
            expected_output="""A comprehensive research document containing:
            - Detailed findings and discoveries
            - Statistical data and trends
            - Key stakeholder information
            - Market/industry context
            - Relevant sources and citations""",
            agent=researcher
        ),
        Task(
            description=f"""
            Analyze the research findings with focus on:
            1. Key patterns and trends
            2. Impact assessment
            3. Comparative analysis
            4. Future projections
            Depth Level: {depth_level}
            """,
            expected_output="""An analytical report containing:
            - Key trends and patterns identified
            - Impact analysis and implications
            - Data-driven insights
            - Strategic recommendations
            - Risk assessment""",
            agent=analyst
        ),
        Task(
            description=f"""
            Create a comprehensive report synthesizing the research and analysis.
            Research Type: {research_type}
            Depth: {depth_level}
            Format: Professional business report
            """,
            expected_output="""A well-structured final report including:
            - Executive summary
            - Key findings and insights
            - Detailed analysis
            - Supporting data and visuals
            - Conclusions and recommendations
            - References and sources""",
            agent=writer
        )
    ]
    return tasks

def render_workflow_diagram():
    """Render the Research Assistant workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'architecture.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')
                                      
def main():
    # Page configuration
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="🔍",
        layout="wide"
    )

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

    # Main content
    st.header("🔍 Research Assistant")

    render_workflow_diagram()
    
    # Research form with session state
    research_topic = st.text_area(
        "Research Topic",
        value=DEFAULT_RESEARCH_TOPIC,
        height=70,
        key='research_topic'
    )

    col1, col2 = st.columns(2)
    with col1:
        research_type = st.selectbox(
            "Research Type",
            ["General Research", "Market Analysis", "Technical Review", "Trend Analysis"]
        )

    with col2:
        depth_level = st.select_slider(
            "Research Depth",
            options=["Basic", "Moderate", "Comprehensive", "Expert"],
            value="Moderate"
        )

    specific_requirements = st.text_area(
        "Specific Requirements",
        value=DEFAULT_REQUIREMENTS,
        height=140,
        key='specific_requirements'
    )

    # Start Research button
    if st.button("Start Research", type="primary"):
        if not research_topic:
            st.error("Please enter a research topic")
            return

        try:
            with st.spinner("Initializing research crew..."):
                researcher, analyst, writer = create_research_crew(
                    llm_provider, selected_model
                )
                
                tasks = create_tasks(
                    researcher, analyst, writer,
                    research_topic, research_type,
                    depth_level, specific_requirements
                )

                crew = Crew(
                    agents=[researcher, analyst, writer],
                    tasks=tasks,
                    verbose=True
                )

            with st.spinner("Conducting research..."):
                crew_result = crew.kickoff()
                result = str(crew_result)

            # Display results
            st.success("Research completed!")
            
            tab1, tab2 = st.tabs(["📊 Research Results", "📋 Download Report"])
            
            with tab1:
                st.markdown("### Research Results")
                st.markdown(result)
            
            with tab2:
                st.markdown("### Download Report")
                st.download_button(
                    label="Download Report",
                    data=result,
                    file_name=f"research_report_{research_topic[:30]}.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check if all required API keys are correctly configured.")

if __name__ == "__main__":
    main()