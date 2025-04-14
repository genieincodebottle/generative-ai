import streamlit as st
import pytz
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone, timedelta
import io
import calendar

# Set page config
st.set_page_config(page_title="Your AI Chat Analytics", page_icon="💀", layout="wide")

def add_sidebar_documentation():
    """Add documentation and download links to the sidebar"""
    st.sidebar.subheader("How to Download Your Data")
    
    # ChatGPT documentation
    st.sidebar.subheader("For ChatGPT Users")
    st.sidebar.write("1. Go to [ChatGPT Settings](https://chatgpt.com/#settings/DataControls)")
    st.sidebar.write("2. Click on 'Export Data'")
    st.sidebar.write("3. Wait for the export email and download your data")
    st.sidebar.write("4. Extract the conversations.json file from the downloaded archive")
    
    st.sidebar.markdown("---")

    # Claude documentation
    st.sidebar.subheader("For Claude Users")
    st.sidebar.write("1. Go to [Claude Settings](https://claude.ai/settings/account)")
    st.sidebar.write("2. Click on 'Export Data'")
    st.sidebar.write("3. Wait for the export email and download your data")
    st.sidebar.write("4. Extract the conversations.json file from the downloaded archive")
    
    st.sidebar.markdown("---")

def parse_conversation_times(convs, format_type, local_tz):
    """Parse conversation times based on specified format"""
    convo_times = []
    
    for conv in convs:
        try:
            if format_type == 'claude':
                created_at = pd.Timestamp(conv['created_at'])
                dt = created_at.to_pydatetime()
            else:  # chatgpt
                unix_timestamp = conv['create_time']
                dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
            
            # Convert to local timezone
            local_dt = dt.astimezone(pytz.timezone(local_tz))
            convo_times.append(local_dt)
        except (KeyError, ValueError) as e:
            st.warning(f"Skipping invalid conversation entry: {str(e)}")
            continue
    
    return convo_times

def calculate_statistics(convo_times, year):
    """Calculate various statistics from conversation times"""
    # Filter conversations for the selected year
    yearly_convos = [ct for ct in convo_times if ct.year == year]
    if not yearly_convos:
        return None
    
    # Convert to pandas datetime for easier analysis
    df = pd.DataFrame({'timestamp': yearly_convos})
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    
    # Calculate weekly stats using the new function
    weekly_stats = calculate_weekly_stats(df)
    
    # Calculate other statistics
    stats = {
        'total_conversations': len(yearly_convos),
        'daily_average': len(yearly_convos) / len(df['date'].unique()),
        'weekly_average': weekly_stats['weekly_average'],
        'weekly_median': weekly_stats['weekly_median'],
        'weekly_std': weekly_stats['weekly_std'],
        'busiest_day': df.groupby('date').size().idxmax(),
        'busiest_day_count': df.groupby('date').size().max(),
        'hourly_distribution': df.groupby('hour').size().to_dict(),
        'weekly_distribution': weekly_stats['weekly_distribution'],
        'monthly_distribution': df.groupby('month').size().to_dict(),
        'peak_week': weekly_stats['peak_week'],
        'peak_week_count': weekly_stats['peak_week_count'],
        'weekly_conversations': weekly_stats['conversations_by_week'],
        'rolling_averages': weekly_stats['rolling_average']
    }
    
    return stats

def create_enhanced_weekly_plot(stats, title="Weekly Usage Pattern"):
    """Create an enhanced weekly visualization"""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    weeks = range(1, 53)  # Standard weeks in a year
    counts = [stats['weekly_conversations'].get(w, 0) for w in weeks]
    rolling_avg = [stats['rolling_averages'].get(w, 0) for w in weeks]
    
    # Plot weekly counts
    bars = ax.bar(weeks, counts, alpha=0.6, label='Weekly Count')
    
    # Plot rolling average
    ax.plot(weeks, rolling_avg, color='red', linewidth=2, 
            label='4-Week Rolling Average')
    
    # Add reference lines
    ax.axhline(y=stats['weekly_average'], color='green', linestyle='--', 
               label=f'Average ({stats["weekly_average"]:.1f})')
    
    # Add value labels on bars
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:  # Only show labels for non-zero values
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Week of Year')
    ax.set_ylabel('Number of Conversations')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_usage_patterns_plot(claude_stats, chatgpt_stats):
    """Create plots comparing usage patterns between Claude and ChatGPT"""
    if not claude_stats or not chatgpt_stats:
        return None
        
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Hourly Distribution
    ax1 = plt.subplot(231)
    hours = range(24)
    claude_hourly = [claude_stats['hourly_distribution'].get(h, 0) for h in hours]
    chatgpt_hourly = [chatgpt_stats['hourly_distribution'].get(h, 0) for h in hours]
    
    ax1.plot(hours, claude_hourly, label='Claude', marker='o')
    ax1.plot(hours, chatgpt_hourly, label='ChatGPT', marker='o')
    ax1.set_title('Hourly Usage Pattern')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Conversations')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Weekly Day Distribution
    ax2 = plt.subplot(232)
    days = range(7)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    claude_weekly = [claude_stats['weekly_distribution'].get(d, 0) for d in days]
    chatgpt_weekly = [chatgpt_stats['weekly_distribution'].get(d, 0) for d in days]
    
    width = 0.35
    ax2.bar(np.array(days) - width/2, claude_weekly, width, label='Claude')
    ax2.bar(np.array(days) + width/2, chatgpt_weekly, width, label='ChatGPT')
    ax2.set_title('Weekly Day Distribution')
    ax2.set_xticks(days)
    ax2.set_xticklabels(day_names)
    ax2.set_ylabel('Number of Conversations')
    ax2.legend()
    
    # 3. Monthly Distribution
    ax3 = plt.subplot(233)
    months = range(1, 13)
    month_names = [calendar.month_abbr[m] for m in months]
    claude_monthly = [claude_stats['monthly_distribution'].get(m, 0) for m in months]
    chatgpt_monthly = [chatgpt_stats['monthly_distribution'].get(m, 0) for m in months]
    
    width = 0.35
    ax3.bar(np.array(months) - width/2, claude_monthly, width, label='Claude')
    ax3.bar(np.array(months) + width/2, chatgpt_monthly, width, label='ChatGPT')
    ax3.set_title('Monthly Distribution')
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names)
    ax3.set_ylabel('Number of Conversations')
    ax3.legend()
    
    # 4. Weekly Trends Comparison
    ax4 = plt.subplot(212)
    weeks = range(1, 53)
    claude_weekly = [claude_stats['weekly_conversations'].get(w, 0) for w in weeks]
    chatgpt_weekly = [chatgpt_stats['weekly_conversations'].get(w, 0) for w in weeks]
    claude_rolling = [claude_stats['rolling_averages'].get(w, 0) for w in weeks]
    chatgpt_rolling = [chatgpt_stats['rolling_averages'].get(w, 0) for w in weeks]
    
    ax4.plot(weeks, claude_weekly, alpha=0.3, color='blue', label='Claude Weekly')
    ax4.plot(weeks, chatgpt_weekly, alpha=0.3, color='orange', label='ChatGPT Weekly')
    ax4.plot(weeks, claude_rolling, color='blue', linewidth=2, label='Claude 4-Week Avg')
    ax4.plot(weeks, chatgpt_rolling, color='orange', linewidth=2, label='ChatGPT 4-Week Avg')
    
    ax4.set_title('Weekly Usage Trends Comparison')
    ax4.set_xlabel('Week of Year')
    ax4.set_ylabel('Number of Conversations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_single_bot_patterns(stats):
    """Create usage pattern plots for single bot analysis"""
    if not stats:
        return None
        
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Hourly Distribution
    ax1 = plt.subplot(221)
    hours = range(24)
    hourly_counts = [stats['hourly_distribution'].get(h, 0) for h in hours]
    
    ax1.plot(hours, hourly_counts, marker='o', color='#2E86C1', linewidth=2)
    ax1.fill_between(hours, hourly_counts, alpha=0.3, color='#2E86C1')
    ax1.set_title('Hourly Usage Pattern')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Conversations')
    ax1.grid(True, alpha=0.3)
    
    # 2. Weekly Day Distribution
    ax2 = plt.subplot(222)
    days = range(7)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_counts = [stats['weekly_distribution'].get(d, 0) for d in days]
    
    bars = ax2.bar(days, weekly_counts, color='#2E86C1')
    ax2.set_title('Weekly Day Distribution')
    ax2.set_xticks(days)
    ax2.set_xticklabels(day_names)
    ax2.set_ylabel('Number of Conversations')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 3. Monthly Distribution
    ax3 = plt.subplot(223)
    months = range(1, 13)
    month_names = [calendar.month_abbr[m] for m in months]
    monthly_counts = [stats['monthly_distribution'].get(m, 0) for m in months]
    
    bars = ax3.bar(months, monthly_counts, color='#2E86C1')
    ax3.set_title('Monthly Distribution')
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names)
    ax3.set_ylabel('Number of Conversations')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 4. Weekly Trends
    ax4 = plt.subplot(224)
    weeks = range(1, 53)
    weekly_counts = [stats['weekly_conversations'].get(w, 0) for w in weeks]
    rolling_avg = [stats['rolling_averages'].get(w, 0) for w in weeks]
    
    ax4.bar(weeks, weekly_counts, alpha=0.4, color='#2E86C1', label='Weekly Count')
    ax4.plot(weeks, rolling_avg, color='red', linewidth=2, label='4-Week Rolling Average')
    ax4.axhline(y=stats['weekly_average'], color='green', linestyle='--', 
                label=f'Average ({stats["weekly_average"]:.1f})')
    
    ax4.set_title('Weekly Usage Trends')
    ax4.set_xlabel('Week of Year')
    ax4.set_ylabel('Number of Conversations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_weekly_stats(df):
    """Calculate more detailed weekly statistics"""
    # Current week number and day of week
    df['week_number'] = df['timestamp'].dt.isocalendar().week
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Calculate weekly stats
    weekly_stats = {
        'conversations_by_week': df.groupby('week_number').size().to_dict(),
        'weekly_average': df.groupby('week_number').size().mean(),
        'weekly_median': df.groupby('week_number').size().median(),
        'weekly_std': df.groupby('week_number').size().std(),
        'peak_week': df.groupby('week_number').size().idxmax(),
        'peak_week_count': df.groupby('week_number').size().max(),
        'weekly_distribution': df.groupby('day_of_week').size().to_dict()
    }
    
    # Calculate rolling averages for trend analysis
    weekly_series = df.groupby('week_number').size()
    weekly_stats['rolling_average'] = weekly_series.rolling(window=4, min_periods=1).mean().to_dict()
    
    return weekly_stats

def create_year_heatmap(convo_times, year, format_type, fig=None, subplot_idx=None):
    """Create heatmap for a specific format type"""
    just_dates = [convo.date() for convo in convo_times if convo.year == year]
    date_counts = Counter(just_dates)
    
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    
    total_days = (end_date - start_date).days + 1
    date_range = [start_date + timedelta(days=i) for i in range(total_days)]
    
    data = []
    for date in date_range:
        week = ((date - start_date).days + start_date.weekday()) // 7
        day_of_week = date.weekday()
        count = date_counts.get(date, 0)
        data.append((week, day_of_week, count, date))
    
    weeks_in_year = (end_date - start_date).days // 7 + 1
    
    if subplot_idx is None:
        fig, ax = plt.subplots(figsize=(15, 8))
    else:
        ax = fig.add_subplot(1, 2, subplot_idx)
    
    ax.set_aspect('equal')
    
    max_count_date = max(date_counts, key=date_counts.get) if date_counts else start_date
    max_count = date_counts[max_count_date] if date_counts else 0
    p90_count = np.percentile(list(date_counts.values()), 90) if date_counts else 1
    
    # Draw rectangles and add date numbers
    for week, day_of_week, count, date in data:
        color = plt.cm.Greens((count + 1) / p90_count) if count > 0 else 'lightgray'
        rect = patches.Rectangle((week, day_of_week), 1, 1, linewidth=0.5, 
                               edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Add the date number inside each box
        date_text = date.strftime('%d')
        text_color = 'black' if count == 0 else 'white' if count > p90_count/2 else 'black'
        plt.text(week + 0.5, day_of_week + 0.5, date_text, 
                ha='center', va='center', color=text_color, fontsize=8)
    
    # Month labels at bottom
    month_starts = [start_date + timedelta(days=i) for i in range(total_days)
                   if (start_date + timedelta(days=i)).day == 1]
    for month_start in month_starts:
        week = (month_start - start_date).days // 7
        plt.text(week + 0.5, 7.75, month_start.strftime('%b'), 
                ha='center', va='center', fontsize=10, rotation=0)
    
    ax.set_xlim(-0.5, weeks_in_year + 0.5)
    ax.set_ylim(-0.5, 8.5)
    
    format_name = "Claude" if format_type == 'claude' else "ChatGPT"
    plt.title(
        f'{year} {format_name} Conversation Heatmap (total={sum(date_counts.values())}).\n'
        f'Most active day: {max_count_date.strftime("%Y-%m-%d")} with {max_count} conversations.',
        fontsize=12
    )
    
    plt.xticks([])
    plt.yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.gca().invert_yaxis()
    
    return fig

def display_comparative_metrics(claude_stats, chatgpt_stats):
    """Display key comparative metrics"""
    if not claude_stats or not chatgpt_stats:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Conversations",
            f"Claude: {claude_stats['total_conversations']}",
            f"ChatGPT: {chatgpt_stats['total_conversations']}"
        )
        
    with col2:
        st.metric(
            "Daily Average",
            f"Claude: {claude_stats['daily_average']:.1f}",
            f"ChatGPT: {chatgpt_stats['daily_average']:.1f}"
        )
        
    with col3:
        st.metric(
            "Weekly Average",
            f"Claude: {claude_stats['weekly_average']:.1f}",
            f"ChatGPT: {chatgpt_stats['weekly_average']:.1f}"
        )

def set_page_style():
    """Set custom CSS styles for better UI appearance"""
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            font-size: 1.4rem !important;
            padding-bottom: 0.5rem;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.3rem !important;
            padding-top: 0.5rem;
            padding-bottom: 0.3rem;
        }
        
        /* Subsection headers */
        .subsection-header {
            font-size: 1rem !important;
            padding-top: 0.3rem;
            padding-bottom: 0.2rem;
        }
        
        /* Regular text */
        .stMarkdown {
            font-size: 0.5rem !important;
            margin-bottom: 0.5rem;
        }
        
        /* Metrics styling */
        .metric-container {
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin: 0.3rem 0;
            border: 1px solid #e0e0e0;  /* Added gray border */
        }
        
        /* Reduce tab padding */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        .sidebar .stMarkdown {
            font-size: 0.85rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, delta=None):
    """Create a styled metric card"""
    html = f"""
        <div class="metric-container">
            <p style="font-size: 0.8rem; color: #666; margin-bottom: 0.2rem;">{title}</p>
            <p style="font-size: 1.2rem; font-weight: bold; margin: 0;">{value}</p>
            {f'<p style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">{delta}</p>' if delta else ''}
        </div>
    """
    return st.markdown(html, unsafe_allow_html=True)


def main():
    # Set custom page styling
    set_page_style()
    
    # Initialize session state
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'single'
    if 'format_type' not in st.session_state:
        st.session_state.format_type = 'claude'
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    # Main title with custom styling
    st.markdown('<p class="main-title">💀 ChatGPT x Claude: How Much You Outsource Your Brain to AI</p>', unsafe_allow_html=True)
    
    # Sidebar controls with better spacing
    with st.sidebar:
        st.markdown("### Dashboard Settings", help="Configure your analysis preferences")
        view_mode = st.selectbox(
            label="Analysis Mode",
            options=['single', 'comparison'],
            format_func=lambda x: "Single Bot Analysis" if x == 'single' else "Compare Both Bots",
            key='view_mode'
        )
        
        year = st.selectbox(
            label="Analysis Year",
            options=list(range(2023, 2026)),
            index=1
        )
        
        timezone = st.selectbox(
            label="Timezone",
            options=['Asia/Kolkata','UTC','America/New_York', 'Europe/London'],
            index=0
        )
        
        st.markdown("---")
        add_sidebar_documentation()
    
    # Main content area
    if view_mode == 'single':
        # Single bot mode
        st.markdown('<p class="section-header">🤖 Select AI Assistant</p>', unsafe_allow_html=True)
        selected_bot = st.selectbox(
            label="Select AI Assistant",
            options=['chatgpt', 'claude'],
            format_func=lambda x: "Claude" if x == 'claude' else "ChatGPT",
            key='single_bot_selector',
            label_visibility="collapsed"
        )
        
        if selected_bot != st.session_state.format_type:
            st.session_state.format_type = selected_bot
            st.session_state.reset_counter += 1
        
        help_text = f"Upload your {selected_bot.upper()} conversations.json file"
        uploaded_file = st.file_uploader(
            label="Upload conversations.json",
            type=['json'],
            help=help_text,
            key=f'file_uploader_{st.session_state.reset_counter}',
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                convs = json.load(uploaded_file)
                if not convs:
                    st.error("No conversations found in the file.")
                    return
                
                with st.spinner("Processing conversations..."):
                    convo_times = parse_conversation_times(convs, selected_bot, timezone)
                
                if not convo_times:
                    st.error("No valid conversation times found.")
                    return
                
                # Calculate statistics
                stats = calculate_statistics(convo_times, year)
                if stats:
                    st.markdown('<p class="section-header">📊 Usage Statistics</p>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        create_metric_card(
                            "Total Conversations",
                            f"{stats['total_conversations']:,}"
                        )
                    with col2:
                        create_metric_card(
                            "Daily Average",
                            f"{stats['daily_average']:.1f}"
                        )
                    with col3:
                        create_metric_card(
                            "Weekly Average",
                            f"{stats['weekly_average']:.1f}"
                        )
                    
                    # Create tabs for visualizations
                    tab1, tab2, tab3 = st.tabs(["Activity Heatmap", "Usage Patterns", "Key Insights"])
                    
                    with tab1:
                        fig = create_year_heatmap(convo_times, year, selected_bot)
                        st.pyplot(fig)
                        
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                        st.markdown("* Numbers in the boxes are day of the month")
                        st.download_button(
                            label="📥 Download Heatmap",
                            data=buf.getvalue(),
                            file_name=f"{selected_bot}_heatmap_{year}.png",
                            mime="image/png"
                        )

                    with tab2:
                        patterns_fig = create_single_bot_patterns(stats)
                        if patterns_fig:
                            st.pyplot(patterns_fig)
                            
                            buf = io.BytesIO()
                            patterns_fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                            st.download_button(
                                label="📥 Download Usage Patterns",
                                data=buf.getvalue(),
                                file_name=f"{selected_bot}_patterns_{year}.png",
                                mime="image/png"
                            )
                    with tab3:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Daily Activity")
                            st.markdown(f"""
                                - Average: {stats['daily_average']:.1f} conversations/day
                                - Peak: {stats['busiest_day_count']} conversations on {stats['busiest_day']}
                            """)
                        
                        with col2:
                            st.markdown("#### Peak Times")
                            # Peak hour
                            peak_hour = max(stats['hourly_distribution'].items(), key=lambda x: x[1])[0]
                            # Peak day
                            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            peak_day = max(stats['weekly_distribution'].items(), key=lambda x: x[1])[0]
                            # Peak month
                            peak_month = max(stats['monthly_distribution'].items(), key=lambda x: x[1])[0]
                            month_name = calendar.month_name[peak_month]
                            
                            st.markdown(f"""
                                - Most active hour: {peak_hour}:00
                                - Most active day: {days[peak_day]}
                                - Most active month: {month_name}
                            """)
                            
            except json.JSONDecodeError:
                st.error("Invalid JSON format in the uploaded file.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
        else:
            # Show format-specific instructions
            if selected_bot == 'claude':
                st.info("Upload your Claude conversations.json file to generate the usage statistics.")
            elif selected_bot == 'chatgpt':
                st.info("Upload your ChatGPT conversations.json file to generate the usage statistics.")
    else:
        # Comparison mode
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="subsection-header">ChatGPT</p>', unsafe_allow_html=True)
            chatgpt_file = st.file_uploader(
                label="Upload ChatGPT conversations.json",
                type=['json'],
                key='chatgpt_uploader',
                label_visibility="collapsed"
            )
            st.info("Upload your ChatGPT conversations.json file to generate the usage statistics.")

        with col2:
            st.markdown('<p class="subsection-header">Claude</p>', unsafe_allow_html=True)
            claude_file = st.file_uploader(
                label="Upload Claude conversations.json",
                type=['json'],
                key='claude_uploader',
                label_visibility="collapsed"
            )
            st.info("Upload your Claude conversations.json file to generate the usage statistics.")
            
        if claude_file is not None and chatgpt_file is not None:
            try:
                # Process both files
                claude_convs = json.load(claude_file)
                chatgpt_convs = json.load(chatgpt_file)
                
                with st.spinner("Processing conversations..."):
                    claude_times = parse_conversation_times(claude_convs, 'claude', timezone)
                    chatgpt_times = parse_conversation_times(chatgpt_convs, 'chatgpt', timezone)
                
                if not claude_times or not chatgpt_times:
                    st.error("No valid conversation times found in one or both files.")
                    return
                
                # Calculate statistics
                claude_stats = calculate_statistics(claude_times, year)
                chatgpt_stats = calculate_statistics(chatgpt_times, year)
                
                # Display comparative metrics
                st.markdown('<p class="section-header">📊 Usage Statistics</p>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    create_metric_card(
                        "Total Conversations",
                        f"Claude: {claude_stats['total_conversations']:,}",
                        f"ChatGPT: {chatgpt_stats['total_conversations']:,}"
                    )
                with col2:
                    create_metric_card(
                        "Daily Average",
                        f"Claude: {claude_stats['daily_average']:.1f}",
                        f"ChatGPT: {chatgpt_stats['daily_average']:.1f}"
                    )
                with col3:
                    create_metric_card(
                        "Weekly Average",
                        f"Claude: {claude_stats['weekly_average']:.1f}",
                        f"ChatGPT: {chatgpt_stats['weekly_average']:.1f}"
                    )
                
                # Create tabs for visualizations
                tab1, tab2, tab3 = st.tabs(["Activity Heatmaps", "Usage Patterns", "Key Insights"])
                
                with tab1:
                    fig = plt.figure(figsize=(20, 8))
                    fig = create_year_heatmap(claude_times, year, 'claude', fig, 1)
                    fig = create_year_heatmap(chatgpt_times, year, 'chatgpt', fig, 2)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                    st.markdown("* Numbers in the boxes are day of the month")
                    st.download_button(
                        label="📥 Download Heatmaps",
                        data=buf.getvalue(),
                        file_name=f"comparison_heatmaps_{year}.png",
                        mime="image/png"
                    )
                
                with tab2:
                    patterns_fig = create_usage_patterns_plot(claude_stats, chatgpt_stats)
                    if patterns_fig:
                        st.pyplot(patterns_fig)
                        
                        buf = io.BytesIO()
                        patterns_fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                        st.download_button(
                            label="📥 Download Usage Patterns",
                            data=buf.getvalue(),
                            file_name=f"comparison_patterns_{year}.png",
                            mime="image/png"
                        )
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Usage Distribution
                        total_convos = claude_stats['total_conversations'] + chatgpt_stats['total_conversations']
                        claude_percentage = (claude_stats['total_conversations'] / total_convos * 100)
                        chatgpt_percentage = (chatgpt_stats['total_conversations'] / total_convos * 100)
                        
                        st.markdown("#### Usage Distribution")
                        st.markdown(f"""
                            - Claude: {claude_percentage:.1f}%
                            - ChatGPT: {chatgpt_percentage:.1f}%
                        """)
                    
                    with col2:
                        st.markdown("#### Peak Usage Times")
                        st.markdown(f"""
                            Claude:
                            - Busiest day: {claude_stats['busiest_day']} ({claude_stats['busiest_day_count']} conversations)
                            
                            ChatGPT:
                            - Busiest day: {chatgpt_stats['busiest_day']} ({chatgpt_stats['busiest_day_count']} conversations)
                        """)
                    
                    st.markdown("#### Usage Patterns")
                    st.markdown(f"""
                        Daily Averages:
                        - Claude: {claude_stats['daily_average']:.1f} conversations/day
                        - ChatGPT: {chatgpt_stats['daily_average']:.1f} conversations/day
                        
                        Weekly Averages:
                        - Claude: {claude_stats['weekly_average']:.1f} conversations/week
                        - ChatGPT: {chatgpt_stats['weekly_average']:.1f} conversations/week
                    """)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON format in one or both uploaded files.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()