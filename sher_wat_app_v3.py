import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from sqlalchemy import create_engine, text
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# Import your agents (assuming they're in the same directory or properly imported)
from test_agent_sher_wat_2 import (
    SherlockPyDictAgent, WatsonPyDictAgent, 
    SherlockSQLDictAgent, WatsonSQLDictAgent
)

# Page config
st.set_page_config(
    page_title="Talk2Table",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Chat message styling */
    .user-message {
    background-color: #3b82f6;  /* Blue-500 */
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    }
    .assistant-message {
    background-color: #6b7280;  /* Gray-500 */
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    }
    /* Sidebar styling */
    .sidebar-chat-item {
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .sidebar-chat-item:hover {
        background-color: #e0e0e0;
    }
    /* Table preview styling */
    .table-preview {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    /* Plot styling */
    .plot-container {
        margin: 15px 0;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_engine' not in st.session_state:
    DB_URL = st.secrets["SUPABASE_DB_URL"]
    if DB_URL:
        st.session_state.db_engine = create_engine(DB_URL, pool_pre_ping=True)
        # Create tables if they don't exist
        with st.session_state.db_engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  SERIAL PRIMARY KEY,
                    title       TEXT    DEFAULT 'Untitled',
                    data_type   TEXT    DEFAULT 'Mixed',
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS messages (
                    id          SERIAL PRIMARY KEY,
                    session_id  INT     REFERENCES sessions(session_id) ON DELETE CASCADE,
                    idx         INT     NOT NULL,
                    role        TEXT    NOT NULL CHECK (role IN ('user','assistant')),
                    content     TEXT,
                    code        TEXT,
                    agent_type  TEXT,
                    ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.commit()
    else:
        st.session_state.db_engine = None

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'use_watson' not in st.session_state:
    st.session_state.use_watson = False
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# Helper functions
def create_new_session(title: str = "Untitled", data_type: str = "Mixed") -> int:
    """Create a new chat session in the database"""
    if st.session_state.db_engine:
        with st.session_state.db_engine.connect() as conn:
            result = conn.execute(
                text("INSERT INTO sessions (title, data_type) VALUES (:title, :data_type) RETURNING session_id"),
                {"title": title, "data_type": data_type}
            )
            conn.commit()
            return result.fetchone()[0]
    return None

def save_message(session_id: int, idx: int, role: str, content: str, 
                code: str = None, agent_type: str = None):
    """Save a message to the database"""
    if st.session_state.db_engine and session_id:
        with st.session_state.db_engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO messages (session_id, idx, role, content, code, agent_type)
                    VALUES (:session_id, :idx, :role, :content, :code, :agent_type)
                """),
                {
                    "session_id": session_id,
                    "idx": idx,
                    "role": role,
                    "content": content,
                    "code": code,
                    "agent_type": agent_type
                }
            )
            conn.commit()

def get_sessions() -> List[Dict]:
    """Get all chat sessions from the database"""
    if st.session_state.db_engine:
        with st.session_state.db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT session_id, title, data_type, created_at FROM sessions ORDER BY created_at DESC")
            )
            return [{"session_id": row[0], "title": row[1], "data_type": row[2], "created_at": row[3]} 
                   for row in result]
    return []

def get_session_messages(session_id: int) -> List[Dict]:
    """Get all messages for a session"""
    if st.session_state.db_engine:
        with st.session_state.db_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT idx, role, content, code, agent_type 
                    FROM messages 
                    WHERE session_id = :session_id 
                    ORDER BY idx
                """),
                {"session_id": session_id}
            )
            return [{"idx": row[0], "role": row[1], "content": row[2], 
                    "code": row[3], "agent_type": row[4]} for row in result]
    return []

def update_session_title(session_id: int, new_title: str):
    """Update session title"""
    if st.session_state.db_engine:
        with st.session_state.db_engine.connect() as conn:
            conn.execute(
                text("UPDATE sessions SET title = :title WHERE session_id = :session_id"),
                {"title": new_title, "session_id": session_id}
            )
            conn.commit()

def delete_session(session_id: int):
    """Delete a session"""
    if st.session_state.db_engine:
        with st.session_state.db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id}
            )
            conn.commit()

def load_data_from_file(uploaded_file) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Load ALL tables/sheets from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            return {'data': df}, 'csv'
            
        elif uploaded_file.name.endswith('.xlsx'):
            # Read ALL sheets
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            return all_sheets, 'xlsx'
            
        elif uploaded_file.name.endswith(('.db', '.sqlite')):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            conn = sqlite3.connect(tmp_file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            # Read ALL tables
            all_tables = {}
            for table in tables:
                all_tables[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            
            conn.close()
            os.unlink(tmp_file_path)
            return all_tables, 'sqlite'
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return {}, None

def create_agent(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], data_type: str, use_watson: bool):
    """Create the appropriate agent based on data type and preference"""
    
    # For CSV and Excel files, use Python agents
    if data_type in ['csv', 'xlsx']:
        if use_watson:
            if isinstance(data, pd.DataFrame):
                return WatsonPyDictAgent(df=data, max_steps=10, verbose=True)
            else:
                return WatsonPyDictAgent(dataframes=data, max_steps=10, verbose=True)
        else:
            if isinstance(data, pd.DataFrame):
                return SherlockPyDictAgent(df=data, max_steps=10, verbose=True)
            else:
                return SherlockPyDictAgent(dataframes=data, max_steps=10, verbose=True)
    
    # For SQLite databases, store data and create a wrapper
    elif data_type in ['sqlite', 'db']:
        # Store the data in session state for thread-safe access
        if isinstance(data, pd.DataFrame):
            st.session_state.sql_data = {'data_table': data}
        else:
            st.session_state.sql_data = data
        
        # Create a wrapper class that creates fresh connections
        class ThreadSafeSQLAgent:
            def __init__(self, agent_class, use_watson):
                self.agent_class = agent_class
                self.use_watson = use_watson
                self.max_steps = 10
                self.verbose = not use_watson
            
            def query(self, question: str):
                # Create a fresh temporary database for this query
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                    tmp_file_path = tmp_file.name
                
                try:
                    # Create connection with thread-safe settings
                    import sqlite3
                    conn = sqlite3.connect(tmp_file_path, check_same_thread=False)
                    
                    # Load all tables
                    for table_name, table_df in st.session_state.sql_data.items():
                        table_df.to_sql(table_name, conn, index=False, if_exists='replace')
                    
                    conn.close()
                    
                    # Create agent with the temp file
                    if self.use_watson:
                        agent = WatsonSQLDictAgent(db_path=tmp_file_path, max_steps=self.max_steps, verbose=self.verbose)
                    else:
                        agent = SherlockSQLDictAgent(db_path=tmp_file_path, max_steps=self.max_steps)
                    
                    # Execute query
                    result = agent.query(question)
                    
                    # Clean up
                    if hasattr(agent, 'close'):
                        agent.close()
                    
                    return result
                    
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
        
        # Return the wrapper
        return ThreadSafeSQLAgent(
            SherlockSQLDictAgent if not use_watson else WatsonSQLDictAgent,
            use_watson
        )
    
    return None

def display_dataframe_preview(data, max_rows: int = 5):
    """Display a compact preview of the dataframe(s)"""
    
    if isinstance(data, dict):
        # Multiple DataFrames - show selectbox
        if len(data) > 1:
            selected_table = st.selectbox(
                "Select table to preview:",
                options=list(data.keys()),
                key="table_selector"
            )
            df = data[selected_table]
            st.markdown(f"#### 📊 Data Preview - {selected_table}")
        else:
            # Single table in dict
            df = list(data.values())[0]
            table_name = list(data.keys())[0]
            st.markdown(f"#### 📊 Data Preview - {table_name}")
    else:
        # Single DataFrame
        df = data
        st.markdown("#### 📊 Data Preview")
    
    # Display the selected/single dataframe
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.dataframe(df.head(max_rows), height=150)
    
    with col2:
        st.markdown("#### 📐 Shape")
        st.metric("Rows", f"{df.shape[0]:,}")
        st.metric("Columns", f"{df.shape[1]:,}")
    
    with col3:
        st.markdown("#### 📋 Schema")
        schema_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str)
        })
        st.dataframe(schema_info, height=150)

def cleanup_agent():
    """Clean up any temporary files and database connections created by the agent"""
    if hasattr(st.session_state.agent, '_engine'):
        try:
            st.session_state.agent._engine.dispose()
        except:
            pass
    
    if hasattr(st.session_state.agent, '_temp_db_path'):
        try:
            os.unlink(st.session_state.agent._temp_db_path)
        except:
            pass

def execute_code_on_current_data(code: str, code_type: str) -> Dict[str, Any]:
    """Execute code on the current loaded data"""
    if st.session_state.uploaded_data is None:
        return {"error": "No data loaded"}
    
    data = st.session_state.uploaded_data
    
    if code_type == 'python':
        # Execute Python code
        try:
            # Create safe execution environment
            safe_globals = {
                'pd': pd,
                'result': None
            }
            
            # Handle single vs multiple dataframes
            if isinstance(data, pd.DataFrame):
                safe_globals['df'] = data
            else:
                # Multiple dataframes
                safe_globals['dataframes'] = data
                # Also add individual dataframes by name
                for name, df in data.items():
                    safe_globals[name] = df
                # Set 'df' to the main dataframe if it exists, or the first one
                safe_globals['df'] = data.get('main', list(data.values())[0])
            
            # Import numpy if available
            try:
                import numpy as np
                safe_globals['np'] = np
            except:
                pass
            
            # Import matplotlib if available
            try:
                import matplotlib.pyplot as plt
                safe_globals['plt'] = plt
            except:
                pass
            
            # Execute code
            exec(code, safe_globals)
            result = safe_globals.get('result')
            
            if isinstance(result, pd.DataFrame):
                return {"success": True, "result": result}
            elif result is not None:
                # Convert non-dataframe results to dataframe
                if isinstance(result, pd.Series):
                    result_df = result.to_frame()
                elif isinstance(result, (int, float, str)):
                    result_df = pd.DataFrame({"Result": [result]})
                elif isinstance(result, dict):
                    result_df = pd.DataFrame([result])
                elif isinstance(result, list):
                    result_df = pd.DataFrame({"Values": result})
                else:
                    result_df = pd.DataFrame({"Result": [str(result)]})
                return {"success": True, "result": result_df}
            else:
                return {"error": "No result produced"}
                
        except Exception as e:
            return {"error": str(e)}
    
    elif code_type == 'sql':
        # Execute SQL code with SQLAlchemy
        try:
            # Create SQLAlchemy engine with thread-safe settings
            engine = create_engine(
                'sqlite:///:memory:',
                connect_args={'check_same_thread': False},
                poolclass=StaticPool
            )
            
            # Load all tables into the in-memory database
            if isinstance(data, pd.DataFrame):
                data.to_sql('data_table', engine, index=False, if_exists='replace')
            else:
                # Multiple tables
                for table_name, table_df in data.items():
                    table_df.to_sql(table_name, engine, index=False, if_exists='replace')
            
            # Execute query using pandas with SQLAlchemy engine
            result_df = pd.read_sql_query(code, engine)
            
            # Dispose of the engine
            engine.dispose()
            
            return {"success": True, "result": result_df}
            
        except Exception as e:
            return {"error": str(e)}
    
    return {"error": "Unknown code type"}

# Plot display functions
def display_plots_from_message(message, message_index):
    """Display plots stored in message - FIXED to avoid duplicates"""
    plots = message.get('plots', [])
    plot_images = message.get('plot_images', [])
    
    # CHANGE: Only use plots if available, otherwise fall back to plot_images
    # This prevents showing the same plot twice
    plots_to_show = plots if plots else plot_images
    
    if not plots_to_show:
        return
        
    st.markdown("### 📊 Generated Plots")
    
    # Display plots (either from 'plots' field or 'plot_images' field, not both)
    for i, plot_item in enumerate(plots_to_show):
        # Handle both formats: dict with plot info, or just base64 string
        if isinstance(plot_item, dict):
            # Full plot info format
            if plot_item.get('success') and plot_item.get('base64'):
                try:
                    img_data = base64.b64decode(plot_item['base64'])
                    img = Image.open(BytesIO(img_data))
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.image(img, caption=f"Plot {i+1}", use_column_width=True)
                    
                    with col2:
                        # Download button
                        st.download_button(
                            label="💾 Download",
                            data=img_data,
                            file_name=f"plot_{message_index}_{i+1}.png",
                            mime="image/png",
                            key=f"download_plot_{message_index}_{i}"
                        )
                        
                        # Show code button
                        if plot_item.get('code'):
                            with st.expander("📝 Code"):
                                st.code(plot_item['code'], language='python')
                                
                except Exception as e:
                    st.error(f"Error displaying plot {i+1}: {str(e)}")
        
        elif isinstance(plot_item, str):
            # Just base64 string format
            try:
                img_data = base64.b64decode(plot_item)
                img = Image.open(BytesIO(img_data))
                st.image(img, caption=f"Plot {i+1}", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying plot image {i+1}: {str(e)}")
                
def clear_agent_plots(agent):
    """Clear accumulated plots from agent state before new query"""
    if hasattr(agent, 'state') and 'plots' in agent.state:
        agent.state['plots'] = []
    # Also clear plots if agent has this method
    if hasattr(agent, 'clear_plots'):
        agent.clear_plots()

def get_plots_from_agent_state(agent):
    """Extract plots directly from agent state"""
    if hasattr(agent, 'state') and 'plots' in agent.state:
        return agent.state.get('plots', [])
    elif hasattr(agent, 'get_all_plots'):
        return agent.get_all_plots()
    return []

def display_message_with_plots(message, message_index, agent=None):
    """Display a message with proper plot handling"""
    if message['role'] == 'user':
        st.markdown(f"<div class='user-message'>🧑 {message['content']}</div>", 
                   unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f"<div class='assistant-message'>🤖 <b>{message.get('agent_type', 'Assistant')}</b></div>", 
                       unsafe_allow_html=True)
            
            # Check if this is a new message with full response data
            if 'output' in message and message.get('output') is not None:
                # Display output table if available
                if not message['output'].empty:
                    st.dataframe(message['output'])
                
                # Display explanation
                if message.get('explanation'):
                    st.markdown(message['explanation'])
                
                # Display plots if available in the message
                if message.get('plots') or message.get('plot_images'):
                    display_plots_from_message(message, message_index)
                
                # Expanders for additional details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if message.get('code'):
                        with st.expander("📝 Code"):
                            code_type = 'sql' if 'sql_query' in message else 'python'
                            st.code(message['code'], language=code_type)
                
                with col2:
                    if message.get('plan'):
                        with st.expander("📋 Plan"):
                            st.write(message['plan'])
                
                with col3:
                    if message.get('verbose'):
                        with st.expander("🔍 Verbose"):
                            st.text(message['verbose'])
            
            else:
                # This is a historical message, show code with run button
                if message.get('code'):
                    with st.expander("📝 Code", expanded=True):
                        code_type = 'sql' if st.session_state.data_type in ['sqlite', 'db'] else 'python'
                        st.code(message['code'], language=code_type)
                        
                        if st.button(f"▶️ Run Again", key=f"run_{message_index}"):
                            result = execute_code_on_current_data(message['code'], code_type)
                            if result.get('success'):
                                st.success("✅ Code executed successfully!")
                                st.dataframe(result['result'])
                            else:
                                st.error(f"❌ Error: {result.get('error')}")

# Sidebar
with st.sidebar:
    # Logo placeholder
    st.image("ChatGPT Image Apr 29, 2025, 01_12_56 PM.png", width=200, caption="Experimental")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your data",
        type=['csv', 'xlsx', 'db', 'sqlite'],
        help="Upload CSV, Excel, or SQLite database files"
    )
    
    if uploaded_file:
        df, data_type = load_data_from_file(uploaded_file)
        if df is not None:
            st.session_state.uploaded_data = df
            st.session_state.data_type = data_type
            st.session_state.uploaded_filename = uploaded_file.name
            
            if isinstance(df, dict):
                total_rows = sum(table_df.shape[0] for table_df in df.values())
                st.success(f"✅ Loaded {len(df)} table(s) from {data_type.upper()} file: {total_rows:,} total rows")
                
                with st.expander("📋 Table Details"):
                    for name, table_df in df.items():
                        st.write(f"**{name}**: {table_df.shape[0]:,} rows × {table_df.shape[1]} columns")
            else:
                st.success(f"✅ Loaded {data_type.upper()} file: {df.shape[0]:,} rows")
    
    # Agent toggle
    st.markdown("### 🔍 Agent Selection")
    use_watson = st.toggle(
        "Use Watson (Blind Analysis)",
        value=st.session_state.use_watson,
        help="Toggle between Sherlock (with data preview) and Watson (blind analysis)"
    )
    st.session_state.use_watson = use_watson
    
    current_agent = "Watson" if use_watson else "Sherlock"
    agent_mode = "Python" if st.session_state.data_type in ['csv', 'xlsx'] else "SQL"
    st.info(f"Using: **{current_agent} ({agent_mode})**")
    
    st.markdown("---")
    
    # Chat management
    st.markdown("### 💬 Chat Sessions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            if st.session_state.uploaded_data is not None:
                # Clean up previous agent if exists
                if st.session_state.agent:
                    cleanup_agent()
                
                # Create a meaningful title with filename
                filename = getattr(st.session_state, 'uploaded_filename', 'Unknown')
                # Remove file extension for cleaner title
                file_base_name = os.path.splitext(filename)[0]
                timestamp = datetime.now().strftime('%m/%d %H:%M')
                
                session_title = f"{file_base_name} - {timestamp}"
                
                session_id = create_new_session(
                    title=session_title,
                    data_type=st.session_state.data_type or "Mixed"
                )
                st.session_state.current_session_id = session_id
                st.session_state.messages = []
                # Create agent with proper data handling
                st.session_state.agent = create_agent(
                    st.session_state.uploaded_data,
                    st.session_state.data_type,
                    st.session_state.use_watson
                )
                st.rerun()
            else:
                st.warning("Please upload data first")
    
    # List previous chats
    sessions = get_sessions()
    
    st.markdown("#### Previous Chats")
    for session in sessions:
        with st.container():
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                if st.button(
                    f"📝 {session['title']}", 
                    key=f"session_{session['session_id']}",
                    use_container_width=True
                ):
                    st.session_state.current_session_id = session['session_id']
                    # Load messages for this session
                    messages = get_session_messages(session['session_id'])
                    st.session_state.messages = messages
                    st.rerun()
            
            with col2:
                if st.button("✏️", key=f"edit_{session['session_id']}"):
                    # Show rename dialog
                    new_name = st.text_input(
                        "New name:", 
                        value=session['title'],
                        key=f"rename_{session['session_id']}"
                    )
                    if st.button("Save", key=f"save_{session['session_id']}"):
                        update_session_title(session['session_id'], new_name)
                        st.rerun()
            
            with col3:
                if st.button("🗑️", key=f"delete_{session['session_id']}"):
                    delete_session(session['session_id'])
                    if st.session_state.current_session_id == session['session_id']:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()

# Main area
if st.session_state.uploaded_data is not None:
    # Display data preview
    with st.expander("📊 Data Overview", expanded=True):
        display_dataframe_preview(st.session_state.uploaded_data)
    
    # Chat interface
    st.markdown("## 💬 Chat")
    
    # Display messages with plot support
    for i, message in enumerate(st.session_state.messages):
        display_message_with_plots(message, i, st.session_state.agent)
    
    # Chat input
    if st.session_state.current_session_id:
        query = st.chat_input("Ask a question about your data...")
        
        if query:
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'content': query
            })
            
            # Save user message
            save_message(
                st.session_state.current_session_id,
                len(st.session_state.messages) - 1,
                'user',
                query
            )
            # st.rerun()  # ← ADD THIS LINE
            # Display user message immediately using container
            user_container = st.container()
            with user_container:
                st.markdown(f"<div class='user-message'>🧑 {query}</div>", unsafe_allow_html=True)
            # Get agent response
            if st.session_state.agent:
                with st.spinner(f"🔍 {current_agent} is analyzing..."):
                    try:
                        # CHANGE: Clear previous plots before new query
                        clear_agent_plots(st.session_state.agent)
                        
                        # Get response as dictionary (this includes plots)
                        response = st.session_state.agent.query(query)
                        
                        # CHANGE: Extract only NEW plots from this specific query
                        current_plots = response.get('plots', [])
                        current_plot_images = response.get('plot_images', [])
                        
                        # Process response with only current plots
                        assistant_message = {
                            'role': 'assistant',
                            'agent_type': f"{current_agent} ({agent_mode})",
                            'content': response.get('final_result', ''),
                            'output': response.get('output'),
                            'code': response.get('code'),
                            'explanation': response.get('explanation'),
                            'plan': response.get('plan'),
                            'verbose': response.get('verbose'),
                            'plots': current_plots,  # Only current plots
                            'plot_images': current_plot_images if not current_plots else []  # Only if no plots dict
                        }
                        
                        st.session_state.messages.append(assistant_message)
                        
                        # Save assistant message
                        save_message(
                            st.session_state.current_session_id,
                            len(st.session_state.messages) - 1,
                            'assistant',
                            response.get('final_result', ''),
                            response.get('code'),
                            f"{current_agent} ({agent_mode})"
                        )
                        
                        # st.rerun()
                        # Display assistant message immediately
                        with st.container():
                            display_message_with_plots(assistant_message, len(st.session_state.messages) - 1, st.session_state.agent)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Agent not initialized. Please create a new chat.")
    else:
        st.info("👈 Please create a new chat session to start")
        
else:
    # Show notification at app startup
    st.info("""
    📱 **For the best experience:** Go to the three dots menu (⋮) in the top right → Settings → Theme → Dark mode
    
    This app is optimized for dark mode!
    """)
    # No data uploaded
    st.markdown("""
    # 🔍 Talk2Table
    
    Welcome to the Talk2Table Chat interface! This tool allows you to:
    
    - 📊 Upload CSV, Excel, or SQLite database files
    - 🤖 Choose between Sherlock (with data preview) or Watson (blind analysis) agents
    - 💬 Have natural language conversations about your data
    - 📈 Get insights, visualizations, and code explanations
    - 💾 Save and revisit previous chat sessions
    - 📊 **Generate and view plots automatically**
    
    **To get started:**
    1. Upload a data file using the sidebar
    2. Choose your preferred agent (Sherlock or Watson)
    3. Create a new chat session
    4. Start asking questions about your data!
    
    """)
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - "What are the top 5 categories by revenue?"
        - "Show me the trend of sales over time"
        - "Find correlations between different columns"
        - "Calculate the average price by product type"
        - "Which customers have the highest order values?"
        - "Create a summary statistics table"
        - "Plot a bar chart of sales by region"
        - "Show me a scatter plot of price vs quantity"
        - "Create a histogram of customer ages"
        """)

# Add some utility functions for debugging plots
if st.session_state.agent and st.sidebar.button("🔍 Debug Agent Plots"):
    plots = get_plots_from_agent_state(st.session_state.agent)
    if plots:
        st.sidebar.write(f"Found {len(plots)} plots in agent state")
        for i, plot in enumerate(plots):
            st.sidebar.write(f"Plot {i+1}: Success={plot.get('success', False)}")
    else:
        st.sidebar.write("No plots found in agent state")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🔍 Talk2Table - Powered by Sherlock & Watson Agents | 
    📊 Supports CSV, Excel, and SQLite | 
    🎨 Automatic Plot Generation
</div>
""", unsafe_allow_html=True)
