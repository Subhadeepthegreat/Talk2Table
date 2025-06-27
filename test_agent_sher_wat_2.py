"""
Enhanced Data Analysis Agents: Sherlock and Watson
- Sherlock: Full data access with preview
- Watson: Blind analysis without data access
- Both support Python and SQL variants
- All agents support multi-table operations and ReAct cycles
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import re
import ast
import sys
import sqlite3
import traceback
from io import StringIO
from typing import Dict, Any, List, Tuple, TypedDict, Annotated, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# SHARED COMPONENTS
# =============================================================================

class ResultFormatter:
    """Format analysis results in user-friendly way"""
    
    @staticmethod
    def convert_to_dataframe(result: Any) -> pd.DataFrame:
        """Convert any result type to a DataFrame for display"""
        if result is None:
            return pd.DataFrame({"Result": ["No data returned"]})
        
        if isinstance(result, pd.DataFrame):
            return result
        
        if isinstance(result, pd.Series):
            return result.to_frame()
        
        if isinstance(result, dict):
            try:
                return pd.DataFrame([result])
            except:
                return pd.json_normalize(result)
        
        if isinstance(result, list):
            if len(result) == 0:
                return pd.DataFrame({"Result": ["Empty list"]})
            
            if all(isinstance(item, dict) for item in result):
                return pd.DataFrame(result)
            
            return pd.DataFrame({"Values": result})
        
        if isinstance(result, (int, float, str, bool)):
            return pd.DataFrame({"Result": [result]})
        
        return pd.DataFrame({"Result": [str(result)]})
    
    @staticmethod
    def format_dataframe_result(df_result: pd.DataFrame, max_rows: int = 10) -> str:
        """Format DataFrame results for display"""
        if df_result is None:
            return "No data returned"
        
        if len(df_result) == 0:
            return "Empty result set"
        
        shape_info = f"📊 **Result Shape:** {df_result.shape[0]} rows × {df_result.shape[1]} columns\n\n"
        
        if len(df_result) <= max_rows:
            data_display = df_result.to_string(index=True, max_cols=10)
        else:
            data_display = df_result.head(max_rows).to_string(index=True, max_cols=10)
            data_display += f"\n... ({len(df_result) - max_rows} more rows)"
        
        return shape_info + "```\n" + data_display + "\n```"

# =============================================================================
# SHARED STATE DEFINITIONS
# =============================================================================

class BaseAgentState(TypedDict, total=False):
    """Base state shared by all agents"""
    file_name: str
    file_type: str
    schema: str
    table_shape: tuple[int, int]
    
    messages: Annotated[List[AnyMessage], add_messages]
    code: str
    execution_result: str
    plots: List[str]
    error: str
    plan: str
    step: int
    max_steps: int
    
    # Enhanced state
    analysis_summary: str
    is_complete: bool
    user_friendly_result: str
    progress_status: str
    verbose: bool
    retry_count: int
    max_retries: int
    
    # New fields for structured output
    final_output: Any
    explanation: str
    raw_verbose: str
    
    # ReAct cycle tracking
    react_history: List[Dict[str, Any]]
    verification_status: str

# =============================================================================
# PYTHON AGENT COMPONENTS
# =============================================================================

class PythonAgentState(BaseAgentState):
    """State for Python-based agents (Sherlock and Watson)"""
    df: pd.DataFrame  # Single DataFrame for backward compatibility
    dataframes: Dict[str, pd.DataFrame]  # Multiple DataFrames support
    preview_md: str  # Only for Sherlock
    column_info: Dict[str, str]  # For Watson
    python: str  # Executed Python code
    
    # Watson-specific
    error_history: List[str]
    successful_patterns: List[str]
    current_strategy: str

class SafeCodeExecutor:
    """Safe code execution with restricted imports and operations"""
    
    ALLOWED_IMPORTS = {
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scipy', 'sklearn',
        'math', 'statistics', 'datetime', 'collections', 'itertools', 're', 'json'
    }
    
    FORBIDDEN_PATTERNS = [
        r'import\s+os', r'import\s+sys', r'import\s+subprocess', r'import\s+requests',
        r'open\s*\(', r'exec\s*\(', r'eval\s*\(', r'__import__',
        r'getattr', r'setattr', r'delattr', r'globals\s*\(', r'locals\s*\(',
        r'vars\s*\(', r'dir\s*\(',
    ]
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame], watson_mode: bool = False):
        """
        Initialize executor with multiple dataframes support
        
        Args:
            dataframes: Dictionary of dataframe names to DataFrames
            watson_mode: If True, enforce Watson's blind analysis restrictions
        """
        self.dataframes = {name: df.copy() for name, df in dataframes.items()}
        self.watson_mode = watson_mode
        
        # Watson-specific forbidden patterns
        if watson_mode:
            self.FORBIDDEN_PATTERNS.extend([
                r'\.head\s*\(', r'\.tail\s*\(', r'\.sample\s*\(',
                r'\.iloc\s*\[.*:.*\]', r'\.loc\s*\[.*:.*\]',
                r'print\s*\(\s*df', r'display\s*\(\s*df',
            ])
    
    def is_code_safe(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden operation detected: {pattern}"
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in self.ALLOWED_IMPORTS:
                        return False, f"Import not allowed: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.ALLOWED_IMPORTS:
                    return False, f"Import not allowed: {node.module}"
        
        return True, "Code is safe"
    
    def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Safely execute code with timeout and capture output"""
        is_safe, safety_msg = self.is_code_safe(code)
        if not is_safe:
            return {
                'success': False,
                'error': f"Security check failed: {safety_msg}",
                'output': '',
                'result': None,
                'error_type': 'security'
            }
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            safe_builtins = {
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
                'sorted': sorted, 'reversed': reversed, 'print': print,
                'type': type, 'isinstance': isinstance, 'hasattr': hasattr
            }
            
            safe_globals = {
                '__builtins__': safe_builtins,
                'pd': pd,
                'dataframes': self.dataframes,  # Multiple dataframes
                'df': self.dataframes.get('main', pd.DataFrame()),  # Backward compatibility
            }
            
            # Add optional libraries
            try:
                import numpy as np
                safe_globals['np'] = np
            except ImportError:
                pass
            
            try:
                import matplotlib.pyplot as plt
                safe_globals['plt'] = plt
            except ImportError:
                pass
                
            try:
                import seaborn as sns
                safe_globals['sns'] = sns
            except ImportError:
                pass
            
            safe_locals = {}
            exec(code, safe_globals, safe_locals)
            result = safe_locals.get('result')
            output = captured_output.getvalue()
            
            return {
                'success': True,
                'error': None,
                'output': output,
                'result': result,
                'error_type': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'output': captured_output.getvalue(),
                'result': None,
                'error_type': type(e).__name__
            }
        finally:
            sys.stdout = old_stdout

# =============================================================================
# SQL AGENT COMPONENTS
# =============================================================================

class SQLAgentState(BaseAgentState):
    """State for SQL-based agents (Sherlock and Watson)"""
    table_name: str  # Single table for backward compatibility
    table_names: List[str]  # Multiple tables support
    db_connection: sqlite3.Connection
    preview_md: str  # Only for Sherlock
    column_info: Dict[str, str]  # For Watson
    sql_query: str  # Executed SQL query
    
    # Watson-specific
    error_history: List[str]
    successful_patterns: List[str]
    current_strategy: str

class SafeSQLExecutor:
    """Safe SQL execution with restricted operations"""
    
    FORBIDDEN_KEYWORDS = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE INDEX',
        'TRUNCATE', 'REPLACE', 'PRAGMA', 'ATTACH', 'DETACH'
    ]
    
    def __init__(self, db_connection: sqlite3.Connection, table_names: List[str], watson_mode: bool = False):
        """
        Initialize executor with multiple tables support
        
        Args:
            db_connection: SQLite database connection
            table_names: List of table names available for querying
            watson_mode: If True, enforce Watson's blind analysis restrictions
        """
        self.db_connection = db_connection
        self.table_names = table_names
        self.watson_mode = watson_mode
        
        # Watson-specific forbidden patterns
        self.FORBIDDEN_PATTERNS = []
        if watson_mode:
            self.FORBIDDEN_PATTERNS = [
                r'SELECT\s+\*\s+FROM\s+\w+\s*;?\s*$',  # SELECT * without WHERE/LIMIT
                r'SELECT\s+.*\s+FROM\s+\w+\s+LIMIT\s+[5-9]\d*',  # Large LIMIT values
                r'SELECT\s+.*\s+FROM\s+\w+\s+ORDER BY\s+RANDOM\(\)',  # Random sampling
            ]
    
    def is_sql_safe(self, sql: str) -> tuple[bool, str]:
        """Check if SQL query is safe to execute"""
        sql_upper = sql.upper().strip()
        
        # Remove comments and normalize whitespace
        sql_clean = re.sub(r'--.*$', '', sql_upper, flags=re.MULTILINE)
        sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
        sql_clean = ' '.join(sql_clean.split())
        
        # Check for forbidden keywords
        for forbidden in self.FORBIDDEN_KEYWORDS:
            if re.search(r'\b' + forbidden + r'\b', sql_clean):
                return False, f"Forbidden operation detected: {forbidden}"
        
        # Ensure query starts with SELECT or WITH
        if not sql_clean.startswith('SELECT') and not sql_clean.startswith('WITH'):
            return False, "Only SELECT queries and CTEs are allowed"
        
        # Check Watson-specific patterns
        if self.watson_mode:
            for pattern in self.FORBIDDEN_PATTERNS:
                if re.search(pattern, sql, re.IGNORECASE):
                    return False, f"Data peeking pattern detected - Watson cannot see raw data"
        
        # Check for multiple statements
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        if len(statements) > 1:
            return False, "Multiple statements not allowed"
        
        return True, "SQL query is safe"
    
    def execute_sql(self, sql: str, return_df: bool = True) -> Dict[str, Any]:
        """Safely execute SQL query and return results"""
        is_safe, safety_msg = self.is_sql_safe(sql)
        if not is_safe:
            return {
                'success': False,
                'error': f"Security check failed: {safety_msg}",
                'result': None,
                'row_count': 0,
                'error_type': 'security'
            }
        
        try:
            if return_df:
                result_df = pd.read_sql_query(sql, self.db_connection)
                return {
                    'success': True,
                    'error': None,
                    'result': result_df,
                    'row_count': len(result_df),
                    'error_type': None
                }
            else:
                cursor = self.db_connection.cursor()
                cursor.execute(sql)
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return {
                    'success': True,
                    'error': None,
                    'result': {'data': results, 'columns': columns},
                    'row_count': len(results),
                    'error_type': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'result': None,
                'row_count': 0,
                'error_type': type(e).__name__
            }

# =============================================================================
# REACT CYCLE MANAGER
# =============================================================================

class ReactCycleManager:
    """Manages ReAct cycles for error correction and verification"""
    
    def __init__(self, max_retries: int = 3, enable_verification: bool = True):
        self.max_retries = max_retries
        self.enable_verification = enable_verification
        self.history = []
    
    def should_retry(self, state: BaseAgentState) -> bool:
        """Determine if another retry should be attempted"""
        retry_count = state.get("retry_count", 0)
        has_error = state.get("error") is not None
        
        return has_error and retry_count < self.max_retries
    
    def should_verify(self, state: BaseAgentState) -> bool:
        """Determine if verification step should be performed"""
        if not self.enable_verification:
            return False
        
        has_result = state.get("final_output") is not None
        not_verified = state.get("verification_status") != "verified"
        no_error = state.get("error") is None
        
        return has_result and not_verified and no_error
    
    def create_retry_instruction(self, error: str, error_type: str, context: str) -> str:
        """Create instruction for retry based on error"""
        return f"""
ERROR ANALYSIS AND CORRECTION REQUIRED:
- Error: {error}
- Error Type: {error_type}
- Context: {context}

INSTRUCTIONS:
1. Carefully analyze the error message
2. Identify the root cause of the failure
3. Adjust your approach to avoid this error
4. Provide a corrected solution
5. Explain what you changed and why

Please provide the corrected code/query below:
"""
    
    def create_verification_instruction(self, result: Any, code: str) -> str:
        """Create instruction for verification step"""
        return f"""
VERIFICATION STEP - Please verify your previous result:

Your code/query produced a result. Please:
1. Review the logic of your solution
2. Check if the result makes sense given the question
3. Verify edge cases are handled
4. Confirm the output format is appropriate
5. If any issues found, provide corrected code; otherwise confirm the result

Previous code/query:
```
{code}
```

Result summary: {type(result).__name__} with shape/size {getattr(result, 'shape', 'N/A')}

Please verify and respond with either:
- "VERIFIED: [explanation]" if correct
- Corrected code if issues found
"""
    
    def update_history(self, state: BaseAgentState, action: str):
        """Update ReAct history"""
        self.history.append({
            'step': state.get('step', 0),
            'action': action,
            'error': state.get('error'),
            'retry_count': state.get('retry_count', 0),
            'verification_status': state.get('verification_status'),
            'timestamp': pd.Timestamp.now()
        })

# =============================================================================
# PYTHON AGENT NODES
# =============================================================================

def create_python_assistant_node(sherlock_mode: bool = True):
    """Factory function to create Python assistant nodes for Sherlock or Watson"""
    
    def assistant_node(state: PythonAgentState) -> dict:
        """Python assistant node with multi-table support and ReAct cycles"""
        
        verbose = state.get("verbose", False)
        agent_name = "SHERLOCK" if sherlock_mode else "WATSON"
        
        if verbose:
            print(f"\n🔍 {agent_name} PYTHON STEP {state.get('step', 0) + 1}: Starting analysis...")
        
        # Initialize ReAct manager
        react_manager = ReactCycleManager(
            max_retries=state.get("max_retries", 3),
            enable_verification=True
        )
        
        # Check if we should verify previous result
        if react_manager.should_verify(state) and state.get("final_output") is not None:
            verification_msg = react_manager.create_verification_instruction(
                state.get("final_output"),
                state.get("python", "")
            )
            state["messages"].append(HumanMessage(content=verification_msg))
            react_manager.update_history(state, "verification")
        
        msgs = state["messages"].copy()
        current_step = state.get("step", 0)
        max_steps = state.get("max_steps", 10)
        
        # Build system prompt based on agent type
        if sherlock_mode:
            sys_prompt = _build_sherlock_python_prompt(state)
        else:
            sys_prompt = _build_watson_python_prompt(state)
        
        # Add system message if not present
        if not any(m.type == "system" for m in msgs):
            msgs.insert(0, SystemMessage(content=sys_prompt))
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini")
            assistant_reply = llm.invoke(msgs)
            
            # Initialize output variables
            user_friendly_result = ""
            execution_result = ""
            error = None
            executed_code = ""
            final_output = None
            explanation = ""
            verification_status = state.get("verification_status", "pending")
            
            # Check for verification confirmation
            if "VERIFIED:" in assistant_reply.content:
                verification_status = "verified"
                explanation = re.search(r"VERIFIED:\s*(.*)", assistant_reply.content, re.DOTALL).group(1)
                return {
                    "messages": [assistant_reply],
                    "verification_status": verification_status,
                    "explanation": explanation,
                    "step": current_step + 1
                }
            
            # Extract and execute code
            code_blocks = re.findall(r'```python\n(.*?)\n```', assistant_reply.content, re.DOTALL)
            
            if code_blocks and state.get("dataframes"):
                executor = SafeCodeExecutor(state["dataframes"], watson_mode=not sherlock_mode)
                
                for code_block in code_blocks:
                    executed_code = code_block.strip()
                    exec_result = executor.execute_code(executed_code)
                    
                    if exec_result['success']:
                        final_output = exec_result.get('result')
                        user_friendly_result = ResultFormatter.format_dataframe_result(final_output) if isinstance(final_output, pd.DataFrame) else f"✅ **Code executed successfully**"
                        execution_result = str(final_output) if final_output is not None else ""
                        
                        # Extract explanation
                        explanation_match = re.search(r'\*\*Explanation\*\*:?(.*?)(?=\*\*|$)', 
                                                    assistant_reply.content, re.DOTALL | re.IGNORECASE)
                        if explanation_match:
                            explanation = explanation_match.group(1).strip()
                    else:
                        error = exec_result['error']
                        error_type = exec_result.get('error_type', 'unknown')
                        
                        # Handle retry logic
                        if react_manager.should_retry(state):
                            retry_instruction = react_manager.create_retry_instruction(
                                error, error_type, 
                                f"{agent_name} Python analysis"
                            )
                            
                            # Recursive retry
                            state["messages"].append(assistant_reply)
                            state["messages"].append(HumanMessage(content=retry_instruction))
                            state["retry_count"] = state.get("retry_count", 0) + 1
                            state["error"] = error
                            
                            react_manager.update_history(state, "retry")
                            return assistant_node(state)
                        
                        user_friendly_result = f"❌ **Final Error:** {error}"
                        break
            
            # Build response
            raw_verbose = _build_verbose_log(agent_name, "PYTHON", state, assistant_reply, executed_code, execution_result, error)
            
            full_response = assistant_reply.content
            if user_friendly_result:
                full_response += f"\n\n{user_friendly_result}"
            
            return {
                "messages": [AIMessage(content=full_response)],
                "step": current_step + 1,
                "python": executed_code,
                "execution_result": execution_result,
                "user_friendly_result": user_friendly_result,
                "error": error,
                "final_output": final_output,
                "explanation": explanation or user_friendly_result,
                "raw_verbose": raw_verbose,
                "verification_status": verification_status,
                "react_history": react_manager.history
            }
            
        except Exception as e:
            error_msg = f"{agent_name} assistant error: {str(e)}"
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
                "step": current_step + 1,
                "error": error_msg
            }
    
    return assistant_node

# =============================================================================
# SQL AGENT NODES
# =============================================================================

def create_sql_assistant_node(sherlock_mode: bool = True):
    """Factory function to create SQL assistant nodes for Sherlock or Watson"""
    
    def assistant_node(state: SQLAgentState) -> dict:
        """SQL assistant node with multi-table support and ReAct cycles"""
        
        verbose = state.get("verbose", False)
        agent_name = "SHERLOCK" if sherlock_mode else "WATSON"
        
        if verbose:
            print(f"\n🔍 {agent_name} SQL STEP {state.get('step', 0) + 1}: Starting analysis...")
        
        # Initialize ReAct manager
        react_manager = ReactCycleManager(
            max_retries=state.get("max_retries", 3),
            enable_verification=True
        )
        
        # Check if we should verify previous result
        if react_manager.should_verify(state) and state.get("final_output") is not None:
            verification_msg = react_manager.create_verification_instruction(
                state.get("final_output"),
                state.get("sql_query", "")
            )
            state["messages"].append(HumanMessage(content=verification_msg))
            react_manager.update_history(state, "verification")
        
        msgs = state["messages"].copy()
        current_step = state.get("step", 0)
        max_steps = state.get("max_steps", 10)
        
        # Build system prompt based on agent type
        if sherlock_mode:
            sys_prompt = _build_sherlock_sql_prompt(state)
        else:
            sys_prompt = _build_watson_sql_prompt(state)
        
        # Add system message if not present
        if not any(m.type == "system" for m in msgs):
            msgs.insert(0, SystemMessage(content=sys_prompt))
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini")
            assistant_reply = llm.invoke(msgs)
            
            # Initialize output variables
            user_friendly_result = ""
            execution_result = ""
            error = None
            sql_query = ""
            final_output = None
            explanation = ""
            verification_status = state.get("verification_status", "pending")
            
            # Check for verification confirmation
            if "VERIFIED:" in assistant_reply.content:
                verification_status = "verified"
                explanation = re.search(r"VERIFIED:\s*(.*)", assistant_reply.content, re.DOTALL).group(1)
                return {
                    "messages": [assistant_reply],
                    "verification_status": verification_status,
                    "explanation": explanation,
                    "step": current_step + 1
                }
            
            # Extract and execute SQL
            sql_blocks = re.findall(r'```sql\n(.*?)\n```', assistant_reply.content, re.DOTALL)
            
            if sql_blocks and state.get("db_connection"):
                executor = SafeSQLExecutor(
                    state["db_connection"], 
                    state.get("table_names", [state.get("table_name", "data_table")]),
                    watson_mode=not sherlock_mode
                )
                
                for sql_block in sql_blocks:
                    sql_query = sql_block.strip()
                    exec_result = executor.execute_sql(sql_query)
                    
                    if exec_result['success']:
                        final_output = exec_result.get('result')
                        user_friendly_result = _format_sql_result(exec_result, sql_query, sherlock_mode)
                        execution_result = str(final_output) if final_output is not None else ""
                        
                        # Extract explanation
                        explanation_match = re.search(r'\*\*Explanation\*\*:?(.*?)(?=\*\*|$)', 
                                                    assistant_reply.content, re.DOTALL | re.IGNORECASE)
                        if explanation_match:
                            explanation = explanation_match.group(1).strip()
                    else:
                        error = exec_result['error']
                        error_type = exec_result.get('error_type', 'unknown')
                        
                        # Handle retry logic
                        if react_manager.should_retry(state):
                            retry_instruction = react_manager.create_retry_instruction(
                                error, error_type,
                                f"{agent_name} SQL analysis"
                            )
                            
                            # Recursive retry
                            state["messages"].append(assistant_reply)
                            state["messages"].append(HumanMessage(content=retry_instruction))
                            state["retry_count"] = state.get("retry_count", 0) + 1
                            state["error"] = error
                            
                            react_manager.update_history(state, "retry")
                            return assistant_node(state)
                        
                        user_friendly_result = f"❌ **Final SQL Error:** {error}"
                        break
            
            # Build response
            raw_verbose = _build_verbose_log(agent_name, "SQL", state, assistant_reply, sql_query, execution_result, error)
            
            full_response = assistant_reply.content
            if user_friendly_result:
                full_response += f"\n\n{user_friendly_result}"
            
            return {
                "messages": [AIMessage(content=full_response)],
                "step": current_step + 1,
                "sql_query": sql_query,
                "execution_result": execution_result,
                "user_friendly_result": user_friendly_result,
                "error": error,
                "final_output": final_output,
                "explanation": explanation or user_friendly_result,
                "raw_verbose": raw_verbose,
                "verification_status": verification_status,
                "react_history": react_manager.history
            }
            
        except Exception as e:
            error_msg = f"{agent_name} SQL assistant error: {str(e)}"
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
                "step": current_step + 1,
                "error": error_msg
            }
    
    return assistant_node

# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def _build_sherlock_python_prompt(state: PythonAgentState) -> str:
    """Build system prompt for Sherlock Python agent"""
    
    # Get info about available dataframes
    df_info = []
    for name, df in state.get("dataframes", {}).items():
        df_info.append(f"- '{name}': {df.shape[0]} rows × {df.shape[1]} columns")
    
    dataframes_str = "\n".join(df_info) if df_info else "No dataframes available"
    
    return f"""You are Sherlock, a data analysis assistant with full data access.

Available DataFrames:
{dataframes_str}

Schema Information:
{state.get("schema", "Not available")}

Preview (first 5 rows of main dataframe):
{state.get("preview_md", "Not available")}

IMPORTANT GUIDELINES:
1. Access dataframes using: dataframes['name'] or df for the main dataframe
2. Write clear, well-commented Python code
3. ALWAYS assign your final result to a variable called `result`
4. Format your response with these sections:
   - **Plan**: Explain your approach
   - **Code**: Show the executable Python code
   - **Explanation**: Interpret the results and their meaning
5. When working with multiple tables, explain relationships and joins
6. Handle edge cases and potential errors gracefully
7. For plotting: use matplotlib/seaborn and save to result if needed

Available libraries:
- pandas as pd
- numpy as np (if available)
- matplotlib.pyplot as plt (if available)
- seaborn as sns (if available)

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Retry count: {state.get('retry_count', 0)}/{state.get('max_retries', 3)}"""

def _build_watson_python_prompt(state: PythonAgentState) -> str:
    """Build system prompt for Watson Python agent"""
    
    # Get schema info without revealing data
    df_info = []
    for name, df in state.get("dataframes", {}).items():
        col_info = []
        for col in df.columns:
            dtype_info = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            col_info.append(f"    - {col}: {dtype_info}, {null_count} nulls")
        
        df_info.append(f"- '{name}': {df.shape[0]} rows × {df.shape[1]} columns\n" + "\n".join(col_info))
    
    dataframes_str = "\n".join(df_info) if df_info else "No dataframes available"
    
    return f"""You are Watson, a blind data analysis assistant. You cannot see raw data values.

CRITICAL CONSTRAINTS:
1. NO data peeking: no .head(), .tail(), .sample(), print(df), or slicing
2. Use only aggregations, statistics, and structural operations
3. Learn from errors to refine your approach
4. Make educated guesses based on column names and types

Available DataFrames (structure only):
{dataframes_str}

Column Information:
{state.get("column_info", "Not available")}

IMPORTANT GUIDELINES:
1. Access dataframes using: dataframes['name'] or df for the main dataframe
2. Use defensive coding to handle potential issues
3. ALWAYS assign your final result to a variable called `result`
4. Format your response with these sections:
   - **Plan**: Explain your blind analysis approach
   - **Code**: Show the executable Python code
   - **Explanation**: Interpret what the results reveal
5. Use .shape, .dtypes, .columns, .info() for structure
6. Use .isnull(), .describe(), .value_counts() for analysis
7. When errors occur, analyze them to learn about the data

Error Learning Strategy:
- KeyError: column name might be different
- TypeError: data type assumption incorrect
- ValueError: data format unexpected

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Previous errors: {state.get('error_history', [])[-3:]}
Current strategy: {state.get('current_strategy', 'Initial exploration')}"""

def _build_sherlock_sql_prompt(state: SQLAgentState) -> str:
    """Build system prompt for Sherlock SQL agent"""
    
    tables_str = ", ".join(state.get("table_names", [state.get("table_name", "data_table")]))
    
    return f"""You are Sherlock, a SQL data analysis assistant with full data access.

Available Tables: {tables_str}

Database Schema:
{state.get("schema", "Not available")}

Preview (first 5 rows):
{state.get("preview_md", "Not available")}

IMPORTANT GUIDELINES:
1. Write efficient SQL queries using available tables
2. Use only SELECT statements - no modifications
3. Format SQL queries in ```sql code blocks
4. Format your response with these sections:
   - **Plan**: Explain your SQL approach
   - **Query**: Show the SQL query
   - **Explanation**: Interpret the results
5. Use JOINs when working with multiple tables
6. Handle NULL values and edge cases properly
7. Use CTEs for complex queries
8. Optimize for performance with appropriate indexes

Available SQL features:
- All SELECT operations
- Aggregate functions: COUNT, SUM, AVG, MIN, MAX
- Window functions
- CTEs (WITH clauses)
- JOINs (INNER, LEFT, RIGHT, FULL)
- String and date functions

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Retry count: {state.get('retry_count', 0)}/{state.get('max_retries', 3)}"""

def _build_watson_sql_prompt(state: SQLAgentState) -> str:
    """Build system prompt for Watson SQL agent"""
    
    tables_str = ", ".join(state.get("table_names", [state.get("table_name", "data_table")]))
    
    return f"""You are Watson, a blind SQL analysis assistant. You cannot query raw data directly.

CRITICAL CONSTRAINTS:
1. NO raw data queries: no SELECT *, no large LIMIT, no sampling
2. Use only aggregations: COUNT, SUM, AVG, GROUP BY, etc.
3. Maximum LIMIT 20 for TOP N queries only
4. Learn from SQL errors to refine queries

Available Tables: {tables_str}

Database Schema:
{state.get("schema", "Not available")}

Column Information:
{state.get("column_info", "Not available")}

IMPORTANT GUIDELINES:
1. Write aggregation-focused SQL queries
2. Format SQL queries in ```sql code blocks
3. Format your response with these sections:
   - **Plan**: Explain your blind analysis approach
   - **Query**: Show the SQL query
   - **Explanation**: Interpret the aggregate results
4. Use COUNT(*) to understand data volume
5. Use GROUP BY for categorical analysis
6. Use statistical functions for numeric columns
7. Handle NULLs with COALESCE or filters

SQL Error Learning:
- Column name errors: verify from schema
- Type errors: adjust data type assumptions
- Syntax errors: check SQL structure

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Previous errors: {state.get('error_history', [])[-3:]}
Current strategy: {state.get('current_strategy', 'Initial exploration')}"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_sql_result(exec_result: Dict[str, Any], sql_query: str, sherlock_mode: bool) -> str:
    """Format SQL execution results based on agent type"""
    if not exec_result['success']:
        return f"❌ **SQL Error:** {exec_result['error']}"
    
    output_parts = []
    output_parts.append(f"🔍 **Executed Query:**\n```sql\n{sql_query}\n```")
    
    result = exec_result['result']
    row_count = exec_result.get('row_count', 0)
    
    if result is not None:
        if isinstance(result, pd.DataFrame):
            if sherlock_mode or (result.shape[0] <= 10 and result.shape[1] <= 3):
                # Show data for Sherlock or small aggregated results for Watson
                output_parts.append(ResultFormatter.format_dataframe_result(result))
            else:
                # Watson sees only metadata
                shape_info = f"📊 **Query Result:** {result.shape[0]} rows × {result.shape[1]} columns"
                cols_info = f"**Columns:** {list(result.columns)}"
                output_parts.append(f"{shape_info}\n{cols_info}")
    
    output_parts.append(f"📈 **Rows returned:** {row_count:,}")
    return "\n\n".join(output_parts)

def _build_verbose_log(agent_name: str, agent_type: str, state: Dict, reply: Any, 
                      code: str, result: str, error: Optional[str]) -> str:
    """Build verbose log for debugging"""
    return "\n".join([
        f"{'='*50}",
        f"{agent_name} {agent_type} STEP {state.get('step', 0) + 1}/{state.get('max_steps', 10)}",
        f"{'='*50}",
        f"USER QUERY: {state['messages'][-1].content if state['messages'] else 'N/A'}",
        f"ASSISTANT RESPONSE: {reply.content}",
        f"EXECUTED CODE: {code}",
        f"EXECUTION RESULT: {result}",
        f"ERROR: {error or 'None'}",
        f"RETRY COUNT: {state.get('retry_count', 0)}",
        f"VERIFICATION STATUS: {state.get('verification_status', 'pending')}",
        f"{'='*50}"
    ])

def check_termination_condition(state: BaseAgentState) -> bool:
    """Check if the agent should terminate"""
    if state.get("step", 0) >= state.get("max_steps", 10):
        return True
    
    if state.get("is_complete", False):
        return True
    
    if state.get("verification_status") == "verified":
        return True
    
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and hasattr(last_message, 'content'):
        completion_phrases = [
            "analysis complete", "task completed", "final result",
            "verification complete", "VERIFIED:"
        ]
        if any(phrase in last_message.content.lower() for phrase in completion_phrases):
            return True
    
    return False

# =============================================================================
# AGENT GRAPH BUILDERS
# =============================================================================

def create_python_agent_graph(sherlock_mode: bool = True):
    """Create Python agent graph with ReAct capabilities"""
    builder = StateGraph(PythonAgentState)
    
    # Create the appropriate assistant node
    assistant = create_python_assistant_node(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    
    builder.add_edge(START, "assistant")
    
    def should_continue(state: PythonAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()

def create_sql_agent_graph(sherlock_mode: bool = True):
    """Create SQL agent graph with ReAct capabilities"""
    builder = StateGraph(SQLAgentState)
    
    # Create the appropriate assistant node
    assistant = create_sql_assistant_node(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    
    builder.add_edge(START, "assistant")
    
    def should_continue(state: SQLAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()

# =============================================================================
# MAIN AGENT CLASSES
# =============================================================================

class TableAnalysisAgent:
    """Sherlock Python: Full data access agent with multi-table support"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None, 
                 dataframes: Optional[Dict[str, pd.DataFrame]] = None,
                 file_name: str = "uploaded_data", 
                 max_steps: int = 10, 
                 max_retries: int = 3,
                 verbose: bool = False):
        """
        Initialize Sherlock Python agent
        
        Args:
            df: Single DataFrame for backward compatibility
            dataframes: Dictionary of DataFrames for multi-table support
            file_name: Name of the file/dataset
            max_steps: Maximum conversation steps
            max_retries: Maximum retry attempts for errors
            verbose: Enable verbose logging
        """
        # Handle backward compatibility
        if dataframes is None and df is not None:
            dataframes = {"main": df}
        elif dataframes is None:
            raise ValueError("Either 'df' or 'dataframes' must be provided")
        
        self.dataframes = dataframes
        self.df = dataframes.get("main", list(dataframes.values())[0])  # For compatibility
        self.file_name = file_name
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.agent = create_python_agent_graph(sherlock_mode=True)
        
        # Build schema information
        schema_info = {}
        for name, df in dataframes.items():
            schema_info[name] = str(df.dtypes.to_dict())
        
        self.state = PythonAgentState(
            file_name=file_name,
            file_type="csv",
            schema=str(schema_info),
            preview_md=self.df.head().to_markdown(),
            table_shape=self.df.shape,
            df=self.df,
            dataframes=dataframes,
            messages=[],
            step=0,
            max_steps=max_steps,
            max_retries=max_retries,
            retry_count=0,
            is_complete=False,
            verbose=verbose,
            final_output=None,
            explanation="",
            raw_verbose="",
            verification_status="pending"
        )
    
    def query(self, question: str) -> str:
        """Query the data with natural language"""
        user_message = HumanMessage(content=question)
        self.state["messages"].append(user_message)
        
        try:
            result = self.agent.invoke(self.state)
            self.state.update(result)
            
            last_message = result["messages"][-1] if result["messages"] else None
            return last_message.content if last_message else "No response generated"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_output_dataframe(self) -> pd.DataFrame:
        """Get the final output as a DataFrame"""
        return ResultFormatter.convert_to_dataframe(self.state.get("final_output"))

class WatsonTableAnalysisAgent:
    """Watson Python: Blind analysis agent with multi-table support"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 dataframes: Optional[Dict[str, pd.DataFrame]] = None,
                 file_name: str = "uploaded_data", 
                 max_steps: int = 10,
                 max_retries: int = 3,
                 verbose: bool = False):
        """
        Initialize Watson Python agent
        
        Args:
            df: Single DataFrame for backward compatibility
            dataframes: Dictionary of DataFrames for multi-table support
            file_name: Name of the file/dataset
            max_steps: Maximum conversation steps
            max_retries: Maximum retry attempts for errors
            verbose: Enable verbose logging
        """
        # Handle backward compatibility
        if dataframes is None and df is not None:
            dataframes = {"main": df}
        elif dataframes is None:
            raise ValueError("Either 'df' or 'dataframes' must be provided")
        
        self.dataframes = dataframes
        self.df = dataframes.get("main", list(dataframes.values())[0])
        self.file_name = file_name
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.agent = create_python_agent_graph(sherlock_mode=False)
        
        # Extract schema without revealing data
        column_info = {}
        schema_info = {}
        for name, df in dataframes.items():
            schema_info[name] = str(df.dtypes.to_dict())
            column_info[name] = {}
            for col in df.columns:
                dtype_info = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                column_info[name][col] = f"dtype: {dtype_info}, null_count: {null_count}"
        
        self.state = PythonAgentState(
            file_name=file_name,
            file_type="csv",
            schema=str(schema_info),
            column_info=column_info,
            table_shape=self.df.shape,
            df=self.df,
            dataframes=dataframes,
            messages=[],
            step=0,
            max_steps=max_steps,
            max_retries=max_retries,
            retry_count=0,
            is_complete=False,
            verbose=verbose,
            error_history=[],
            successful_patterns=[],
            current_strategy="Initial blind exploration",
            final_output=None,
            explanation="",
            raw_verbose="",
            verification_status="pending"
        )
    
    def query(self, question: str) -> str:
        """Query with blind analysis approach"""
        user_message = HumanMessage(content=question)
        self.state["messages"].append(user_message)
        
        try:
            result = self.agent.invoke(self.state)
            self.state.update(result)
            
            last_message = result["messages"][-1] if result["messages"] else None
            return last_message.content if last_message else "No response generated"
            
        except Exception as e:
            return f"Error during blind analysis: {str(e)}"

class SQLTableAnalysisAgent:
    """Sherlock SQL: Full data access SQL agent with multi-table support"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 dataframes: Optional[Dict[str, pd.DataFrame]] = None,
                 file_name: str = "uploaded_data", 
                 table_name: Optional[str] = None,
                 table_names: Optional[Dict[str, str]] = None,
                 max_steps: int = 10,
                 max_retries: int = 3):
        """
        Initialize Sherlock SQL agent
        
        Args:
            df: Single DataFrame for backward compatibility
            dataframes: Dictionary of DataFrames for multi-table support
            file_name: Name of the file/dataset
            table_name: Single table name for backward compatibility
            table_names: Dictionary mapping df names to SQL table names
            max_steps: Maximum conversation steps
            max_retries: Maximum retry attempts for errors
        """
        # Handle backward compatibility
        if dataframes is None and df is not None:
            dataframes = {"main": df}
        elif dataframes is None:
            raise ValueError("Either 'df' or 'dataframes' must be provided")
        
        if table_names is None and table_name is not None:
            table_names = {"main": table_name}
        elif table_names is None:
            table_names = {name: f"table_{name}" for name in dataframes.keys()}
        
        self.dataframes = dataframes
        self.df = dataframes.get("main", list(dataframes.values())[0])
        self.file_name = file_name
        self.table_names = table_names
        self.max_steps = max_steps
        self.max_retries = max_retries
        
        # Create in-memory SQLite database
        self.db_connection = sqlite3.connect(':memory:')
        
        # Load all DataFrames into SQLite
        schema_parts = []
        for df_name, df in dataframes.items():
            sql_table_name = table_names.get(df_name, f"table_{df_name}")
            df.to_sql(sql_table_name, self.db_connection, index=False, if_exists='replace')
            
            # Get schema for this table
            cursor = self.db_connection.cursor()
            cursor.execute(f"PRAGMA table_info({sql_table_name})")
            schema_info = cursor.fetchall()
            schema_str = f"\nTable: {sql_table_name}\n"
            schema_str += "\n".join([f"  {col[1]} ({col[2]})" for col in schema_info])
            schema_parts.append(schema_str)
        
        self.agent = create_sql_agent_graph(sherlock_mode=True)
        
        self.state = SQLAgentState(
            file_name=file_name,
            file_type="csv",
            schema="\n".join(schema_parts),
            preview_md=self.df.head().to_markdown(),
            table_shape=self.df.shape,
            table_name=list(table_names.values())[0],  # For compatibility
            table_names=list(table_names.values()),
            db_connection=self.db_connection,
            messages=[],
            step=0,
            max_steps=max_steps,
            max_retries=max_retries,
            retry_count=0,
            is_complete=False,
            final_output=None,
            explanation="",
            raw_verbose="",
            verification_status="pending"
        )
    
    def query(self, question: str) -> str:
        """Query with SQL"""
        user_message = HumanMessage(content=question)
        self.state["messages"].append(user_message)
        
        try:
            result = self.agent.invoke(self.state)
            self.state.update(result)
            
            last_message = result["messages"][-1] if result["messages"] else None
            return last_message.content if last_message else "No response generated"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def close(self):
        """Clean up database connection"""
        if self.db_connection:
            self.db_connection.close()

class WatsonSQLTableAnalysisAgent:
    """Watson SQL: Blind SQL analysis agent with multi-table support"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 dataframes: Optional[Dict[str, pd.DataFrame]] = None,
                 file_name: str = "uploaded_data", 
                 table_name: Optional[str] = None,
                 table_names: Optional[Dict[str, str]] = None,
                 max_steps: int = 10,
                 max_retries: int = 3,
                 verbose: bool = False):
        """
        Initialize Watson SQL agent
        
        Args:
            df: Single DataFrame for backward compatibility
            dataframes: Dictionary of DataFrames for multi-table support
            file_name: Name of the file/dataset
            table_name: Single table name for backward compatibility
            table_names: Dictionary mapping df names to SQL table names
            max_steps: Maximum conversation steps
            max_retries: Maximum retry attempts for errors
            verbose: Enable verbose logging
        """
        # Handle backward compatibility
        if dataframes is None and df is not None:
            dataframes = {"main": df}
        elif dataframes is None:
            raise ValueError("Either 'df' or 'dataframes' must be provided")
        
        if table_names is None and table_name is not None:
            table_names = {"main": table_name}
        elif table_names is None:
            table_names = {name: f"table_{name}" for name in dataframes.keys()}
        
        self.dataframes = dataframes
        self.df = dataframes.get("main", list(dataframes.values())[0])
        self.file_name = file_name
        self.table_names = table_names
        self.max_steps = max_steps
        self.max_retries = max_retries
        
        # Create in-memory SQLite database
        self.db_connection = sqlite3.connect(':memory:')
        
        # Load all DataFrames into SQLite and get schema without data
        schema_parts = []
        column_info = {}
        
        for df_name, df in dataframes.items():
            sql_table_name = table_names.get(df_name, f"table_{df_name}")
            df.to_sql(sql_table_name, self.db_connection, index=False, if_exists='replace')
            
            # Get schema for this table
            cursor = self.db_connection.cursor()
            cursor.execute(f"PRAGMA table_info({sql_table_name})")
            schema_info = cursor.fetchall()
            schema_str = f"\nTable: {sql_table_name}\n"
            schema_str += "\n".join([f"  {col[1]} ({col[2]})" for col in schema_info])
            schema_parts.append(schema_str)
            
            # Extract column info without data
            column_info[sql_table_name] = {}
            for col in df.columns:
                dtype_info = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                column_info[sql_table_name][col] = f"dtype: {dtype_info}, nulls: {null_count}, unique: {unique_count}"
        
        self.agent = create_sql_agent_graph(sherlock_mode=False)
        
        self.state = SQLAgentState(
            file_name=file_name,
            file_type="csv",
            schema="\n".join(schema_parts),
            column_info=column_info,
            table_shape=self.df.shape,
            table_name=list(table_names.values())[0],
            table_names=list(table_names.values()),
            db_connection=self.db_connection,
            messages=[],
            step=0,
            max_steps=max_steps,
            max_retries=max_retries,
            retry_count=0,
            is_complete=False,
            verbose=verbose,
            error_history=[],
            successful_patterns=[],
            current_strategy="Initial blind SQL exploration",
            final_output=None,
            explanation="",
            raw_verbose="",
            verification_status="pending"
        )
    
    def query(self, question: str) -> str:
        """Query with blind SQL analysis"""
        user_message = HumanMessage(content=question)
        self.state["messages"].append(user_message)
        
        try:
            result = self.agent.invoke(self.state)
            self.state.update(result)
            
            last_message = result["messages"][-1] if result["messages"] else None
            return last_message.content if last_message else "No response generated"
            
        except Exception as e:
            return f"Error during blind SQL analysis: {str(e)}"
    
    def close(self):
        """Clean up database connection"""
        if self.db_connection:
            self.db_connection.close()
            
            
            
# =============================================================================
# DROP-IN ADJUSTMENTS FOR .DB FILE SUPPORT
# =============================================================================

def load_db_file(db_path: str, table_name: Optional[str] = None) -> Tuple[sqlite3.Connection, List[str], Dict[str, pd.DataFrame]]:
    """
    Load a .db file and return connection, table names, and dataframes
    
    Args:
        db_path: Path to the .db file
        table_name: Optional specific table to load
        
    Returns:
        Tuple of (connection, table_names, dataframes_dict)
    """
    # Connect to the database file
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0] for row in cursor.fetchall()]
    
    # Filter tables if specific table requested
    if table_name:
        if table_name in all_tables:
            tables_to_load = [table_name]
        else:
            raise ValueError(f"Table '{table_name}' not found in database. Available tables: {all_tables}")
    else:
        tables_to_load = all_tables
    
    # Load tables as dataframes for preview/schema
    dataframes = {}
    for table in tables_to_load:
        dataframes[table] = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1000", conn)
    
    return conn, tables_to_load, dataframes


# Override the original classes with enhanced versions
class SQLTableAnalysisAgent(SQLTableAnalysisAgent):
    """Enhanced SQLTableAnalysisAgent with .db file support"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 dataframes: Optional[Dict[str, pd.DataFrame]] = None,
                 db_path: Optional[str] = None,
                 file_name: str = "uploaded_data", 
                 table_name: Optional[str] = None,
                 table_names: Optional[Dict[str, str]] = None,
                 max_steps: int = 10,
                 max_retries: int = 3):
        
        # If db_path provided, load from file
        if db_path:
            conn, loaded_tables, loaded_dataframes = load_db_file(db_path, table_name)
            
            # Use loaded data
            dataframes = loaded_dataframes
            self.db_connection = conn
            self.external_db = True
            
            # Set table names to match what's in the DB
            if not table_names:
                table_names = {name: name for name in loaded_tables}
        else:
            self.external_db = False
            self.db_connection = None
            
        # Call parent constructor WITHOUT db_path
        super().__init__(
            df=df,
            dataframes=dataframes,
            file_name=file_name if not db_path else db_path,
            table_name=table_name,
            table_names=table_names,
            max_steps=max_steps,
            max_retries=max_retries
        )
        
        # If we loaded from DB, override the connection
        if db_path and self.external_db:
            self.state["db_connection"] = conn
            # Don't create new in-memory DB, use the existing one
            

class WatsonSQLTableAnalysisAgent(WatsonSQLTableAnalysisAgent):
    """Enhanced WatsonSQLTableAnalysisAgent with .db file support"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 dataframes: Optional[Dict[str, pd.DataFrame]] = None,
                 db_path: Optional[str] = None,
                 file_name: str = "uploaded_data", 
                 table_name: Optional[str] = None,
                 table_names: Optional[Dict[str, str]] = None,
                 max_steps: int = 10,
                 max_retries: int = 3,
                 verbose: bool = False):
        
        # If db_path provided, load from file
        if db_path:
            conn, loaded_tables, loaded_dataframes = load_db_file(db_path, table_name)
            
            # Use loaded data
            dataframes = loaded_dataframes
            self.db_connection = conn
            self.external_db = True
            
            # Set table names to match what's in the DB
            if not table_names:
                table_names = {name: name for name in loaded_tables}
        else:
            self.external_db = False
            self.db_connection = None
            
        # Call parent constructor WITHOUT db_path
        super().__init__(
            df=df,
            dataframes=dataframes,
            file_name=file_name if not db_path else db_path,
            table_name=table_name,
            table_names=table_names,
            max_steps=max_steps,
            max_retries=max_retries,
            verbose=verbose
        )
        
        # If we loaded from DB, override the connection
        if db_path and self.external_db:
            self.state["db_connection"] = conn


# =============================================================================
# RECREATE THE DICT WRAPPER CLASSES TO USE ENHANCED VERSIONS
# =============================================================================




# =============================================================================
# CONVENIENCE WRAPPER UPDATES
# =============================================================================

# Update the existing classes to use enhanced versions
# SQLTableAnalysisAgent = SQLTableAnalysisAgentEnhanced
# WatsonSQLTableAnalysisAgent = WatsonSQLTableAnalysisAgentEnhanced

# =============================================================================
# DICT RETURNING MIXIN
# =============================================================================

class DictReturningMixin:
    """
    A mix-in that converts string-returning query() to dict-returning one,
    with enhanced structured output for all agent types.
    """
    def _extract_code(self, plan_or_reply: str) -> Optional[str]:
        """Extract code from the response"""
        # Try Python first, then SQL
        python_match = re.search(r"```python\n(.*?)\n```", plan_or_reply, re.DOTALL)
        if python_match:
            return python_match.group(1).strip()
        
        sql_match = re.search(r"```sql\n(.*?)\n```", plan_or_reply, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        return None

    def query(self, question: str) -> Dict[str, Any]:
        """Query with dictionary return format"""
        final_text: str = super().query(question)
        state = self.state
        
        # Get the output as a DataFrame using the ResultFormatter
        final_output = state.get("final_output")
        output_df = ResultFormatter.convert_to_dataframe(final_output)
        
        # Extract code from various sources
        code = (state.get("python") or 
                state.get("sql_query") or 
                self._extract_code(state.get("plan", "")) or
                self._extract_code(final_text))
        
        return {
            "output": output_df,  # DataFrame representation of the result
            "final_result": final_text,
            "code": code,
            "explanation": state.get("explanation") or state.get("user_friendly_result", ""),
            "plan": state.get("plan", ""),
            "verbose": state.get("raw_verbose") or "\n".join(
                m.content for m in state.get("messages", [])
            ),
            "react_history": state.get("react_history", []),
            "verification_status": state.get("verification_status", "pending"),
            "error_history": state.get("error_history", []),
            "retry_count": state.get("retry_count", 0)
        }
        
        
# =============================================================================
# DROP-IN ADJUSTMENTS FOR SQL AGENT PLOT EXECUTION
# =============================================================================

import matplotlib.pyplot as plt
import io
import base64
from typing import Optional, Dict, Any, List

class PlotExecutor:
    """Execute plotting code and save plots"""
    
    def __init__(self):
        self.plots = []  # Store plot objects
        
    def execute_plot_code(self, code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute plotting code with the query result data
        
        Args:
            code: Python plotting code
            data: DataFrame from SQL query result
            
        Returns:
            Dict with plot information
        """
        # Reset matplotlib
        plt.close('all')
        
        # Create a new figure
        fig = plt.figure(figsize=(10, 6))
        
        # Prepare execution environment
        plot_globals = {
            'plt': plt,
            'pd': pd,
            'np': np,
            'df': data,  # Make query result available as 'df'
            'data': data,  # Also available as 'data'
            'result': data,  # And as 'result' for consistency
        }
        
        try:
            # Execute the plotting code
            exec(code, plot_globals)
            
            # Capture the current figure
            if plt.get_fignums():
                # Save plot to bytes
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                
                # Create base64 encoded image
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                # Store plot info
                plot_info = {
                    'figure': fig,
                    'base64': img_base64,
                    'timestamp': pd.Timestamp.now(),
                    'success': True,
                    'error': None
                }
                
                self.plots.append(plot_info)
                
                return plot_info
            else:
                return {
                    'success': False,
                    'error': 'No plot was generated'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Plot execution error: {str(e)}"
            }
        finally:
            plt.close('all')
    
    def display_plot(self, index: int = -1):
        """Display a saved plot"""
        if not self.plots:
            print("No plots available")
            return
            
        plot_info = self.plots[index]
        if plot_info['success']:
            # Display the plot
            plt.figure(plot_info['figure'].number)
            plt.show()
    
    def get_plot_as_base64(self, index: int = -1) -> Optional[str]:
        """Get plot as base64 string"""
        if not self.plots or index >= len(self.plots):
            return None
        return self.plots[index].get('base64')


# Enhanced SQL assistant node with plot execution
def create_sql_assistant_node_with_plots(sherlock_mode: bool = True):
    """Factory function to create SQL assistant nodes with plot execution"""
    
    # Get the original node function
    original_node = create_sql_assistant_node(sherlock_mode)
    
    def assistant_node_with_plots(state: SQLAgentState) -> dict:
        """SQL assistant node with plot execution capability"""
        
        # Call the original node
        result = original_node(state)
        
        # Check if we have a successful query result and look for plot code
        if (result.get('final_output') is not None and 
            isinstance(result.get('final_output'), pd.DataFrame)):
            
            # Look for Python plot code in the response
            last_message = result['messages'][-1] if result['messages'] else None
            if last_message and hasattr(last_message, 'content'):
                content = last_message.content
                
                # Extract Python plotting code
                plot_matches = re.findall(
                    r'```python\n(.*?)```', 
                    content, 
                    re.DOTALL
                )
                
                # Execute plot code if found
                if plot_matches:
                    plot_executor = PlotExecutor()
                    plots_created = []
                    
                    for plot_code in plot_matches:
                        # Skip if it's the SQL query result display code
                        if 'import matplotlib' in plot_code or 'plt.' in plot_code:
                            plot_result = plot_executor.execute_plot_code(
                                plot_code.strip(),
                                result['final_output']
                            )
                            
                            if plot_result['success']:
                                plots_created.append(plot_result)
                                
                                # Add plot info to result
                                if 'plots' not in result:
                                    result['plots'] = []
                                result['plots'].append(plot_result)
                    
                    # Update the response with plot info
                    if plots_created:
                        plot_msg = f"\n\n📊 **Plots Generated:** {len(plots_created)} plot(s) created successfully"
                        result['messages'][-1].content += plot_msg
                        result['plot_executor'] = plot_executor
        
        return result
    
    return assistant_node_with_plots


# Override the graph builders to use plot-enabled nodes
def create_sql_agent_graph_with_plots(sherlock_mode: bool = True):
    """Create SQL agent graph with plot execution capabilities"""
    builder = StateGraph(SQLAgentState)
    
    # Create the plot-enabled assistant node
    assistant = create_sql_assistant_node_with_plots(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    
    builder.add_edge(START, "assistant")
    
    def should_continue(state: SQLAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()


# Enhanced SQL agent classes with plot support
class SQLTableAnalysisAgent(SQLTableAnalysisAgent):
    """Enhanced SQL agent with plot execution"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the agent with plot-enabled version
        self.agent = create_sql_agent_graph_with_plots(sherlock_mode=True)
        self.plot_executor = None
    
    def get_plots(self) -> List[Dict[str, Any]]:
        """Get all generated plots"""
        return self.state.get('plots', [])
    
    def display_plot(self, index: int = -1):
        """Display a generated plot"""
        if hasattr(self, 'plot_executor') and self.plot_executor:
            self.plot_executor.display_plot(index)
        else:
            plots = self.get_plots()
            if plots and index < len(plots):
                # Recreate the plot from base64
                import matplotlib.image as mpimg
                img_data = base64.b64decode(plots[index]['base64'])
                img = mpimg.imread(io.BytesIO(img_data))
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.show()


class WatsonSQLTableAnalysisAgent(WatsonSQLTableAnalysisAgent):
    """Enhanced Watson SQL agent with plot execution"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the agent with plot-enabled version
        self.agent = create_sql_agent_graph_with_plots(sherlock_mode=False)
        self.plot_executor = None
    
    def get_plots(self) -> List[Dict[str, Any]]:
        """Get all generated plots"""
        return self.state.get('plots', [])
    
    def display_plot(self, index: int = -1):
        """Display a generated plot"""
        if hasattr(self, 'plot_executor') and self.plot_executor:
            self.plot_executor.display_plot(index)
        else:
            plots = self.get_plots()
            if plots and index < len(plots):
                # Recreate the plot from base64
                import matplotlib.image as mpimg
                img_data = base64.b64decode(plots[index]['base64'])
                img = mpimg.imread(io.BytesIO(img_data))
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.show()


# Update DictReturningMixin to include plot info
class DictReturningMixin(DictReturningMixin):
    """Enhanced mixin with plot support"""
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query with dictionary return format including plots"""
        result = super().query(question)
        
        # Add plot information if available
        if hasattr(self, 'get_plots'):
            plots = self.get_plots()
            if plots:
                result['plots'] = plots
                result['plot_count'] = len(plots)
                # Include base64 images for easy access
                result['plot_images'] = [p.get('base64') for p in plots if p.get('success')]
        
        return result



# =============================================================================
# DROP-IN FIX FOR SQL AGENT PLOTTING
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional, Dict, Any, List, Tuple

# Enhanced SQL assistant node with integrated plot execution
def create_sql_assistant_node_with_integrated_plots(sherlock_mode: bool = True):
    """Factory function to create SQL assistant nodes with integrated plot execution"""
    
    def assistant_node(state: SQLAgentState) -> dict:
        """SQL assistant node with integrated plot execution capability"""
        
        verbose = state.get("verbose", False)
        agent_name = "SHERLOCK" if sherlock_mode else "WATSON"
        
        if verbose:
            print(f"\n🔍 {agent_name} SQL STEP {state.get('step', 0) + 1}: Starting analysis...")
        
        # Initialize ReAct manager
        react_manager = ReactCycleManager(
            max_retries=state.get("max_retries", 3),
            enable_verification=True
        )
        
        # Check if we should verify previous result
        if react_manager.should_verify(state) and state.get("final_output") is not None:
            verification_msg = react_manager.create_verification_instruction(
                state.get("final_output"),
                state.get("sql_query", "")
            )
            state["messages"].append(HumanMessage(content=verification_msg))
            react_manager.update_history(state, "verification")
        
        msgs = state["messages"].copy()
        current_step = state.get("step", 0)
        max_steps = state.get("max_steps", 10)
        
        # Build system prompt based on agent type
        if sherlock_mode:
            sys_prompt = _build_sherlock_sql_prompt_with_plot(state)
        else:
            sys_prompt = _build_watson_sql_prompt_with_plot(state)
        
        # Add system message if not present
        if not any(m.type == "system" for m in msgs):
            msgs.insert(0, SystemMessage(content=sys_prompt))
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini")
            assistant_reply = llm.invoke(msgs)
            
            # Initialize output variables
            user_friendly_result = ""
            execution_result = ""
            error = None
            sql_query = ""
            final_output = None
            explanation = ""
            verification_status = state.get("verification_status", "pending")
            plots_info = []
            
            # Check for verification confirmation
            if "VERIFIED:" in assistant_reply.content:
                verification_status = "verified"
                explanation = re.search(r"VERIFIED:\s*(.*)", assistant_reply.content, re.DOTALL).group(1)
                return {
                    "messages": [assistant_reply],
                    "verification_status": verification_status,
                    "explanation": explanation,
                    "step": current_step + 1
                }
            
            # Extract and execute SQL
            sql_blocks = re.findall(r'```sql\n(.*?)\n```', assistant_reply.content, re.DOTALL)
            
            if sql_blocks and state.get("db_connection"):
                executor = SafeSQLExecutor(
                    state["db_connection"], 
                    state.get("table_names", [state.get("table_name", "data_table")]),
                    watson_mode=not sherlock_mode
                )
                
                for sql_block in sql_blocks:
                    sql_query = sql_block.strip()
                    exec_result = executor.execute_sql(sql_query)
                    
                    if exec_result['success']:
                        final_output = exec_result.get('result')
                        
                        # Now check for plotting code AFTER we have the SQL result
                        if isinstance(final_output, pd.DataFrame) and len(final_output) > 0:
                            # Look for Python plotting code in the assistant's response
                            python_blocks = re.findall(r'```python\n(.*?)\n```', assistant_reply.content, re.DOTALL)
                            
                            for python_block in python_blocks:
                                # Execute plot code with the SQL result
                                plot_result = execute_plot_with_data(python_block.strip(), final_output)
                                if plot_result['success']:
                                    plots_info.append(plot_result)
                        
                        user_friendly_result = _format_sql_result(exec_result, sql_query, sherlock_mode)
                        execution_result = str(final_output) if final_output is not None else ""
                        
                        # Extract explanation
                        explanation_match = re.search(r'\*\*Explanation\*\*:?(.*?)(?=\*\*|$)', 
                                                    assistant_reply.content, re.DOTALL | re.IGNORECASE)
                        if explanation_match:
                            explanation = explanation_match.group(1).strip()
                    else:
                        error = exec_result['error']
                        error_type = exec_result.get('error_type', 'unknown')
                        
                        # Handle retry logic
                        if react_manager.should_retry(state):
                            retry_instruction = react_manager.create_retry_instruction(
                                error, error_type,
                                f"{agent_name} SQL analysis"
                            )
                            
                            # Recursive retry
                            state["messages"].append(assistant_reply)
                            state["messages"].append(HumanMessage(content=retry_instruction))
                            state["retry_count"] = state.get("retry_count", 0) + 1
                            state["error"] = error
                            
                            react_manager.update_history(state, "retry")
                            return assistant_node(state)
                        
                        user_friendly_result = f"❌ **Final SQL Error:** {error}"
                        break
            
            # Build response
            raw_verbose = _build_verbose_log(agent_name, "SQL", state, assistant_reply, sql_query, execution_result, error)
            
            full_response = assistant_reply.content
            if user_friendly_result:
                full_response += f"\n\n{user_friendly_result}"
            
            # Add plot info if plots were created
            if plots_info:
                full_response += f"\n\n📊 **Plots Generated:** {len(plots_info)} plot(s) created successfully"
            
            return {
                "messages": [AIMessage(content=full_response)],
                "step": current_step + 1,
                "sql_query": sql_query,
                "execution_result": execution_result,
                "user_friendly_result": user_friendly_result,
                "error": error,
                "final_output": final_output,
                "explanation": explanation or user_friendly_result,
                "raw_verbose": raw_verbose,
                "verification_status": verification_status,
                "react_history": react_manager.history,
                "plots": plots_info  # Add plots to state
            }
            
        except Exception as e:
            error_msg = f"{agent_name} SQL assistant error: {str(e)}"
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
                "step": current_step + 1,
                "error": error_msg
            }
    
    return assistant_node


def execute_plot_with_data(code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute plotting code with SQL query result data
    
    Args:
        code: Python plotting code
        data: DataFrame from SQL query result
        
    Returns:
        Dict with plot information
    """
    # Reset matplotlib
    plt.close('all')
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 6))
    
    # Prepare execution environment
    plot_globals = {
        'plt': plt,
        'pd': pd,
        'np': np,
        'sns': sns,
        'df': data,  # Make query result available as 'df'
        'data': data,  # Also available as 'data'
        'result': data,  # And as 'result' for consistency
        'print': print,  # Allow print statements
    }
    
    try:
        # Execute the plotting code
        exec(code, plot_globals)
        
        # Capture the current figure
        if plt.get_fignums():
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # Create base64 encoded image
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Store plot info
            plot_info = {
                'figure': fig,
                'base64': img_base64,
                'timestamp': pd.Timestamp.now(),
                'success': True,
                'error': None,
                'code': code
            }
            
            return plot_info
        else:
            return {
                'success': False,
                'error': 'No plot was generated',
                'code': code
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Plot execution error: {str(e)}",
            'code': code
        }
    finally:
        plt.close('all')


def _build_sherlock_sql_prompt_with_plot(state: SQLAgentState) -> str:
    """Build system prompt for Sherlock SQL agent with plotting instructions"""
    
    tables_str = ", ".join(state.get("table_names", [state.get("table_name", "data_table")]))
    
    return f"""You are Sherlock, a SQL data analysis assistant with full data access and plotting capabilities.

Available Tables: {tables_str}

Database Schema:
{state.get("schema", "Not available")}

Preview (first 5 rows):
{state.get("preview_md", "Not available")}

IMPORTANT GUIDELINES:
1. Write efficient SQL queries using available tables
2. Use only SELECT statements - no modifications
3. Format SQL queries in ```sql code blocks
4. If visualization would help, include Python plotting code in ```python blocks
5. Format your response with these sections:
   - **Plan**: Explain your SQL approach
   - **Query**: Show the SQL query
   - **Visualization** (optional): Python code to plot the results
   - **Explanation**: Interpret the results
6. Use JOINs when working with multiple tables
7. Handle NULL values and edge cases properly
8. Use CTEs for complex queries

For plotting:
- The SQL result will be available as 'df', 'data', or 'result'
- Use matplotlib (plt) or seaborn (sns)
- Keep plots clear and well-labeled
- Example:
  ```python
  plt.figure(figsize=(10, 6))
  plt.bar(df['category'], df['count'])
  plt.xlabel('Category')
  plt.ylabel('Count')
  plt.title('Distribution by Category')
  plt.xticks(rotation=45)
  plt.tight_layout()
  ```

Available SQL features:
- All SELECT operations
- Aggregate functions: COUNT, SUM, AVG, MIN, MAX
- Window functions
- CTEs (WITH clauses)
- JOINs (INNER, LEFT, RIGHT, FULL)

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Retry count: {state.get('retry_count', 0)}/{state.get('max_retries', 3)}"""


def _build_watson_sql_prompt_with_plot(state: SQLAgentState) -> str:
    """Build system prompt for Watson SQL agent with plotting instructions"""
    
    tables_str = ", ".join(state.get("table_names", [state.get("table_name", "data_table")]))
    
    return f"""You are Watson, a blind SQL analysis assistant with plotting capabilities. You cannot query raw data directly.

CRITICAL CONSTRAINTS:
1. NO raw data queries: no SELECT *, no large LIMIT, no sampling
2. Use only aggregations: COUNT, SUM, AVG, GROUP BY, etc.
3. Maximum LIMIT 20 for TOP N queries only
4. Learn from SQL errors to refine queries

Available Tables: {tables_str}

Database Schema:
{state.get("schema", "Not available")}

Column Information:
{state.get("column_info", "Not available")}

IMPORTANT GUIDELINES:
1. Write aggregation-focused SQL queries
2. Format SQL queries in ```sql code blocks
3. If visualization would help with aggregated results, include Python plotting code in ```python blocks
4. Format your response with these sections:
   - **Plan**: Explain your blind analysis approach
   - **Query**: Show the SQL query
   - **Visualization** (optional): Python code to plot the aggregated results
   - **Explanation**: Interpret the aggregate results
5. Use COUNT(*) to understand data volume
6. Use GROUP BY for categorical analysis
7. Use statistical functions for numeric columns

For plotting aggregated results:
- The SQL result will be available as 'df', 'data', or 'result'
- Use matplotlib (plt) or seaborn (sns)
- Focus on aggregate visualizations (bar charts, pie charts, etc.)
- Example for aggregated data:
  ```python
  plt.figure(figsize=(10, 6))
  plt.pie(df['count'], labels=df['category'], autopct='%1.1f%%')
  plt.title('Distribution of Categories')
  ```

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Previous errors: {state.get('error_history', [])[-3:]}
Current strategy: {state.get('current_strategy', 'Initial exploration')}"""


# Override the graph creation functions
def create_sql_agent_graph(sherlock_mode: bool = True):
    """Create SQL agent graph with integrated plot capabilities"""
    builder = StateGraph(SQLAgentState)
    
    # Use the new integrated assistant node
    assistant = create_sql_assistant_node_with_integrated_plots(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    
    builder.add_edge(START, "assistant")
    
    def should_continue(state: SQLAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()


# Enhanced display_plot method for SQL agents
def enhanced_display_plot(self, index: int = -1):
    """Display a generated plot from base64 data"""
    plots = self.state.get('plots', [])
    
    if not plots:
        print("No plots available")
        return
    
    if abs(index) > len(plots):
        print(f"Invalid plot index. Available plots: {len(plots)}")
        return
    
    plot_info = plots[index]
    
    if plot_info.get('success') and plot_info.get('base64'):
        # Decode and display the plot
        import matplotlib.image as mpimg
        from IPython.display import Image, display
        
        # For Jupyter environments
        try:
            img_data = base64.b64decode(plot_info['base64'])
            display(Image(img_data))
        except:
            # For non-Jupyter environments
            img_data = base64.b64decode(plot_info['base64'])
            img = mpimg.imread(io.BytesIO(img_data))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Plot {abs(index) + 1} of {len(plots)}")
            plt.show()
    else:
        print(f"Plot generation failed: {plot_info.get('error', 'Unknown error')}")


# Patch the existing SQL agent classes
SQLTableAnalysisAgent.display_plot = enhanced_display_plot
WatsonSQLTableAnalysisAgent.display_plot = enhanced_display_plot

# Add a get_plot_image method for easy access
def get_plot_image(self, index: int = -1) -> Optional[str]:
    """Get plot as base64 string"""
    plots = self.state.get('plots', [])
    
    if not plots or abs(index) > len(plots):
        return None
    
    return plots[index].get('base64')

SQLTableAnalysisAgent.get_plot_image = get_plot_image
WatsonSQLTableAnalysisAgent.get_plot_image = get_plot_image

# Add save_plot method
def save_plot(self, filename: str, index: int = -1):
    """Save plot to file"""
    plots = self.state.get('plots', [])
    
    if not plots:
        print("No plots available")
        return
    
    if abs(index) > len(plots):
        print(f"Invalid plot index. Available plots: {len(plots)}")
        return
    
    plot_info = plots[index]
    
    if plot_info.get('success') and plot_info.get('base64'):
        img_data = base64.b64decode(plot_info['base64'])
        with open(filename, 'wb') as f:
            f.write(img_data)
        print(f"Plot saved to {filename}")
    else:
        print(f"Plot generation failed: {plot_info.get('error', 'Unknown error')}")

SQLTableAnalysisAgent.save_plot = save_plot
WatsonSQLTableAnalysisAgent.save_plot = save_plot


# =============================================================================
# COMPLETE DROP-IN FIX FOR SQL AGENT PLOTTING
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional, Dict, Any, List, Tuple

# First, update the SQLAgentState to include plots
def patch_sql_agent_state():
    """Patch SQLAgentState to include plots field"""
    # Add plots field to SQLAgentState if not already present
    if 'plots' not in SQLAgentState.__annotations__:
        SQLAgentState.__annotations__['plots'] = List[Dict[str, Any]]

# Apply the patch
patch_sql_agent_state()

# Enhanced SQL assistant node with integrated plot execution
def create_sql_assistant_node_with_integrated_plots(sherlock_mode: bool = True):
    """Factory function to create SQL assistant nodes with integrated plot execution"""
    
    def assistant_node(state: SQLAgentState) -> dict:
        """SQL assistant node with integrated plot execution capability"""
        
        verbose = state.get("verbose", False)
        agent_name = "SHERLOCK" if sherlock_mode else "WATSON"
        
        if verbose:
            print(f"\n🔍 {agent_name} SQL STEP {state.get('step', 0) + 1}: Starting analysis...")
        
        # Initialize plots if not present
        if 'plots' not in state:
            state['plots'] = []
        
        # Initialize ReAct manager
        react_manager = ReactCycleManager(
            max_retries=state.get("max_retries", 3),
            enable_verification=True
        )
        
        # Check if we should verify previous result
        if react_manager.should_verify(state) and state.get("final_output") is not None:
            verification_msg = react_manager.create_verification_instruction(
                state.get("final_output"),
                state.get("sql_query", "")
            )
            state["messages"].append(HumanMessage(content=verification_msg))
            react_manager.update_history(state, "verification")
        
        msgs = state["messages"].copy()
        current_step = state.get("step", 0)
        max_steps = state.get("max_steps", 10)
        
        # Build system prompt based on agent type
        if sherlock_mode:
            sys_prompt = _build_sherlock_sql_prompt_with_plot(state)
        else:
            sys_prompt = _build_watson_sql_prompt_with_plot(state)
        
        # Add system message if not present
        if not any(m.type == "system" for m in msgs):
            msgs.insert(0, SystemMessage(content=sys_prompt))
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini")
            assistant_reply = llm.invoke(msgs)
            
            # Initialize output variables
            user_friendly_result = ""
            execution_result = ""
            error = None
            sql_query = ""
            final_output = None
            explanation = ""
            verification_status = state.get("verification_status", "pending")
            new_plots = []
            
            # Check for verification confirmation
            if "VERIFIED:" in assistant_reply.content:
                verification_status = "verified"
                explanation = re.search(r"VERIFIED:\s*(.*)", assistant_reply.content, re.DOTALL).group(1)
                return {
                    "messages": [assistant_reply],
                    "verification_status": verification_status,
                    "explanation": explanation,
                    "step": current_step + 1
                }
            
            # Extract and execute SQL
            sql_blocks = re.findall(r'```sql\n(.*?)\n```', assistant_reply.content, re.DOTALL)
            
            if sql_blocks and state.get("db_connection"):
                executor = SafeSQLExecutor(
                    state["db_connection"], 
                    state.get("table_names", [state.get("table_name", "data_table")]),
                    watson_mode=not sherlock_mode
                )
                
                for sql_block in sql_blocks:
                    sql_query = sql_block.strip()
                    exec_result = executor.execute_sql(sql_query)
                    
                    if exec_result['success']:
                        final_output = exec_result.get('result')
                        
                        # Now check for plotting code AFTER we have the SQL result
                        if isinstance(final_output, pd.DataFrame) and len(final_output) > 0:
                            # Look for Python plotting code in the assistant's response
                            python_blocks = re.findall(r'```python\n(.*?)\n```', assistant_reply.content, re.DOTALL)
                            
                            for python_block in python_blocks:
                                # Execute plot code with the SQL result
                                plot_result = execute_plot_with_data(python_block.strip(), final_output)
                                if plot_result['success']:
                                    new_plots.append(plot_result)
                                    if verbose:
                                        print(f"✅ Plot generated successfully")
                                else:
                                    if verbose:
                                        print(f"❌ Plot failed: {plot_result['error']}")
                        
                        user_friendly_result = _format_sql_result(exec_result, sql_query, sherlock_mode)
                        execution_result = str(final_output) if final_output is not None else ""
                        
                        # Extract explanation
                        explanation_match = re.search(r'\*\*Explanation\*\*:?(.*?)(?=\*\*|$)', 
                                                    assistant_reply.content, re.DOTALL | re.IGNORECASE)
                        if explanation_match:
                            explanation = explanation_match.group(1).strip()
                    else:
                        error = exec_result['error']
                        error_type = exec_result.get('error_type', 'unknown')
                        
                        # Handle retry logic
                        if react_manager.should_retry(state):
                            retry_instruction = react_manager.create_retry_instruction(
                                error, error_type,
                                f"{agent_name} SQL analysis"
                            )
                            
                            # Recursive retry
                            state["messages"].append(assistant_reply)
                            state["messages"].append(HumanMessage(content=retry_instruction))
                            state["retry_count"] = state.get("retry_count", 0) + 1
                            state["error"] = error
                            
                            react_manager.update_history(state, "retry")
                            return assistant_node(state)
                        
                        user_friendly_result = f"❌ **Final SQL Error:** {error}"
                        break
            
            # Build response
            raw_verbose = _build_verbose_log(agent_name, "SQL", state, assistant_reply, sql_query, execution_result, error)
            
            full_response = assistant_reply.content
            if user_friendly_result:
                full_response += f"\n\n{user_friendly_result}"
            
            # Add plot info if plots were created
            if new_plots:
                full_response += f"\n\n📊 **Plots Generated:** {len(new_plots)} plot(s) created successfully"
            
            # Append new plots to existing plots in state
            existing_plots = state.get('plots', [])
            all_plots = existing_plots + new_plots
            
            return {
                "messages": [AIMessage(content=full_response)],
                "step": current_step + 1,
                "sql_query": sql_query,
                "execution_result": execution_result,
                "user_friendly_result": user_friendly_result,
                "error": error,
                "final_output": final_output,
                "explanation": explanation or user_friendly_result,
                "raw_verbose": raw_verbose,
                "verification_status": verification_status,
                "react_history": react_manager.history,
                "plots": all_plots  # Return all plots (existing + new)
            }
            
        except Exception as e:
            error_msg = f"{agent_name} SQL assistant error: {str(e)}"
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
                "step": current_step + 1,
                "error": error_msg,
                "plots": state.get('plots', [])  # Preserve existing plots
            }
    
    return assistant_node


def execute_plot_with_data(code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute plotting code with SQL query result data
    
    Args:
        code: Python plotting code
        data: DataFrame from SQL query result
        
    Returns:
        Dict with plot information
    """
    # Reset matplotlib
    plt.close('all')
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 6))
    
    # Prepare execution environment
    plot_globals = {
        'plt': plt,
        'pd': pd,
        'np': np,
        'sns': sns,
        'df': data.copy(),  # Make query result available as 'df'
        'data': data.copy(),  # Also available as 'data'
        'result': data.copy(),  # And as 'result' for consistency
        'print': print,  # Allow print statements
        'fig': fig,  # Access to figure
    }
    
    # Add matplotlib components
    plot_globals.update({
        'Figure': plt.Figure,
        'subplot': plt.subplot,
        'subplots': plt.subplots,
    })
    
    try:
        # Execute the plotting code
        exec(code, plot_globals)
        
        # Force a draw to ensure plot is rendered
        plt.draw()
        
        # Capture the current figure
        if plt.get_fignums():
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # Create base64 encoded image
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Store plot info
            plot_info = {
                'figure': fig,
                'base64': img_base64,
                'timestamp': pd.Timestamp.now(),
                'success': True,
                'error': None,
                'code': code
            }
            
            return plot_info
        else:
            return {
                'success': False,
                'error': 'No plot was generated - check if plt.show() or plt.savefig() is missing',
                'code': code
            }
            
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f"Plot execution error: {str(e)}\nTraceback: {traceback.format_exc()}",
            'code': code
        }
    finally:
        plt.close('all')


def _build_sherlock_sql_prompt_with_plot(state: SQLAgentState) -> str:
    """Build system prompt for Sherlock SQL agent with plotting instructions"""
    
    tables_str = ", ".join(state.get("table_names", [state.get("table_name", "data_table")]))
    
    return f"""You are Sherlock, a SQL data analysis assistant with full data access and plotting capabilities.

Available Tables: {tables_str}

Database Schema:
{state.get("schema", "Not available")}

Preview (first 5 rows):
{state.get("preview_md", "Not available")}

IMPORTANT GUIDELINES:
1. Write efficient SQL queries using available tables
2. Use only SELECT statements - no modifications
3. Format SQL queries in ```sql code blocks
4. If visualization would help, include Python plotting code in ```python blocks
5. Format your response with these sections:
   - **Plan**: Explain your SQL approach
   - **Query**: Show the SQL query
   - **Visualization** (optional): Python code to plot the results
   - **Explanation**: Interpret the results

For plotting:
- The SQL result will be available as 'df', 'data', or 'result'
- Use matplotlib (plt) or seaborn (sns)
- Always include plt.tight_layout() before finishing
- The plot will be automatically saved, no need to call plt.show()
- Example:
  ```python
  # Bar chart example
  plt.figure(figsize=(10, 6))
  plt.bar(df['category'], df['count'])
  plt.xlabel('Category')
  plt.ylabel('Count')
  plt.title('Distribution by Category')
  plt.xticks(rotation=45)
  plt.tight_layout()
  ```

Available SQL features:
- All SELECT operations
- Aggregate functions: COUNT, SUM, AVG, MIN, MAX
- Window functions
- CTEs (WITH clauses)
- JOINs (INNER, LEFT, RIGHT, FULL)

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Retry count: {state.get('retry_count', 0)}/{state.get('max_retries', 3)}"""


def _build_watson_sql_prompt_with_plot(state: SQLAgentState) -> str:
    """Build system prompt for Watson SQL agent with plotting instructions"""
    
    tables_str = ", ".join(state.get("table_names", [state.get("table_name", "data_table")]))
    
    return f"""You are Watson, a blind SQL analysis assistant with plotting capabilities. You cannot query raw data directly.

CRITICAL CONSTRAINTS:
1. NO raw data queries: no SELECT *, no large LIMIT, no sampling
2. Use only aggregations: COUNT, SUM, AVG, GROUP BY, etc.
3. Maximum LIMIT 20 for TOP N queries only
4. Learn from SQL errors to refine queries

Available Tables: {tables_str}

Database Schema:
{state.get("schema", "Not available")}

Column Information:
{state.get("column_info", "Not available")}

IMPORTANT GUIDELINES:
1. Write aggregation-focused SQL queries
2. Format SQL queries in ```sql code blocks
3. If visualization would help with aggregated results, include Python plotting code in ```python blocks
4. Format your response with these sections:
   - **Plan**: Explain your blind analysis approach
   - **Query**: Show the SQL query
   - **Visualization** (optional): Python code to plot the aggregated results
   - **Explanation**: Interpret the aggregate results

For plotting aggregated results:
- The SQL result will be available as 'df', 'data', or 'result'
- Use matplotlib (plt) or seaborn (sns)
- Focus on aggregate visualizations
- Example for aggregated data:
  ```python
  # Pie chart for aggregated data
  plt.figure(figsize=(10, 6))
  plt.pie(df['count'], labels=df['category'], autopct='%1.1f%%')
  plt.title('Distribution of Categories')
  plt.tight_layout()
  ```

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}"""


# Override the graph creation functions
def create_sql_agent_graph(sherlock_mode: bool = True):
    """Create SQL agent graph with integrated plot capabilities"""
    builder = StateGraph(SQLAgentState)
    
    # Use the new integrated assistant node
    assistant = create_sql_assistant_node_with_integrated_plots(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    
    builder.add_edge(START, "assistant")
    
    def should_continue(state: SQLAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()


# Patch the __init__ methods to initialize plots
original_sql_init = SQLTableAnalysisAgent.__init__
original_watson_sql_init = WatsonSQLTableAnalysisAgent.__init__

def patched_sql_init(self, *args, **kwargs):
    original_sql_init(self, *args, **kwargs)
    # Initialize plots in state
    self.state['plots'] = []
    # Replace agent with plot-enabled version
    self.agent = create_sql_agent_graph(sherlock_mode=True)

def patched_watson_sql_init(self, *args, **kwargs):
    original_watson_sql_init(self, *args, **kwargs)
    # Initialize plots in state
    self.state['plots'] = []
    # Replace agent with plot-enabled version
    self.agent = create_sql_agent_graph(sherlock_mode=False)

SQLTableAnalysisAgent.__init__ = patched_sql_init
WatsonSQLTableAnalysisAgent.__init__ = patched_watson_sql_init


# Enhanced display_plot method for SQL agents
def enhanced_display_plot(self, index: int = -1):
    """Display a generated plot from base64 data"""
    plots = self.state.get('plots', [])
    
    if not plots:
        print("No plots available")
        return
    
    if abs(index) > len(plots):
        print(f"Invalid plot index. Available plots: {len(plots)}")
        return
    
    plot_info = plots[index]
    
    if plot_info.get('success') and plot_info.get('base64'):
        # Decode and display the plot
        import matplotlib.image as mpimg
        
        # Check if we're in Jupyter
        try:
            from IPython.display import Image, display
            img_data = base64.b64decode(plot_info['base64'])
            display(Image(img_data))
        except:
            # For non-Jupyter environments
            img_data = base64.b64decode(plot_info['base64'])
            img = mpimg.imread(io.BytesIO(img_data))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Plot {abs(index) + 1} of {len(plots)}")
            plt.show()
    else:
        print(f"Plot generation failed: {plot_info.get('error', 'Unknown error')}")


# Patch the existing SQL agent classes
SQLTableAnalysisAgent.display_plot = enhanced_display_plot
WatsonSQLTableAnalysisAgent.display_plot = enhanced_display_plot
TableAnalysisAgent.display_plot = enhanced_display_plot
WatsonTableAnalysisAgent.display_plot = enhanced_display_plot

# Add a get_plot_image method for easy access
def get_plot_image(self, index: int = -1) -> Optional[str]:
    """Get plot as base64 string"""
    plots = self.state.get('plots', [])
    
    if not plots or abs(index) > len(plots):
        return None
    
    return plots[index].get('base64')

SQLTableAnalysisAgent.get_plot_image = get_plot_image
WatsonSQLTableAnalysisAgent.get_plot_image = get_plot_image

# Add save_plot method
def save_plot(self, filename: str, index: int = -1):
    """Save plot to file"""
    plots = self.state.get('plots', [])
    
    if not plots:
        print("No plots available")
        return False
    
    if abs(index) > len(plots):
        print(f"Invalid plot index. Available plots: {len(plots)}")
        return False
    
    plot_info = plots[index]
    
    if plot_info.get('success') and plot_info.get('base64'):
        img_data = base64.b64decode(plot_info['base64'])
        with open(filename, 'wb') as f:
            f.write(img_data)
        print(f"Plot saved to {filename}")
        return True
    else:
        print(f"Plot generation failed: {plot_info.get('error', 'Unknown error')}")
        return False

SQLTableAnalysisAgent.save_plot = save_plot
WatsonSQLTableAnalysisAgent.save_plot = save_plot

# Add method to get all plots
def get_all_plots(self) -> List[Dict[str, Any]]:
    """Get all generated plots"""
    return self.state.get('plots', [])

SQLTableAnalysisAgent.get_all_plots = get_all_plots
WatsonSQLTableAnalysisAgent.get_all_plots = get_all_plots

# Debug method to check plot status
def debug_plots(self):
    """Debug method to check plot status"""
    plots = self.state.get('plots', [])
    print(f"Total plots in state: {len(plots)}")
    for i, plot in enumerate(plots):
        print(f"Plot {i+1}:")
        print(f"  - Success: {plot.get('success', False)}")
        print(f"  - Has base64: {'base64' in plot}")
        print(f"  - Error: {plot.get('error', 'None')}")
        if 'code' in plot:
            print(f"  - Code preview: {plot['code'][:100]}...")

SQLTableAnalysisAgent.debug_plots = debug_plots
WatsonSQLTableAnalysisAgent.debug_plots = debug_plots


import re
from typing import Optional, Dict, Any

def extract_plan_from_response(response_content: str) -> Optional[str]:
    """Extract the plan section from assistant's response"""
    # Handle case where response_content might not be a string
    if not isinstance(response_content, str):
        return None
        
    # Look for **Plan**: or similar patterns
    plan_patterns = [
        r'\*\*Plan\*\*:\s*(.*?)(?=\*\*(?:Query|SQL|Code|Visualization|Explanation)|```|$)',
        r'Plan:\s*(.*?)(?=Query:|SQL:|Code:|Visualization:|Explanation:|```|$)',
        r'\*\*Plan\*\*\s*\n(.*?)(?=\*\*|Query:|SQL:|Code:|```|$)',
    ]
    
    for pattern in plan_patterns:
        match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)
        if match:
            plan = match.group(1).strip()
            # Clean up the plan text
            plan = re.sub(r'\n+', '\n', plan)  # Remove multiple newlines
            plan = re.sub(r'^\s*[-*]\s*', '', plan, flags=re.MULTILINE)  # Remove bullet points
            return plan
    
    return None


# Override the _build_verbose_log to include plan extraction
original_build_verbose_log = _build_verbose_log

def _build_verbose_log(agent_name: str, agent_type: str, state: Dict, reply: Any, 
                      code: str, result: str, error: Optional[str]) -> str:
    """Build verbose log and extract plan"""
    # Call original function
    verbose_log = original_build_verbose_log(agent_name, agent_type, state, reply, code, result, error)
    
    # Extract plan from assistant response if available
    if reply and hasattr(reply, 'content'):
        plan = extract_plan_from_response(reply.content)
        if plan and 'plan' not in state:
            state['plan'] = plan
    
    return verbose_log


# Enhanced SQL assistant nodes to properly store plan
def create_sql_assistant_node_enhanced(sherlock_mode: bool = True):
    """Enhanced SQL assistant node that extracts and stores plan"""
    
    # Get the original function
    original_node = create_sql_assistant_node(sherlock_mode)
    
    def enhanced_node(state: SQLAgentState) -> dict:
        # Call original node
        result = original_node(state)
        
        # Extract plan from messages if not already set
        if 'plan' not in result and result.get('messages'):
            for msg in result['messages']:
                if hasattr(msg, 'content'):
                    plan = extract_plan_from_response(msg.content)
                    if plan:
                        result['plan'] = plan
                        break
        
        return result
    
    return enhanced_node


# Enhanced Python assistant nodes to properly store plan
def create_python_assistant_node_enhanced(sherlock_mode: bool = True):
    """Enhanced Python assistant node that extracts and stores plan"""
    
    # Get the original function
    original_node = create_python_assistant_node(sherlock_mode)
    
    def enhanced_node(state: PythonAgentState) -> dict:
        # Call original node
        result = original_node(state)
        
        # Extract plan from messages if not already set
        if 'plan' not in result and result.get('messages'):
            for msg in result['messages']:
                if hasattr(msg, 'content'):
                    plan = extract_plan_from_response(msg.content)
                    if plan:
                        result['plan'] = plan
                        break
        
        return result
    
    return enhanced_node


# Patch the graph creation to use enhanced nodes
original_create_python_agent_graph = create_python_agent_graph
original_create_sql_agent_graph = create_sql_agent_graph

def create_python_agent_graph(sherlock_mode: bool = True):
    """Create Python agent graph with plan extraction"""
    builder = StateGraph(PythonAgentState)
    assistant = create_python_assistant_node_enhanced(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    
    def should_continue(state: PythonAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()

def create_sql_agent_graph(sherlock_mode: bool = True):
    """Create SQL agent graph with plan extraction"""
    builder = StateGraph(SQLAgentState)
    
    # Use both plotting and plan extraction
    assistant = create_sql_assistant_node_with_integrated_plots(sherlock_mode=sherlock_mode)
    
    def assistant_with_plan(state):
        result = assistant(state)
        # Extract plan if not already present
        if 'plan' not in result and result.get('messages'):
            for msg in result['messages']:
                if hasattr(msg, 'content'):
                    plan = extract_plan_from_response(msg.content)
                    if plan:
                        result['plan'] = plan
                        break
        return result
    
    builder.add_node("assistant", assistant_with_plan)
    builder.add_edge(START, "assistant")
    
    def should_continue(state: SQLAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()


# Fix the DictReturningMixin to work with your existing code
def fixed_dict_query(self, question: str) -> Dict[str, Any]:
    """Fixed query method that properly handles plan extraction"""
    # Call the parent class query (not DictReturningMixin's query)
    # This calls the actual agent's query method which returns a string
    final_text = super(DictReturningMixin, self).query(question)
    state = self.state
    
    # Get the output as a DataFrame using the ResultFormatter
    final_output = state.get("final_output")
    output_df = ResultFormatter.convert_to_dataframe(final_output)
    
    # Extract code from various sources
    code = (state.get("python") or 
            state.get("sql_query") or 
            self._extract_code(state.get("plan", "")) or
            self._extract_code(final_text))
    
    # Extract plan from state or from final_text
    plan = state.get("plan", "")
    if not plan and isinstance(final_text, str):
        plan = extract_plan_from_response(final_text) or ""
        # Also check in messages
        if not plan:
            for msg in reversed(state.get("messages", [])):
                if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                    plan = extract_plan_from_response(msg.content)
                    if plan:
                        break
    
    result = {
        "output": output_df,
        "final_result": final_text,
        "code": code,
        "sql_query": state.get("sql_query", ""),
        "explanation": state.get("explanation") or state.get("user_friendly_result", ""),
        "plan": plan or "",
        "verbose": state.get("raw_verbose") or "\n".join(
            m.content for m in state.get("messages", []) if hasattr(m, 'content')
        ),
        "react_history": state.get("react_history", []),
        "verification_status": state.get("verification_status", "pending"),
        "error_history": state.get("error_history", []),
        "retry_count": state.get("retry_count", 0)
    }
    
    # Add plot information if available
    if hasattr(self, 'get_plots'):
        plots = self.get_plots()
        if plots:
            result['plots'] = plots
            result['plot_count'] = len(plots)
            result['plot_images'] = [p.get('base64') for p in plots if p.get('success')]
    
    return result

# Apply the fix to DictReturningMixin
DictReturningMixin.query = fixed_dict_query


# Add helper methods to all agents
def get_plan(self) -> str:
    """Get the extracted plan from the last query"""
    plan = self.state.get('plan', '')
    
    # If not in state, try to extract from messages
    if not plan and self.state.get('messages'):
        for msg in reversed(self.state['messages']):
            if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                extracted = extract_plan_from_response(msg.content)
                if extracted:
                    self.state['plan'] = extracted  # Cache it
                    return extracted
    
    return plan

# Add to all agent classes
TableAnalysisAgent.get_plan = get_plan
WatsonTableAnalysisAgent.get_plan = get_plan
SQLTableAnalysisAgent.get_plan = get_plan
WatsonSQLTableAnalysisAgent.get_plan = get_plan


# Debug helper
def debug_components(self):
    """Debug method to show all extracted components"""
    print("=== EXTRACTED COMPONENTS ===")
    print(f"Plan: {self.get_plan()[:200]}..." if self.get_plan() else "Plan: None")
    
    if hasattr(self.state, 'sql_query'):
        print(f"SQL Query: {self.state.get('sql_query', 'None')[:200]}...")
    if hasattr(self.state, 'python'):
        print(f"Python Code: {self.state.get('python', 'None')[:200]}...")
        
    print(f"Explanation: {self.state.get('explanation', 'None')[:200]}...")
    
    if hasattr(self.state, 'plots'):
        print(f"Plots: {len(self.state.get('plots', []))} plots")
        
    final_output = self.state.get('final_output')
    if hasattr(final_output, 'shape'):
        print(f"Final Output Shape: {final_output.shape}")
    elif final_output is not None:
        print(f"Final Output Type: {type(final_output).__name__}")
    else:
        print("Final Output: None")
    print("===========================")

# Add to all agent classes
TableAnalysisAgent.debug_components = debug_components
WatsonTableAnalysisAgent.debug_components = debug_components
SQLTableAnalysisAgent.debug_components = debug_components
WatsonSQLTableAnalysisAgent.debug_components = debug_components

import re
from typing import Dict, Any, Optional, Union

# Enhanced _extract_code method that handles various input types
def safe_extract_code(self, plan_or_reply: Union[str, dict, Any]) -> Optional[str]:
    """Safely extract code from the response, handling various input types"""
    # Convert to string if not already
    if isinstance(plan_or_reply, dict):
        # If it's a dict, try to get a string representation
        plan_or_reply = str(plan_or_reply.get('content', '')) if 'content' in plan_or_reply else str(plan_or_reply)
    elif not isinstance(plan_or_reply, str):
        # Convert any other type to string
        plan_or_reply = str(plan_or_reply) if plan_or_reply is not None else ""
    
    # Now safely search for code blocks
    # Try Python first
    python_match = re.search(r"```python\n(.*?)```", plan_or_reply, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()
    
    # Try SQL
    sql_match = re.search(r"```sql\n(.*?)```", plan_or_reply, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try generic code block
    generic_match = re.search(r"```\n(.*?)```", plan_or_reply, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    
    return None

# Replace the _extract_code method in DictReturningMixin
DictReturningMixin._extract_code = safe_extract_code


# Enhanced extract_plan_from_response that's more robust
def extract_plan_from_response(response_content: Union[str, Any]) -> Optional[str]:
    """Extract the plan section from assistant's response"""
    # Handle case where response_content might not be a string
    if not isinstance(response_content, str):
        if hasattr(response_content, 'content'):
            response_content = response_content.content
        else:
            response_content = str(response_content) if response_content else ""
    
    # Look for **Plan**: or similar patterns
    plan_patterns = [
        r'\*\*Plan\*\*:\s*(.*?)(?=\*\*(?:Query|SQL|Code|Visualization|Explanation)|```|$)',
        r'Plan:\s*(.*?)(?=Query:|SQL:|Code:|Visualization:|Explanation:|```|$)',
        r'\*\*Plan\*\*\s*\n(.*?)(?=\*\*|Query:|SQL:|Code:|```|$)',
        r'## Plan\s*\n(.*?)(?=##|```|$)',
    ]
    
    for pattern in plan_patterns:
        match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)
        if match:
            plan = match.group(1).strip()
            # Clean up the plan text
            plan = re.sub(r'\n\s*\n', '\n', plan)  # Remove empty lines
            plan = re.sub(r'^\s*[-*]\s*', '', plan, flags=re.MULTILINE)  # Remove bullet points
            return plan
    
    return None


# Comprehensive fixed query method for DictReturningMixin
def comprehensive_dict_query(self, question: str) -> Dict[str, Any]:
    """Comprehensive query method that handles all cases properly"""
    try:
        # Call the parent class query method
        final_text = super(DictReturningMixin, self).query(question)
        state = self.state
        
        # Initialize result dictionary with defaults
        result = {
            "output": pd.DataFrame(),  # Empty DataFrame as default
            "final_result": "",
            "code": "",
            "explanation": "",
            "plan": "",
            "verbose": "",
            "react_history": [],
            "verification_status": "pending",
            "error_history": [],
            "retry_count": 0
        }
        
        # Update final_result
        if isinstance(final_text, str):
            result["final_result"] = final_text
        else:
            result["final_result"] = str(final_text)
        
        # Get the output DataFrame
        final_output = state.get("final_output")
        if final_output is not None:
            try:
                result["output"] = ResultFormatter.convert_to_dataframe(final_output)
            except:
                # If conversion fails, keep empty DataFrame
                pass
        
        # Extract code safely
        code_sources = [
            state.get("python"),
            state.get("sql_query"),
            state.get("code")
        ]
        
        # Try to get code from state first
        for code_source in code_sources:
            if code_source:
                result["code"] = code_source
                break
        
        # If no code in state, try extracting from content
        if not result["code"]:
            # Try from plan
            if state.get("plan"):
                extracted = self._extract_code(state.get("plan"))
                if extracted:
                    result["code"] = extracted
            
            # Try from final_text
            if not result["code"] and final_text:
                extracted = self._extract_code(final_text)
                if extracted:
                    result["code"] = extracted
            
            # Try from messages
            if not result["code"]:
                for msg in reversed(state.get("messages", [])):
                    if hasattr(msg, 'content'):
                        extracted = self._extract_code(msg.content)
                        if extracted:
                            result["code"] = extracted
                            break
        
        # Extract plan
        plan = state.get("plan", "")
        if not plan:
            # Try from final_text
            if isinstance(final_text, str):
                plan = extract_plan_from_response(final_text) or ""
            
            # Try from messages
            if not plan:
                for msg in reversed(state.get("messages", [])):
                    if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                        plan = extract_plan_from_response(msg.content)
                        if plan:
                            break
        
        result["plan"] = plan
        
        # SQL-specific fields
        if hasattr(state, 'sql_query') and state.get('sql_query'):
            result["sql_query"] = state.get('sql_query', '')
        
        # Get explanation
        explanation = state.get("explanation", "")
        if not explanation:
            explanation = state.get("user_friendly_result", "")
        if not explanation and isinstance(final_text, str):
            # Try to extract explanation from final_text
            exp_match = re.search(r'\*\*Explanation\*\*:?\s*(.*?)(?=\*\*|$)', final_text, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()
        result["explanation"] = explanation
        
        # Build verbose log
        verbose_parts = []
        if state.get("raw_verbose"):
            verbose_parts.append(state.get("raw_verbose"))
        else:
            # Build from messages
            for msg in state.get("messages", []):
                if hasattr(msg, 'content'):
                    msg_type = getattr(msg, 'type', 'unknown')
                    verbose_parts.append(f"[{msg_type.upper()}]: {msg.content}")
        
        result["verbose"] = "\n\n".join(verbose_parts)
        
        # Copy other state fields
        result["react_history"] = state.get("react_history", [])
        result["verification_status"] = state.get("verification_status", "pending")
        result["error_history"] = state.get("error_history", [])
        result["retry_count"] = state.get("retry_count", 0)
        
        # Add plot information if available
        if hasattr(self, 'get_plots'):
            plots = self.get_plots() if hasattr(self, 'get_plots') else state.get('plots', [])
            if plots:
                result['plots'] = plots
                result['plot_count'] = len(plots)
                result['plot_images'] = [p.get('base64') for p in plots if p.get('success')]
        
        # Special handling for cases where only verbose contains the answer
        # (like schema queries)
        if not result["output"].empty or result["code"] or final_output is not None:
            # We have structured output
            pass
        else:
            # No structured output - the answer might be in the text
            # Create a DataFrame with the response
            if final_text and isinstance(final_text, str):
                # Check if this looks like a schema or table listing
                if any(keyword in final_text.lower() for keyword in ['table', 'schema', 'column', 'structure']):
                    # Create a simple DataFrame with the text response
                    result["output"] = pd.DataFrame({"Response": [final_text]})
        
        return result
        
    except Exception as e:
        # If anything fails, return a safe default with error info
        return {
            "output": pd.DataFrame({"Error": [str(e)]}),
            "final_result": f"Error occurred: {str(e)}",
            "code": "",
            "explanation": f"An error occurred while processing the query: {str(e)}",
            "plan": "",
            "verbose": f"Error: {str(e)}\n\nTraceback: {traceback.format_exc()}",
            "react_history": [],
            "verification_status": "error",
            "error_history": [str(e)],
            "retry_count": 0
        }

# Apply the comprehensive fix
DictReturningMixin.query = comprehensive_dict_query


# Add a method to get raw text response for schema-type queries
def get_text_response(self) -> str:
    """Get the raw text response from the last query"""
    # Check final_result from state
    if hasattr(self, 'state') and self.state.get('final_result'):
        return self.state.get('final_result')
    
    # Check last AI message
    if hasattr(self, 'state') and self.state.get('messages'):
        for msg in reversed(self.state['messages']):
            if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                return msg.content
    
    return "No response available"

# Add to all agent classes
TableAnalysisAgent.get_text_response = get_text_response
WatsonTableAnalysisAgent.get_text_response = get_text_response
SQLTableAnalysisAgent.get_text_response = get_text_response
WatsonSQLTableAnalysisAgent.get_text_response = get_text_response


# Enhanced debug method
def enhanced_debug_components(self):
    """Enhanced debug method to show all extracted components"""
    print("=== EXTRACTED COMPONENTS ===")
    print(f"Plan: {self.get_plan()[:200]}..." if self.get_plan() else "Plan: None")
    
    if hasattr(self.state, 'sql_query'):
        print(f"SQL Query: {self.state.get('sql_query', 'None')}")
    if hasattr(self.state, 'python'):
        print(f"Python Code: {(self.state.get('python', '') or '')[:200]}...")
        
    print(f"Explanation: {(self.state.get('explanation', '') or '')[:200]}...")
    
    if hasattr(self.state, 'plots'):
        print(f"Plots: {len(self.state.get('plots', []))} plots")
        
    final_output = self.state.get('final_output')
    if final_output is not None:
        if hasattr(final_output, 'shape'):
            print(f"Final Output Shape: {final_output.shape}")
        else:
            print(f"Final Output Type: {type(final_output).__name__}")
    else:
        print("Final Output: None")
    
    # Show text response for schema queries
    print(f"\nText Response Preview: {self.get_text_response()[:200]}...")
    print("===========================")

# Replace debug method
TableAnalysisAgent.debug_components = enhanced_debug_components
WatsonTableAnalysisAgent.debug_components = enhanced_debug_components
SQLTableAnalysisAgent.debug_components = enhanced_debug_components
WatsonSQLTableAnalysisAgent.debug_components = enhanced_debug_components

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional, Dict, Any, List

# First, patch the PythonAgentState to include plots field
def patch_python_agent_state():
    """Patch PythonAgentState to include plots field"""
    if 'plots' not in PythonAgentState.__annotations__:
        PythonAgentState.__annotations__['plots'] = List[Dict[str, Any]]

# Apply the patch
patch_python_agent_state()

# Enhanced Python assistant node with integrated plot execution
def create_python_assistant_node_with_plots(sherlock_mode: bool = True):
    """Factory function to create Python assistant nodes with integrated plot execution"""
    
    def assistant_node(state: PythonAgentState) -> dict:
        """Python assistant node with integrated plot execution capability"""
        
        verbose = state.get("verbose", False)
        agent_name = "SHERLOCK" if sherlock_mode else "WATSON"
        
        if verbose:
            print(f"\n🔍 {agent_name} PYTHON STEP {state.get('step', 0) + 1}: Starting analysis...")
        
        # Initialize plots if not present
        if 'plots' not in state:
            state['plots'] = []
        
        # Initialize ReAct manager
        react_manager = ReactCycleManager(
            max_retries=state.get("max_retries", 3),
            enable_verification=True
        )
        
        # Check if we should verify previous result
        if react_manager.should_verify(state) and state.get("final_output") is not None:
            verification_msg = react_manager.create_verification_instruction(
                state.get("final_output"),
                state.get("python", "")
            )
            state["messages"].append(HumanMessage(content=verification_msg))
            react_manager.update_history(state, "verification")
        
        msgs = state["messages"].copy()
        current_step = state.get("step", 0)
        max_steps = state.get("max_steps", 10)
        
        # Build system prompt based on agent type
        if sherlock_mode:
            sys_prompt = _build_sherlock_python_prompt_with_plot(state)
        else:
            sys_prompt = _build_watson_python_prompt_with_plot(state)
        
        # Add system message if not present
        if not any(m.type == "system" for m in msgs):
            msgs.insert(0, SystemMessage(content=sys_prompt))
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini")
            assistant_reply = llm.invoke(msgs)
            
            # Initialize output variables
            user_friendly_result = ""
            execution_result = ""
            error = None
            executed_code = ""
            final_output = None
            explanation = ""
            verification_status = state.get("verification_status", "pending")
            new_plots = []
            
            # Check for verification confirmation
            if "VERIFIED:" in assistant_reply.content:
                verification_status = "verified"
                explanation = re.search(r"VERIFIED:\s*(.*)", assistant_reply.content, re.DOTALL).group(1)
                return {
                    "messages": [assistant_reply],
                    "verification_status": verification_status,
                    "explanation": explanation,
                    "step": current_step + 1
                }
            
            # Extract and execute code
            code_blocks = re.findall(r'```python\n(.*?)\n```', assistant_reply.content, re.DOTALL)
            
            if code_blocks and state.get("dataframes"):
                executor = SafeCodeExecutor(state["dataframes"], watson_mode=not sherlock_mode)
                
                for code_block in code_blocks:
                    executed_code = code_block.strip()
                    
                    # Check if this is plotting code
                    is_plotting_code = any(keyword in executed_code.lower() for keyword in 
                                         ['plt.', 'matplotlib', 'seaborn', 'sns.', 'plot(', 'bar(', 'scatter(', 'hist('])
                    
                    if is_plotting_code:
                        # Execute plotting code with plot capture
                        plot_result = execute_python_plot_code(executed_code, state["dataframes"])
                        if plot_result['success']:
                            new_plots.append(plot_result)
                            if verbose:
                                print(f"✅ Plot generated successfully")
                            # Set plot as final output for display
                            final_output = plot_result.get('data_used', pd.DataFrame())
                            user_friendly_result = f"✅ **Plot generated successfully**\n\n📊 Plot created with {len(final_output)} data points"
                            execution_result = "Plot generated and saved"
                        else:
                            error = plot_result['error']
                            if verbose:
                                print(f"❌ Plot failed: {error}")
                    else:
                        # Regular code execution
                        exec_result = executor.execute_code(executed_code)
                        
                        if exec_result['success']:
                            final_output = exec_result.get('result')
                            user_friendly_result = ResultFormatter.format_dataframe_result(final_output) if isinstance(final_output, pd.DataFrame) else f"✅ **Code executed successfully**"
                            execution_result = str(final_output) if final_output is not None else ""
                            
                            # Extract explanation
                            explanation_match = re.search(r'\*\*Explanation\*\*:?(.*?)(?=\*\*|$)', 
                                                        assistant_reply.content, re.DOTALL | re.IGNORECASE)
                            if explanation_match:
                                explanation = explanation_match.group(1).strip()
                        else:
                            error = exec_result['error']
                            error_type = exec_result.get('error_type', 'unknown')
                            
                            # Handle retry logic
                            if react_manager.should_retry(state):
                                retry_instruction = react_manager.create_retry_instruction(
                                    error, error_type, 
                                    f"{agent_name} Python analysis"
                                )
                                
                                # Recursive retry
                                state["messages"].append(assistant_reply)
                                state["messages"].append(HumanMessage(content=retry_instruction))
                                state["retry_count"] = state.get("retry_count", 0) + 1
                                state["error"] = error
                                
                                react_manager.update_history(state, "retry")
                                return assistant_node(state)
                            
                            user_friendly_result = f"❌ **Final Error:** {error}"
                            break
            
            # Build response
            raw_verbose = _build_verbose_log(agent_name, "PYTHON", state, assistant_reply, executed_code, execution_result, error)
            
            full_response = assistant_reply.content
            if user_friendly_result:
                full_response += f"\n\n{user_friendly_result}"
            
            # Add plot info if plots were created
            if new_plots:
                full_response += f"\n\n📊 **Plots Generated:** {len(new_plots)} plot(s) created successfully"
            
            # Append new plots to existing plots in state
            existing_plots = state.get('plots', [])
            all_plots = existing_plots + new_plots
            
            return {
                "messages": [AIMessage(content=full_response)],
                "step": current_step + 1,
                "python": executed_code,
                "execution_result": execution_result,
                "user_friendly_result": user_friendly_result,
                "error": error,
                "final_output": final_output,
                "explanation": explanation or user_friendly_result,
                "raw_verbose": raw_verbose,
                "verification_status": verification_status,
                "react_history": react_manager.history,
                "plots": all_plots  # Return all plots (existing + new)
            }
            
        except Exception as e:
            error_msg = f"{agent_name} assistant error: {str(e)}"
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
                "step": current_step + 1,
                "error": error_msg,
                "plots": state.get('plots', [])  # Preserve existing plots
            }
    
    return assistant_node


def execute_python_plot_code(code: str, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Execute Python plotting code with dataframes
    
    Args:
        code: Python plotting code
        dataframes: Dictionary of available dataframes
        
    Returns:
        Dict with plot information
    """
    # Reset matplotlib
    plt.close('all')
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 6))
    
    # Prepare execution environment
    safe_builtins = {
        'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        'range': range, 'enumerate': enumerate, 'zip': zip,
        'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
        'sorted': sorted, 'reversed': reversed, 'print': print,
        'type': type, 'isinstance': isinstance, 'hasattr': hasattr
    }
    
    plot_globals = {
        '__builtins__': safe_builtins,
        'plt': plt,
        'pd': pd,
        'np': np,
        'sns': sns,
        'dataframes': {name: df.copy() for name, df in dataframes.items()},
        'df': dataframes.get('main', list(dataframes.values())[0]).copy() if dataframes else pd.DataFrame(),
        'fig': fig,
    }
    
    # Add individual dataframes to globals for easy access
    for name, df in dataframes.items():
        plot_globals[name] = df.copy()
    
    try:
        # Execute the plotting code
        plot_locals = {}
        exec(code, plot_globals, plot_locals)
        
        # Force a draw to ensure plot is rendered
        plt.draw()
        
        # Capture the current figure
        if plt.get_fignums():
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # Create base64 encoded image
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Try to determine what data was used
            data_used = plot_locals.get('result', dataframes.get('main', pd.DataFrame()) if dataframes else pd.DataFrame())
            
            # Store plot info
            plot_info = {
                'figure': fig,
                'base64': img_base64,
                'timestamp': pd.Timestamp.now(),
                'success': True,
                'error': None,
                'code': code,
                'data_used': data_used
            }
            
            return plot_info
        else:
            return {
                'success': False,
                'error': 'No plot was generated - ensure plotting commands are included',
                'code': code
            }
            
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f"Plot execution error: {str(e)}\nTraceback: {traceback.format_exc()}",
            'code': code
        }
    finally:
        plt.close('all')


def _build_sherlock_python_prompt_with_plot(state: PythonAgentState) -> str:
    """Build system prompt for Sherlock Python agent with plotting instructions"""
    
    # Get info about available dataframes
    df_info = []
    for name, df in state.get("dataframes", {}).items():
        df_info.append(f"- '{name}': {df.shape[0]} rows × {df.shape[1]} columns")
    
    dataframes_str = "\n".join(df_info) if df_info else "No dataframes available"
    
    return f"""You are Sherlock, a data analysis assistant with full data access and plotting capabilities.

Available DataFrames:
{dataframes_str}

Schema Information:
{state.get("schema", "Not available")}

Preview (first 5 rows of main dataframe):
{state.get("preview_md", "Not available")}

IMPORTANT GUIDELINES:
1. Access dataframes using: dataframes['name'] or df for the main dataframe
2. Write clear, well-commented Python code
3. ALWAYS assign your final result to a variable called `result`
4. For plotting: use matplotlib (plt) or seaborn (sns) - plots will be automatically captured
5. Format your response with these sections:
   - **Plan**: Explain your approach
   - **Code**: Show the executable Python code
   - **Explanation**: Interpret the results and their meaning
6. When working with multiple tables, explain relationships and joins
7. Handle edge cases and potential errors gracefully

For plotting:
- Use matplotlib.pyplot as plt or seaborn as sns
- The plot will be automatically saved and displayed
- Include proper labels, titles, and formatting
- Example plotting code:
  ```python
  import matplotlib.pyplot as plt
  
  # Your analysis code here
  result = df.groupby('category')['value'].sum()
  
  # Plotting
  plt.figure(figsize=(10, 6))
  plt.bar(result.index, result.values)
  plt.xlabel('Category')
  plt.ylabel('Value')
  plt.title('Sum of Values by Category')
  plt.xticks(rotation=45)
  plt.tight_layout()
  ```

Available libraries:
- pandas as pd
- numpy as np (if available)
- matplotlib.pyplot as plt (if available)
- seaborn as sns (if available)

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Retry count: {state.get('retry_count', 0)}/{state.get('max_retries', 3)}"""


def _build_watson_python_prompt_with_plot(state: PythonAgentState) -> str:
    """Build system prompt for Watson Python agent with plotting instructions"""
    
    # Get schema info without revealing data
    df_info = []
    for name, df in state.get("dataframes", {}).items():
        col_info = []
        for col in df.columns:
            dtype_info = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            col_info.append(f"    - {col}: {dtype_info}, {null_count} nulls")
        
        df_info.append(f"- '{name}': {df.shape[0]} rows × {df.shape[1]} columns\n" + "\n".join(col_info))
    
    dataframes_str = "\n".join(df_info) if df_info else "No dataframes available"
    
    return f"""You are Watson, a blind data analysis assistant with plotting capabilities. You cannot see raw data values.

CRITICAL CONSTRAINTS:
1. NO data peeking: no .head(), .tail(), .sample(), print(df), or slicing
2. Use only aggregations, statistics, and structural operations
3. Learn from errors to refine your approach
4. Make educated guesses based on column names and types

Available DataFrames (structure only):
{dataframes_str}

Column Information:
{state.get("column_info", "Not available")}

IMPORTANT GUIDELINES:
1. Access dataframes using: dataframes['name'] or df for the main dataframe
2. Use defensive coding to handle potential issues
3. ALWAYS assign your final result to a variable called `result`
4. For plotting aggregated data: use matplotlib (plt) or seaborn (sns)
5. Format your response with these sections:
   - **Plan**: Explain your blind analysis approach
   - **Code**: Show the executable Python code
   - **Explanation**: Interpret what the results reveal
6. Use .shape, .dtypes, .columns, .info() for structure
7. Use .isnull(), .describe(), .value_counts() for analysis
8. When errors occur, analyze them to learn about the data

For plotting (blind analysis only):
- Only plot aggregated results (counts, sums, means, etc.)
- No raw data visualization
- Example blind plotting:
  ```python
  # Aggregated analysis
  result = df.groupby('category').size()
  
  # Plot aggregated results
  plt.figure(figsize=(10, 6))
  plt.bar(result.index, result.values)
  plt.xlabel('Category')
  plt.ylabel('Count')
  plt.title('Distribution by Category')
  plt.xticks(rotation=45)
  plt.tight_layout()
  ```

Error Learning Strategy:
- KeyError: column name might be different
- TypeError: data type assumption incorrect
- ValueError: data format unexpected

Current step: {state.get('step', 0)}/{state.get('max_steps', 10)}
Previous errors: {state.get('error_history', [])[-3:]}
Current strategy: {state.get('current_strategy', 'Initial exploration')}"""


# Override the Python agent graph creation to use plot-enabled nodes
def create_python_agent_graph_with_plots(sherlock_mode: bool = True):
    """Create Python agent graph with integrated plot capabilities"""
    builder = StateGraph(PythonAgentState)
    
    # Use the new plot-enabled assistant node
    assistant = create_python_assistant_node_with_plots(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    
    builder.add_edge(START, "assistant")
    
    def should_continue(state: PythonAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()


# Patch the __init__ methods to initialize plots and use plot-enabled graphs
original_python_init = TableAnalysisAgent.__init__
original_watson_python_init = WatsonTableAnalysisAgent.__init__

def patched_python_init(self, *args, **kwargs):
    original_python_init(self, *args, **kwargs)
    # Initialize plots in state
    self.state['plots'] = []
    # Replace agent with plot-enabled version
    self.agent = create_python_agent_graph_with_plots(sherlock_mode=True)

def patched_watson_python_init(self, *args, **kwargs):
    original_watson_python_init(self, *args, **kwargs)
    # Initialize plots in state
    self.state['plots'] = []
    # Replace agent with plot-enabled version
    self.agent = create_python_agent_graph_with_plots(sherlock_mode=False)

TableAnalysisAgent.__init__ = patched_python_init
WatsonTableAnalysisAgent.__init__ = patched_watson_python_init


# Add plotting methods to Python agents
def get_plots(self) -> List[Dict[str, Any]]:
    """Get all generated plots"""
    return self.state.get('plots', [])

def enhanced_display_plot_python(self, index: int = -1):
    """Display a generated plot from base64 data"""
    plots = self.state.get('plots', [])
    
    if not plots:
        print("No plots available")
        return
    
    if abs(index) > len(plots):
        print(f"Invalid plot index. Available plots: {len(plots)}")
        return
    
    plot_info = plots[index]
    
    if plot_info.get('success') and plot_info.get('base64'):
        # Decode and display the plot
        import matplotlib.image as mpimg
        
        # Check if we're in Jupyter
        try:
            from IPython.display import Image, display
            img_data = base64.b64decode(plot_info['base64'])
            display(Image(img_data))
        except:
            # For non-Jupyter environments
            img_data = base64.b64decode(plot_info['base64'])
            img = mpimg.imread(io.BytesIO(img_data))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Plot {abs(index) + 1} of {len(plots)}")
            plt.show()
    else:
        print(f"Plot generation failed: {plot_info.get('error', 'Unknown error')}")

def get_plot_image_python(self, index: int = -1) -> Optional[str]:
    """Get plot as base64 string"""
    plots = self.state.get('plots', [])
    
    if not plots or abs(index) > len(plots):
        return None
    
    return plots[index].get('base64')

def save_plot_python(self, filename: str, index: int = -1):
    """Save plot to file"""
    plots = self.state.get('plots', [])
    
    if not plots:
        print("No plots available")
        return False
    
    if abs(index) > len(plots):
        print(f"Invalid plot index. Available plots: {len(plots)}")
        return False
    
    plot_info = plots[index]
    
    if plot_info.get('success') and plot_info.get('base64'):
        img_data = base64.b64decode(plot_info['base64'])
        with open(filename, 'wb') as f:
            f.write(img_data)
        print(f"Plot saved to {filename}")
        return True
    else:
        print(f"Plot generation failed: {plot_info.get('error', 'Unknown error')}")
        return False

def get_all_plots_python(self) -> List[Dict[str, Any]]:
    """Get all generated plots"""
    return self.state.get('plots', [])

def debug_plots_python(self):
    """Debug method to check plot status"""
    plots = self.state.get('plots', [])
    print(f"Total plots in state: {len(plots)}")
    for i, plot in enumerate(plots):
        print(f"Plot {i+1}:")
        print(f"  - Success: {plot.get('success', False)}")
        print(f"  - Has base64: {'base64' in plot}")
        print(f"  - Error: {plot.get('error', 'None')}")
        if 'code' in plot:
            print(f"  - Code preview: {plot['code'][:100]}...")

# Apply methods to Python agents
TableAnalysisAgent.get_plots = get_plots
TableAnalysisAgent.display_plot = enhanced_display_plot_python
TableAnalysisAgent.get_plot_image = get_plot_image_python
TableAnalysisAgent.save_plot = save_plot_python
TableAnalysisAgent.get_all_plots = get_all_plots_python
TableAnalysisAgent.debug_plots = debug_plots_python

WatsonTableAnalysisAgent.get_plots = get_plots
WatsonTableAnalysisAgent.display_plot = enhanced_display_plot_python
WatsonTableAnalysisAgent.get_plot_image = get_plot_image_python
WatsonTableAnalysisAgent.save_plot = save_plot_python
WatsonTableAnalysisAgent.get_all_plots = get_all_plots_python
WatsonTableAnalysisAgent.debug_plots = debug_plots_python

# =============================================================================
# DROP-IN FIX FOR PYTHON AGENT PLOTTING EXECUTION
# =============================================================================

def execute_python_plot_code_fixed(code: str, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Fixed version of execute_python_plot_code with proper import handling
    
    Args:
        code: Python plotting code
        dataframes: Dictionary of available dataframes
        
    Returns:
        Dict with plot information
    """
    # Reset matplotlib
    plt.close('all')
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 6))
    
    # Prepare execution environment with ALL necessary builtins
    safe_builtins = {
        'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        'range': range, 'enumerate': enumerate, 'zip': zip,
        'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
        'sorted': sorted, 'reversed': reversed, 'print': print,
        'type': type, 'isinstance': isinstance, 'hasattr': hasattr,
        '__import__': __import__,  # THIS WAS MISSING!
        'getattr': getattr, 'setattr': setattr,
        'map': map, 'filter': filter, 'any': any, 'all': all,
        'iter': iter, 'next': next, 'open': open  # Add more needed builtins
    }
    
    plot_globals = {
        '__builtins__': safe_builtins,
        'plt': plt,
        'pd': pd,
        'np': np,
        'sns': sns,
        'dataframes': {name: df.copy() for name, df in dataframes.items()},
        'df': dataframes.get('main', list(dataframes.values())[0]).copy() if dataframes else pd.DataFrame(),
        'main': dataframes.get('main', list(dataframes.values())[0]).copy() if dataframes else pd.DataFrame(),  # Add 'main' reference
        'fig': fig,
        # Pre-import common modules to avoid import issues
        'matplotlib': plt.matplotlib,
        'pyplot': plt
    }
    
    # Add individual dataframes to globals for easy access
    for name, df in dataframes.items():
        plot_globals[name] = df.copy()
    
    try:
        # Execute the plotting code
        plot_locals = {}
        exec(code, plot_globals, plot_locals)
        
        # Force a draw to ensure plot is rendered
        plt.draw()
        
        # Capture the current figure
        if plt.get_fignums():
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # Create base64 encoded image
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Try to determine what data was used
            data_used = plot_locals.get('result', dataframes.get('main', pd.DataFrame()) if dataframes else pd.DataFrame())
            
            # Store plot info
            plot_info = {
                'figure': fig,
                'base64': img_base64,
                'timestamp': pd.Timestamp.now(),
                'success': True,
                'error': None,
                'code': code,
                'data_used': data_used
            }
            
            return plot_info
        else:
            return {
                'success': False,
                'error': 'No plot was generated - ensure plotting commands are included',
                'code': code
            }
            
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f"Plot execution error: {str(e)}\nTraceback: {traceback.format_exc()}",
            'code': code
        }
    finally:
        plt.close('all')


# Also fix the SafeCodeExecutor to include __import__ for regular code execution
class SafeCodeExecutorFixed(SafeCodeExecutor):
    """Fixed version of SafeCodeExecutor with proper import handling"""
    
    def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Safely execute code with timeout and capture output - FIXED VERSION"""
        is_safe, safety_msg = self.is_code_safe(code)
        if not is_safe:
            return {
                'success': False,
                'error': f"Security check failed: {safety_msg}",
                'output': '',
                'result': None,
                'error_type': 'security'
            }
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            safe_builtins = {
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
                'sorted': sorted, 'reversed': reversed, 'print': print,
                'type': type, 'isinstance': isinstance, 'hasattr': hasattr,
                '__import__': __import__,  # ADD THIS!
                'getattr': getattr, 'setattr': setattr,
                'map': map, 'filter': filter, 'any': any, 'all': all
            }
            
            safe_globals = {
                '__builtins__': safe_builtins,
                'pd': pd,
                'dataframes': self.dataframes,
                'df': self.dataframes.get('main', pd.DataFrame()),
                'main': self.dataframes.get('main', pd.DataFrame()),  # Add 'main' reference
            }
            
            # Add individual dataframes to globals
            for name, df in self.dataframes.items():
                safe_globals[name] = df
            
            # Add optional libraries
            try:
                import numpy as np
                safe_globals['np'] = np
            except ImportError:
                pass
            
            try:
                import matplotlib.pyplot as plt
                safe_globals['plt'] = plt
                safe_globals['matplotlib'] = plt.matplotlib
            except ImportError:
                pass
                
            try:
                import seaborn as sns
                safe_globals['sns'] = sns
            except ImportError:
                pass
            
            safe_locals = {}
            exec(code, safe_globals, safe_locals)
            result = safe_locals.get('result')
            output = captured_output.getvalue()
            
            return {
                'success': True,
                'error': None,
                'output': output,
                'result': result,
                'error_type': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'output': captured_output.getvalue(),
                'result': None,
                'error_type': type(e).__name__
            }
        finally:
            sys.stdout = old_stdout


# Replace the execute_python_plot_code function
execute_python_plot_code = execute_python_plot_code_fixed

# Also need to update the assistant node to use the fixed executor
def create_python_assistant_node_with_plots_fixed(sherlock_mode: bool = True):
    """Factory function to create Python assistant nodes with FIXED plot execution"""
    
    def assistant_node(state: PythonAgentState) -> dict:
        """Python assistant node with FIXED plot execution capability"""
        
        verbose = state.get("verbose", False)
        agent_name = "SHERLOCK" if sherlock_mode else "WATSON"
        
        if verbose:
            print(f"\n🔍 {agent_name} PYTHON STEP {state.get('step', 0) + 1}: Starting analysis...")
        
        # Initialize plots if not present
        if 'plots' not in state:
            state['plots'] = []
        
        # Initialize ReAct manager
        react_manager = ReactCycleManager(
            max_retries=state.get("max_retries", 3),
            enable_verification=True
        )
        
        # Check if we should verify previous result
        if react_manager.should_verify(state) and state.get("final_output") is not None:
            verification_msg = react_manager.create_verification_instruction(
                state.get("final_output"),
                state.get("python", "")
            )
            state["messages"].append(HumanMessage(content=verification_msg))
            react_manager.update_history(state, "verification")
        
        msgs = state["messages"].copy()
        current_step = state.get("step", 0)
        max_steps = state.get("max_steps", 10)
        
        # Build system prompt based on agent type
        if sherlock_mode:
            sys_prompt = _build_sherlock_python_prompt_with_plot(state)
        else:
            sys_prompt = _build_watson_python_prompt_with_plot(state)
        
        # Add system message if not present
        if not any(m.type == "system" for m in msgs):
            msgs.insert(0, SystemMessage(content=sys_prompt))
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini")
            assistant_reply = llm.invoke(msgs)
            
            # Initialize output variables
            user_friendly_result = ""
            execution_result = ""
            error = None
            executed_code = ""
            final_output = None
            explanation = ""
            verification_status = state.get("verification_status", "pending")
            new_plots = []
            
            # Check for verification confirmation
            if "VERIFIED:" in assistant_reply.content:
                verification_status = "verified"
                explanation = re.search(r"VERIFIED:\s*(.*)", assistant_reply.content, re.DOTALL).group(1)
                return {
                    "messages": [assistant_reply],
                    "verification_status": verification_status,
                    "explanation": explanation,
                    "step": current_step + 1
                }
            
            # Extract and execute code
            code_blocks = re.findall(r'```python\n(.*?)\n```', assistant_reply.content, re.DOTALL)
            
            if code_blocks and state.get("dataframes"):
                # Use the FIXED executor
                executor = SafeCodeExecutorFixed(state["dataframes"], watson_mode=not sherlock_mode)
                
                for code_block in code_blocks:
                    executed_code = code_block.strip()
                    
                    # Check if this is plotting code
                    is_plotting_code = any(keyword in executed_code.lower() for keyword in 
                                         ['plt.', 'matplotlib', 'seaborn', 'sns.', 'plot(', 'bar(', 'scatter(', 'hist('])
                    
                    if is_plotting_code:
                        # Execute plotting code with plot capture using FIXED function
                        plot_result = execute_python_plot_code_fixed(executed_code, state["dataframes"])
                        if plot_result['success']:
                            new_plots.append(plot_result)
                            if verbose:
                                print(f"✅ Plot generated successfully")
                            # Set plot as final output for display
                            final_output = plot_result.get('data_used', pd.DataFrame())
                            user_friendly_result = f"✅ **Plot generated successfully**\n\n📊 Plot created with {len(final_output)} data points"
                            execution_result = "Plot generated and saved"
                        else:
                            error = plot_result['error']
                            if verbose:
                                print(f"❌ Plot failed: {error}")
                    else:
                        # Regular code execution
                        exec_result = executor.execute_code(executed_code)
                        
                        if exec_result['success']:
                            final_output = exec_result.get('result')
                            user_friendly_result = ResultFormatter.format_dataframe_result(final_output) if isinstance(final_output, pd.DataFrame) else f"✅ **Code executed successfully**"
                            execution_result = str(final_output) if final_output is not None else ""
                            
                            # Extract explanation
                            explanation_match = re.search(r'\*\*Explanation\*\*:?(.*?)(?=\*\*|$)', 
                                                        assistant_reply.content, re.DOTALL | re.IGNORECASE)
                            if explanation_match:
                                explanation = explanation_match.group(1).strip()
                        else:
                            error = exec_result['error']
                            error_type = exec_result.get('error_type', 'unknown')
                            
                            # Handle retry logic
                            if react_manager.should_retry(state):
                                retry_instruction = react_manager.create_retry_instruction(
                                    error, error_type, 
                                    f"{agent_name} Python analysis"
                                )
                                
                                # Recursive retry
                                state["messages"].append(assistant_reply)
                                state["messages"].append(HumanMessage(content=retry_instruction))
                                state["retry_count"] = state.get("retry_count", 0) + 1
                                state["error"] = error
                                
                                react_manager.update_history(state, "retry")
                                return assistant_node(state)
                            
                            user_friendly_result = f"❌ **Final Error:** {error}"
                            break
            
            # Build response
            raw_verbose = _build_verbose_log(agent_name, "PYTHON", state, assistant_reply, executed_code, execution_result, error)
            
            full_response = assistant_reply.content
            if user_friendly_result:
                full_response += f"\n\n{user_friendly_result}"
            
            # Add plot info if plots were created
            if new_plots:
                full_response += f"\n\n📊 **Plots Generated:** {len(new_plots)} plot(s) created successfully"
            
            # Append new plots to existing plots in state
            existing_plots = state.get('plots', [])
            all_plots = existing_plots + new_plots
            
            return {
                "messages": [AIMessage(content=full_response)],
                "step": current_step + 1,
                "python": executed_code,
                "execution_result": execution_result,
                "user_friendly_result": user_friendly_result,
                "error": error,
                "final_output": final_output,
                "explanation": explanation or user_friendly_result,
                "raw_verbose": raw_verbose,
                "verification_status": verification_status,
                "react_history": react_manager.history,
                "plots": all_plots  # Return all plots (existing + new)
            }
            
        except Exception as e:
            error_msg = f"{agent_name} assistant error: {str(e)}"
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
                "step": current_step + 1,
                "error": error_msg,
                "plots": state.get('plots', [])  # Preserve existing plots
            }
    
    return assistant_node


# Override the graph creation function with fixed version
def create_python_agent_graph_with_plots_fixed(sherlock_mode: bool = True):
    """Create Python agent graph with FIXED plot capabilities"""
    builder = StateGraph(PythonAgentState)
    
    # Use the FIXED plot-enabled assistant node
    assistant = create_python_assistant_node_with_plots_fixed(sherlock_mode=sherlock_mode)
    builder.add_node("assistant", assistant)
    
    builder.add_edge(START, "assistant")
    
    def should_continue(state: PythonAgentState) -> str:
        if check_termination_condition(state):
            return "__end__"
        return "assistant"
    
    builder.add_conditional_edges("assistant", should_continue)
    return builder.compile()


# Replace the patched init methods with fixed versions
def patched_python_init_fixed(self, *args, **kwargs):
    original_python_init(self, *args, **kwargs)
    # Initialize plots in state
    self.state['plots'] = []
    # Replace agent with FIXED plot-enabled version
    self.agent = create_python_agent_graph_with_plots_fixed(sherlock_mode=True)

def patched_watson_python_init_fixed(self, *args, **kwargs):
    original_watson_python_init(self, *args, **kwargs)
    # Initialize plots in state
    self.state['plots'] = []
    # Replace agent with FIXED plot-enabled version
    self.agent = create_python_agent_graph_with_plots_fixed(sherlock_mode=False)

# Apply the fixed init methods
TableAnalysisAgent.__init__ = patched_python_init_fixed
WatsonTableAnalysisAgent.__init__ = patched_watson_python_init_fixed


# =============================================================================
# ROBUST MULTI-SOURCE DATA HANDLER FOR PYTHON AGENTS
# =============================================================================

import pandas as pd
from typing import Dict, Any, Union, Optional, List
import os

def normalize_dataframes_input(
    df: Optional[pd.DataFrame] = None,
    dataframes: Optional[Union[Dict[str, pd.DataFrame], pd.ExcelFile, str, List[str]]] = None,
    file_path: Optional[str] = None,
    sheet_names: Optional[List[str]] = None,
    max_sheets: int = 10,
    sample_rows: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Robust function to normalize various input types into a dictionary of DataFrames
    
    Args:
        df: Single DataFrame (backward compatibility)
        dataframes: Various types - Dict, ExcelFile, file path, list of paths
        file_path: Path to Excel/CSV file
        sheet_names: Specific sheet names to load (for Excel)
        max_sheets: Maximum number of sheets to load
        sample_rows: Number of rows to sample from each sheet (None = all rows)
    
    Returns:
        Dictionary of DataFrames with standardized names
    
    Handles:
        - Single DataFrame
        - Dictionary of DataFrames  
        - pd.ExcelFile object
        - Excel file path (string)
        - CSV file path (string)
        - List of file paths
        - Directory path (loads all Excel/CSV files)
    """
    
    result_dfs = {}
    
    # Case 1: Single DataFrame provided
    if df is not None and isinstance(df, pd.DataFrame):
        result_dfs['main'] = df.copy()
        if dataframes is None:
            return result_dfs
    
    # Case 2: Already a dictionary of DataFrames
    if isinstance(dataframes, dict) and all(isinstance(v, pd.DataFrame) for v in dataframes.values()):
        result_dfs.update({k: v.copy() for k, v in dataframes.items()})
        return result_dfs
    
    # Case 3: pd.ExcelFile object
    if isinstance(dataframes, pd.ExcelFile):
        excel_file = dataframes
        sheets_to_load = sheet_names if sheet_names else excel_file.sheet_names[:max_sheets]
        
        for sheet_name in sheets_to_load:
            try:
                sheet_df = excel_file.parse(sheet_name)
                if sample_rows:
                    sheet_df = sheet_df.head(sample_rows)
                
                # Clean sheet name for use as dict key
                clean_name = clean_sheet_name(sheet_name)
                result_dfs[clean_name] = sheet_df
                
                print(f"✅ Loaded sheet '{sheet_name}' as '{clean_name}': {sheet_df.shape}")
            except Exception as e:
                print(f"⚠️ Failed to load sheet '{sheet_name}': {e}")
        
        return result_dfs
    
    # Case 4: File path provided (either as dataframes or file_path)
    file_to_process = dataframes if isinstance(dataframes, str) else file_path
    
    if isinstance(file_to_process, str) and os.path.exists(file_to_process):
        if file_to_process.endswith(('.xlsx', '.xls')):
            # Excel file
            try:
                excel_file = pd.ExcelFile(file_to_process)
                sheets_to_load = sheet_names if sheet_names else excel_file.sheet_names[:max_sheets]
                
                for sheet_name in sheets_to_load:
                    try:
                        sheet_df = pd.read_excel(file_to_process, sheet_name=sheet_name)
                        if sample_rows:
                            sheet_df = sheet_df.head(sample_rows)
                        
                        clean_name = clean_sheet_name(sheet_name)
                        result_dfs[clean_name] = sheet_df
                        
                        print(f"✅ Loaded sheet '{sheet_name}' as '{clean_name}': {sheet_df.shape}")
                    except Exception as e:
                        print(f"⚠️ Failed to load sheet '{sheet_name}': {e}")
                        
            except Exception as e:
                print(f"❌ Failed to read Excel file: {e}")
                
        elif file_to_process.endswith('.csv'):
            # CSV file
            try:
                csv_df = pd.read_csv(file_to_process)
                if sample_rows:
                    csv_df = csv_df.head(sample_rows)
                
                filename = os.path.splitext(os.path.basename(file_to_process))[0]
                result_dfs[filename] = csv_df
                print(f"✅ Loaded CSV '{filename}': {csv_df.shape}")
                
            except Exception as e:
                print(f"❌ Failed to read CSV file: {e}")
    
    # Case 5: List of file paths
    elif isinstance(dataframes, list):
        for i, file_path in enumerate(dataframes):
            if isinstance(file_path, str) and os.path.exists(file_path):
                try:
                    if file_path.endswith(('.xlsx', '.xls')):
                        # For Excel, load first sheet only
                        df_temp = pd.read_excel(file_path)
                        if sample_rows:
                            df_temp = df_temp.head(sample_rows)
                        
                        filename = os.path.splitext(os.path.basename(file_path))[0]
                        result_dfs[filename] = df_temp
                        print(f"✅ Loaded Excel '{filename}': {df_temp.shape}")
                        
                    elif file_path.endswith('.csv'):
                        df_temp = pd.read_csv(file_path)
                        if sample_rows:
                            df_temp = df_temp.head(sample_rows)
                        
                        filename = os.path.splitext(os.path.basename(file_path))[0]
                        result_dfs[filename] = df_temp
                        print(f"✅ Loaded CSV '{filename}': {df_temp.shape}")
                        
                except Exception as e:
                    print(f"⚠️ Failed to load file {file_path}: {e}")
    
    # Case 6: Directory path (load all Excel/CSV files)
    elif isinstance(dataframes, str) and os.path.isdir(dataframes):
        directory = dataframes
        files = [f for f in os.listdir(directory) 
                if f.endswith(('.xlsx', '.xls', '.csv'))][:max_sheets]
        
        for file in files:
            file_path = os.path.join(directory, file)
            try:
                if file.endswith(('.xlsx', '.xls')):
                    df_temp = pd.read_excel(file_path)
                elif file.endswith('.csv'):
                    df_temp = pd.read_csv(file_path)
                
                if sample_rows:
                    df_temp = df_temp.head(sample_rows)
                
                filename = os.path.splitext(file)[0]
                result_dfs[filename] = df_temp
                print(f"✅ Loaded '{filename}': {df_temp.shape}")
                
            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")
    
    # Ensure we have at least one DataFrame
    if not result_dfs:
        if df is not None:
            result_dfs['main'] = df.copy()
        else:
            print("⚠️ No valid DataFrames found, creating empty DataFrame")
            result_dfs['main'] = pd.DataFrame()
    
    # Ensure 'main' exists (use first DataFrame if 'main' not present)
    if 'main' not in result_dfs and result_dfs:
        first_key = list(result_dfs.keys())[0]
        result_dfs['main'] = result_dfs[first_key].copy()
        print(f"📋 Set '{first_key}' as 'main' DataFrame")
    
    return result_dfs


def clean_sheet_name(sheet_name: str) -> str:
    """Clean sheet name to be a valid Python identifier"""
    import re
    # Replace spaces and special characters with underscores
    clean = re.sub(r'[^\w\s]', '_', sheet_name)
    clean = re.sub(r'\s+', '_', clean)
    # Remove leading/trailing underscores
    clean = clean.strip('_')
    # Ensure it doesn't start with a number
    if clean and clean[0].isdigit():
        clean = f"sheet_{clean}"
    # Fallback for empty names
    if not clean:
        clean = "unnamed_sheet"
    return clean.lower()


def display_dataframes_summary(dataframes: Dict[str, pd.DataFrame]) -> None:
    """Display a summary of loaded DataFrames"""
    print(f"\n📊 Loaded {len(dataframes)} DataFrames:")
    print("=" * 60)
    
    for name, df in dataframes.items():
        print(f"📋 {name}:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        print()


# Enhanced Agent Classes with Robust Data Handling
class TableAnalysisAgentEnhanced(TableAnalysisAgent):
    """Enhanced TableAnalysisAgent with robust multi-source data handling"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 dataframes: Optional[Union[Dict[str, pd.DataFrame], pd.ExcelFile, str, List[str]]] = None,
                 file_path: Optional[str] = None,
                 sheet_names: Optional[List[str]] = None,
                 file_name: str = "uploaded_data",
                 max_steps: int = 10,
                 max_retries: int = 3,
                 verbose: bool = False,
                 sample_rows: Optional[int] = None,
                 max_sheets: int = 10):
        """
        Initialize with robust data handling
        
        Args:
            df: Single DataFrame (backward compatibility)
            dataframes: Various input types (Dict, ExcelFile, file path, etc.)
            file_path: Path to Excel/CSV file
            sheet_names: Specific sheet names to load
            file_name: Name identifier for the dataset
            max_steps: Maximum conversation steps
            max_retries: Maximum retry attempts
            verbose: Enable verbose logging
            sample_rows: Sample N rows from each sheet (None = all rows)
            max_sheets: Maximum number of sheets to load
        """
        
        # Normalize all input types to dictionary of DataFrames
        normalized_dfs = normalize_dataframes_input(
            df=df,
            dataframes=dataframes,
            file_path=file_path,
            sheet_names=sheet_names,
            max_sheets=max_sheets,
            sample_rows=sample_rows
        )
        
        if verbose:
            display_dataframes_summary(normalized_dfs)
        
        # Call parent constructor with normalized data
        super().__init__(
            dataframes=normalized_dfs,
            file_name=file_name,
            max_steps=max_steps,
            max_retries=max_retries,
            verbose=verbose
        )


class WatsonTableAnalysisAgentEnhanced(WatsonTableAnalysisAgent):
    """Enhanced WatsonTableAnalysisAgent with robust multi-source data handling"""
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 dataframes: Optional[Union[Dict[str, pd.DataFrame], pd.ExcelFile, str, List[str]]] = None,
                 file_path: Optional[str] = None,
                 sheet_names: Optional[List[str]] = None,
                 file_name: str = "uploaded_data",
                 max_steps: int = 10,
                 max_retries: int = 3,
                 verbose: bool = False,
                 sample_rows: Optional[int] = None,
                 max_sheets: int = 10):
        """
        Initialize with robust data handling
        
        Args:
            df: Single DataFrame (backward compatibility)
            dataframes: Various input types (Dict, ExcelFile, file path, etc.)
            file_path: Path to Excel/CSV file
            sheet_names: Specific sheet names to load
            file_name: Name identifier for the dataset
            max_steps: Maximum conversation steps
            max_retries: Maximum retry attempts
            verbose: Enable verbose logging
            sample_rows: Sample N rows from each sheet (None = all rows)
            max_sheets: Maximum number of sheets to load
        """
        
        # Normalize all input types to dictionary of DataFrames
        normalized_dfs = normalize_dataframes_input(
            df=df,
            dataframes=dataframes,
            file_path=file_path,
            sheet_names=sheet_names,
            max_sheets=max_sheets,
            sample_rows=sample_rows
        )
        
        if verbose:
            display_dataframes_summary(normalized_dfs)
        
        # Call parent constructor with normalized data
        super().__init__(
            dataframes=normalized_dfs,
            file_name=file_name,
            max_steps=max_steps,
            max_retries=max_retries,
            verbose=verbose
        )


# Enhanced Dict-returning wrappers
class SherlockPyDictAgentEnhanced(DictReturningMixin, TableAnalysisAgentEnhanced):
    """Enhanced Sherlock Python agent with robust data handling and dict output"""
    pass

class WatsonPyDictAgentEnhanced(DictReturningMixin, WatsonTableAnalysisAgentEnhanced):
    """Enhanced Watson Python agent with robust data handling and dict output"""
    pass


# Convenience functions for common use cases
def load_excel_sheets(file_path: str, 
                     sheet_names: Optional[List[str]] = None,
                     max_sheets: int = 10,
                     sample_rows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load Excel sheets into a dictionary
    
    Args:
        file_path: Path to Excel file
        sheet_names: Specific sheets to load (None = all sheets)
        max_sheets: Maximum number of sheets to load
        sample_rows: Sample N rows from each sheet
    
    Returns:
        Dictionary of DataFrames
    """
    return normalize_dataframes_input(
        dataframes=file_path,
        sheet_names=sheet_names,
        max_sheets=max_sheets,
        sample_rows=sample_rows
    )


def create_agent_from_excel(file_path: str, 
                           agent_type: str = 'sherlock',
                           sheet_names: Optional[List[str]] = None,
                           max_sheets: int = 10,
                           sample_rows: Optional[int] = None,
                           verbose: bool = True) -> Union[SherlockPyDictAgentEnhanced, WatsonPyDictAgentEnhanced]:
    """
    Convenience function to create agents directly from Excel files
    
    Args:
        file_path: Path to Excel file
        agent_type: 'sherlock' or 'watson'
        sheet_names: Specific sheets to load
        max_sheets: Maximum sheets to load
        sample_rows: Sample N rows from each sheet
        verbose: Show loading summary
    
    Returns:
        Configured agent ready to use
    """
    if agent_type.lower() == 'sherlock':
        return SherlockPyDictAgentEnhanced(
            dataframes=file_path,
            sheet_names=sheet_names,
            max_sheets=max_sheets,
            sample_rows=sample_rows,
            verbose=verbose
        )
    elif agent_type.lower() == 'watson':
        return WatsonPyDictAgentEnhanced(
            dataframes=file_path,
            sheet_names=sheet_names,
            max_sheets=max_sheets,
            sample_rows=sample_rows,
            verbose=verbose
        )
    else:
        raise ValueError("agent_type must be 'sherlock' or 'watson'")


# Backward compatibility - patch original classes to use enhanced versions
def patch_original_classes():
    """Patch original classes to use enhanced data handling"""
    global TableAnalysisAgent, WatsonTableAnalysisAgent
    global SherlockPyDictAgent, WatsonPyDictAgent
    
    # Store original classes
    TableAnalysisAgent._original_init = TableAnalysisAgent.__init__
    WatsonTableAnalysisAgent._original_init = WatsonTableAnalysisAgent.__init__
    
    # Replace with enhanced versions
    TableAnalysisAgent.__init__ = TableAnalysisAgentEnhanced.__init__
    WatsonTableAnalysisAgent.__init__ = WatsonTableAnalysisAgentEnhanced.__init__
    
    # print("✅ Original agent classes patched with enhanced data handling!")


# Apply the patches
patch_original_classes()


# =============================================================================
# SIMPLE DROP-IN FIX FOR MULTIPLE DATAFRAMES INPUT
# =============================================================================

import pandas as pd
from typing import Dict, Any, Union, Optional, List
import os
import re

def normalize_dataframes_input(
    df: Optional[pd.DataFrame] = None,
    dataframes: Optional[Union[Dict[str, pd.DataFrame], pd.ExcelFile, str, List[str]]] = None,
    file_path: Optional[str] = None,
    sheet_names: Optional[List[str]] = None,
    max_sheets: int = 10,
    sample_rows: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Robust function to normalize various input types into a dictionary of DataFrames
    """
    
    result_dfs = {}
    
    # Case 1: Single DataFrame provided
    if df is not None and isinstance(df, pd.DataFrame):
        result_dfs['main'] = df.copy()
        if dataframes is None:
            return result_dfs
    
    # Case 2: Already a dictionary of DataFrames
    if isinstance(dataframes, dict) and all(isinstance(v, pd.DataFrame) for v in dataframes.values()):
        result_dfs.update({k: v.copy() for k, v in dataframes.items()})
        return result_dfs
    
    # Case 3: pd.ExcelFile object - YOUR USE CASE!
    if isinstance(dataframes, pd.ExcelFile):
        excel_file = dataframes
        sheets_to_load = sheet_names if sheet_names else excel_file.sheet_names[:max_sheets]
        
        for sheet_name in sheets_to_load:
            try:
                sheet_df = excel_file.parse(sheet_name)
                if sample_rows:
                    sheet_df = sheet_df.head(sample_rows)
                
                # Clean sheet name for use as dict key
                clean_name = clean_sheet_name(sheet_name)
                result_dfs[clean_name] = sheet_df
                
                print(f"✅ Loaded sheet '{sheet_name}' as '{clean_name}': {sheet_df.shape}")
            except Exception as e:
                print(f"⚠️ Failed to load sheet '{sheet_name}': {e}")
        
        # Ensure 'main' exists (use first sheet as main)
        if result_dfs and 'main' not in result_dfs:
            first_key = list(result_dfs.keys())[0]
            result_dfs['main'] = result_dfs[first_key].copy()
            print(f"📋 Set '{first_key}' as 'main' DataFrame")
        
        return result_dfs
    
    # Case 4: File path provided as string
    if isinstance(dataframes, str) and os.path.exists(dataframes):
        if dataframes.endswith(('.xlsx', '.xls')):
            # Excel file
            try:
                excel_file = pd.ExcelFile(dataframes)
                sheets_to_load = sheet_names if sheet_names else excel_file.sheet_names[:max_sheets]
                
                for sheet_name in sheets_to_load:
                    try:
                        sheet_df = pd.read_excel(dataframes, sheet_name=sheet_name)
                        if sample_rows:
                            sheet_df = sheet_df.head(sample_rows)
                        
                        clean_name = clean_sheet_name(sheet_name)
                        result_dfs[clean_name] = sheet_df
                        
                        print(f"✅ Loaded sheet '{sheet_name}' as '{clean_name}': {sheet_df.shape}")
                    except Exception as e:
                        print(f"⚠️ Failed to load sheet '{sheet_name}': {e}")
                        
            except Exception as e:
                print(f"❌ Failed to read Excel file: {e}")
                
        elif dataframes.endswith('.csv'):
            # CSV file
            try:
                csv_df = pd.read_csv(dataframes)
                if sample_rows:
                    csv_df = csv_df.head(sample_rows)
                
                filename = os.path.splitext(os.path.basename(dataframes))[0]
                result_dfs[filename] = csv_df
                result_dfs['main'] = csv_df.copy()
                print(f"✅ Loaded CSV '{filename}': {csv_df.shape}")
                
            except Exception as e:
                print(f"❌ Failed to read CSV file: {e}")
    
    # Fallback: if nothing worked and we have a df, use it
    if not result_dfs and df is not None:
        result_dfs['main'] = df.copy()
    
    # Final fallback: empty DataFrame
    if not result_dfs:
        print("⚠️ No valid DataFrames found, creating empty DataFrame")
        result_dfs['main'] = pd.DataFrame()
    
    return result_dfs


def clean_sheet_name(sheet_name: str) -> str:
    """Clean sheet name to be a valid Python identifier"""
    # Replace spaces and special characters with underscores
    clean = re.sub(r'[^\w\s]', '_', sheet_name)
    clean = re.sub(r'\s+', '_', clean)
    # Remove leading/trailing underscores
    clean = clean.strip('_')
    # Ensure it doesn't start with a number
    if clean and clean[0].isdigit():
        clean = f"sheet_{clean}"
    # Fallback for empty names
    if not clean:
        clean = "unnamed_sheet"
    return clean.lower()


# Store original __init__ methods
_original_table_analysis_init = TableAnalysisAgent.__init__
_original_watson_table_analysis_init = WatsonTableAnalysisAgent.__init__

def enhanced_table_analysis_init(self, 
                                df: Optional[pd.DataFrame] = None,
                                dataframes: Optional[Union[Dict[str, pd.DataFrame], pd.ExcelFile, str]] = None,
                                file_name: str = "uploaded_data",
                                max_steps: int = 10,
                                max_retries: int = 3,
                                verbose: bool = False,
                                sheet_names: Optional[List[str]] = None,
                                sample_rows: Optional[int] = None,
                                max_sheets: int = 10):
    """Enhanced TableAnalysisAgent __init__ with robust data handling"""
    
    # Normalize the input to a dictionary of DataFrames
    normalized_dfs = normalize_dataframes_input(
        df=df,
        dataframes=dataframes,
        sheet_names=sheet_names,
        max_sheets=max_sheets,
        sample_rows=sample_rows
    )
    
    if verbose:
        print(f"\n📊 Loaded {len(normalized_dfs)} DataFrames:")
        for name, df_item in normalized_dfs.items():
            print(f"   📋 {name}: {df_item.shape}")
    
    # Set up the instance attributes
    self.dataframes = normalized_dfs
    self.df = normalized_dfs.get("main", list(normalized_dfs.values())[0]) if normalized_dfs else pd.DataFrame()
    self.file_name = file_name
    self.max_steps = max_steps
    self.max_retries = max_retries
    self.agent = create_python_agent_graph_with_plots_fixed(sherlock_mode=True)
    
    # Build schema information
    schema_info = {}
    for name, df_item in normalized_dfs.items():
        schema_info[name] = str(df_item.dtypes.to_dict())
    
    self.state = PythonAgentState(
        file_name=file_name,
        file_type="csv",
        schema=str(schema_info),
        preview_md=self.df.head().to_markdown() if not self.df.empty else "No data",
        table_shape=self.df.shape,
        df=self.df,
        dataframes=normalized_dfs,
        messages=[],
        step=0,
        max_steps=max_steps,
        max_retries=max_retries,
        retry_count=0,
        is_complete=False,
        verbose=verbose,
        final_output=None,
        explanation="",
        raw_verbose="",
        verification_status="pending",
        plots=[]  # Initialize plots
    )


def enhanced_watson_table_analysis_init(self,
                                       df: Optional[pd.DataFrame] = None,
                                       dataframes: Optional[Union[Dict[str, pd.DataFrame], pd.ExcelFile, str]] = None,
                                       file_name: str = "uploaded_data",
                                       max_steps: int = 10,
                                       max_retries: int = 3,
                                       verbose: bool = False,
                                       sheet_names: Optional[List[str]] = None,
                                       sample_rows: Optional[int] = None,
                                       max_sheets: int = 10):
    """Enhanced WatsonTableAnalysisAgent __init__ with robust data handling"""
    
    # Normalize the input to a dictionary of DataFrames
    normalized_dfs = normalize_dataframes_input(
        df=df,
        dataframes=dataframes,
        sheet_names=sheet_names,
        max_sheets=max_sheets,
        sample_rows=sample_rows
    )
    
    if verbose:
        print(f"\n📊 Loaded {len(normalized_dfs)} DataFrames:")
        for name, df_item in normalized_dfs.items():
            print(f"   📋 {name}: {df_item.shape}")
    
    # Set up the instance attributes
    self.dataframes = normalized_dfs
    self.df = normalized_dfs.get("main", list(normalized_dfs.values())[0]) if normalized_dfs else pd.DataFrame()
    self.file_name = file_name
    self.max_steps = max_steps
    self.max_retries = max_retries
    self.agent = create_python_agent_graph_with_plots_fixed(sherlock_mode=False)
    
    # Extract schema without revealing data
    column_info = {}
    schema_info = {}
    for name, df_item in normalized_dfs.items():
        schema_info[name] = str(df_item.dtypes.to_dict())
        column_info[name] = {}
        for col in df_item.columns:
            dtype_info = str(df_item[col].dtype)
            null_count = df_item[col].isnull().sum()
            column_info[name][col] = f"dtype: {dtype_info}, null_count: {null_count}"
    
    self.state = PythonAgentState(
        file_name=file_name,
        file_type="csv",
        schema=str(schema_info),
        column_info=column_info,
        table_shape=self.df.shape,
        df=self.df,
        dataframes=normalized_dfs,
        messages=[],
        step=0,
        max_steps=max_steps,
        max_retries=max_retries,
        retry_count=0,
        is_complete=False,
        verbose=verbose,
        error_history=[],
        successful_patterns=[],
        current_strategy="Initial blind exploration",
        final_output=None,
        explanation="",
        raw_verbose="",
        verification_status="pending",
        plots=[]  # Initialize plots
    )


# Apply the patches to existing classes
TableAnalysisAgent.__init__ = enhanced_table_analysis_init
WatsonTableAnalysisAgent.__init__ = enhanced_watson_table_analysis_init

# Add helper method to show available dataframes
def show_dataframes(self):
    """Show all available dataframes in this agent"""
    print(f"\n📊 Available DataFrames in {self.__class__.__name__}:")
    print("=" * 50)
    for name, df in self.dataframes.items():
        print(f"📋 '{name}':")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        if name == 'main':
            print("   ⭐ (This is the main/default DataFrame)")
        print()

# Add the method to both agent classes
TableAnalysisAgent.show_dataframes = show_dataframes
WatsonTableAnalysisAgent.show_dataframes = show_dataframes

# Convenience function
def load_excel_as_agent(file_path: str, 
                       agent_type: str = 'sherlock', 
                       sheet_names: Optional[List[str]] = None,
                       verbose: bool = True) -> Union[SherlockPyDictAgent, WatsonPyDictAgent]:
    """
    Convenience function to create agents directly from Excel files
    
    Args:
        file_path: Path to Excel file or pd.ExcelFile object
        agent_type: 'sherlock' or 'watson'
        sheet_names: Specific sheets to load (None = all sheets)
        verbose: Show loading information
    
    Returns:
        Ready-to-use agent
    """
    if agent_type.lower() == 'sherlock':
        return SherlockPyDictAgent(
            dataframes=file_path,
            sheet_names=sheet_names,
            verbose=verbose
        )
    elif agent_type.lower() == 'watson':
        return WatsonPyDictAgent(
            dataframes=file_path,
            sheet_names=sheet_names,
            verbose=verbose
        )
    else:
        raise ValueError("agent_type must be 'sherlock' or 'watson'")


# print("✅ Simple multi-DataFrame support added!")
# print("\nNow you can use:")
# print("📁 Excel file object: SherlockPyDictAgent(dataframes=pd.ExcelFile('file.xlsx'))")
# print("📁 Excel file path: SherlockPyDictAgent(dataframes='file.xlsx')")
# print("📁 Multiple sheets: SherlockPyDictAgent(dataframes=excel_file, sheet_names=['Sheet1', 'Sheet2'])")
# print("🎯 Convenience: load_excel_as_agent('file.xlsx', 'sherlock')")
# print("\nYour code should now work:")
# print("df = pd.ExcelFile('finance.xlsx')")
# print("agent = SherlockPyDictAgent(dataframes=df)")
# print("agent.show_dataframes()  # See what was loaded")


# concrete wrappers ----------------------------------------------------------
class SherlockPyDictAgent(DictReturningMixin, TableAnalysisAgent):
    pass
class WatsonPyDictAgent(DictReturningMixin, WatsonTableAnalysisAgent): pass
# class SherlockSQLDictAgent  (DictReturningMixin, SQLTableAnalysisAgentEnhanced):    pass
# class WatsonSQLDictAgent    (DictReturningMixin, WatsonSQLTableAnalysisAgentEnhanced): pass

# Recreate wrapper classes
class SherlockSQLDictAgent(DictReturningMixin, SQLTableAnalysisAgent):
    """Sherlock SQL agent with dictionary output and plot support"""
    pass

class WatsonSQLDictAgent(DictReturningMixin, WatsonSQLTableAnalysisAgent):
    """Watson SQL agent with dictionary output and plot support"""
    pass