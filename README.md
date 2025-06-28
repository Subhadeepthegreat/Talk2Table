# 🔍 Talk2Table: Sherlock & Watson Data Detective Agents

*Transform your data into insights through natural conversation*

---

## 🎭 Meet Your Data Detectives

Inspired by Sir Arthur Conan Doyle's legendary detective duo, **Talk2Table** brings you two distinct AI agents that approach data analysis in fascinatingly different ways:

### 🕵️ **Sherlock Holmes Agent**
Just like the master detective himself, Sherlock has **full access** to examine every detail of your data. He can peek at rows, explore columns, and use his complete arsenal of analytical tools to deduce insights from your datasets.

### 👨‍⚕️ **Watson Agent** 
Following the trusted doctor's approach, Watson operates with **limited visibility** - he can only see data structures and aggregate patterns, never the raw details. Like Watson reporting back to Holmes from the field, this agent must use clever deductive reasoning to solve your data mysteries without directly observing the evidence.

---

## 🚀 What Can Talk2Table Do?

- 📊 **Multi-format Support**: Upload CSV, Excel, or SQLite files
- 💬 **Natural Language Queries**: Ask questions in plain English
- 📈 **Automatic Visualizations**: Get charts and plots without coding
- 🔄 **Smart Code Generation**: See the Python/SQL behind every answer
- 💾 **Session Management**: Save and revisit your analysis sessions
- 🎯 **Dual Analysis Modes**: Choose your detective based on your needs

---

## 🛠️ Quick Start

### For Everyone (No Coding Required!)

1. **Upload Your Data** 📁
   - Drag and drop your CSV, Excel, or database file
   - Watch as the app automatically detects all tables/sheets

2. **Choose Your Detective** 🎭
   - **Sherlock**: When you want thorough, detailed analysis
   - **Watson**: When you need to respect data privacy or work with aggregated insights

3. **Start Asking Questions** 💭
   ```
   "What are the top 5 products by sales?"
   "Show me a trend of revenue over time"
   "Create a chart comparing regions"
   ```

4. **Get Instant Results** ✨
   - See data tables, explanations, and visualizations
   - Download generated charts
   - View the code that created your results

### For Developers 👨‍💻

```python
from test_agent_sher_wat_2 import SherlockPyDictAgent, WatsonPyDictAgent

# Load your data
df = pd.read_csv('your_data.csv')

# Choose your agent
sherlock = SherlockPyDictAgent(df=df)
watson = WatsonPyDictAgent(df=df)

# Query naturally
result = sherlock.query("What's the correlation between price and sales?")
print(result['explanation'])
```

---

## 🔍 Sherlock vs Watson: The Detective Comparison

| Feature | 🕵️ **Sherlock Holmes** | 👨‍⚕️ **Watson** |
|---------|------------------------|------------------|
| **Data Access** | Full access to raw data | Structure and aggregates only |
| **Analysis Style** | Direct examination & exploration | Deductive reasoning from patterns |
| **Privacy** | Sees all data details | Privacy-preserving approach |
| **Speed** | Faster with direct access | May require multiple iterations |
| **Best For** | Detailed exploration, EDA | Sensitive data, general insights |
| **Accuracy** | High (can verify assumptions) | Good (relies on statistical inference) |
| **Use Case** | Research, development | Production, compliance scenarios |

### 🎯 When to Use Which?

**Choose Sherlock when:**
- Exploring new datasets
- Need detailed, granular insights
- Data privacy isn't a primary concern
- Want the fastest, most direct analysis

**Choose Watson when:**
- Working with sensitive/confidential data
- Need privacy-preserving analysis
- Want to test analysis approaches
- Demonstrating data science methods

---

## 📊 Features in Detail

### 🤖 **Intelligent Agents**
- **ReAct Methodology**: Both agents use reasoning and action cycles
- **Error Recovery**: Automatic retry logic with learning
- **Code Verification**: Self-checking mechanisms for accuracy

### 📈 **Visualization Engine**
- **Auto-plotting**: Agents automatically generate relevant charts
- **Multiple Formats**: Bar charts, line plots, scatter plots, histograms
- **Export Ready**: Download plots as PNG files

### 💾 **Session Management**
- **Persistent Chats**: Save and resume analysis sessions
- **Cloud Storage**: PostgreSQL backend for session data
- **Team Collaboration**: Share sessions with colleagues

### 🔒 **Security & Privacy**
- **Safe Code Execution**: Sandboxed environment for code running
- **Data Privacy**: Watson agent ensures sensitive data never leaves aggregated form
- **Input Validation**: Comprehensive safety checks on all user inputs

---

## 📝 Example Use Cases

### 📈 **Business Analytics**
```
"Show me monthly revenue trends"
"Which products have declining sales?"
"Create a dashboard of key metrics"
```

### 🔬 **Data Science**
```
"Find outliers in the price column"
"What's the correlation matrix for numeric columns?"
"Perform clustering analysis on customer data"
```

### 📊 **Quick Insights**
```
"Summarize this dataset"
"Show me the distribution of ages"
"Which category has the highest average value?"
```

---

## 🎭 A Word from the Detectives

*As Holmes himself might have observed to his faithful companion:*

**Holmes**: "You see, Watson, but you do not observe. The data before us contains a multitude of patterns, each telling its own story. While I examine every detail directly, your approach of gathering only the essential facts and deducing the larger picture has its own merit."

**Watson**: "Indeed, Holmes. Sometimes the forest becomes clearer when one cannot see every individual tree. My method may be slower, but it forces a discipline of reasoning that can reveal insights your direct examination might overlook."

**Holmes**: "Precisely! Each approach serves its purpose - mine for speed and thoroughness, yours for principled deduction and privacy. Together, we offer any data investigator the tools they need."

**Watson**: "And just as I have learned to observe the smallest details in my medical practice, I've adapted those skills to see patterns in data structures without compromising the privacy of individual records."

**Holmes**: "Excellent point, my dear fellow. Your blind analysis approach mirrors how we often solved cases with limited information - using logic, patterns, and statistical reasoning to reach sound conclusions."

---


## 🐛 Known Issues & Limitations

- Large datasets (>100MB) may require patience
- Complex multi-table joins work best with explicit instructions
- Plot generation requires matplotlib/seaborn compatibility

---


## 🙏 Acknowledgments

- Sir Arthur Conan Doyle for the inspiring character archetypes
- The LangChain community for excellent agent frameworks
- Streamlit team for making beautiful web apps accessible

---

**Ready to solve your data mysteries? Upload a file and let Sherlock and Watson get to work!** 🕵️‍♂️📊

*"The game is afoot!"* - Sherlock Holmes
