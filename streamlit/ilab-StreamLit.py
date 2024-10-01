import streamlit as st
import sqlite3
import os
import subprocess
import time

# Database setup for storing parameters
conn = sqlite3.connect('instructlab_parameters.db')
c = conn.cursor()

# Function to create a table with a structured schema
def create_table():
    c.execute('''
        CREATE TABLE IF NOT EXISTS parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            timestamp TEXT,
            apikey TEXT,
            endpoint_url TEXT,
            model TEXT,
            output_dir TEXT,
            batch_size INTEGER,
            chunk_word_count INTEGER,
            num_cpus INTEGER,
            num_gpus INTEGER,
            pipeline TEXT,
            sdg_scale_factor INTEGER
        )
    ''')
    conn.commit()

# Function to save parameters to SQLite with structured schema
def save_parameters(param_type, options):
    # Clear existing entries of the same type
    c.execute("DELETE FROM parameters WHERE type = ?", (param_type,))
    
    # Get the current timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save the parameters into the database
    c.execute('''
        INSERT INTO parameters (type, timestamp, apikey, endpoint_url, model, output_dir, batch_size, chunk_word_count, num_cpus, num_gpus, pipeline, sdg_scale_factor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (param_type, timestamp, options['--api-key'], options['--endpoint-url'], options['--model'], options['--output-dir'],
          options['--batch-size'], options['--chunk-word-count'], options['--num-cpus'], options['--gpus'], options['--pipeline'], options['--sdg-scale-factor']))
    
    conn.commit()

# Function to load parameters from SQLite
def load_parameters(param_type):
    c.execute("SELECT * FROM parameters WHERE type = ?", (param_type,))
    result = c.fetchone()
    if result:
        # Return the parameters as a dictionary
        return {
            "--api-key": result[3],
            "--endpoint-url": result[4],
            "--model": result[5],
            "--output-dir": result[6],
            "--batch-size": result[7],
            "--chunk-word-count": result[8],
            "--num-cpus": result[9],
            "--gpus": result[10],
            "--pipeline": result[11],
            "--sdg-scale-factor": result[12]
        }
    return {}

# Function to generate indicator for non-default values
def generate_indicator(current_value, default_value):
    if current_value != default_value:
        return f"<span style='color: orange;'>Current {current_value}</span> - Default: {default_value}"
    return f"Default: {default_value}"

# Function to execute command and capture the output
def execute_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout + result.stderr

# Function to clear and recreate the database
def reset_database():
    c.execute('DROP TABLE IF EXISTS parameters')
    create_table()
    st.success("Database reset successfully!")

def build_command(base_command, options):
    command = base_command
    for option, value in options.items():
        if value:
            command += f" \\\n{option} {value}"  # Add line break and prefix -- before each option
    return command

# Initialize session state for fields
if 'batch_size' not in st.session_state:
    st.session_state['batch_size'] = 0
if 'chunk_word_count' not in st.session_state:
    st.session_state['chunk_word_count'] = 1000
if 'num_cpus' not in st.session_state:
    st.session_state['num_cpus'] = 10
if 'num_gpus' not in st.session_state:
    st.session_state['num_gpus'] = 0
if 'selected_endpoint' not in st.session_state:
    st.session_state['selected_endpoint'] = 'Local Model'  # Track currently selected endpoint

# Main app structure
st.title("InstructLab Command Controller")

# Sidebar navigation
option = st.sidebar.selectbox("Select InstructLab Command", ['Generate Data'])

# Button to reset the database
if st.sidebar.button("Reset Database"):
    reset_database()

# Create table if it doesn't exist
create_table()

# Endpoint-specific saved parameters
endpoint_choice = st.selectbox("Endpoint", ["Local Model", "OpenAI", "watsonx.ai", "Own Connection"], index=["Local Model", "OpenAI", "watsonx.ai", "Own Connection"].index(st.session_state['selected_endpoint']))

# Load saved parameters for the newly selected endpoint
saved_params = load_parameters(endpoint_choice)

# Ensure all parameters are defined, even if not saved previously
api_key = saved_params.get("--api-key", "")
endpoint_url = saved_params.get("--endpoint-url", "")
model = saved_params.get("--model", "")
output_dir = saved_params.get("--output-dir", os.path.join(os.getcwd(), f"outputdir-{endpoint_choice.replace(' ', '').lower()}"))
batch_size = saved_params.get("--batch-size", 0)
chunk_word_count = saved_params.get("--chunk_word_count", 1000)
num_cpus = saved_params.get("--num_cpus", 10)
num_gpus = saved_params.get("--num_gpus", 0)
pipeline = saved_params.get("--pipeline", "simple")
sdg_scale_factor = saved_params.get("--sdg_scale_factor", 100)

# Apply background colors based on endpoint choice
if endpoint_choice == "OpenAI":
    st.markdown(
        """
        <style>
        .stSelectbox, .stTextInput  {background-color: #50b02c !important; color: white !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    if not endpoint_url:
        endpoint_url = "https://api.openai.com/v1"  # Set default OpenAI endpoint

elif endpoint_choice == "watsonx.ai":
    st.markdown(
        """
        <style>
        .stSelectbox, .stTextInput {background-color: lightblue !important; color: white !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    if not endpoint_url:
        endpoint_url = "http://localhost:8000/v1"  # Set default Watsonx.ai endpoint

elif endpoint_choice == "Own Connection":
    st.markdown(
        """
        <style>
        .stSelectbox, .stTextInput {background-color: #eccdfe !important; color: white !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    if not endpoint_url:
        endpoint_url = "https://api.openai.com/v1/chat"  # Set default Custom endpoint

# Organize fields into rows and columns for better layout
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Generation Settings")
        pipeline = st.selectbox("Pipeline (simple/full)", ["simple", "full"], index=0 if pipeline == "simple" else 1)
        st.markdown("Defines pipeline. Default: simple")
        
        sdg_scale_factor = st.selectbox("SDG Scale Factor", [10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], index=4)
        st.markdown(generate_indicator(sdg_scale_factor, 100), unsafe_allow_html=True)

        num_cpus = st.selectbox("--num-cpus", list(range(1, 61)), index=list(range(1, 61)).index(num_cpus))
        st.markdown(generate_indicator(num_cpus, 10), unsafe_allow_html=True)
    
    with col2:
        # Processing Configuration
        st.subheader("Processing Configuration")
        batch_size = st.selectbox("--batch-size", list(range(0, 51)), index=list(range(0, 51)).index(batch_size))
        st.markdown(generate_indicator(batch_size, 0), unsafe_allow_html=True)

        chunk_word_count = st.selectbox("--chunk-word-count", list(range(100, 11000, 100)), index=list(range(100, 11000, 100)).index(chunk_word_count))
        st.markdown(generate_indicator(chunk_word_count, 1000), unsafe_allow_html=True)
        
        num_gpus = st.selectbox("--num_gpus", list(range(0, 10)), index=list(range(0, 10)).index(num_gpus))
        st.markdown(generate_indicator(num_gpus, 0), unsafe_allow_html=True)
            
# Endpoint and Model Configuration
st.subheader("Endpoint and Model Configuration")
with st.container():
    col3, col4 = st.columns(2)

    with col3:
        if endpoint_choice == "Local Model":
            model = st.text_input("Local Model Path", value=model)
            st.markdown(f"Path to the local model. Default: ~/.cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf")
            
        elif endpoint_choice == "OpenAI":
            endpoint_url = st.text_input("OpenAI Endpoint URL", value=endpoint_url)
            api_key = st.text_input("API Key", value=api_key)
            model = st.selectbox("Model", ["gpt-3.5-turbo-instruct-0914","gpt-3.5-turbo-instruct", "babbage-002", "davinci-002", "text-embedding-ada-002"], index=2)
            
        elif endpoint_choice == "watsonx.ai":
            endpoint_url = st.text_input("Watsonx.ai Endpoint URL", value=endpoint_url)
            api_key = st.text_input("API Key", value=api_key)
            model = st.text_input("Model", value=model)

        elif endpoint_choice == "Own Connection":
            endpoint_url = st.text_input("Custom Endpoint URL", value=endpoint_url)
            api_key = st.text_input("API Key", value=api_key)
            model = st.text_input("Custom Model Path", value=model)

    with col4:
        output_dir = st.text_input("--output-dir", output_dir)
        st.markdown(f"Path to output generated files. Default: {output_dir}")

# Show IAM Caching option only for watsonx.ai
if endpoint_choice == "watsonx.ai":
    iam_caching = st.checkbox("Enable IAM Caching?", value=True)

# Build the command string as preview
base_command = "ilab data generate"
options = {
    "--pipeline": pipeline,
    "--sdg-scale-factor": sdg_scale_factor,
    "--model": model,
    "--endpoint-url": endpoint_url if endpoint_choice != "Local Model" else "",
    "--api-key": api_key if endpoint_choice != "Local Model" else "",
    "--output-dir": output_dir,
    "--batch-size": batch_size,
    "--chunk-word-count": chunk_word_count,
    "--num-cpus": num_cpus,
    "--gpus": num_gpus
}
preview_command = build_command(base_command, options)

# Display the preview command in Markdown (```bash)
st.markdown(f"### Preview Command:")
st.markdown(f"```bash\n{preview_command}\n```")

# Add Save button to manually save parameters
if st.button("Save Parameters"):
    save_parameters(endpoint_choice, options)
    st.success(f"Parameters for {endpoint_choice} saved successfully!")

# Add button to execute the command
if st.button("Execute Command"):
    st.markdown(f"### Executing Command:")
    st.code(preview_command, language="bash")  # Show command in code block for easy copy/paste
    output = execute_command(preview_command)
    st.text_area("Command Output", output, height=300)  # Display the output
