import shlex
import subprocess
from pathlib import Path
import os
import modal

streamlit_script_local_path = Path(__file__).parent / "streamlit_run_ver_2.py"
streamlit_script_remote_path = "/root/streamlit_run_ver_2.py"
image = (
    modal.Image.debian_slim(python_version="3.9")
    .uv_pip_install(
        "streamlit",
        "supabase",
        "pandas",
        "numpy",
        "requests",
        "pdfplumber",
        "beautifulsoup4",
        "openai",
        "nltk",
        "python-dotenv",
        "scikit-learn",
        "streamlit-extras"
    )
    .env({"FORCE_REBUILD": "true"})  # 🚨 Add this line to force a rebuild
    .add_local_file(streamlit_script_local_path, streamlit_script_remote_path)
    # Add all PNG files used in Streamlit code:
    .add_local_file("fraud_group_counts.png", "/root/assets/fraud_group_counts.png")
    .add_local_file("loan_fraud_secondary_counts.png", "/root/assets/loan_fraud_secondary_counts.png")
    .add_local_file("New Visuals/Semantic_relation_drift_of_fraud_types2019-2025.png", "/root/assets/Semantic_relation_drift_of_fraud_types2019-2025.png")
    .add_local_file("New Visuals/2019_umap_plot.png", "/root/assets/2019_umap_plot.png")
    .add_local_file("New Visuals/2024_umap_plot.png", "/root/assets/2024_umap_plot.png")
    .add_local_file("New Visuals/covid_case_count.png", "/root/assets/covid_case_count.png")
    .add_local_file("New Visuals/loan_fraud_detection_by_cluster.png", "/root/assets/loan_fraud_detection_by_cluster.png")
    .add_local_file("New Visuals/loan_fraud_umap_clusters.png", "/root/assets/loan_fraud_umap_clusters.png")
    .add_local_file("FDIC_scraper.py", "/root/FDIC_scraper.py")
    .add_local_file("fdicOIG_scraper.py", "/root/fdicOIG_scraper.py")
)
app = modal.App(name="usaa_fraud_dashboard", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "Hey your starter streamlit isnt working"
    )

@app.function(
    allow_concurrent_inputs=100,
    secrets=[modal.Secret.from_name("usaa-project")]
)
@modal.web_server(8000)
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    # Build environment variables, filtering out None values
    env_vars = {}
    if os.getenv("SUPABASE_KEY"):
        env_vars["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY")
    if os.getenv("SUPABASE_URL"):
        env_vars["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
    if os.getenv("OPENAI_API_KEY"):
        env_vars["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Include current environment to ensure PATH and other essential vars are available
    env_vars.update(os.environ)
        
    subprocess.Popen(cmd, shell=True, env=env_vars)