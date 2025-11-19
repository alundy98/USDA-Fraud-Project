def main():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import supabase as supabase
    from supabase import create_client, Client
    import os
    import plotly.express as px

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    st.title("USAA Fraud Detection Dashboard")


    tab1, tab2, tab3 = st.tabs(["Overview", "Interactive Query", "Placeholder"])
    @st.cache_data    
    def load_all_data():
        response = supabase.table("marathon_times").select("*").execute()
        data = response.data
        return pd.DataFrame(data)

if __name__ == "__main__":
    main()