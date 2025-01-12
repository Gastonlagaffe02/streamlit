# power_bi/power_bi_page.py
import streamlit as st

def run():
    # Embed Power BI Report
    st.title("Power BI Dashboard Integration")
    power_bi_embed_url = "https://app.powerbi.com/reportEmbed?reportId=9e45b639-5d8b-4e56-8332-eabf3c892adc&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"

    # Add iframe to Streamlit app
    st.markdown(
        f"""
        <iframe width="1080" height="600" src="{power_bi_embed_url}" frameborder="0" allowfullscreen="true"></iframe>
        """,
        unsafe_allow_html=True,
    )
