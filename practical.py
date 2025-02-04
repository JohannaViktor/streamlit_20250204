import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")
st.title(":earth_africa: Worldwide Analysis of Quality of Life and Economic Factors")
st.subheader(
    "This app enables you to explore the relationships between poverty, "
    "life expectancy, and GDP across various countries and years. "
    "Use the panels to select options and interact with the data."
)
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])
with tab1:
    st.write("### :earth_americas: Global Overview")
    st.write("This section provides a global perspective on quality of life indicators.")
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")
with tab3:
    #df = pd.read_csv('global_development_data.csv')
    url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
    df = pd.read_csv(url)
    #df.head()
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    unique_countries = df["country"].dropna().unique().tolist()
    st.write("### :open_file_folder: Data Explorer")
    st.write("Explore raw data and trends over time.")
    st.subheader("Data explorer")
    st.write("This is the complete dataset:")
    selected_year = st.slider("Select a year", min_value=min_year, max_value=max_year, value=min_year)
    # Country multiselect
    selected_countries = st.multiselect("Select countries", unique_countries, default=unique_countries[:3])  # Default selects first 3
    # Filter dataset based on selected year and countries
    filtered_df = df[(df["year"] == selected_year) & (df["country"].isin(selected_countries))]
    st.dataframe(filtered_df)
    # Convert filtered DataFrame to CSV
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    # Download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"filtered_data_{selected_year}.csv",
        mime="text/csv"
    )