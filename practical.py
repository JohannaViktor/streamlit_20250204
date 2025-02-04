import streamlit as st
import pandas as pd
import plotly as plt
from model import train_model, load_model, predict_life_expectancy

import plotly.express as px
import streamlit as st

url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
df = pd.read_csv(url)

st.set_page_config(layout="wide")
st.title(":earth_africa: Worldwide Analysis of Quality of Life and Economic Factors")
st.subheader(
    "This app enables you to explore the relationships between poverty, "
    "life expectancy, and GDP across various countries and years. "
    "Use the panels to select options and interact with the data."
)
# Cache the model and data to prevent reloading/training on every UI change
@st.cache_resource
def get_model():
    """Loads or trains the model once, then caches it."""
    try:
        model = load_model()
        feature_importances = model.feature_importances_
    except:
        model, feature_importances = train_model(df)
    return model, feature_importances

@st.cache_data
def get_feature_importance(feature_importances):
    """Returns cached feature importance values as a DataFrame."""
    feature_names = ['GDP per capita', 'Poverty Ratio', 'Year']
    return pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Load cached model
model, feature_importances = get_model()

# Load cached feature importance
feature_importance_df = get_feature_importance(feature_importances)











def plot_data(df, year):
    fig = px.scatter(
        #df.query(f”year=={year}“),
        df,
        x="GDP per capita",
        y="Life Expectancy (IHME)",
        size="Population",
        color="country",
        hover_name="country",
        log_x=True,
        size_max=60,
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)





tab1, tab2, tab3, tab4 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer","Predictions"])
with tab1:
    st.write("### :earth_americas: Global Overview")
    st.write("This section provides a global perspective on quality of life indicators.")
    # Year selection slider
    year_selected = st.slider("Select a year:", int(df['survey_year'].min()), int(df['survey_year'].max()), int(df['survey_year'].max()))
    # Filter data for selected year
    year_df = df[df['survey_year'] == year_selected]
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Mean Life Expectancy", value=round(year_df["Life Expectancy (IHME)"].mean(), 2))
    with col2:
        st.metric(label="Median GDP per Capita", value=round(year_df["GDP per capita"].median(), 2))
    with col3:
        st.metric(label="Mean Poverty Rate (Upper Middle Income)", value=round(year_df["headcount_ratio_upper_mid_income_povline"].mean(), 2))
    with col4:
        st.metric(label="Number of Countries", value=year_df["country"].nunique())
    plot_data(year_df,year_selected)
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")
with tab3:
    #df = pd.read_csv('global_development_data.csv')
    
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
with tab4:
    st.header("Predict Life Expectancy")
    col_left, col_right = st.columns([1, 1])

    with col_left:
            # User Input Fields
        gdp_per_capita = st.slider("Enter GDP per capita", min_value=float(df["GDP per capita"].min()), max_value=float(df["GDP per capita"].max()), value=float(df["GDP per capita"].median()))
        poverty_ratio = st.slider("Enter Poverty Ratio", min_value=float(df["headcount_ratio_upper_mid_income_povline"].min()), max_value=float(df["headcount_ratio_upper_mid_income_povline"].max()), value=float(df["headcount_ratio_upper_mid_income_povline"].median()))
        year = st.slider("Select Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))

        # Predict Life Expectancy
        if st.button("Predict Life Expectancy"):
            prediction = predict_life_expectancy(model, gdp_per_capita, poverty_ratio, year)
            st.success(f"Predicted Life Expectancy: {prediction:.2f} years")

    with col_right:
        # Feature Importance Plot
        st.subheader("Feature Importance")
        feature_names = ['GDP per capita', 'Poverty Ratio', 'year']
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        fig = px.bar(feature_importance_df, x="Feature", y="Importance", title="Feature Importance", labels={"Importance": "Relative Importance"})
        st.plotly_chart(fig)