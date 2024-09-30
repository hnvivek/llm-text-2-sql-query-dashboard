import streamlit as st
import pandas as pd
from advanced_query_processor import AdvancedQueryProcessor
from dotenv import load_dotenv
import os
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go
import time

load_dotenv()

# Initialize the AdvancedQueryProcessor
db_path = "food_delivery.db"
llm_api_url = "https://api.openai.com/v1/chat/completions"
llm_api_key = os.getenv('openai_api_key')  # Store your API key in Streamlit secrets
model = os.getenv('model')
csv_dir = "data"

processor = AdvancedQueryProcessor(db_path, llm_api_url, llm_api_key, model)
processor.connect_to_db()

# Streamlit app
st.title("Food Delivery Data Explorer")

# Define timeout duration in seconds (e.g., 300 seconds = 5 minutes)
TIMEOUT_DURATION = 300

# Check if last activity time is in session state
if "last_active" not in st.session_state:
    st.session_state.last_active = time.time()  # Initialize last active time

# Check for timeout and clear session state if exceeded
if time.time() - st.session_state.last_active > TIMEOUT_DURATION:
    st.session_state.clear()  # Clear session state
    st.session_state.last_active = time.time()  # Reinitialize last active time

# Update last active time upon any user action
def update_last_active():
    st.session_state.last_active = time.time()


# Define the tabs
tabs = st.tabs(["Query Runner", "Database Schema", "Sample Questions"])

# Tab 1: Query Runner
with tabs[0]:
    st.subheader("Generate SQL Query")
    st.markdown("""
    <span style="color:red;">**Disclaimer:**</span> This application is read-only and restricts queries to `SELECT` statements. 
    Any queries attempting to `INSERT`, `UPDATE`, or `DELETE` data will not be processed.
    Please ensure your queries are for data retrieval purposes only.
    """, unsafe_allow_html=True)

    user_query = st.text_area("Enter your question:", height=100)

    # Check if there's a query in session state
    if "result_df" not in st.session_state:
        st.session_state.result_df = None
        st.session_state.chart = None

    if st.button("Run Query"):
        if user_query:
            # Process the query and store the result in session state
            result_df, initial_chart_type, fig = processor.process_query(user_query)
            st.session_state.result_df = result_df
            st.session_state.chart = fig

            # Store the SQL query in session state (for re-display)
            st.session_state.sql_query = processor.last_executed_query

    # Check if result data exists in session state
    if st.session_state.result_df is not None:
        # Display the generated SQL query
        st.subheader("Generated SQL Query")
        st.code(st.session_state.sql_query, language="sql")

        # Download button above the table
        csv = st.session_state.result_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv",
        )

        # Display the result table with sorting, filtering, and pagination using st-aggrid
        st.subheader("Query Results")
        gb = GridOptionsBuilder.from_dataframe(st.session_state.result_df)
        gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
        gb.configure_side_bar()  # Add a sidebar with filtering options
        gb.configure_default_column(editable=True, groupable=True, sortable=True, filter=True)
        gridOptions = gb.build()
        AgGrid(st.session_state.result_df, gridOptions=gridOptions, enable_enterprise_modules=False)

        # Display the Plotly chart below the table
        st.subheader("Data Visualization")
        if len(st.session_state.result_df) == 1 and len(
                st.session_state.result_df.columns) == 1:
            # Single value result (like total count)
            column_name = st.session_state.result_df.columns[0]
            value = st.session_state.result_df.iloc[0, 0]
            st.metric(label=column_name, value=value)

        elif len(st.session_state.result_df.columns) == 1:
            # Single-column, multi-row result, display as a table (list of values)
            st.table(st.session_state.result_df)

        elif st.session_state.chart and isinstance(st.session_state.chart, go.Figure):
            # If a valid chart is generated, display the chart
            st.plotly_chart(st.session_state.chart, use_container_width=True)

        else:
            # Display a message if no chart or valid display type is available
            st.info("No chart generated for the current query.")

# Tab 2: Database Schema
with tabs[1]:
    st.subheader("Database Schema")
    st.write("Here are the available tables and fields you can query:")

    schema_info = processor.get_db_schema()
    for table, columns in schema_info.items():
        st.write(f"### Table: `{table}`")
        for column in columns:
            st.write(f"- **{column[1]}**: {column[2]}")
        st.markdown("---")

# Tab 3: Sample Queries
with tabs[2]:
    st.subheader("Sample Queries")
    st.write("You can use the following queries to test the database:")
    sample_queries = processor.get_sample_queries()
    for query in sample_queries:
        st.code(query)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Plotly by [Vivek Reddy](https://github.com/hnvivek/llm-text-2-sql-query-dashboard)")
# Close the database connection when the app is done
processor.close_connection()
