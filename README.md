# LLM Text-to-SQL Query Dashboard 

## Overview

[**LLM Text-to-SQL Query Dashboard**](https://llm-text-2-sql-query-dashboard-production.up.railway.app/) is a web-based tool that allows users to run SQL queries on a database using a natural language interface powered by large language models (LLMs). It seamlessly translates user queries into SQL and provides visualizations or tabular representations of the results in an intuitive and interactive way. 

This dashboard is built with **Streamlit** and **Plotly** for a smooth user interface and dynamic visualizations. It is particularly useful for those who are familiar with SQL databases but prefer to interact using simple language queries or for those who want to query databases without deep SQL expertise.
---

## Purpose of the Project

The **LLM Text-to-SQL Query Dashboard** aims to democratize access to data stored in SQL databases by removing the barrier of complex SQL syntax. Using this platform, users can:

- Write queries in natural language, which are converted into SQL automatically.
- Run SQL queries directly on the database with a simple user interface.
- View and interact with the results, whether it's a table of data or visualized charts (e.g., bar charts, line charts, scatter plots).
- Download the results of their queries in CSV format for further analysis or reporting.

This project bridges the gap between non-technical users and data stored in relational databases, making it easier to access and analyze the data.

---

## Key Features

- **Natural Language to SQL Translation**: Type your queries in plain English, and the system automatically converts them into SQL commands.
- **Dynamic Visualizations**: Automatically generate the most appropriate chart or table based on the query results.
- **Interactive Query Runner**: Users can directly input SQL queries if they prefer and get results instantly.
- **Downloadable Data**: Export the query results in CSV format for further analysis.
- **Database Schema Browser**: Easily view the database schema, including tables and columns, to understand what data is available for querying.
- **Sample Queries**: Predefined sample queries to help users quickly understand how to interact with the data.

---

## How This Project is Useful

The **LLM Text-to-SQL Query Dashboard** is designed to be useful for:

- **Business Analysts**: Analysts who need to extract insights from a database but may not have advanced SQL skills can use natural language queries.
- **Data Teams**: Empower non-technical members of data teams to run queries on company data without needing to learn SQL.
- **Developers**: Developers can run SQL queries without leaving the dashboard and get visualizations, saving time in writing code for data analysis.
- **Project Managers**: Project managers can get quick insights from the database using natural language instead of relying on a data team or learning SQL.
- **SQL Learners**: People who are learning SQL can benefit by comparing natural language queries with the generated SQL, improving their understanding of database querying.

---

## How to Use

### Prerequisites

Before running this project, ensure that you have the following installed:

- **Python 3.x**
- **Streamlit**
- **Plotly**
- **Pandas**
- **SQLite** (for testing purposes, replace with your preferred database)

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/llm-text-to-sql-query-dashboard.git
   ```

2. Navigate to the project directory:

   ```bash
   cd llm-text-to-sql-query-dashboard
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have a running database (e.g., SQLite) and update the `db_path` in your environment settings.

5. Set up the `.env` file with your API key (e.g., OpenAI API for LLM) and database configuration.

6. Run the application:

   ```bash
   streamlit run app.py
   ```

### Usage

- Open your browser and navigate to `http://localhost:8501` (or whatever port Streamlit indicates).
- You will see a dashboard with the following sections:
  - **Query Runner**: Enter your query in natural language or SQL and view the results.
  - **Database Schema**: View the schema of the connected database to understand available tables and fields.
  - **Sample Questions**: Use sample questions to explore the data quickly.

## Architecture

This project is built using the following technologies:

- **Streamlit**: For building the web interface and providing a simple way to display and interact with query results.
- **Plotly**: For generating interactive charts and graphs.
- **Pandas**: For handling and manipulating data frames.
- **LLMs (Large Language Models)**: Converts natural language queries into SQL using API calls (e.g., OpenAI).

---
