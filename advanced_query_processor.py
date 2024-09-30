import sqlite3
import pandas as pd
import requests
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import show

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedQueryProcessor:
    def __init__(self, db_path, llm_api_url, llm_api_key, model):
        self.db_path = db_path
        self.llm_api_url = llm_api_url
        self.llm_api_key = llm_api_key
        self.model = model
        self.conn = None
        self.cursor = None
        self.table_info = self.get_table_info()
        self.last_executed_query = ""

    def get_table_info(self):
        return {
            "users": {
                "description": "Contains information about the users of the food delivery application.",
                "columns": {
                    "user_id": "Unique identifier for each user",
                    "name": "Full name of the user",
                    "gender": "Gender of the user",
                    "email": "Email address of the user",
                    "phone_number": "Phone number of the user",
                    "delivery_address": "Delivery address of the user"
                }
            },
            "restaurants": {
                "description": "Contains information about the restaurants listed in the food delivery application.",
                "columns": {
                    "restaurant_id": "Unique identifier for each restaurant",
                    "name": "Name of the restaurant",
                    "address": "Address of the restaurant",
                    "phone_number": "Phone number of the restaurant",
                    "email": "Email address of the restaurant",
                    "cuisine_type": "Type of cuisine the restaurant specializes in"
                }
            },
            "menu_items": {
                "description": "Contains information about the menu items offered by the restaurants.",
                "columns": {
                    "menu_item_id": "Unique identifier for each menu item",
                    "restaurant_id": "Unique identifier of the restaurant offering the menu item",
                    "item_name": "Name of the menu item",
                    "price": "Price of the menu item"
                }
            },
            "orders": {
                "description": "Contains information about the orders placed by users.",
                "columns": {
                    "order_id": "Unique identifier for each order",
                    "user_id": "Unique identifier of the user who placed the order",
                    "order_time": "The datetime when the order was placed",
                    "delivery_address": "The address where the order is to be delivered",
                    "order_status": "The status of the order (e.g., 'Pending', 'Delivered', 'Cancelled', 'Undelivered')",
                    "restaurant_id": "Unique identifier of the restaurant from which the order was placed",
                    "total_amount": "Total amount of the order"
                }
            },
            "order_details": {
                "description": "Contains detailed information about the items in each order.",
                "columns": {
                    "order_details_id": "Unique identifier for each order detail",
                    "order_id": "Unique identifier of the order",
                    "menu_item_id": "Unique identifier of the menu item",
                    "quantity": "Quantity of the menu item ordered",
                    "price": "Price of the menu item"
                }
            },
            "payments": {
                "description": "Contains information about the payments for orders.",
                "columns": {
                    "payment_id": "Unique identifier for each payment",
                    "order_id": "Unique identifier of the order",
                    "payment_method": "Method used for payment (e.g., 'Credit Card', 'Debit Card', 'PayPal')",
                    "amount_paid": "Total amount paid for the order",
                    "payment_date": "The datetime when the payment was made",
                    "payment_status": "Status of the payment (e.g., 'Successful', 'Failed')",
                    "refund": "Amount refunded based on the rating"
                }
            },
            "reviews": {
                "description": "Contains information about the reviews given by users for their orders.",
                "columns": {
                    "review_id": "Unique identifier for each review",
                    "user_id": "Unique identifier of the user who placed the order",
                    "order_id": "Unique identifier of the order",
                    "restaurant_id": "Unique identifier of the restaurant from which the order was placed",
                    "rating": "Rating given by the user (may be null)",
                    "comments": "Review comments (may be null, especially if rating is null)",
                    "review_date": "Date when the review was written"
                }
            }
        }

    def connect_to_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close_connection(self):
        if self.conn:
            self.conn.close()

    def create_tables(self):
        for table_name, table_info in self.table_info.items():
            columns = ", ".join([f"{col} TEXT" for col in table_info['columns']])
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {columns}
                )
            ''')
        self.conn.commit()

    def delete_all_tables(self):
        """
        Deletes all tables in the database.
        """
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        for table in tables:
            table_name = table[0]
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

        self.conn.commit()
        print("All tables have been deleted.")

    def load_data(self, csv_dir):
        for table_name in self.table_info.keys():
            df = pd.read_csv(os.path.join(csv_dir, f"{table_name.lower()}.csv"))
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)

    def generate_sql(self, user_query):
        table_descriptions = json.dumps(self.table_info, indent=2)
        prompt = f"""
        Given the following table information:
        {table_descriptions}

        And the user query:
        "{user_query}"

        Generate a SQL query to answer the user's question. 
        Ensure the query is a read-only operation (i.e., SELECT statements only). 
        Do not generate any DELETE, DROP, UPDATE, or INSERT queries.
        Return only the SQL query without any additional explanation.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(self.llm_api_url, headers=headers, data=json.dumps(data))
        sql_query = response.json()['choices'][0]['message']['content'].strip()

        # Check if the generated query is read-only
        disallowed_keywords = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "REPLACE"]
        if any(keyword in sql_query.upper() for keyword in disallowed_keywords):
            raise ValueError("Only read-only queries (SELECT) are allowed.")

        return sql_query

    def fix_sql_error(self, sql_query, error_message):
        table_descriptions = json.dumps(self.table_info, indent=2)
        prompt = f"""
        Given the following table information:
        {table_descriptions}

        The following SQL query resulted in an error:
        {sql_query}

        Error message:
        {error_message}

        Please provide a corrected SQL query that addresses this error. 
        Ensure that the corrected query is a read-only operation (i.e., SELECT statements only). 
        Do not generate any DELETE, DROP, UPDATE, or INSERT queries. 
        Return only the corrected SQL query without any additional explanation.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(self.llm_api_url, headers=headers, data=json.dumps(data))
        corrected_sql_query = response.json()['choices'][0]['message']['content'].strip()

        # Check if the corrected query is read-only
        disallowed_keywords = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "REPLACE"]
        if any(keyword in corrected_sql_query.upper() for keyword in disallowed_keywords):
            raise ValueError("Only read-only queries (SELECT) are allowed.")

        return corrected_sql_query

    def execute_query(self, sql_query):
        try:
            self.last_executed_query = sql_query  # Store the query
            df = pd.read_sql_query(sql_query, self.conn)
            logger.info(f"Executing SQL query: {sql_query}")
            return df
        except sqlite3.Error as e:
            corrected_sql = self.fix_sql_error(sql_query, str(e))
            self.last_executed_query = corrected_sql  # Store the corrected query
            df = pd.read_sql_query(corrected_sql, self.conn)
            return df

    def determine_best_chart(self, df):
        """
        Analyze the DataFrame and determine the best chart type or visualization method.

        :param df: pandas DataFrame containing the query results
        :return: tuple containing (chart_type, x_column, y_column, category_column, title)
        """
        num_rows = len(df)
        num_columns = len(df.columns)

        # Handle Single-Row Data (one row of data)
        if num_rows == 1:
            if num_columns == 1:
                # If there's only one column, display it as a single value card
                column = df.columns[0]
                return 'single_value', column, None, None, f"{column}"
            elif num_columns == 2:
                # If there are two columns, treat it as a key-value pair
                x_column = df.columns[0]
                y_column = df.columns[1]
                return 'value_card', x_column, y_column, None, f"{y_column} by {x_column}"

        # Handle Single-Column Data (one column with multiple rows)
        if num_columns == 1:
            # Display the single column as a table or a list
            column = df.columns[0]
            return 'table', column, None, None, f"List of {column}"

        # Handle Two-Column Data
        if num_columns == 2:
            # Two columns: one for x-axis and one for y-axis
            x_column = df.columns[0]
            y_column = df.columns[1]

            # Determine chart type based on the data types of the columns
            if df[x_column].dtype in ['int64', 'float64'] and df[y_column].dtype in [
                'int64', 'float64']:
                return 'scatter', x_column, y_column, None, f"{y_column} vs {x_column}"
            elif df[x_column].nunique() <= 10:
                # If x_column has few unique values, use a bar chart
                return 'bar', x_column, y_column, None, f"{y_column} by {x_column}"
            else:
                # Otherwise, use a line chart
                return 'line', x_column, y_column, None, f"{y_column} over {x_column}"

        # Handle DataFrames with 3 or more columns
        if num_columns >= 3:
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = df.select_dtypes(include=['object']).columns

            if len(numeric_columns) >= 1 and len(categorical_columns) >= 1:
                # Choose one numeric and one categorical column for a bar chart
                x_column = categorical_columns[0]
                y_column = numeric_columns[0]
                return 'bar', x_column, y_column, None, f"{y_column} by {x_column}"

        # Default to a table if no suitable chart type is found
        return 'table', None, None, None, "Query Results"

    def plot_chart(self, df, initial_chart_type, x_column, y_column, category_column=None,
                   title=None):

        def create_trace(chart_type):
            if chart_type == 'bar':
                return go.Bar(
                    x=df[x_column],
                    y=df[y_column],
                    name=y_column,
                    text=df[y_column].round(2),
                    textposition='outside',
                    hoverinfo='text',
                    hovertext=[f"{x}: {y:.2f}" for x, y in
                               zip(df[x_column], df[y_column])],
                    marker_color='rgb(158,202,225)',
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5,
                    opacity=0.6
                )
            elif chart_type == 'line':
                return go.Scatter(
                    x=df[x_column],
                    y=df[y_column],
                    mode='lines+markers',
                    name=y_column,
                    text=[f"{x}: {y:.2f}" for x, y in zip(df[x_column], df[y_column])],
                    hoverinfo='text'
                )
            elif chart_type == 'scatter':
                return go.Scatter(
                    x=df[x_column],
                    y=df[y_column],
                    mode='markers',
                    name=y_column,
                    text=[f"{x}: {y:.2f}" for x, y in zip(df[x_column], df[y_column])],
                    hoverinfo='text',
                    marker=dict(size=10)
                )
            elif chart_type == 'area':
                return go.Scatter(
                    x=df[x_column],
                    y=df[y_column],
                    mode='lines',
                    fill='tozeroy',
                    name=y_column,
                    text=[f"{x}: {y:.2f}" for x, y in zip(df[x_column], df[y_column])],
                    hoverinfo='text'
                )
            elif chart_type == 'pie':
                return go.Pie(
                    labels=df[x_column],
                    values=df[y_column],
                    text=df[y_column].round(2),
                    hoverinfo='label+percent+text',
                    textinfo='value',
                    textposition='inside'
                )

        # Check if there's only one row and handle accordingly
        if len(df) == 1:
            if initial_chart_type == 'single_value':
                return None, f"Total: {df.iloc[0][x_column]}"
            elif initial_chart_type == 'value_card':
                return None, f"{df.iloc[0][x_column]}: {df.iloc[0][y_column]}"

        # Create the initial figure
        fig = go.Figure()

        # Check if y_column is valid before proceeding to plot
        if y_column is None:
            return None, "Y column cannot be None for plotting."

        def update_layout_for_chart(chart_type):
            base_layout = {
                "title": dict(text=title, x=0.5),
                "plot_bgcolor": "white",
                "hoverlabel": dict(bgcolor="white", font_size=12),
                "barmode": 'group',
                "bargap": 0.15,
                "bargroupgap": 0.1,
            }

            if chart_type == 'pie':
                base_layout.update({
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "showlegend": True,
                    "height": 600  # Adjust as needed
                })
            else:
                base_layout.update({
                    "xaxis": {
                        "visible": True,
                        "title": x_column.capitalize(),
                        "showgrid": True,
                        "gridwidth": 1,
                        "gridcolor": 'lightgray',
                        "tickangle": 45
                    },
                    "yaxis": {
                        "visible": True,
                        "title": y_column.capitalize(),
                        "showgrid": True,
                        "gridwidth": 1,
                        "gridcolor": 'lightgray'
                    },
                    "showlegend": False,
                    "height": 600  # Adjust as needed
                })
            return base_layout

        # Add all traces in the order of chart_types
        chart_types = ['bar', 'line', 'scatter', 'area', 'pie']
        for chart_type in chart_types:
            trace = create_trace(chart_type)
            trace.visible = (chart_type == initial_chart_type)
            fig.add_trace(trace)

        # Update layout for the initial chart
        fig.update_layout(**update_layout_for_chart(initial_chart_type))

        # Add dropdown menu for chart type selection
        fig.update_layout(
            updatemenus=[
                dict(
                    active=chart_types.index(initial_chart_type),
                    buttons=[
                        dict(
                            label=chart_type.capitalize(),
                            method="update",
                            args=[
                                {"visible": [ct == chart_type for ct in chart_types]},
                                update_layout_for_chart(chart_type)
                            ]
                        )
                        for chart_type in chart_types
                    ],
                    direction="down",
                    pad={"r": 0, "t": 0},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.5,
                    yanchor="top"
                ),
            ]
        )

        return fig

    def process_query(self, user_query):
        sql_query = self.generate_sql(user_query)
        result_df = self.execute_query(sql_query)

        initial_chart_type, x_column, y_column, category_column, title = self.determine_best_chart(
            result_df)

        logger.info(f"Initial chart type determined: {initial_chart_type}")
        logger.info(
            f"X-column: {x_column}, Y-column: {y_column}, Category-column: {category_column}")

        fig = self.plot_chart(result_df, initial_chart_type, x_column, y_column,
                              category_column, title)

        return result_df, initial_chart_type, fig

    def get_db_schema(self):
        """
        Retrieves the database schema information (tables and columns).
        """
        schema_info = {}
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        for table in tables:
            table_name = table[0]
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = self.cursor.fetchall()
            schema_info[table_name] = columns

        return schema_info

    def get_sample_queries(self):
        """
        Provides a list of sample SQL queries for the user to test.
        """
        sample_queries = [
            "Total customers grouped by gender with average spend, order, and rating.",
            "Top 5 restaurants with lowest avg review, show review count, and display "
            "one comment.",
            "Top 5 customers with the most orders.",
            "Get total orders by payment type, status, and percentage.",
            "Unique customer count per restaurant, sorted high to low, with average order amount."
        ]
        return sample_queries