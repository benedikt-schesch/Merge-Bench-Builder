import pandas as pd
import numpy as np

def get_minimal_athena_type(series):
    """
    Analyzes the values in a pandas Series to determine the optimal
    and smallest possible Athena data type.
    """
    dtype = series.dtype

    # --- Integer Types: Check the actual min/max values ---
    if pd.api.types.is_integer_dtype(dtype):
        col_min = series.min()
        col_max = series.max()
        if col_min >= -128 and col_max <= 127:
            return 'TINYINT'
        elif col_min >= -32768 and col_max <= 32767:
            return 'SMALLINT'
        elif col_min >= -2147483648 and col_max <= 2147483647:
            return 'INTEGER'
        else:
            return 'BIGINT'

    # --- Float Types ---
    elif pd.api.types.is_float_dtype(dtype):
        # Using DOUBLE is safest for floats unless you have a specific reason for FLOAT
        if '32' in str(dtype):
            return 'FLOAT'
        return 'DOUBLE'
        
    # --- Other Data Types (no changes needed here) ---
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    elif pd.api.types.is_categorical_dtype(dtype):
        return 'STRING'
    else:
        # Default fallback for object, string, etc.
        return 'STRING'

# --- Example Usage ---
# This will now work correctly on your actual file.
# Replace with the actual path to your file.
try:
    df = pd.read_parquet("~/Downloads/oee_data.parquet")

    # Generate column definitions by analyzing each column (Series)
    column_definitions = [f"`{col}` {get_minimal_athena_type(df[col])}" for col in df.columns]

    # Create the final CREATE TABLE statement
    create_table_statement = f"CREATE EXTERNAL TABLE IF NOT EXISTS my_database.my_oee_table (\n  {', \n  '.join(column_definitions)}\n);"

    print("--- Optimized Athena Table Definition ---")
    print(create_table_statement)

except FileNotFoundError:
    print("Error: The file '~/Downloads/oee_data.parquet' was not found.")
    print("Please update the path to your parquet file.")