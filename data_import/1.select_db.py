import psycopg2
import pandas as pd

# Function to read the db connection info
def read_db_connection_info(filename="db_connection_info.txt"):
    connection_info = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                connection_info[key] = value
    except Exception as e:
        print(f"Failed to read database connection info: {e}")
        raise
    return connection_info

# Load the connection info
connection_info = read_db_connection_info()

try:
    # Connect to the database
    conn = psycopg2.connect(
        host=connection_info["host"],
        user=connection_info["user"],
        password=connection_info["password"],
        dbname=connection_info["database"],
        port=connection_info["port"]
    )
    # This ensures that resources are cleaned up properly
    with conn:
        with conn.cursor() as cursor:
            # Export data_mapping table
            query_mapping = """
                SELECT * FROM data_mapping
                WHERE ships_idx BETWEEN 1000 AND 1999
            """
            cursor.execute(query_mapping)
            results_mapping = cursor.fetchall()
            columns_mapping = [desc[0] for desc in cursor.description]
            df_mapping = pd.DataFrame(results_mapping, columns=columns_mapping)
            df_mapping.to_csv('data_import/data_mapping.csv', index=False, encoding='utf-8-sig')
            
            # Export data_master_model table
            query_master = """
                SELECT * FROM data_model_master
            """
            cursor.execute(query_master)
            results_master = cursor.fetchall()
            columns_master = [desc[0] for desc in cursor.description]
            df_master = pd.DataFrame(results_master, columns=columns_master)
            df_master.to_csv('data_import/data_model_master_export.csv', index=False, encoding='utf-8-sig')

    print("Data exported successfully to 'data_import/data_mapping.csv' and 'data_import/data_model_master_export.csv'")

except (Exception, psycopg2.DatabaseError) as error:
    print(f"An error occurred: {error}")
finally:
    if conn is not None:
        conn.close()
