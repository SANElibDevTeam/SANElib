from jinja2 import Template
import numpy as np

tmpl_mysql = {}

tmpl_mysql['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}';
            ''')


def get_column_names(database, table):
    sql_statement = tmpl_mysql['table_columns'].render(database=database.database_name, table=table)
    data = database.execute_query(sql_statement)
    return np.asarray(data)


# Multiply matrixA/tableA by matrixB/tableB: C(result_table) = AB, tables must contain id's!
def multiply_matrices(database, tableA, tableB, result_table):
    # Init table a_transposed, Transpose table A
    number_of_b_columns = len(get_column_names(database, tableB)) - 1
    print(number_of_b_columns)

    # Init matrix_multiplication_calculation, fill table with values
    # Init result_table, calculate results
    pass
