from jinja2 import Template
import numpy as np

tmpl = {}

tmpl['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}';
            ''')

tmpl['init_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
                PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')


def get_column_names(database, table):
    sql_statement = tmpl['table_columns'].render(database=database.database_name, table=table)
    data = database.execute_query(sql_statement)
    return np.asarray(data)


# Multiply matrixA/tableA by matrixB/tableB: C(result_table) = AB, tables must contain id's!
def multiply_matrices(database, tableA, tableB, result_table):
    # Init table a_transposed, Transpose table A
    number_of_a_columns = len(get_column_names(database, tableA)) - 1
    x_columns = []
    for i in range(number_of_a_columns):
        x_columns.append("x"+str(i+1))
    sql_statement = tmpl['init_table'].render(database=database.database_name, table="a_transposed", x_columns=x_columns)
    database.execute(sql_statement)


    # Init matrix_multiplication_calculation, fill table with values
    # Init result_table, calculate results

