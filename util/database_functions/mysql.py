from jinja2 import Template
import numpy as np

tmpl = {}

tmpl['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}';
            ''')

tmpl['drop_table'] = Template('''
            DROP TABLE IF EXISTS {{ table }};
            ''')

tmpl['select_count'] = Template('''
            SELECT COUNT(*) FROM {{ table }};
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

tmpl['transpose_table'] = Template('''
            INSERT INTO {{ table }}({{ x_column_string }}) VALUES({{ selection_string }});
            ''')


def get_column_names(database, table):
    sql_statement = tmpl['table_columns'].render(database=database.database_name, table=table)
    data = database.execute_query(sql_statement)
    return np.asarray(data)


# Matrix-Multiply tableA with tableB: result_table = AB, input tables must contain id's!
def multiply_matrices(database, tableA, tableB, result_table_name):
    # Init calculation table (nxm)
    sql_statement = tmpl['select_count'].render(table=tableA)
    n = np.asarray(database.execute_query(sql_statement))[0][0]
    m = len(get_column_names(database, tableB)) - 1

    sql_statement = tmpl['drop_table'].render(table="matmul_calculation")
    database.execute(sql_statement)

    calculation_columns = []
    for i in range(n):
        calculation_columns.append("a" + str(i + 1))
    for i in range(m):
        calculation_columns.append("b" + str(i + 1))

    sql_statement = tmpl['init_table'].render(database=database.database_name, table="matmul_calculation",
                                              x_columns=calculation_columns)
    database.execute(sql_statement)

    # Transpose tableA
    sql_statement = tmpl['drop_table'].render(table="a_transposed")
    database.execute(sql_statement)

    number_of_a_columns = len(get_column_names(database, tableA)) - 1
    x_columns = []
    for i in range(number_of_a_columns):
        x_columns.append("x" + str(i + 1))
    sql_statement = tmpl['init_table'].render(database=database.database_name, table="a_transposed",
                                              x_columns=x_columns)
    database.execute(sql_statement)

    x_column_string = ""
    for i in range(number_of_a_columns):
        x_column_string = x_column_string + "x" + str(i + 1)
        if i < number_of_a_columns - 1:
            x_column_string = x_column_string + ","

    for i in range(n):
        selection_string = ""
        for j in range(number_of_a_columns):
            if j == 0:
                limit = "1"
            else:
                limit = str(j) + "," + str(j)
            selection_string = selection_string + "(SELECT x" + str(i + 1) + " FROM " + str(
                tableA) + " LIMIT " + limit + ")"
            if j < number_of_a_columns - 1:
                selection_string = selection_string + ","
        sql_statement = tmpl['transpose_table'].render(table="a_transposed", x_column_string=x_column_string,
                                                       selection_string=selection_string)
        database.execute(sql_statement)

    # Calculate results
        # Init result_table, calculate results

    # Drop temporary tables
