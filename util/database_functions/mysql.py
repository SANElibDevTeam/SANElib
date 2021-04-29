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

tmpl['fill_matmul_calulation_table'] = Template('''
            INSERT INTO {{ table }} ({{ column_string }}) 
            SELECT {% for x in a_columns %}{% if x != "id" %}a.{% endif %}{{ x if x != "id" else "" }}{% if x != "id" %},{% endif %}{% endfor %} 
            {% for x in b_columns %}{% if x != "id" %}b.{% endif %}{{ x if x != "id" else "" }}{{ ", " if not loop.last and x != "id" else "" }} {% endfor %} 
            FROM {{ table_a }} as a
            LEFT JOIN {{ table_b }} as b ON a.id = b.id;
            ''')

tmpl['transpose_table'] = Template('''
            INSERT INTO {{ table }}({{ x_column_string }}) VALUES({{ selection_string }});
            ''')

tmpl['calculate_matmul'] = Template('''
            INSERT INTO {{ table }} ({{ x_column_string }}) 
            VALUES({% for x in sum_statements %}{{ x }}{% endfor %});
            ''')


def get_column_names(database, table):
    sql_statement = tmpl['table_columns'].render(database=database.database_name, table=table)
    data = database.execute_query(sql_statement)
    return np.asarray(data)


def get_number_of_rows(database, table):
    sql_statement = tmpl['select_count'].render(table=table)
    data = database.execute_query(sql_statement)
    return np.asarray(data)[0][0]


# Matrix-Multiply tableA with tableB: result_table = AB, input tables must contain id's (column id)!
def multiply_matrices(database, table_a, table_b, result_table_name):
    # Get column names & Init m, n & Check compatibility
    a_column_names = get_column_names(database, table_a)[:, 0]
    b_column_names = get_column_names(database, table_b)[:, 0]
    a_column_names_string = ""
    for i in range(len(a_column_names)):
        a_column_names_string = a_column_names_string + str(a_column_names[i])
        if i < len(a_column_names) - 1:
            a_column_names_string = a_column_names_string + ","
    b_column_names_string = ""
    for i in range(len(b_column_names)):
        b_column_names_string = b_column_names_string + str(b_column_names[i])
        if i < len(b_column_names) - 1:
            b_column_names_string = b_column_names_string + ","

    number_of_a_columns = len(a_column_names) - 1
    number_of_rows_b = get_number_of_rows(database, table_b)

    if number_of_a_columns != number_of_rows_b:
        raise Exception('Unable to matrix multiplicate A & B! Dimension mismatch!')

    number_of_rows_a = get_number_of_rows(database, table_a)
    n = number_of_rows_a
    m = len(b_column_names) - 1

    # Transpose tableA
    sql_statement = tmpl['drop_table'].render(table="matmul_a_transposed")
    database.execute(sql_statement)

    transposed_a_columns = []
    for i in range(n):
        transposed_a_columns.append("x" + str(i + 1))
    sql_statement = tmpl['init_table'].render(database=database.database_name, table="matmul_a_transposed",
                                              x_columns=transposed_a_columns)
    database.execute(sql_statement)

    x_column_string = ""
    for i in range(n):
        x_column_string = x_column_string + "x" + str(i + 1)
        if i < n - 1:
            x_column_string = x_column_string + ","

    for i in range(number_of_a_columns):
        selection_string = ""
        for j in range(number_of_rows_a):
            if j == 0:
                limit = "1"
            else:
                limit = str(j) + "," + str(j)
            selection_string = selection_string + "(SELECT x" + str(i + 1) + " FROM " + str(
                table_a) + " LIMIT " + limit + ")"
            if j < number_of_rows_a - 1:
                selection_string = selection_string + ","
        sql_statement = tmpl['transpose_table'].render(table="matmul_a_transposed", x_column_string=x_column_string,
                                                       selection_string=selection_string)
        database.execute(sql_statement)

    # Init calculation table
    sql_statement = tmpl['drop_table'].render(table="matmul_calculation")
    database.execute(sql_statement)

    calculation_columns = []
    calculation_columns_a = []
    calculation_columns_b = []
    for i in range(n):
        calculation_columns.append("a" + str(i + 1))
        calculation_columns_a.append("a" + str(i + 1))
    for i in range(m):
        calculation_columns.append("b" + str(i + 1))
        calculation_columns_b.append("b" + str(i + 1))
    calculation_columns_string = ""
    for i in range(len(calculation_columns)):
        calculation_columns_string = calculation_columns_string + str(calculation_columns[i])
        if i < len(calculation_columns) - 1:
            calculation_columns_string = calculation_columns_string + ","

    sql_statement = tmpl['init_table'].render(database=database.database_name, table="matmul_calculation",
                                              x_columns=calculation_columns)
    database.execute(sql_statement)
    sql_statement = tmpl['fill_matmul_calulation_table'].render(table="matmul_calculation",
                                                                column_string=calculation_columns_string,
                                                                a_columns=transposed_a_columns,
                                                                b_columns=b_column_names, table_a="matmul_a_transposed",
                                                                table_b=table_b)
    database.execute(sql_statement)

    # Calculate results
    sql_statement = tmpl['drop_table'].render(table=result_table_name)
    database.execute(sql_statement)
    x_columns = []
    for i in range(m):
        x_columns.append("x" + str(i + 1))
    sql_statement = tmpl['init_table'].render(database=database.database_name, table=result_table_name,
                                              x_columns=x_columns)
    database.execute(sql_statement)

    x_column_string = ""
    for i in range(m):
        x_column_string = x_column_string + "x" + str(i + 1)
        if i < m - 1:
            x_column_string = x_column_string + ","

    for i in range(n):
        sum_statements = []
        for j in range(m):
            statement = "(SELECT sum(" + str(calculation_columns_a[i]) + "*" + str(calculation_columns_b[j]) + ") FROM matmul_calculation)"
            if j < m - 1:
                statement = statement + ","
            sum_statements.append(statement)

        sql_statement = tmpl['calculate_matmul'].render(table=result_table_name, x_column_string=x_column_string,
                                                        sum_statements=sum_statements)
        database.execute(sql_statement)

    # Drop temporary tables
    sql_statement = tmpl['drop_table'].render(table="matmul_a_transposed")
    database.execute(sql_statement)
    sql_statement = tmpl['drop_table'].render(table="matmul_calculation")
    database.execute(sql_statement)
