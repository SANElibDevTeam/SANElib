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


def get_column_names(database, table):
    sql_statement = tmpl['table_columns'].render(database=database.database_name, table=table)
    data = database.execute_query(sql_statement)
    return np.asarray(data)


# Matrix-Multiply tableA with tableB: result_table = AB, input tables must contain id's!
def multiply_matrices(database, table_a, table_b, result_table_name):
    # Get column names
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

    n = len(a_column_names) - 1
    m = len(b_column_names) - 1

    # Transpose tableA
    sql_statement = tmpl['drop_table'].render(table="a_transposed")
    database.execute(sql_statement)

    number_of_a_columns = len(a_column_names) - 1
    transposed_a_columns = []
    for i in range(n):
        transposed_a_columns.append("x" + str(i + 1))
    sql_statement = tmpl['init_table'].render(database=database.database_name, table="a_transposed",
                                                  x_columns=transposed_a_columns)
    database.execute(sql_statement)

    x_column_string = ""
    for i in range(n):
        x_column_string = x_column_string + "x" + str(i + 1)
        if i < n - 1:
            x_column_string = x_column_string + ","

    for i in range(n):
        selection_string = ""
        for j in range(number_of_a_columns):
            if j == 0:
                limit = "1"
            else:
                limit = str(j) + "," + str(j)
            selection_string = selection_string + "(SELECT x" + str(i + 1) + " FROM " + str(
                table_a) + " LIMIT " + limit + ")"
            if j < number_of_a_columns - 1:
                selection_string = selection_string + ","
        sql_statement = tmpl['transpose_table'].render(table="a_transposed", x_column_string=x_column_string,
                                                           selection_string=selection_string)
        database.execute(sql_statement)

    # Init calculation table (nxm)
    sql_statement = tmpl['drop_table'].render(table="matmul_calculation")
    database.execute(sql_statement)

    calculation_columns = []
    for i in range(n):
        calculation_columns.append("a" + str(i + 1))
    for i in range(m):
        calculation_columns.append("b" + str(i + 1))
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
                                                                a_columns=a_column_names,
                                                                b_columns=b_column_names, table_a=table_a,
                                                                table_b=table_b)
    database.execute(sql_statement)



    # Calculate results
    # Init result_table, calculate results

    # Drop temporary tables
