from jinja2 import Template

tmpl_sqlite = {}

tmpl_sqlite['select_x_from'] = Template('''
            SELECT {{ x }} FROM {{ table }};
            ''')

tmpl_sqlite['get_all_from'] = Template('''
            SELECT * FROM {{ table }};
            ''')

tmpl_sqlite['table_columns'] = Template('''
            SELECT name FROM pragma_table_info('{{ table }}');
            ''')

tmpl_sqlite['add_ones_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column }} INT DEFAULT 1;
            ''')

tmpl_sqlite['drop_table'] = Template('''
            DROP TABLE IF EXISTS {{ table }};
            ''')

tmpl_sqlite['init_calculation_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ table }} (
                id DOUBLE NULL,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
                y DOUBLE NULL);
            ''')

tmpl_sqlite['init_result_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ table }} (
                theta DOUBLE NULL);
            ''')

tmpl_sqlite['calculate_equations'] = Template('''
            INSERT INTO {{ table }}({% for x in x_columns %}{{ x }}, {% endfor %}y) 
            VALUES(
                {% for sum_statement in sum_statements %}
                    (SELECT {{ sum_statement }}
                {% endfor %}
            );
            ''')

tmpl_sqlite['save_theta'] = Template('''
            INSERT INTO {{ table }} (theta)
            VALUES ({{ value }});
            ''')

tmpl_sqlite['init_model_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ table }} (
                name VARCHAR(45) NULL,
                state INT NULL,
                input_table VARCHAR(45) NULL,
                prediction_table VARCHAR(45) NULL,
                x_columns TEXT NULL,
                y_column VARCHAR(45) NULL,
                prediction_columns TEXT NULL,
                input_size INT NULL,
                ohe_columns TEXT NULL);
            ''')

tmpl_sqlite['save_model'] = Template('''
            REPLACE INTO linreg_model (name, state, input_table, prediction_table, x_columns, y_column, prediction_columns, input_size, ohe_columns)
                VALUES(
                '{{ name }}',
                {{ state }},
                '{{ input_table }}',
                '{{ prediction_table }}',
                '{{ x_columns }}',
                '{{ y_column }}',
                '{{ prediction_columns }}',
                {{ input_size }},
                '{{ ohe_columns }}');
            ''')