from jinja2 import Template

tmpl = {}

tmpl['select_x_from'] = Template('''
            SELECT {{ x }} FROM {{ database }}.{{ table }};
            ''')

tmpl['get_all_from'] = Template('''
            SELECT * FROM {{ database }}.{{ table }};
            ''')

tmpl['get_all_from_where_id'] = Template('''
            SELECT * FROM {{ database }}.{{ table }} WHERE id='{{ where_statement }}';
            ''')

tmpl['delete_from_table_where_id'] = Template('''
            DELETE FROM {{ table }} WHERE id='{{ where_statement }}';
            ''')
tmpl['set_safe_updates'] = Template('''
            SET SQL_SAFE_UPDATES = {{ value }};
            ''')

tmpl['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}';
            ''')

tmpl['column_type'] = Template('''
            SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE table_name = '{{ table }}' and column_name='{{ column }}';
            ''')

tmpl['drop_table'] = Template('''
            DROP TABLE IF EXISTS {{ table }};
            ''')

tmpl['add_ones_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column }} INT DEFAULT 1;
            ''')

tmpl['add_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column }} {{ type }};
            ''')

tmpl['set_ohe_column'] = Template('''
            UPDATE {{ table }}
            SET {{ ohe_column }} = IF({{ input_column }}='{{ value }}', 1, 0);
            ''')

tmpl['save_model'] = Template('''
            REPLACE INTO linreg_model
                SET id = '{{ id }}',
                name = '{{ name }}',
                state = {{ state }},
                input_table = '{{ input_table }}',
                prediction_table = '{{ prediction_table }}',
                x_columns = '{{ x_columns }}',
                y_column = '{{ y_column }}',
                prediction_columns = '{{ prediction_columns }}',
                input_size = {{ input_size }};
            ''')

tmpl['get_model_list'] = Template('''
            SELECT id, name FROM {{ database }}.{{ table }};
            ''')

tmpl['init_model_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id VARCHAR(20) NOT NULL,
                name VARCHAR(45) NULL,
                state INT NULL,
                input_table VARCHAR(45) NULL,
                prediction_table VARCHAR(45) NULL,
                x_columns TEXT NULL,
                y_column VARCHAR(45) NULL,
                prediction_columns TEXT NULL,
                input_size INT NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl['init_calculation_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
                y DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl['init_result_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                theta DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl['init_prediction_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                y_prediction DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl['init_score_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                score DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl['calculate_equations'] = Template('''
            INSERT INTO {{ table }}({% for x in x_columns %}{{ x }}, {% endfor %}y) 
            VALUES(
                {% for sum_statement in sum_statements %}
                    (SELECT {{ sum_statement }}
                {% endfor %}
            );
            ''')

tmpl['predict'] = Template('''
            INSERT INTO {{ table }} (y_prediction) 
            SELECT {{ prediction_statement }} FROM {{ input_table }};;
            ''')

tmpl['save_theta'] = Template('''
            INSERT INTO {{ table }} (theta)
            VALUES ({{ value }});
            ''')

tmpl['calculate_save_score'] = Template('''
            INSERT INTO {{ table_id }}_score (score)
            SELECT 1-((sum(({{ y }}-y_prediction)*({{ y }}-y_prediction)))/(sum(({{ y }}-y_avg)*({{ y }}-y_avg)))) FROM
            (SELECT {{ y }}, y_prediction FROM {{ input_table }}
            LEFT JOIN {{ table_id }}_prediction
            ON {{ table_id }}_prediction.id = {{ input_table }}.id
            ) AS subquery1
            CROSS JOIN
            (SELECT avg({{ y }}) as y_avg FROM {{ input_table }}) as subquery2;
            ''')
