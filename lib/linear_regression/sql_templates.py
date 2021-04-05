from jinja2 import Template

tmpl = {}

tmpl['select_x_from'] = Template('''
            SELECT {{ x }} FROM {{ database }}.{{ table }};
            ''')

tmpl['get_all_from'] = Template('''
            SELECT * FROM {{ database }}.{{ table }};
            ''')

tmpl['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}';
            ''')

tmpl['drop_table'] = Template('''
            DROP TABLE IF EXISTS {{ table }};
            ''')

tmpl['add_ones_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column }} INT DEFAULT 1;
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
            INSERT INTO {{ table }}(x0,x1,x2,y) 
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



