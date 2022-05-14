from jinja2 import Template

tmpl_mysql = {}

tmpl_mysql['select_x_from'] = Template('''
            SELECT {{ x }} FROM {{ database }}.{{ table }};
            ''')

tmpl_mysql['select_x_from_where'] = Template('''
            SELECT {{ x }} FROM {{ table }} WHERE id = '{{ where_statement }}';
            ''')
tmpl_mysql['get_all_from'] = Template('''
            SELECT * FROM {{ database }}.{{ table }};
            ''')

tmpl_mysql['get_all_from_where_id'] = Template('''
            SELECT * FROM {{ database }}.{{ table }} WHERE id='{{ where_statement }}';
            ''')

tmpl_mysql['delete_from_table_where_id'] = Template('''
            DELETE FROM {{ table }} WHERE id='{{ where_statement }}';
            ''')
tmpl_mysql['set_safe_updates'] = Template('''
            SET SQL_SAFE_UPDATES = {{ value }};
            ''')

tmpl_mysql['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}';
            ''')

tmpl_mysql['column_type'] = Template('''
            SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE table_name = '{{ table }}' and column_name='{{ column }}';
            ''')

tmpl_mysql['drop_table'] = Template('''
            DROP TABLE IF EXISTS {{ table }};
            ''')

tmpl_mysql['add_ones_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column }} INT DEFAULT 1;
            ''')

tmpl_mysql['add_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column }} {{ type }};
            ''')

tmpl_mysql['set_ohe_column'] = Template('''
            UPDATE {{ table }}
            SET {{ ohe_column }} = IF({{ input_column }}='{{ value }}', 1, 0);
            ''')

tmpl_mysql['get_targets'] = Template('''
            SELECT DISTINCT({{ y }}) FROM {{ table }} order by {{ y }} ASC;
            
            ''')

tmpl_mysql['save_model'] = Template('''
            REPLACE INTO gaussian_model
                SET id = '{{ id }}',
                name = '{{ name }}',
                state = {{ state }},
                input_table = '{{ input_table }}',
                prediction_table = '{{ prediction_table }}',
                x_columns = '{{ x_columns }}',
                y_column = '{{ y_column }}',
                prediction_columns = '{{ prediction_columns }}',
                input_size = {{ input_size }},
                ohe_columns = '{{ ohe_columns }}';
            ''')

tmpl_mysql['get_model_list'] = Template('''
            SELECT id, name FROM {{ database }}.{{ table }};
            ''')

tmpl_mysql['init_model_table'] = Template('''
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
                ohe_columns TEXT NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_calculation_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
                y DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_uni_gauss_prob_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                row_no INT NULL,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
                y DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_covariance_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ table }} (
                id varchar(255) NOT NULL,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
                
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_inverse_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                {{ row_name }} DOUBLE NULL,
                {{ col_name }} DOUBLE NULL,
                actual_value DOUBLE NULL,
                

            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_determinante_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ table }} (
                id varchar(255) NOT NULL,
                determinante DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')
# tmpl_mysql['init_prediction_table'] = Template('''
#             CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
#                 id INT NOT NULL AUTO_INCREMENT,
#                 y_prediction DOUBLE NULL,
#             PRIMARY KEY (id),
#             UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
#             ''')

# tmpl_mysql['init_score_table'] = Template('''
#             CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
#                 id INT NOT NULL AUTO_INCREMENT,
#                 score DOUBLE NULL,
#             PRIMARY KEY (id),
#             UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
#             ''')

tmpl_mysql['calculate_means'] = Template('''
            INSERT INTO {{ table }}({% for x in x_columns_means %}{{ x }}, {% endfor %}y) 
            VALUES
                {% for class in y_classes%}
                    
                    ({% for x in x_columns %}
                        (SELECT AVG({{ x }}) FROM {{ input_table }} WHERE {{ target }} = {{ class }}),{% endfor %}{{class}}),
                    
                {% endfor %}
                ({% for x in x_columns_means %}999,{% endfor %}999);
            ''')

tmpl_mysql['calculate_variances'] =  Template('''
            INSERT INTO {{ table }}({% for x in x_columns_variances %}{{ x }}, {% endfor %}y) 
            VALUES
                {% for class in y_classes%}
                    
                    ({% for x in x_columns %}
                        (SELECT VARIANCE({{ x }}) FROM {{ input_table }} WHERE {{ target }} = {{ class }}),{% endfor %}{{class}}),
                    
                {% endfor %}
                ({% for x in x_columns_variances %}999,{% endfor %}999);
            ''')

tmpl_mysql['drop_row'] = Template('''
            DELETE FROM {{table}} WHERE y = 999;
            ''')

tmpl_mysql['calculate_gauss_prob_univariate'] = Template('''
            INSERT INTO {{ table }}(row_no,{% for x in x_columns %}{{ x }},{% endfor %}y) 
            VALUES
                {% for gauss_statement in gauss_statements %}
                    {{ gauss_statement }}
                {% endfor %}
                ({% for x in x_columns %}999,{% endfor %}999,999);    
            
            ''')

tmpl_mysql['get_no_of_rows'] = Template('''
SELECT COUNT(*) FROM {{ table }};
''')

tmpl_mysql['calculate_covariance'] = Template('''
SELECT SUM(({{ feature_1 }}
    -(SELECT{{ feature_1 }}
    from {{ mean_table }}
    where y={{ class }})) 
    * 
    ({{ feature2 }}
    -(SELECT {{ feature_2 }}
    from {{ mean_table }}
    where y={{ class }})))
     / (select count(*)
     from {{ input_table }}
     where {{ y_column }} = {{ class }}) as Covariance
from {{ input_table }} where {{ y_column }} = {{ class }};
''')

tmpl_mysql['create_table_like'] = Template('''
CREATE TABLE {{new_table}} LIKE {{original_table}};
''')
tmpl_mysql['copy_table'] = Template('''
INSERT INTO {{new_table}} SELECT * FROM {{original_table}};
''')

tmpl_mysql['diff_mean']= Template('''
UPDATE {{ table }} SET {{column}} = ({{ feature_1 }}
    -(SELECT {{ feature_1 }}
    from {{ mean_table }}
    where y={{ y_class }})) 
''')

tmpl_mysql["insert_id"]= Template('''
INSERT INTO {{ table }} (id) VALUES({{ id }})''')

tmpl_mysql['multiply_columns']= Template('''
UPDATE {{ table }} SET {{column}} = ({{ feature_1 }} * {{ feature_2 }}
    ) 
''')

tmpl_mysql['rename_column']= Template('''
ALTER TABLE {{ table }} RENAME {{ column }} TO {{ new_column }};
''')

tmpl_mysql['divide_by_total']= Template('''
UPDATE {{ table }} SET {{column}} = ({{ column }} / (select count(*)
     from {{ input_table }}
     where {{ y_column }} = {{ y_class }}) 
    ) 
''')

tmpl_mysql['fill_covariance_matrix'] = Template('''
UPDATE {{ table }} SET {{ feature_2 }} = ((SELECT SUM({{ covariance }}) FROM {{ gaussian_input_table }})/(select count(*)
     from {{ gaussian_input_table }}) ) WHERE id= {{ feature_1 }};
''')

tmpl_mysql['determinante']= Template('''
SELECT ({{ m00 }}) * ({{ m11 }}) - ({{ m10 }}) * ({{ m01 }}) as determinante
     from {{ covariance_matrix }} 
''')

tmpl_mysql['m0c'] = Template(''' SELECT {{ m0c1 }} from {{ covariance_matrix }} WHERE id= {{ m0c0 }};  ''')

tmpl_mysql['transpose_matrix'] = Template('''
REPLACE INTO {{ inverse_matrix }} (id,{% for feature in features %} {% if loop.index > 1 %}, {% endif %} {{feature}} {%endfor%} )
select id_t, {% for feature in features %}
    {% if loop.index > 1 %}, {% endif %}
    max(IF(id = {{"'"}}{{ feature }}{{"'"}}, amount, null)) AS {{ feature }}
    {% endfor %}
    from (
    {% for feature in features %}
    {% if loop.index > 1 %}union {% endif %}
        select id, {{"'"}}{{ feature }}{{"'"}} id_t, {{ feature }} as amount from {{ covariance_matrix }}
        
    {% endfor %}
  ) t
  group by id_t;
''')
tmpl_mysql['insert_inverse_2_2'] = Template('''
INSERT INTO {{ inverse_matrix }} (id,{% for feature in features %} {% if loop.index > 1 %}, {% endif %} {{feature}} {%endfor%} )
VALUES ({{"'"}}{{features[0]}}{{"'"}},({{ m11 }}), ({{ m01 }})),
        ({{"'"}}{{features[1]}}{{"'"}},({{ m10 }}), ({{ m00 }}))
''')

tmpl_mysql['insert_inverse_n_n'] = Template('''
INSERT INTO {{ inverse_matrix }} (id,{% for feature in features %} {% if loop.index > 1 %}, {% endif %} {{feature}} {%endfor%} )
VALUES 
    {% for row in cofactor_matrix %}
    {% if loop.index > 1 %}, {% endif %}
    ({{"'"}}{{features[loop.index-1]}}{{"'"}},{% for element in row %} {% if loop.index > 1 %}, {% endif %} {{element}} {%endfor%} )
    {%endfor%}
    

''')

tmpl_mysql['insert_determinante'] = Template('''
INSERT INTO {{ determinante_table }} (id,determinante)
VALUES ({{ table }}, {{ determinante }})
''')


tmpl_mysql['insert_inverse'] = Template('''
INSERT INTO {{ inverse_table }} (k, j, actual_value)
VALUES ({{ row }}, {{ column }}, {{ actual_value }})
''')


tmpl_mysql['insert_vector'] = Template('''
{% for n in row_no %}
{% for y in y_classes%}
INSERT INTO {{ vector_table }}{{ n }}_{{y}} (i, k, actual_value)
VALUES
{% for element in x_columns %}
{% if loop.index > 1 %}, {% endif %}
(1, {{ loop.index }},(SELECT {{ element }}_diff_mean_{{ y }} FROM {{ input_table }} LIMIT {{ n }},1))
{% endfor %};
{% endfor %}
{% endfor %}

''')


tmpl_mysql['calculate_mahalonobis_distance'] = Template('''

{% for n in row_no %}
{% for y in y_classes %}
INSERT INTO {{ estimation_table }} (row_no, y, mahalonobis_distance)
SELECT  {{ n }} as row_no,  {{ y }} as y, SUM(MatrixD.actual_value * vector_transformed.actual_value) as mahalonobis_distance
  FROM 

 (SELECT i, j as k, SUM({{ vector_table }}{{ n }}_{{y}}.actual_value * {{ covariance_matrix }}_{{ y }}_inverse.actual_value) as actual_value
  FROM {{ vector_table }}{{ n }}_{{y}}, {{ covariance_matrix }}_{{ y }}_inverse
 WHERE {{ vector_table }}{{ n }}_{{y}}.k ={{ covariance_matrix }}_{{ y }}_inverse.k
 GROUP BY i, j) AS MatrixD,
 (SELECT k, i as j,actual_value FROM {{ vector_table }}{{ n }}_{{y}}) as vector_transformed
 
 WHERE MatrixD.k = vector_transformed.k
 GROUP BY i, j ;
 
 {% endfor %}
{% endfor %}
''')

tmpl_mysql['init_multi_gauss_prob_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                row_no INT NULL,
                mahalonobis_distance DOUBLE NULL,
                probability DOUBLE NULL,
                y DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')


tmpl_mysql['calculate_multivariate_density'] = Template('''
{% for n in row_no %}
{% for y in y_classes %}
UPDATE {{ table }} SET probability =
(SELECT (1/(power(2*PI(),{{ k }})*(SELECT determinante from {{ determinante_table}} where id = "gaussian_m0_covariance_matrix_{{ y }}")))
* EXP(0.5 * (SELECT mahalonobis_distance from {{ table }} where y = {{ y }} and row_no ={{ n }})))
WHERE y = {{ y }} and row_no ={{ n }};
{% endfor %}
{% endfor %}

''')
# tmpl_mysql['predict'] = Template('''
#             INSERT INTO {{ table }} (y_prediction)
#             SELECT {{ prediction_statement }} FROM {{ input_table }};;
#             ''')
#
# tmpl_mysql['save_theta'] = Template('''
#             INSERT INTO {{ table }} (theta)
#             VALUES ({{ value }});
#             ''')
#
# tmpl_mysql['calculate_save_score'] = Template('''
#             INSERT INTO {{ table_id }}_score (score)
#             SELECT 1-((sum(({{ y }}-y_prediction)*({{ y }}-y_prediction)))/(sum(({{ y }}-y_avg)*({{ y }}-y_avg)))) FROM
#             (SELECT {{ y }}, y_prediction FROM {{ input_table }}
#             LEFT JOIN {{ table_id }}_prediction
#             ON {{ table_id }}_prediction.id = {{ input_table }}.id
#             ) AS subquery1
#             CROSS JOIN
#             (SELECT avg({{ y }}) as y_avg FROM {{ input_table }}) as subquery2;
#             ''')