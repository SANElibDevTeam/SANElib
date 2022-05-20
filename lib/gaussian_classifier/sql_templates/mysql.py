from jinja2 import Template

tmpl_mysql = {}

tmpl_mysql["_train"] = Template('''
create table {{ train_table }} as 
(select * from {{ database }}.{{ input_table }} where rand ({{ input_seed }}) < {{ input_ratio }});
''')

tmpl_mysql["_eval"] = Template('''
create table {{ train_table }} as 
(select * from {{ database }}.{{ input_table }} where rand ({{ input_seed }}) >= {{ input_ratio }});
''')
tmpl_mysql['select_x_from'] = Template('''
            SELECT {{ x }} FROM {{ database }}.{{ table }};
            ''')

tmpl_mysql['get_accuracy'] = Template('''
            SELECT SUM({{ x }})/{{ no_rows_prediction }} FROM {{ database }}.{{ table }};
            ''')

tmpl_mysql['select_x_from_where'] = Template('''
            SELECT {{ x }} FROM {{ database }}.{{ table }} WHERE id = '{{ where_statement }}';
            ''')
tmpl_mysql['get_all_from'] = Template('''
            SELECT * FROM {{ database }}.{{ table }};
            ''')

tmpl_mysql['get_all_from_where_id'] = Template('''
            SELECT * FROM {{ database }}.{{ table }} WHERE id='{{ where_statement }}';
            ''')

tmpl_mysql['delete_from_table_where_id'] = Template('''
            DELETE FROM {{ database }}.{{ table }} WHERE id='{{ where_statement }}';
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
            DROP TABLE IF EXISTS {{ database }}.{{ table }};
            ''')

tmpl_mysql['add_ones_column'] = Template('''
            ALTER TABLE {{ database }}.{{ table }} ADD COLUMN {{ column }} INT DEFAULT 1;
            ''')

tmpl_mysql['add_column'] = Template('''
            ALTER TABLE {{ database }}.{{ table }} ADD COLUMN {{ column }} {{ type }};
            ''')

tmpl_mysql['set_ohe_column'] = Template('''
            UPDATE {{ database }}.{{ table }}
            SET {{ ohe_column }} = IF({{ input_column }}='{{ value }}', 1, 0);
            ''')

tmpl_mysql['get_targets'] = Template('''
            SELECT DISTINCT({{ y }}) FROM {{ database }}.{{ table }} order by {{ y }} ASC;
            
            ''')

tmpl_mysql['get_target_array'] = Template('''
            SELECT {{ y }} FROM {{ database }}.{{ table }};

            ''')

tmpl_mysql['save_model'] = Template('''
            REPLACE INTO gaussian_model
                SET id = '{{ id }}',
                name = '{{ name }}',
                state = {{ state }},
                input_table = '{{ input_table }}',
                prediction_table = '{{ prediction_table }}',
                x_columns = '{{ x_columns }}',
                y_classes = '{{ y_classes }}',
                y_column = '{{ y_column }}',
                prediction_columns = '{{ prediction_columns }}',
                input_size = {{ input_size }},
                no_of_rows_input = {{ no_of_rows_input }},
                no_of_rows_prediction = {{ no_of_rows_prediction }};
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
                y_classes TEXT NULL,
                y_column VARCHAR(45) NULL,
                prediction_columns TEXT NULL,
                input_size INT NULL,
                no_of_rows_input INT NULL,
                no_of_rows_prediction INT NULL,
                
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

tmpl_mysql['init_mean_overall_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
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
                gaussian_distribution DOUBLE NULL,
                p_y DOUBLE NULL,
                probability DOUBLE NULL,
                y DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_covariance_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id varchar(255) NOT NULL,
                {% for x in x_columns %}
                    {{ x }} DOUBLE NULL,
                {% endfor %}
                
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_inverse_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                {{ row_name }} DOUBLE NULL,
                {{ col_name }} DOUBLE NULL,
                actual_value DOUBLE NULL,
                

            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_determinante_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id varchar(255) NOT NULL,
                determinante DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['init_prediction_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL,
                y_prediction DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

# tmpl_mysql['init_score_table'] = Template('''
#             CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
#                 id INT NOT NULL AUTO_INCREMENT,
#                 y_prediction DOUBLE NULL,
#                 y DOUBLE NULL,
#                 score DOUBLE NULL,
#             PRIMARY KEY (id),
#             UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
#             ''')

tmpl_mysql['calculate_means'] = Template('''
            INSERT INTO {{ database }}.{{ table }}({% for x in x_columns_means %}{{ x }}, {% endfor %}y) 
            VALUES
                {% for class in y_classes%}
                    
                    ({% for x in x_columns %}
                        (SELECT AVG({{ x }}) FROM {{ database }}.{{ input_table }} WHERE {{ target }} = {{ class }}),{% endfor %}{{class}}),
                    
                {% endfor %}
                ({% for x in x_columns_means %}999,{% endfor %}999);
            ''')

tmpl_mysql['calculate_means_overall'] = Template('''
            INSERT INTO {{ database }}.{{ table }}({% for x in x_columns_means %}{% if loop.index > 1 %}, {% endif %}{{ x }} {% endfor %}) 
            VALUES
                    ({% for x in x_columns %}
                    {% if loop.index > 1 %}, {% endif %}
                        (SELECT AVG({{ x }}) FROM {{ input_table }})

                {% endfor %} );
                
            ''')

tmpl_mysql['calculate_variances'] = Template('''
            INSERT INTO {{ database }}.{{ table }}({% for x in x_columns_variances %}{{ x }}, {% endfor %}y) 
            VALUES
                {% for class in y_classes%}
                    
                    ({% for x in x_columns %}
                        (SELECT VARIANCE({{ x }}) FROM {{ database }}.{{ input_table }} WHERE {{ target }} = {{ class }}),{% endfor %}{{class}}),
                    
                {% endfor %}
                ({% for x in x_columns_variances %}999,{% endfor %}999);
            ''')

tmpl_mysql['drop_row'] = Template('''
            DELETE FROM {{ database }}.{{table}} WHERE y = 999;
            ''')

tmpl_mysql['calculate_gauss_prob_univariate'] = Template('''
            INSERT INTO {{ database }}.{{ table }}(row_no,{% for x in x_columns %}{{ x }},{% endfor %}y) 
            VALUES
                {% for gauss_statement in gauss_statements %}
                    {{ gauss_statement }}
                {% endfor %}
                ;    
            
            ''')

tmpl_mysql['get_no_of_rows'] = Template('''
SELECT COUNT(*) FROM {{ database }}.{{ table }};
''')

tmpl_mysql['calculate_covariance'] = Template('''
SELECT SUM(({{ feature_1 }}
    -(SELECT{{ feature_1 }}
    from {{ database }}.{{ mean_table }}
    where y={{ class }})) 
    * 
    ({{ feature2 }}
    -(SELECT {{ feature_2 }}
    from {{ database }}.{{ mean_table }}
    where y={{ class }})))
     / (select count(*)
     from {{ database }}.{{ input_table }}
     where {{ y_column }} = {{ class }}) as Covariance
from {{ database }}.{{ input_table }} where {{ y_column }} = {{ class }};
''')

tmpl_mysql['create_table_like'] = Template('''
CREATE TABLE {{new_table}} LIKE {{original_table}};
''')
tmpl_mysql['copy_table'] = Template('''
INSERT INTO {{ database }}.{{new_table}} SELECT * FROM {{ database }}.{{original_table}};
''')

tmpl_mysql['diff_mean'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{column}} = ({{ feature_1 }}
    -(SELECT {{ feature_1 }}
    from {{ mean_table }}
    where y={{ y_class }})) 
''')

tmpl_mysql['diff_mean_overall'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{column}} = ({{ feature_1 }}
    -(SELECT {{ feature_1 }}
    from {{ database }}.{{ mean_table }})) 
''')

tmpl_mysql["insert_id"] = Template('''
INSERT INTO {{ database }}.{{ table }} (id) VALUES({{ id }})''')

tmpl_mysql['multiply_columns'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{column}} = ({{ feature_1 }} * {{ feature_2 }}
    ) 
''')

tmpl_mysql['rename_column'] = Template('''
ALTER TABLE {{ database }}.{{ table }} RENAME {{ column }} TO {{ new_column }};
''')

tmpl_mysql['divide_by_total'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{column}} = ({{ column }} / (select count(*)
     from {{ database }}.{{ input_table }}
     where {{ y_column }} = {{ y_class }}) 
    ) 
''')

tmpl_mysql['fill_covariance_matrix'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{ feature_2 }} = ((SELECT SUM({{ covariance }}) FROM {{ database }}.{{ gaussian_input_table }})/ {{ no_rows }} ) WHERE id= {{ feature_1 }};
''')

tmpl_mysql['determinante'] = Template('''
SELECT ({{ m00 }}) * ({{ m11 }}) - ({{ m10 }}) * ({{ m01 }}) as determinante
     from {{ database }}.{{ covariance_matrix }} 
''')

tmpl_mysql['m0c'] = Template(''' SELECT {{ m0c1 }} from {{ database }}.{{ covariance_matrix }} WHERE id= {{ m0c0 }};  ''')

tmpl_mysql['transpose_matrix'] = Template('''
REPLACE INTO {{ database }}.{{ inverse_matrix }} (id,{% for feature in features %} {% if loop.index > 1 %}, {% endif %} {{feature}} {%endfor%} )
select id_t, {% for feature in features %}
    {% if loop.index > 1 %}, {% endif %}
    max(IF(id = {{"'"}}{{ feature }}{{"'"}}, amount, null)) AS {{ feature }}
    {% endfor %}
    from (
    {% for feature in features %}
    {% if loop.index > 1 %}union {% endif %}
        select id, {{"'"}}{{ feature }}{{"'"}} id_t, {{ feature }} as amount from {{ database }}.{{ covariance_matrix }}
        
    {% endfor %}
  ) t
  group by id_t;
''')
tmpl_mysql['insert_inverse_2_2'] = Template('''
INSERT INTO {{ database }}.{{ inverse_matrix }} (id,{% for feature in features %} {% if loop.index > 1 %}, {% endif %} {{feature}} {%endfor%} )
VALUES ({{"'"}}{{features[0]}}{{"'"}},({{ m11 }}), ({{ m01 }})),
        ({{"'"}}{{features[1]}}{{"'"}},({{ m10 }}), ({{ m00 }}))
''')

tmpl_mysql['insert_inverse_n_n'] = Template('''
INSERT INTO {{ database }}.{{ inverse_matrix }} (id,{% for feature in features %} {% if loop.index > 1 %}, {% endif %} {{feature}} {%endfor%} )
VALUES 
    {% for row in cofactor_matrix %}
    {% if loop.index > 1 %}, {% endif %}
    ({{"'"}}{{features[loop.index-1]}}{{"'"}},{% for element in row %} {% if loop.index > 1 %}, {% endif %} {{element}} {%endfor%} )
    {%endfor%}
    

''')

tmpl_mysql['insert_determinante'] = Template('''
INSERT INTO {{ database }}.{{ determinante_table }} (id,determinante)
VALUES ({{ table }}, {{ determinante }})
''')

tmpl_mysql['insert_inverse'] = Template('''
INSERT INTO {{ database }}.{{ inverse_table }} (k, j, actual_value)
VALUES ({{ row }}, {{ column }}, {{ actual_value }})
''')
tmpl_mysql['insert_target'] = Template('''
{%for n in row_no%}
UPDATE {{ database }}.{{ table }} SET {{ column }} = (SELECT {{ y_column }}
FROM {{ input_table }} LIMIT {{ n }},1) WHERE id = {{ n }};
{%endfor%}
''')
#
tmpl_mysql['insert_score'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{ column }} =
CASE
   WHEN {{ column }} = 0 THEN 1
   ELSE 0
END
''')

tmpl_mysql['substract_columns'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{column}} = ({{ feature_1 }} - {{ feature_2 }}
    ) 
''')

tmpl_mysql['insert_vector'] = Template('''
{% for n in row_no %}
{% for y in y_classes%}
INSERT INTO {{ database }}.{{ vector_table }}{{ n }} (i, k, actual_value)
VALUES
{% for element in x_columns %}
{% if loop.index > 1 %}, {% endif %}
(1, {{ loop.index }},(SELECT {{ element }}_diff_mean FROM {{ database }}.{{ input_table }} LIMIT {{ n }},1))
{% endfor %};
{% endfor %}
{% endfor %}

''')

tmpl_mysql['calculate_mahalonobis_distance'] = Template('''

{% for n in row_no %}
{% for y in y_classes %}
INSERT INTO {{ database }}.{{ estimation_table }} (row_no, y, mahalonobis_distance)
SELECT  {{ n }} as row_no,  {{ y }} as y, SUM(MatrixD.actual_value * vector_transformed.actual_value) as mahalonobis_distance
  FROM 

 (SELECT i, j as k, SUM({{ vector_table }}{{ n }}.actual_value * {{ covariance_matrix }}_{{ y }}_inverse.actual_value) as actual_value
  FROM {{ database }}.{{ vector_table }}{{ n }}, {{ database }}.{{ covariance_matrix }}_{{ y }}_inverse
 WHERE {{ vector_table }}{{ n }}.k ={{ covariance_matrix }}_{{ y }}_inverse.k
 GROUP BY i, j) AS MatrixD,
 (SELECT k, i as j,actual_value FROM {{ database }}.{{ vector_table }}{{ n }}) as vector_transformed
 
 WHERE MatrixD.k = vector_transformed.k
 GROUP BY i, j ;
 
 {% endfor %}
{% endfor %}
''')

tmpl_mysql['init_multi_gauss_prob_table'] = Template('''
            CREATE TABLE IF NOT EXISTS {{ database }}.{{ table }} (
                id INT NOT NULL AUTO_INCREMENT,
                row_no INT NULL,
                mahalonobis_distance DOUBLE NULL,
                gaussian_distribution DOUBLE NULL,
                p_y DOUBLE NULL,
                probability DOUBLE NULL,
                y DOUBLE NULL,
            PRIMARY KEY (id),
            UNIQUE INDEX id_UNIQUE (id ASC) VISIBLE);
            ''')

tmpl_mysql['calculate_multivariate_density'] = Template('''
{% for n in row_no %}
{% for y in y_classes %}
UPDATE {{ database }}.{{ table }} SET gaussian_distribution =
(SELECT (1/(power(2*PI(),{{ k }})*(SELECT determinante from {{ database }}.{{ determinante_table}} where id = "{{ id }}_covariance_matrix_{{ y }}")))
* EXP(0.5 * (SELECT mahalonobis_distance from (SELECT * from {{ database }}.{{ table }}) as tmp_table where y = {{ y }} and row_no ={{ n }})))
WHERE y = {{ y }} and row_no ={{ n }};
{% endfor %}
{% endfor %}

''')

tmpl_mysql['calculate_univariate_density'] = Template('''
UPDATE {{ database }}.{{ table }} SET {{column}} = ({{ multiplication_string }})
''')
tmpl_mysql['predict'] = Template('''
INSERT INTO {{ database }}.{{ table }} (id,y_prediction)
VALUES
{% for n in row_no %}
{% if loop.index > 1 %}, {% endif %}

({{ n }},(SELECT y
FROM {{ database }}.{{ estimation_table }}
WHERE (probability) in (select max(probability)
                                from {{ database }}.{{ estimation_table }}
                                WHERE row_no = {{ n }}
                                )
                                AND row_no = {{ n }}
                                ))


{%endfor %};
            ''')

tmpl_mysql['p_y'] = Template('''
{%for y in y_classes %}
UPDATE {{ database }}.{{ table }} SET p_y = ((SELECT COUNT(*) FROM {{ database }}.{{ input_table }} WHERE {{ y_column }} = {{ y }})/ {{ row_no}})
WHERE y = {{ y }};

{%endfor%}
''')
