from jinja2 import Template

tmpl = {}

tmpl['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}'
            ''')

tmpl['add_ones_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column }} INT DEFAULT 1
            ''')

tmpl['init_calculation_table'] = Template('''
            CREATE TABLE IF NOT EXISTS `{{ database }}`.`{{ table }}` (
                `id` INT NOT NULL AUTO_INCREMENT,
                {% for x in x_columns %}
                    `{{ x }}` DOUBLE NULL,
                {% endfor %}
                `y` DOUBLE NULL,
            PRIMARY KEY (`id`),
            UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE)
            ''')
