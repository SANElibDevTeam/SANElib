from jinja2 import Template

tmpl = {}

tmpl['table_columns'] = Template('''
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA='{{ database }}' AND TABLE_NAME='{{ table }}'
            ''')

tmpl['add_ones_column'] = Template('''
            ALTER TABLE {{ table }} ADD COLUMN {{ column_name }} INT DEFAULT 1
            ''')