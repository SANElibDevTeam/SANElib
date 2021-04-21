import sql_templates

result = sql_templates.get_templates("sqlite")
print(result.get_row_count("demo"))
