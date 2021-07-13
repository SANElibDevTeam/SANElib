tmplt = {}

# def train_test_split():
tmplt["_train"] = '''
select * from {{ input.dataset }} ORDER BY RANDOM() LIMIT (SELECT ROUND(COUNT(*) * {{ input.ratio }}) FROM {{ input.dataset }});
'''

tmplt["_eval"] = '''
select * from {{ input.dataset }} EXCEPT SELECT * FROM {{ input.table_train }};
'''

tmplt["_getColumns"] = '''
SELECT * FROM {{ input.dataset }} LIMIT 1
;
'''

tmplt["_distinctValues"] = '''
SELECT DISTINCT {{ column }} FROM {{ table }} ORDER BY {{ column }} ASC
;
'''

tmplt["_renameColumn"] = '''
ALTER TABLE {{ input.dataset }} RENAME COLUMN {{ orig }} TO {{ to }}
;
'''

tmplt["_encodeTableTrainEval"] = [
    '''DROP TABLE IF EXISTS {{ input.table_train }};''',
    '''DROP TABLE IF EXISTS {{ input.table_eval }};''',
    '''
create table {{ input.table_train }} as
select *,
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
(case
    {% for value in input.catFeatures[key] %}
        when {{ key }}_orig = '{{ value[0] }}' then {{ loop.index }}
    {% endfor %}\
end) as {{ key }}
{% endfor %}\
from (select * from {{ input.dataset }} ORDER BY RANDOM() LIMIT (SELECT ROUND(COUNT(*) * {{ input.ratio }}) FROM {{ input.dataset }}) ) as train
''',
    '''
create table {{ input.table_eval }} as
select *,
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
(case
    {% for value in input.catFeatures[key] %}
        when {{ key }}_orig = '{{ value[0] }}' then {{ loop.index }}
    {% endfor %}\
end) as {{ key }}
{% endfor %}\
FROM (SELECT * FROM {{ input.dataset }} EXCEPT SELECT
{% for col in input.numFeatures %}
{% if loop.index > 1 %}, {% endif %}\
{{ col }}
{% endfor %}\
{% if input.catFeatures|length > 0 %},{% endif %}\
{% for col in input.catFeatures %}
{% if loop.index > 1 %},{% endif %}\
{{ col }}_orig
{% endfor %}\
,{{ input.target }}
FROM {{ input.table_train }} ) AS EVAL
''',
'''
CREATE TABLE {{ input.table_train }}_temp as
SELECT
{% for key in input.numFeatures %}
{% if loop.index > 1 %}, {% endif %}\
{{ key }}
{% endfor %}\
,
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
{{ key }}
{% endfor %}\
, {{ input.target }}
FROM {{ input.table_train }}
;
''',
'''
CREATE TABLE {{ input.table_eval }}_temp as
SELECT 
{% for key in input.numFeatures %}
{% if loop.index > 1 %}, {% endif %}\
{{ key }}
{% endfor %}\
,
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
{{ key }}
{% endfor %}\
, {{ input.target }}
FROM {{ input.table_eval }}
;
''',
'''DROP TABLE {{ input.table_train }}''',
'''DROP TABLE {{ input.table_eval }}''',
'''ALTER TABLE {{ input.table_train }}_temp RENAME TO {{ input.table_train }}''',
'''ALTER TABLE {{ input.table_eval }}_temp RENAME TO {{ input.table_eval }}''',
]

tmplt["_addIndex"] = '''
create index if not exists idx{{ enumidx }} on {{ input.table_train }} ({{ idx }})
;
'''

tmplt["_CC_table"] = '''
{% for nf in input.numFeatures %}
{% if loop.index > 1 %}union all {% endif %}\
select '{{ nf }}' as f, {{ nf }} as x, {{ input.target }} as y, count(*)  as nxy from {{ subquery }} as subq group by {{ nf }}, {{ input.target }}\
{% endfor %}\
{% for cf in input.catFeatures %}
{% if loop.index + input.numFeatures|length  > 1 %} union {% endif %}\
select '{{ cf }}' as f, {{ cf }} as x, {{ input.target }} as y, count(*)  as nxy from {{ subquery }} as subq group by {{ cf }}, {{ input.target }}
{% endfor %}\
;'''

tmplt["_predictEval"] ='''
ALTER TABLE {{ input.table_eval }} ADD COLUMN Prediction INT GENERATED ALWAYS AS (
{% for f in input.model %}
{{ f }}
{% endfor %}\
)
;'''

tmplt["_predictionProcedure"] = '''
CREATE PROCEDURE predict_{{ input.dataset }}(
{% for f in input.numFeatures %}
IN {{ f }} INT,
{% endfor %}\
{% for f in input.catFeatures %}
IN {{ f }} INT,
{% endfor %}\
OUT predictedClass INT
)
BEGIN

{% for f in model %}
{{ f }};
END IF;
{% endfor %}\

END
'''
