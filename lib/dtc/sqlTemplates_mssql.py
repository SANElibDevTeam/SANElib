tmplt = {}

# def train_test_split():
tmplt["_train"] = '''
SELECT TOP ({{ input.ratio }} * 100) PERCENT * INTO {{ input.table_train }} FROM {{ input.dataset }} ORDER BY NEWID();
'''


tmplt["_eval"] = '''
SELECT * INTO {{ input.table_eval}} FROM {{ input.dataset }} EXCEPT SELECT * FROM {{ input.table_train }};
'''

tmplt["_getColumns"] = '''
SELECT TOP 1 * FROM {{ input.dataset }}
;
'''

tmplt["_distinctValues"] = '''
SELECT DISTINCT {{ column }} FROM {{ table }} ORDER BY {{ column }} ASC
;
'''

tmplt["_renameColumn"] = '''
EXEC dbo.sp_rename '{{ input.dataset }}.{{ orig }}', '{{ to }}', 'COLUMN';
'''

tmplt["_encodeTableTrainEval"] = [
'''DROP TABLE IF EXISTS {{ input.table_train }};''',
'''DROP TABLE IF EXISTS {{ input.table_eval }};''',
'''
select *,
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
(case
    {% for value in input.catFeatures[key] %}
        when {{ key }}_orig = '{{ value[0] }}' then {{ loop.index }}
    {% endfor %}\
end) as {{ key }}
{% endfor %}\
into {{ input.table_train }}
FROM (SELECT TOP ({{ input.ratio }} * 100) PERCENT * FROM {{ input.dataset }} ORDER BY NEWID()) AS TRAIN
;
''',
'''
select *,
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
(case
    {% for value in input.catFeatures[key] %}
        when {{ key }}_orig = '{{ value[0] }}' then {{ loop.index }}
    {% endfor %}\
end) as {{ key }}
{% endfor %}\
into {{ input.table_eval }}
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
alter table {{ input.table_train }} drop
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
column {{ key }}_orig
{% endfor %}\
;
''',
'''
alter table {{ input.table_eval }} drop
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
column {{ key }}_orig
{% endfor %}\
;
''']

tmplt["_addIndex"] = '''
IF NOT EXISTS(SELECT * FROM sys.indexes WHERE name = 'idx{{ enumidx }}' AND object_id = OBJECT_ID('{{ input.table_train }}'))
    BEGIN
        create index idx{{ enumidx }} on {{ input.table_train }} ({{ idx }});
    END
'''


tmplt["_CC_table"] ='''
select f, x, {{ input.target }} as y, count(*) as nxy into {{ input.dataset }}_CC_table from
{{ subquery }} as p
unpivot
(
x for f in (
{% for nf in input.numFeatures %}
{% if loop.index > 1 %},{% endif %}\
{{ nf }}
{% endfor %}\
{% if input.catFeatures|length > 0 %},{% endif %}\
{% for nf in input.catFeatures %}
{% if loop.index > 1 %},{% endif %}\
{{ nf }}
{% endfor %}\
)
) as unpinv
group by f,x,{{ input.target }}
;'''


tmplt["_predictEval"] ='''
alter table {{ input.table_eval }} add Prediction as (
{% for f in input.model %}
{{ f }}
{% endfor %}\
) PERSISTED
;'''

tmplt["_predictionProcedure"] ='''
CREATE PROCEDURE predict_{{ input.dataset }}
{% for f in input.numFeatures %}
{% if loop.index > 1 %},{% endif %}\
@{{ f }} INT = 0
{% endfor %}\
{% for f in input.catFeatures %}
,
@{{ f }} INT = 0
{% endfor %}\
AS
DECLARE @pred INT
{% for f in model %}
{{ f }}
{% endfor %}\
print(@pred)
'''
