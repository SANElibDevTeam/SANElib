tmplt = {}

# def train_test_split():
tmplt["_train"] = '''
(select * from {{ input.dataset }} where rand ({{ input.seed }}) < {{ input.ratio }});
'''

tmplt["_eval"] = '''
(select * from {{ input.dataset }} where rand ({{ input.seed }}) >= {{ input.ratio }});
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
    '''SET GLOBAL max_allowed_packet=1073741824;''',
    '''DROP TABLE IF EXISTS {{ input.table_train }};''',
    '''DROP TABLE IF EXISTS {{ input.table_eval }};''',
    '''ALTER TABLE {{ input.dataset }} ADD column `rownum` INT NOT NULL AUTO_INCREMENT unique first;''',
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
from (select * from {{ input.dataset }} where rand ({{ input.seed }}) <= {{ input.ratio }}) as train
group by rownum;
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
from (select * from {{ input.dataset }} where rand ({{ input.seed }}) > {{ input.ratio }}) as eval
group by rownum;
''',
'''alter table {{ input.dataset }} drop column `rownum`;''',
'''alter table {{ input.table_train }} drop column `rownum`;''',
'''alter table {{ input.table_eval }} drop column `rownum`;''',
'''
alter table {{ input.table_train }}
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
drop column {{ key }}_orig
{% endfor %}\
;
''',
'''
alter table {{ input.table_eval }}
{% for key in input.catFeatures %}
{% if loop.index > 1 %}, {% endif %}\
drop column {{ key }}_orig
{% endfor %}\
;
'''
]

tmplt["_addIndex"] = '''
alter table {{ input.table_train }} add index ({{ idx }})
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
alter table {{ input.table_eval }} add column Prediction int as (
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
