tmplt = {}

# def train_test_split():
tmplt["_train"] = '''
SELECT * INTO {{ input.table_train }} FROM
(SELECT TOP ({{ input.ratio  }}* 100) PERCENT * from {{ input.dataset }} ORDER BY NEWID()) as train;
'''

tmplt["_eval"] = '''
SELECT * INTO {{ input.table_eval }} FROM
(SELECT TOP ((1 - {{ input.ratio }}) * 100) PERCENT * from {{ input.dataset }} ORDER BY NEWID()) as eval;
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
from (SELECT TOP ({{ input.ratio  }}* 100) PERCENT * from {{ input.dataset }} ORDER BY NEWID()) AS train
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
from (SELECT TOP ((1 - {{ input.ratio }}) * 100) PERCENT * from {{ input.dataset }} ORDER BY NEWID()) as train
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
create index idx{{ enumidx }} on {{ input.table_train }} ({{ idx }})
;
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


tmplt["_sqlMI"] ='''
(select  f, x, mi, cum_nx_, n__-cum_nx_ as alt,  n__ ,
(select sum(nxy) from info where y = 1 and info.f = e.f and info.x <= e.x) ny1,
(select sum(nxy) from info where y = 2 and info.f = e.f and info.x <= e.x) ny2,
(select sum(nxy) from info where y = 3 and info.f = e.f and info.x <= e.x) ny3,
(select sum(nxy) from info where y = 4 and info.f = e.f and info.x <= e.x) ny4,
(select sum(nxy) from info where y = 5 and info.f = e.f and info.x <= e.x) ny5,
(select sum(nxy) from info where y = 6 and info.f = e.f and info.x <= e.x) ny6,
(select sum(nxy) from info where y = 7 and info.f = e.f and info.x <= e.x) ny7,
(select sum(nxy) from info where y = 1 and info.f = e.f and info.x > e.x) alt_ny1,
(select sum(nxy) from info where y = 2 and info.f = e.f and info.x > e.x) alt_ny2,
(select sum(nxy) from info where y = 3 and info.f = e.f and info.x > e.x) alt_ny3,
(select sum(nxy) from info where y = 4 and info.f = e.f and info.x > e.x) alt_ny4,
(select sum(nxy) from info where y = 5 and info.f = e.f and info.x > e.x) alt_ny5,
(select sum(nxy) from info where y = 6 and info.f = e.f and info.x > e.x) alt_ny6,
(select sum(nxy) from info where y = 7 and info.f = e.f and info.x > e.x) alt_ny7,
(select sum(nxy) from info where info.f = e.f and info.x <= e.x) ny

from
(#e
select f, x, mi, cum_nx_, n__-cum_nx_ as alt,  n__
from
(#d
	select f, x, mi, max(mi) over(partition by f) as mx, cum_nx_, n__
	from
	( #c
		select f, x, 
		(sum( 
			(cum_nxy*1.0/n__)
			*log(2,
				(cum_nxy*1.0/n__)
				/( (n_y*1.0/n__) * (cum_nx_*1.0/n__) )
				) ) ) as mi,
		min(cum_nx_) as cum_nx_,
        min(n__) as n__
		from 
		( #b
			select f, x, y,
			sum(nxy) over(partition by f, y order by x) as cum_nxy, 
			round(sum(nx_/ndy_gx) over(partition by f order by x), 0)  as cum_nx_,
			n__, 
			nx_,
			n_y
			from 
			( #a
				SELECT f, x, y, nxy, nx_, n_y, n__, 
                ndy_gx, min_nxy
                FROM info
				join (SELECT f, x,  sum(nxy) as nx_, count(*) as ndy_gx, min(nxy) as min_nxy FROM info group by f, x) a using (f, x)
				join (SELECT f, y, sum(nxy) as n_y FROM info group by f, y) b using (f, y)
				join ( select f, sum(nxy) as n__ FROM info group by f ) c using (f) 
			) as a
			order by f, y, x
		) as b
		group by f, x
		order by f, cast(x as signed)
	) c
) d
where mi = mx
order by mx desc
) e )
;'''


tmplt["_DFmi"] ='''
(select  f, x, mi, cum_nx_, n__-cum_nx_ as alt,  n__ ,
{% for tg in input.targetValues %}
(select sum(nxy) from dtctraintest_info where y = {{ tg }} and dtctraintest_info.f = temp_mutual_inf.f and dtctraintest_info.x <= temp_mutual_inf.x) ny{{ tg }},
{% endfor %}\
{% for tg in input.targetValues %}
{% if loop.index > 1 %},{% endif %}\
(select sum(nxy) from dtctraintest_info where y = {{ tg }} and dtctraintest_info.f = temp_mutual_inf.f and dtctraintest_info.x > temp_mutual_inf.x) alt_ny{{ tg }}
{% endfor %}\
from temp_mutual_inf)
;'''

tmplt["_train_view"] ='''
(select * from {{ input.table_train }} where 
{% for key,value in input.mutual_inf.iterrows() %}
{% if loop.index > 1 %} and {% endif %}\
{{ value['f'] }} {{ input.crit }} {{ value['x'] }}
{% endfor %}\
)
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
