tmplt = {}

# def train_test_split():
tmplt["_train"] = '''
(select * from {{ input.table_train }} where rand ({{ input.seed }}) < {{ input.ratio }});
'''

tmplt["_eval"] = '''
(select * from {{ input.table_train }} where rand ({{ input.seed }}) >= {{ input.ratio }});
'''

#######
tmplt["_agg"] = ''' 
select 
{% for i in input.numFeatures %} \
    max({{ i }}) as mx{{ loop.index }}, min({{ i }})-0.001 as mn{{ loop.index }}, 
    {% endfor %} \
    count(*) as n__
from table_train 
'''

# def train():
tmplt["_qt"] ='''
select y_, 
{% for i in input.numFeatures %} xq{{ loop.index }},  min(xn{{ loop.index }}) as mn{{ loop.index }}, \
max(xn{{ loop.index }}) as mx{{ loop.index }}, {% endfor %}
{% for i in input.catFeatures %} xc{{ loop.index }}, {% endfor %}
count(*) as nxy
 from ( -- a
	select
{% for i in input.numFeatures %}\
CEIL({{ input.bins }}*({{ i }}-mn{{ loop.index }})/((mx{{ loop.index }}-mn{{ loop.index }}) ) )  as xq{{ loop.index }}, 
{% endfor %}
{% for i in input.numFeatures %} {{ i }} as xn{{ loop.index }}, {% endfor %}
{% for i in input.catFeatures %} {{ i }} as xc{{ loop.index }}, {% endfor %}
{{ input.target }} as y_
	from {{input.table_train}} 
	cross join  {{ input.model_id }}_agg )a
group by y_
	{% for i in input.numFeatures %} ,xq{{ loop.index }}  {% endfor %}
    {% for i in input.catFeatures %} ,xc{{ loop.index }}  {% endfor %};
'''

# def predict():

tmplt["_qe"] = '''
select 
{% for i in input.numFeatures %} \
	CEIL({{ input.bins }}*( xn{{ loop.index }}-mn{{ loop.index }})/((mx{{ loop.index }}-mn{{ loop.index }}) ) )  as xq{{ loop.index }}, 
{% endfor %} \
t.*
 	from (select row_number() over() as id, {{ input.target }} as y 
    -- qe_nf
    {% for i in input.numFeatures %} ,{{ i }} as xn{{ loop.index }} {% endfor %}
    {% for i in input.catFeatures %} ,{{ i }} as xc{{ loop.index }} {% endfor %}
 from {{ input.table_eval }}) -- table_eval
    as t
    cross join {{ input.model_id }}_agg 
''';

tmplt["_qe_ix"] = '''create index ix1 on {{ input.model_id }}_qe(
{% for i in input.numFeatures %} xq{{ loop.index }} , {% endfor %}
{% for i in input.catFeatures %} xc{{ loop.index }}(35) {% if not loop.last %},{% endif %} {% endfor %}
);
''';

tmplt["_p"] = '''select *
from ( -- b
	select id, 
	{% for i in input.numFeatures %} xn{{ loop.index }} , {% endfor %}
    {% for i in input.catFeatures %} xc{{ loop.index }} , {% endfor %}
    y, y_, measure,
		case when max(measure) OVER(PARTITION BY 
        {% for i in input.numFeatures %} xq{{ loop.index }}\
        {% if loop.index < (input.numFeatures|length + input.catFeatures|length) %},{% endif %}\
        {% endfor %}
        {% for i in input.catFeatures %} xc{{ loop.index }}\
        {% if loop.index + input.numFeatures|length < (input.numFeatures|length + input.catFeatures|length) %},{% endif %}\
        {% endfor %}
        ) -- arg max y (p_xgy * p_y)
			= measure then 1 else 0 end as prediction,
		nxy
    from --
	(
		select
			e.*, m.y_,
			nxy *1.0/n__  as measure,
            m.nxy
		from {{ input.model_id }}_qe as e -- evaluation  dataset
		cross join {{ input.model_id }}_agg
		left outer join {{ input.model_id }}_qt as m
		using(
        {% for i in input.numFeatures %} xq{{ loop.index }}\
        {% if loop.index < (input.numFeatures|length + input.catFeatures|length) %},{% endif %}\
        {% endfor %}
        {% for i in input.catFeatures %} xc{{ loop.index }}\
        {% if loop.index + input.numFeatures|length < (input.numFeatures|length + input.catFeatures|length) %},{% endif %}\
        {% endfor %}
        )
	) a
    -- where nxy is null
) b
where prediction = 1 or y_ is null;'''

tmplt["_p_update"] ='''update {{ input.model_id }}_p
set y_ =
(select  y_ from (
select y_, sum(nxy) as n from  {{ input.model_id }}_qt group by y_
) a 
where n =  (select max(n) from (select y_, sum(nxy) as n from {{ input.model_id }}_qt group by y_ ) b ))
where y_ is null
;'''

tmplt["_m1d"] ='''
{% for nf in input.numFeatures %}
{% if loop.index > 1 %}union all {% endif %}\
select "{{ nf }}" as f, x, y, count(*)  as nxy 
from 
( select
        "{{ nf }}" as f,
		CEIL({{ input.bins}}*RANK() OVER (ORDER BY {{ nf }} )*1.0/COUNT(*) OVER()) as x,
        {{ input.target }} as y
from table_train ) a  
group by x, y\
{% endfor %}\
{% for cf in input.catFeatures %}
{% if loop.index + input.numFeatures|length  > 1 %} union {% endif %}\
select "{{ cf }}" as f, {{ cf }} as x, {{ input.target }} as y, count(*)  as nxy from {{ input.table_train }} group by {{ cf }}, {{ input.target }}
{% endfor %}\
;'''



tmplt["_m1d_mi"] ='''
select f, 
(sum( 
	(nxy*1.0/n__)
    *log(2,
		(nxy*1.0/n__)
        /( (n_y*1.0/n__) * (nx_*1.0/n__) )
        )  ) ) as mi
from 
(
SELECT f, x, y, nxy, nx_, n_y, n__ FROM {{ input.model_id }}_m1d
join (SELECT f, x,  sum(nxy) as nx_ FROM {{ input.model_id }}_m1d group by f, x) a using (f, x)
join (SELECT f, y, sum(nxy) as n_y FROM {{ input.model_id }}_m1d group by f, y) b using (f, y)
join ( select f, sum(nxy) as n__ FROM {{ input.model_id }}_m1d group by f ) c using (f) 
) as m
group by f 
order by mi desc
'''
