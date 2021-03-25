tmplt = {}

# def train_test_split():
tmplt["_train"] = '''
(select * from {{ input.dataset }} where rand ({{ input.seed }}) < {{ input.ratio }});
'''

tmplt["_table_eval"] = '''
(select * from {{ input.dataset }} where rand ({{ input.seed }}) >= {{ input.ratio }});
'''

# def train():
tmplt["_qt"] ='''
select y, 
{% for i in input.numFeatures %} xq{{ loop.index }},  min(xn{{ loop.index }}) as mn{{ loop.index }}, \
max(xn{{ loop.index }}) as mx{{ loop.index }}, {% endfor %}
{% for i in input.catFeatures %} xc{{ loop.index }}, {% endfor %}
count(*) as n
 from ( -- a
	select
{% for i in input.numFeatures %}\
CEIL({{ input.bins }} *RANK() OVER (ORDER BY {{ i }} )*1.0/COUNT(*) OVER()) as xq{{ loop.index }},\
{% endfor %}
{% for i in input.numFeatures %} {{ i }} as xn{{ loop.index }}, {% endfor %}
{% for i in input.catFeatures %} {{ i }} as xc{{ loop.index }}, {% endfor %}
{{ input.target }} as y
	from {{input.train}} )a
group by y
	{% for i in input.numFeatures %} ,xq{{ loop.index }}  {% endfor %}
    {% for i in input.catFeatures %} ,xc{{ loop.index }}  {% endfor %};
'''

#######

tmplt["_qmt"] = '''select 
a.* ,
min(mn) over(partition by i) as tmn, 
max(mx) over(partition by i) as tmx # num. attr. 1
from (
{% for i in input.numFeatures %} \
{% if not loop.first %} union {% endif %} \
select {{ loop.index }} as i, xq{{ loop.index }} as xq, min(mn{{ loop.index }}) as mn,  max(mx{{ loop.index }}) as mx \
from {{ input.model_id }}_qt group by xq{{ loop.index }} \
{% endfor %}
) a
;'''


#######

tmplt["_m"] = '''
select
	y as y_,
	{% for i in input.numFeatures %} \
	xq{{ loop.index }}, mn{{ loop.index }}, mx{{ loop.index }}, mx_{{ loop.index }},\
	tmn{{ loop.index }} ,tmx{{ loop.index }}, \
	{% endfor %}
    {% for i in input.catFeatures %} \
	xc{{ loop.index }}, \
	{% endfor %}
    n__, n_y,
	sum(n) as nxy
from (
	select
		y,
		{% for i in input.numFeatures %} \
		xq{{ loop.index }}, \
		x{{ loop.index }}.mn as mn{{ loop.index }}, \
		x{{ loop.index }}.mx as mx{{ loop.index }}, \
		x{{ loop.index }}.mx_ as mx_{{ loop.index }}, 
		min(mn{{ loop.index }}) OVER( ) as tmn{{ loop.index }},
		max(mx{{ loop.index }}) OVER( ) as tmx{{ loop.index }},
		{% endfor %}
        {% for i in input.catFeatures %} \
	    xc{{ loop.index }}, \
	    {% endfor %}
        sum(n) over() as n__,
		sum(n) over(partition by y) as n_y,
        n
	from {{ input.model_id }}_qt as t
	{% for i in input.numFeatures %} \
	join (select i, xq, mn, mx, LEAD(mn, 1) OVER (ORDER BY xq) as mx_ from {{ input.model_id }}_qmt 
	where i = {{ loop.index }}) x{{ loop.index }} on (xq{{ loop.index }} = x{{ loop.index }}.xq)
	{% endfor %}
    ) a
group by y,
{% for i in input.numFeatures %} \
	xq{{ loop.index }}, mn{{ loop.index }}, mx{{ loop.index }}, mx_{{ loop.index }}, tmn{{ loop.index }}, tmx{{ loop.index }}, \
	{% endfor %}
    {% for i in input.catFeatures %} \
    xc{{ loop.index }}, \
    {% endfor %}
    n__, n_y
    ;'''

# def predict():

tmplt["_qe"] = '''select t.*
{% for i in input.numFeatures %} ,x{{ loop.index }}.xq as xq{{ loop.index }}{% endfor %}
 	from (select row_number() over() as id, {{ input.target }} as y 
    -- qe_nf
    {% for i in input.numFeatures %} ,{{ i }} as xn{{ loop.index }} {% endfor %}
    {% for i in input.catFeatures %} ,{{ i }} as xc{{ loop.index }} {% endfor %}
 from {{ input.eval }}) -- table_eval
    as t
    {% for i in input.numFeatures %}join (select i, xq, mn, mx, tmn, tmx, \
    LEAD(mn, 1) OVER (ORDER BY xq) as mx_ from {{ input.model_id }}_qmt 
	    where i = {{ loop.index }}) x{{ loop.index }} 
		on ((x{{ loop.index }}.mx_ > xn{{ loop.index }} or x{{ loop.index }}.mx = x{{ loop.index }}.tmx) 
		and (x{{ loop.index }}.mn <= xn{{ loop.index }} or x{{ loop.index }}.mn = x{{ loop.index }}.tmn))
	{% endfor %}
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
		left outer join {{ input.model_id }}_m as m
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
(select  y from (
select y, sum(n) as n from  {{ input.model_id }}_qt group by y
) a 
where n =  (select max(n) from (select y, sum(n) as n from {{ input.model_id }}_qt group by y ) b ))
where y_ is null
;'''

tmplt["_m1d"] ='''
{% for nf in input.numFeatures %}
{% if loop.index > 1 %}union {% endif %}\
select f, x, y, count(*) over(partition by x, y) as nxy, count(*) over (partition by x) as nx_, count(*) over (partition by y) as n_y, count(*) over() as n__ 
from ( select
        "{{ nf }}" as f,
		CEIL({{ input.bins}}*RANK() OVER (ORDER BY {{ nf }} )*1.0/COUNT(*) OVER()) as x,
        {{ input.target }} as y
from table_train ) a 
{% endfor %}
{% for cf in input.catFeatures %}
{% if loop.index + input.numFeatures|length  > 1 %} union {% endif %}\
select f, x, y, count(*) over(partition by x, y) as nxy, count(*) over (partition by x) as nx_, count(*) over (partition by y) as n_y, count(*) over() as n__ 
from ( select
        "{{ cf }}" as f,
		{{ cf }} as x,
        {{ input.target }} as y
from table_train ) a 
{% endfor %}
;'''


tmplt["_m1d_mi"] ='''select f, 
sum(
	(nxy*1.0/n__)
    *log(2,
		(nxy*1.0/n__)
        /( (n_y*1.0/n__) * (nx_*1.0/n__) )
        ) ) as mi
from {{ input.model_id }}_m1d as m
group by f 
order by mi desc
;
'''
