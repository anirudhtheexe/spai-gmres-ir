"""Independent native CPU/GPU comparison, retaining first decision witness."""
import argparse
from decimal import Decimal as D, getcontext
from fractions import Fraction as F
import json
from pathlib import Path

getcontext().prec=80
def load(path):
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    return record.get('result',record)

def singleton_score_witness(construction,column,fit):
    if len(fit['J'])!=1:
        return {'oracle':'not singleton; Decimal QR follow-up required'}
    b=construction['B']
    rows=[{b['indices'][j]:F.from_float(b['values'][j]) for j in range(b['offsets'][i],b['offsets'][i+1])} for i in range(b['rows'])]
    j=fit['J'][0]
    v=[rows[i].get(j,F(0)) for i in fit['I']]
    coefficient=rows[column].get(j,F(0))/sum((x*x for x in v),F(0))
    residual=[x*coefficient-(i==column) for i,x in zip(fit['I'],v)]
    rsq=sum((x*x for x in residual),F(0))
    grouped={}
    candidates=[]
    for candidate,stored in zip(fit['candidates'],fit['scores']):
        values=[rows[i].get(candidate,F(0)) for i in fit['I']]
        denominator=sum((x*x for x in values),F(0))
        dot=sum((x*y for x,y in zip(values,residual)),F(0))
        exact=rsq-dot*dot/denominator
        score=(D(exact.numerator)/D(exact.denominator)).sqrt()
        grouped.setdefault(exact,[]).append(candidate)
        candidates.append({'candidate':candidate,'exact_arithmetic_score_Decimal80':str(score),'recorded_score':stored,'score_difference':str(D.from_float(stored)-score)})
    return {'oracle':'exact Fraction singleton least-squares and squared restricted scores; Decimal80 sqrt only',
            'coefficient_Decimal80':str(D(coefficient.numerator)/D(coefficient.denominator)),
            'exact_squared_score_tie_groups':[group for group in grouped.values() if len(group)>1],
            'candidate_scores':candidates}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('cpu')
    p.add_argument('gpu')
    p.add_argument('--output')
    args=p.parse_args()
    c,g=load(args.cpu),load(args.gpu)
    report={'cpu_status':c['substatus'],'gpu_status':g['substatus'],'same_original_A_b':c.get('A')==g.get('A') and c.get('b')==g.get('b')}
    if 'construction' in c and 'construction' in g:
        cc,gc=c['construction'],g['construction']
        report.update(cpu_P_nnz=cc['P']['nnz'],gpu_P_nnz=gc['P']['nnz'])
        different=[]
        order_witness=None
        set_witness=None
        for k,(x,y) in enumerate(zip(cc['columns'],gc['columns'])):
            if x['pattern']!=y['pattern'] or x['status']!=y['status']:
                different.append(k)
            for h,(a,b) in enumerate(zip(x['history'],y['history'])):
                if order_witness is None and a['selected']!=b['selected']:
                    order_witness={'column':k,'fit':h,'cpu_selected':a['selected'],'gpu_selected':b['selected']}
                if set_witness is None and (set(a['selected'])!=set(b['selected']) or a['J']!=b['J']):
                    set_witness={'column':k,'fit':h,'cpu_fit':a,'gpu_fit':b,
                                 'cpu_exact_score_witness':singleton_score_witness(cc,k,a),
                                 'gpu_exact_score_witness':singleton_score_witness(gc,k,b)}
        report.update(different_final_pattern_or_status_columns=different,first_selected_order_difference=order_witness,first_selected_set_or_pattern_difference=set_witness)
    if 'solver' in c and 'solver' in g:
        cs,gs=c['solver'],g['solver']
        report.update(cpu_outer_steps=cs['outer_steps'],gpu_outer_steps=gs['outer_steps'],
                      cpu_inner_iterations=[x['gmres']['iterations'] for x in cs['inner_history']],
                      gpu_inner_iterations=[x['gmres']['iterations'] for x in gs['inner_history']],
                      solution_infinity_difference=max((abs(x-y) for x,y in zip(cs['x'],gs['x'])),default=0))
    if args.output:
        with Path(args.output).open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2)
    print(json.dumps(report))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
