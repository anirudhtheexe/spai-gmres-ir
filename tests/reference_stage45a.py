"""Independent sparse Decimal80/120 elimination reference; no solver fallback."""
import argparse
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
from assess_stage45a import csr,diagnostics,number

def solve(a,b,precision):
    with localcontext() as context:
        context.prec=precision
        rows=[dict(row) for row in a]
        rhs=b[:]
        n=len(rows)
        for k in range(n):
            pivot=max(range(k,n),key=lambda i:abs(rows[i].get(k,D(0))))
            if not rows[pivot].get(k,D(0)):
                raise ValueError('Reference found zero pivot; no alternate solve fallback')
            rows[k],rows[pivot]=rows[pivot],rows[k]
            rhs[k],rhs[pivot]=rhs[pivot],rhs[k]
            diagonal=rows[k][k]
            entries=[(j,v) for j,v in rows[k].items() if j>k]
            for i in range(k+1,n):
                entry=rows[i].pop(k,D(0))
                if not entry:
                    continue
                factor=entry/diagonal
                rhs[i]-=factor*rhs[k]
                for j,v in entries:
                    value=rows[i].get(j,D(0))-factor*v
                    if value:
                        rows[i][j]=value
                    else:
                        rows[i].pop(j,None)
        x=[D(0)]*n
        for i in reversed(range(n)):
            x[i]=(rhs[i]-sum((v*x[j] for j,v in rows[i].items() if j>i),D(0)))/rows[i][i]
        return x

def main():
    p=argparse.ArgumentParser()
    p.add_argument('cpu_record')
    p.add_argument('gpu_record')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    cpu=json.loads(Path(args.cpu_record).read_text())['result']
    gpu=json.loads(Path(args.gpu_record).read_text())['result']
    a=csr(cpu['A'])
    b=[number(v) for v in cpu['b']]
    assert cpu['A']==gpu['A'] and cpu['b']==gpu['b']
    x80,x120=solve(a,b,80),solve(a,b,120)
    with localcontext() as context:
        context.prec=120
        denominator=max(map(abs,x120),default=D(0)) or D(1)
        consistency=max((abs(v-w) for v,w in zip(x80,x120)),default=D(0))/denominator
        quality=diagnostics(a,b,x120)
        report={'precision_digits':[80,120],'reference_method':'sparse Gaussian elimination with partial row pivoting, independently at both precisions',
                'n':len(a),'reference_precision_relative_difference':str(consistency),
                'reference_nbe':str(quality['nbe']),'reference_residual_inf':str(quality['inf']),
                'precision_consistency_pass':consistency<=D('1e-50'),
                'input_records':[args.cpu_record,args.gpu_record],
                'forward_errors':{}}
        for backend,result in [('cpu',cpu),('gpu',gpu)]:
            if result['solver']['converged']:
                x=[number(v) for v in result['solver']['x']]
                error=max((abs(v-w) for v,w in zip(x,x120)),default=D(0))/denominator
                report['forward_errors'][backend]=str(error)
        with Path(args.output).open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2)
        print(json.dumps(report))
        return 0 if report['precision_consistency_pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
