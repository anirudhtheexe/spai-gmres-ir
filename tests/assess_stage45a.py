"""Independent Decimal80 Stage4.5A assessment. No solver or timing fallback.

Rounding envelopes are diagnostic-only; they never replace production stopping.
"""
from decimal import Decimal as D, getcontext
import argparse
import json
import math
import sys
import hashlib
from pathlib import Path

getcontext().prec = 80
U = D(2) ** -53
TINY = D.from_float(float.fromhex('0x0.0000000000001p-1022'))

def number(x):
    if x is None or not math.isfinite(float(x)):
        raise ValueError('Nonfinite or missing required numerical value')
    return D.from_float(float(x))

def gamma(k):
    z = D(k)*U
    if z >= 1:
        raise ValueError('Unsupported rounding-envelope operation length')
    return z/(1-z)

def norm(x):
    return sum((v*v for v in x), D(0)).sqrt()

def csr(item):
    offsets = item.get('row_offsets', item.get('offsets'))
    indices = item.get('column_indices', item.get('indices'))
    values = item['values']
    return [[(indices[j], number(values[j])) for j in range(offsets[i], offsets[i+1])]
            for i in range(len(offsets)-1)]

def product(a, x):
    return [sum((v*x[j] for j,v in row), D(0)) for row in a]

def product_error(a, x):
    return [max(64*U, gamma(2*len(row)+2))*sum((abs(v)*abs(x[j]) for j,v in row),D(0))
            + D(2*len(row)+2)*TINY for row in a]

def reduced_qr(b_columns, rows, columns, target):
    """Independent Decimal80 MGS QR, never normal equations."""
    m,q=len(rows),len(columns)
    basis=[]
    triangular=[[D(0)]*q for _ in range(q)]
    for j,column in enumerate(columns):
        vector=[b_columns[column].get(i,D(0)) for i in rows]
        for k in range(j):
            triangular[k][j]=sum((v*w for v,w in zip(basis[k],vector)),D(0))
            vector=[v-triangular[k][j]*w for v,w in zip(vector,basis[k])]
        triangular[j][j]=norm(vector)
        if not triangular[j][j]:
            raise ValueError('Exact Decimal reduced rank deficiency')
        basis.append([v/triangular[j][j] for v in vector])
    rhs=[sum((v for i,v in zip(rows,vector) if i==target),D(0)) for vector in basis]
    expected=[D(0)]*q
    for j in reversed(range(q)):
        expected[j]=(rhs[j]-sum((triangular[j][k]*expected[k] for k in range(j+1,q)),D(0)))/triangular[j][j]
    return expected

def fit_residual(b_columns, pattern, coefficients, target):
    residual={target:D(-1)}
    scales={target:D(1)}
    lengths={target:0}
    for column,coefficient in zip(pattern,coefficients):
        for row,value in b_columns[column].items():
            residual[row]=residual.get(row,D(0))+value*coefficient
            scales[row]=scales.get(row,D(0))+abs(value*coefficient)
            lengths[row]=lengths.get(row,0)+1
    errors={i:max(64*U,gamma(2*lengths[i]+3))*s+D(2*lengths[i]+3)*TINY for i,s in scales.items()}
    return residual,norm(list(residual.values())),norm(list(errors.values()))

def exact_recorded_preconditioned(a,p,b,x):
    """Exact stored binary64 input arithmetic plus separate rounding envelope."""
    ax=product(a,x)
    ax_error=product_error(a,x)
    pb=product(p,b)
    pax=product(p,ax)
    residual=[v-w for v,w in zip(pb,pax)]
    pb_error=product_error(p,b)
    pax_error=product_error(p,ax)
    propagated=product([[(j,abs(v)) for j,v in row] for row in p],ax_error)
    error=[be+ae+pe+U*(abs(v)+abs(w))+TINY for be,ae,pe,v,w in zip(pb_error,pax_error,propagated,pb,pax)]
    denominator=norm(pb)
    denominator_error=norm(pb_error)+max(64*U,gamma(3*len(pb)+3))*denominator
    numerator=norm(residual)
    numerator_error=norm(error)+max(64*U,gamma(3*len(residual)+3))*numerator
    if denominator==0:
        return D(0) if numerator==0 else D('Infinity'),D(0)
    relative=numerator/denominator
    # If denominator error is not small, report unbounded diagnostic uncertainty.
    allowance=(numerator_error+relative*denominator_error)/(denominator-denominator_error) if denominator>denominator_error else D('Infinity')
    return relative,allowance

def diagnostics(a, b, x):
    ax = product(a,x)
    residual = [bi-ai for bi,ai in zip(b,ax)]
    scales = [abs(bi)+sum((abs(v)*abs(x[j]) for j,v in row),D(0)) for bi,row in zip(b,a)]
    allowance = [max(64*U,gamma(2*len(row)+3))*s+D(2*len(row)+3)*TINY
                 for row,s in zip(a,scales)]
    rinf = max(map(abs,residual), default=D(0))
    ainf = max((sum((abs(v) for _,v in row),D(0)) for row in a),default=D(0))
    xinf,binf = max(map(abs,x),default=D(0)), max(map(abs,b),default=D(0))
    denominator = ainf*xinf+binf
    eta = rinf/denominator if denominator else (D(0) if rinf==0 else D('Infinity'))
    cbe = max((abs(r)/s if s else (D(0) if r==0 else D('Infinity'))
               for r,s in zip(residual,scales)),default=D(0))
    envelope = max(allowance,default=TINY)/denominator + max(64*U,gamma(max((len(row) for row in a),default=0)+3))*eta if denominator else TINY
    cbe_envelope = max((e/s if s else TINY for e,s in zip(allowance,scales)),default=TINY)+max(64*U,gamma(max((len(row) for row in a),default=0)+3))*cbe
    return dict(residual=residual, inf=rinf, norm2=norm(residual), nbe=eta, cbe=cbe,
                component_allowance=allowance, nbe_allowance=envelope, cbe_allowance=cbe_envelope)

class Audit:
    def __init__(self):
        self.checks=0
        self.failures=[]
    def check(self, condition, label):
        self.checks+=1
        if not condition:
            self.failures.append(label)
    def diagnostic(self,a,b,x,stored,accepted=False):
        expected=diagnostics(a,b,x)
        for r,s,e in zip(expected['residual'],stored.get('residual',stored.get('r')),expected['component_allowance']):
            self.check(abs(r-number(s))<=e,'original residual component envelope')
        self.check(abs(expected['inf']-number(stored['residual_inf']))<=max(expected['component_allowance'],default=TINY),'original residual infinity norm')
        self.check(abs(expected['norm2']-number(stored['residual_2']))<=norm(expected['component_allowance'])+max(64*U,gamma(3*len(b)+3))*expected['norm2'],'original residual two norm')
        self.check(abs(expected['nbe']-number(stored['normwise_backward_error']))<=expected['nbe_allowance'],'normwise BE diagnostic envelope')
        self.check(abs(expected['cbe']-number(stored['componentwise_backward_error']))<=expected['cbe_allowance'],'componentwise BE diagnostic envelope')
        if accepted:
            self.check(number(stored['normwise_backward_error'])<=10*U,'unchanged recorded normwise stopping')
            self.check(expected['nbe']<=10*U+expected['nbe_allowance'],'terminal Decimal BE within rounding envelope')
        return expected

def assess_construction(a, record, audit, controlled):
    result=record['construction']
    if len(result['scaling'])!=len(a):
        audit.check(result['status']!='success','incomplete scaling cannot succeed')
        return {'status':result['status'],'assessment':'no completed scaling/columns'}
    scales=[number(v) for v in result['scaling']]
    b_columns=[{j:number(float(v)*float(s)) for j,v in row} for row,s in zip(a,scales)]
    for row,s in zip(a,scales):
        audit.check(s==number(1/max(abs(float(v)) for _,v in row)),'row-max reciprocal scaling')
    b_rows=csr(result['B'])
    for i,row in enumerate(b_rows):
        audit.check(dict(row)=={j:column[i] for j,column in enumerate(b_columns) if i in column},'B=A transpose D sparse values/support')
    columns=result['columns']
    maximum_error=D(0)
    residuals=[]
    qr_checks=0
    qr_unchecked=0
    sampled={0,len(a)//2,len(a)-1}
    nonempty=[c for c in columns if c['history']]
    if nonempty:
        sampled.add(max(nonempty,key=lambda c:len(c['pattern']))['column'])
        sampled.add(max(nonempty,key=lambda c:c['full_residual'])['column'])
        sampled.add(min(nonempty,key=lambda c:c['pivot_ratio'])['column'])
    packed_p=[{} for _ in a]
    packed_n=[{} for _ in a]
    for c in columns:
        k=c['column']
        history=c['history']
        audit.check(c['fits']==len(history) and c['fits']<=record['options']['alpha']+1,'fit history/budget')
        if not history:
            audit.check(c['status']!='success','successful column has completed fit')
            continue
        audit.check(history[0]['J']==[k],'identity initial pattern')
        audit.check(c['pattern']==history[-1]['J'],'publish only solved pattern')
        for h,f in enumerate(history):
            j,i=f['J'],f['I']
            audit.check(j==sorted(set(j)) and i==sorted(set(i)),'sorted unique reduced sets')
            expected_shadow=sorted(set().union(*(set(b_columns[col]) for col in j)))
            audit.check(i==expected_shadow,'complete sparse shadow')
            coefficients=[number(v) for v in f['coefficients']]
            r,rnorm,envelope=fit_residual(b_columns,j,coefficients,k)
            audit.check(abs(rnorm-number(f['full_residual']))<=envelope,'Decimal full fit residual')
            audit.check(abs(rnorm-number(f['residual']))<=envelope,'Decimal reduced fit residual')
            if h==len(history)-1:
                residuals.append(rnorm)
                if c['status']=='success':
                    audit.check(number(f['residual'])<=number(record['options']['epsilon']),'unchanged recorded epsilon')
                    audit.check(rnorm<=number(record['options']['epsilon'])+envelope,'Decimal epsilon within rounding envelope')
            if f['candidates']:
                union_rows=set(i)|{k}
                expected_candidates=[col for col,values in enumerate(b_columns) if col not in j and union_rows.intersection(values)]
                audit.check(f['candidates']==expected_candidates,'restricted candidate support excluding J')
            if len(f['scores'])==len(f['candidates']) and f['candidates']:
                scores=f['scores']
                binary64_sum=0.0
                for score in scores:
                    binary64_sum+=float(score)
                audit.check(binary64_sum/len(scores)==f['mean_score'],'single full candidate mean in binary64 order')
                expected_selected=[candidate for score,candidate in sorted(zip(scores,f['candidates'])) if score<=f['mean_score']][:record['options']['beta']]
                audit.check(f['selected']==expected_selected,'score/mean/tie ordering')
                for candidate,score in zip(f['candidates'],scores):
                    v=[b_columns[candidate].get(row,D(0)) for row in i]
                    den=sum((z*z for z in v),D(0))
                    dot=sum((r.get(row,D(0))*z for row,z in zip(i,v)),D(0))
                    if den:
                        expected_squared=max(D(0),rnorm*rnorm-dot*dot/den)
                        audit.check(abs(number(score)**2-expected_squared)<=max(1536*U,gamma(4*len(i)+10))*max(D(1),rnorm*rnorm),'Decimal restricted score squared')
            if h+1<len(history):
                audit.check(history[h+1]['J']==sorted(set(j)|set(f['selected'])),'adaptive growth follows selected indices')
            if (controlled or k in sampled) and len(i)*len(j)*len(j)<=100000:
                try:
                    truth=reduced_qr(b_columns,i,j,k)
                    qr_checks+=1
                    for got,expected in zip(coefficients,truth):
                        error=abs(got-expected)/max(D(1),abs(expected))
                        maximum_error=max(maximum_error,error)
                        if controlled:
                            audit.check(error<=max(1536*U,gamma(16*len(i)*len(j)+64)),'controlled Decimal QR coefficient bound')
                except ValueError as e:
                    audit.check(c['status']!='success',str(e))
            else:
                qr_unchecked+=1
        for j,v in zip(c['pattern'],c['coefficients']):
            if float(v)!=0:
                packed_n[j][k]=number(v)
            p=float(v)*float(scales[j])
            if p!=0:
                packed_p[k][j]=number(p)
    if result['status']=='success':
        audit.check(all(c['status']=='success' for c in columns),'all columns successful before P use')
    for name,expected in [('N',packed_n),('P',packed_p)]:
        rows=csr(result[name])
        # A scaling/assembly failure may intentionally return empty outputs.
        if result['status']=='success' or any(rows):
            audit.check([dict(row) for row in rows]==expected,'canonical '+name+' orientation/rounded coefficient assembly')
    p_rows=csr(result['P'])
    pa_squared=D(0)
    for k,row in enumerate(p_rows):
        residual={k:D(-1)}
        for j,coefficient in row:
            for col,value in a[j]:
                residual[col]=residual.get(col,D(0))+coefficient*value
        pa_squared+=sum((value*value for value in residual.values()),D(0))
    return dict(status=result['status'],qr_fits=qr_checks,qr_unchecked_fits=qr_unchecked,
                maximum_scaled_QR_coefficient_difference=str(maximum_error),
                maximum_full_column_residual=str(max(residuals,default=D(0))),
                frobenius_BN_minus_I=str(norm(residuals)),
                frobenius_I_minus_PA=str(pa_squared.sqrt()),sampled_columns=sorted(sampled))

def assess_record(record):
    audit=Audit()
    summary={'classification':record.get('classification'),'substatus':record.get('substatus'),
             'phase':record.get('phase'),'backend':record.get('backend')}
    if 'A' not in record or record.get('classification')=='resource_abort':
        return dict(summary,validation_outcome='unvalidated',checks=0,failures=[])
    a=csr(record['A'])
    b=[number(v) for v in record['b']]
    reference=[number(v) for v in record.get('reference',[])]
    controlled=bool(reference)
    if reference:
        ref=diagnostics(a,b,reference)
        summary['manufactured_reference_original_residual_inf']=str(ref['inf'])
    if 'construction' in record:
        summary['construction_assessment']=assess_construction(a,record,audit,controlled)
    if 'solver' in record:
        solver=record['solver']
        p=csr(record.get('supplied_P',record.get('construction',{}).get('P')))
        states=solver['outer_history']
        for h,state in enumerate(states):
            x=[number(v) for v in state['x']]
            accepted=solver['converged'] and h==len(states)-1
            result=audit.diagnostic(a,b,x,state['diagnostics'],accepted)
            if h==len(states)-1:
                summary['terminal_Decimal_diagnostics']={key:str(result[key]) for key in ['inf','norm2','nbe','cbe','nbe_allowance']}
                summary['terminal_boundary_uncertain']=accepted and result['nbe']>10*U
                if reference:
                    denominator=max(map(abs,reference),default=D(0))
                    summary['manufactured_forward_error']=str(max((abs(u-v) for u,v in zip(x,reference)),default=D(0))/(denominator or D(1)))
        if states:
            initial=[float(v) for v in states[0]['x']]
            expected=[]
            for row in p:
                s=0.0
                for j,v in row:
                    s+=float(v)*float(b[j])
                expected.append(s)
            audit.check(initial==expected,'CPU outer x0=P*b binary64 order')
        applied=0
        for attempt in solver['inner_history']:
            inner=attempt['gmres']
            rhs=[number(v) for v in attempt['rhs']]
            if states:
                audit.check(attempt['rhs']==states[applied]['diagnostics']['r'],'correction RHS retains original recorded residual')
            for item in inner['history']:
                x=[number(v) for v in item['x']]
                relative,envelope=exact_recorded_preconditioned(a,p,rhs,x)
                if item['recomputed'] is not None:
                    audit.check(abs(relative-number(item['recomputed']))<=envelope,'inner independently recomputed preconditioned residual')
                if item['accepted']:
                    audit.check(item['verification_triggered'],'inner true residual verification triggered')
                    audit.check(number(item['recomputed'])<=number(record['options']['tau']),'unchanged recorded inner tau')
                    audit.check(relative<=number(record['options']['tau'])+envelope,'Decimal inner acceptance envelope')
                if item['breakdown']:
                    audit.check(number(item['arnoldi_subdiagonal'])==0,'exact computed Arnoldi breakdown')
            if inner['converged'] and inner['history']:
                audit.check(inner['history'][-1]['accepted'],'converged inner has accepted terminal candidate')
            if attempt['update_applied']:
                audit.check(inner['converged'],'only converged inner updates outer')
                if applied+1<len(states):
                    audit.check(states[applied+1]['x']==[float(u)+float(v) for u,v in zip(states[applied]['x'],inner['x'])],'accepted correction update binary64')
                applied+=1
        audit.check(applied==solver['outer_steps'],'applied updates equal outer count')
        audit.check(len(states)==applied+1 or not states,'outer states retain last accepted iterate')
        if solver['converged']:
            audit.check(solver['status']=='success' and bool(states),'IR successful status/history')
        if record.get('backend','').startswith('gpu') and states:
            audit.check(record.get('context_constructions')==1 and record.get('matrix_uploads')==2,'one persistent A/P context')
            for t in record.get('inner_transfers',[]):
                audit.check(t['arnoldi_vector_uploads']==0 and t['arnoldi_vector_downloads']==0,'inner Arnoldi vector residency')
    summary.update(checks=audit.checks,failures=audit.failures,validation_outcome='discrepancy' if audit.failures else ('pass' if audit.checks else 'unvalidated'))
    return summary

def assess_provenance(path):
    manifest_path=Path(path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    failed=False
    for item in manifest['records']:
        audit=Audit()
        root=manifest_path.parent
        matrix_path=root/item['relative_matrix_path']
        archive_path=root/item['relative_archive_path']
        for actual,key,size_key in [(matrix_path,'matrix_sha256','matrix_bytes'),(archive_path,'archive_sha256','archive_bytes')]:
            data=actual.read_bytes()
            audit.check(hashlib.sha256(data).hexdigest().upper()==item[key],'independent '+key)
            audit.check(len(data)==item[size_key],'independent '+size_key)
        lines=matrix_path.read_text(encoding='utf-8-sig').splitlines()
        audit.check(lines[0].strip()=='%%MatrixMarket matrix coordinate real general','raw admitted Matrix Market banner')
        records=[line for line in lines[1:] if line.strip() and not line.lstrip().startswith('%')]
        n,m,declared=map(int,records[0].split())
        audit.check((n,m,declared)==(item['rows'],item['columns'],item['declared_entries']),'independent declared dimensions/count')
        audit.check(len(records)-1==declared,'independent observed entry count')
        coordinates={}
        zeros=duplicates=0
        for line in records[1:]:
            i,j,v=line.split()
            i,j,v=int(i)-1,int(j)-1,float(v)
            audit.check(0<=i<n and 0<=j<m and math.isfinite(v),'raw coordinate bounds/finite')
            zeros+=v==0
            duplicates+=(i,j) in coordinates
            coordinates[i,j]=coordinates.get((i,j),0.0)+v
        rows=[[] for _ in range(n)]
        for (i,j),v in sorted(coordinates.items()):
            if v!=0:
                rows[i].append((j,v))
        nnz=sum(map(len,rows))
        audit.check(zeros==item['explicit_zeros'] and duplicates==item['duplicate_records'] and nnz==item['canonical_nnz'],'independent zero/duplicate/canonical counts')
        for i,row in enumerate(rows):
            diagonal=dict(row).get(i,0)
            audit.check(math.isfinite(diagonal) and diagonal!=0 and bool(row),'nonzero finite original diagonal/row')
            reciprocal=1/max(abs(v) for _,v in row)
            audit.check(math.isfinite(reciprocal) and reciprocal!=0,'finite nonzero reciprocal row scaling')
            audit.check(all(math.isfinite(v*reciprocal) and v*reciprocal!=0 for _,v in row),'nonzero scaling range retained')
        print(json.dumps({'name':item['name'],'rows':n,'raw_entries':declared,'explicit_zeros':zeros,'duplicates':duplicates,'canonical_nnz':nnz,'checks':audit.checks,'failures':audit.failures,'validation_outcome':'discrepancy' if audit.failures else 'static_preflight_pass_not_rank_certificate'}))
        failed=failed or bool(audit.failures)
    return 1 if failed else 0

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('records')
    parser.add_argument('--provenance',action='store_true')
    args=parser.parse_args()
    if args.provenance:
        return assess_provenance(args.records)
    failed=False
    with open(args.records,encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            if line.strip():
                record=json.loads(line)
                try:
                    summary=assess_record(record.get('result',record))
                except Exception as e:
                    summary={'validation_outcome':'unvalidated','assessment_error':str(e)}
                    failed=True
                summary['record_index']=index
                summary['matrix_path']=record.get('derived_provenance',{}).get('raw_path')
                summary['input_sha256']=hashlib.sha256(line.encode()).hexdigest()
                print(json.dumps(summary))
                failed=failed or summary['validation_outcome']=='discrepancy'
    return 1 if failed else 0

if __name__=='__main__':
    sys.exit(main())
