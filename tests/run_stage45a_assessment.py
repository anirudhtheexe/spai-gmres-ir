"""Sequential untimed pilot execution and independent assessment; no timers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from assess_stage45a import assess_record

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/stage45a_provenance_20261011/MANIFEST.json')
    p.add_argument('--runner',default='build/validation-cuda/untimed_screen.exe')
    p.add_argument('--output-dir',default='results/stage45a_validation')
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--backends',nargs='+',default=['cpu_native','gpu_native'])
    p.add_argument('--fixed-p-from',choices=['cpu_native','gpu_native'])
    p.add_argument('--label',default='')
    p.add_argument('--outercap',default='20')
    p.add_argument('--innercap')
    p.add_argument('--rhs',default='equal_unit2')
    args=p.parse_args()
    manifest_path=Path(args.manifest)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    entries=manifest.get('records',manifest.get('cases',[]))
    lookup={x.get('name',x.get('id')):x for x in entries}
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['PATH']=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64'+os.pathsep+env.get('PATH','')
    env['CUBLAS_EMULATE_DOUBLE_PRECISION']='0'
    for name in args.cases:
        item=lookup[name]
        n=item.get('rows',item.get('n'))
        matrix_relative=item.get('relative_matrix_path',item.get('raw_path'))
        provenance_relative=item.get('provenance_path')
        provenance_path=(manifest_path.parent/provenance_relative if provenance_relative else manifest_path)
        for backend in args.backends:
            prefix=name+('_'+args.label if args.label else '')
            output=root/(prefix+'_'+backend+'.jsonl')
            assessment=root/(prefix+'_'+backend+'_assessment.json')
            if output.exists() or assessment.exists():
                raise RuntimeError('Refuse overwriting retained artifacts: '+str(output))
            command=[sys.executable,'scripts/untimed_screen.py','--runner',args.runner,
                     '--matrix',str(manifest_path.parent/matrix_relative),
                     '--provenance',str(provenance_path),'--backend',backend,
                     '--rhs',args.rhs,'--alpha',str((n+7)//8),'--innercap',args.innercap or str(n),'--outercap',args.outercap,'--output',str(output)]
            if args.fixed_p_from:
                source=json.loads((root/(name+'_'+args.fixed_p_from+'.jsonl')).read_text(encoding='utf-8'))['result']
                if source['construction']['status']!='success':
                    raise RuntimeError('Refuse using partial failed preconditioner')
                matrix=source['construction']['P']
                p_path=root/(name+'_'+args.fixed_p_from+'_P.mtx')
                if not p_path.exists():
                    with p_path.open('x',encoding='utf-8',newline='\n') as stream:
                        stream.write('%%MatrixMarket matrix coordinate real general\n% Validation-only canonical supplied P from retained successful construction\n')
                        stream.write(f"{matrix['rows']} {matrix['cols']} {matrix['nnz']}\n")
                        for i in range(matrix['rows']):
                            for j in range(matrix['offsets'][i],matrix['offsets'][i+1]):
                                stream.write(f"{i+1} {matrix['indices'][j]+1} {matrix['values'][j]:.17g}\n")
                command+=['--P',str(p_path)]
            print('START '+name+' '+backend,flush=True)
            child=subprocess.run(command,capture_output=True,text=True,env=env)
            if child.returncode:
                raise RuntimeError('Infrastructure process failed; retain output: '+child.stderr)
            record=json.loads(output.read_text(encoding='utf-8'))
            try:
                summary=assess_record(record.get('result',record))
            except Exception as error:
                summary={'validation_outcome':'unvalidated','assessment_error':str(error)}
            summary.update(name=name,backend=backend,raw_record=str(output),command=command)
            with assessment.open('x',encoding='utf-8') as stream:
                json.dump(summary,stream,indent=2)
            print(json.dumps(summary),flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
