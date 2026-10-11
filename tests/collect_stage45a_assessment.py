"""Reassess retained untimed raw records without rerunning any numerical method."""
import argparse
import hashlib
import json
from pathlib import Path
from assess_stage45a import assess_record

def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    results=[]
    failed=False
    for path in sorted(Path(args.directory).glob('*.jsonl')):
        wrapper=json.loads(path.read_text())
        try:
            result=assess_record(wrapper.get('result',wrapper))
        except Exception as error:
            result={'validation_outcome':'unvalidated','assessment_error':str(error)}
        result.update(raw_record=str(path),raw_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      resource_classification=wrapper.get('classification'),
                      matrix_sha256=wrapper.get('derived_provenance',{}).get('raw_sha256'))
        results.append(result)
        failed=failed or result['validation_outcome']=='discrepancy' or 'assessment_error' in result
        print(path.name,result['validation_outcome'],result.get('classification'),result.get('substatus'),result.get('checks',0),len(result.get('failures',[])),flush=True)
    with Path(args.output).open('x',encoding='utf-8') as stream:
        json.dump({'results':results,'total_checks':sum(r.get('checks',0) for r in results),
                   'total_discrepancies':sum(len(r.get('failures',[])) for r in results)},stream,indent=2)
    return 1 if failed else 0

if __name__=='__main__':
    raise SystemExit(main())
