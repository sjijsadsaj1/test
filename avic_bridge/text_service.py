"""Dedicated text-belief service. Never constructs an image generation pipeline."""
import os
from pathlib import Path
import threading
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .protocol import SCHEMA_VERSION

if os.environ.get('APEXNAV_TEXT_METHOD')=='dictionary':
    from .frontier_dictionary import CONTRACT, DictionaryFrontierModel as TextFrontierModel
else:
    from .text_frontier import CONTRACT, TextFrontierModel

app=FastAPI(title='ApexNav text frontier belief')
model=TextFrontierModel(os.environ.get('AVIC_TEXT_PROVIDER','local'))
output=Path(os.environ['AVIC_TEXT_OUTPUT_ROOT'])
lock=threading.Lock()

@app.get('/health')
def health():
    return dict(ok=True,backend='text_frontier',contract=CONTRACT,
                provider=model.provider,model=model.client.model,generated_images=0,
                output_root=str(output),alignment_audit_contract=dict(schema_version='alignmentaudit_v1',enabled=True))

@app.post('/v1/review')
def review(payload: dict):
    from .target_review import review_request
    rid=payload.get('request_id','')
    if not isinstance(rid,str) or not rid.isalnum() or len(rid)>64:
        return JSONResponse(dict(error='invalid review ID'),status_code=400)
    with lock:
        return review_request(model.client,payload,output.parent/'target_reviews')

@app.post('/v1/decide')
def decide(payload: dict):
    start=time.monotonic()
    with lock:
        rid=payload.get('request_id','')
        if not isinstance(rid,str) or not rid.isalnum() or len(rid)>64:
            return JSONResponse(dict(error='invalid request ID'),status_code=400)
        try:
            result=model.predict(payload,output/rid)
        except Exception as exc:
            # Do not expose provider credentials/errors; failed trials abort,
            # rather than silently behaving like a baseline.
            detail=getattr(exc,'diagnostic',dict(kind=type(exc).__name__,message=str(exc)))
            print('[TEXT_ERROR] '+str(detail),flush=True)
            return JSONResponse(dict(error=detail),status_code=500)
    return dict(result,schema_version=SCHEMA_VERSION,backend='text_frontier',
                decision='call_wm',selected_action=None,selected_plan_id=None,
                candidate_scores=[],latency_ms=(time.monotonic()-start)*1000)

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ['AVIC_SERVICE_PORT']),workers=1)
