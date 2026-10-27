"""Explicit opt-in scoring contract; verifier diagnostics do not gate it."""
import math
import base64
import io
CONTRACT='relaxed_blend_v1'

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def q25(scores):
    if len(scores)<2 or any(not finite(x) or not -1<=x<=1 for x in scores):
        raise ValueError('relaxed blend requires at least two valid frame ITM scores')
    values=sorted(scores);position=.25*(len(values)-1);lo=int(position);hi=math.ceil(position)
    return values[lo]+(position-lo)*(values[hi]-values[lo])

def blend(real,future,real_weight,predicted_weight):
    if not all(finite(x) for x in [real,future,real_weight,predicted_weight]) or not -1<=future<=1:
        raise ValueError('invalid relaxed blend numbers')
    if real_weight<0 or predicted_weight<0 or real_weight+predicted_weight<=0:raise ValueError('invalid relaxed blend weights')
    if real==future:return float(real)
    return (real_weight*real+predicted_weight*future)/(real_weight+predicted_weight)

def score_prediction(policy,real,scores,frames):
    if policy.get('mode')!='grounded_delta' or policy.get('contract')!=CONTRACT or policy.get('relaxed_imagination') is not True:
        raise ValueError('relaxed blend request contract mismatch')
    if len(frames)!=len(scores):
        raise ValueError('relaxed blend requires actual generated frames')
    from PIL import Image
    for frame in frames:
        try:
            raw=base64.b64decode(frame.get('jpeg_b64',''),validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format!='JPEG':raise ValueError('generated frame is not JPEG')
                image.verify()
        except Exception as exc:raise ValueError('relaxed blend requires valid generated JPEG frames') from exc
    future=q25(scores)
    final=blend(real,future,policy.get('real_weight',.25),policy.get('imagination_weight',.75))
    return dict(score_mode='grounded_delta',relaxed_imagination=True,relaxed_contract=CONTRACT,
        future_itm_q25=future,decision_semantic_score=final,imagination_bonus=final-real,
        score_reason='valid generated frame Q25 blend; verifier diagnostics not vetoes')
