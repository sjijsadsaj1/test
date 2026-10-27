"""Local detector proposals + bundled Qwen3-VL crop checks; no scoring.

Weights must already exist locally. The model is loaded lazily and can be
released before diffusion, so diagnostics do not need both models resident.
"""
import base64
import gc
import io
import json
import os
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps


class LocalSemanticClient:
    model='bundled-qwen3-vl-visible-space'
    def __init__(self,model_path=None,pipe=None):
        self.model_path=model_path or os.environ.get('AVIC_IMAGE_MODEL_DIR',
            str(Path(__file__).resolve().parents[1]/'.runtime/models/Qwen-Image-2.1-Turbo'))
        if not (Path(self.model_path)/'text_encoder/model.safetensors').is_file():
            raise ValueError('local_semantic_weights_missing')
        self.net=getattr(pipe,'text_encoder',None);self.processor=getattr(pipe,'processor',None)
        self.shared=pipe is not None
        self.last_attempts=0;self.last_usage={}
        self.total_requests=0
        self.proposal_diagnostic={}

    def _request(self,body,retries=0):
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        if self.net is None:
            self.processor=AutoProcessor.from_pretrained(str(Path(self.model_path)/'processor'),local_files_only=True)
            self.net=Qwen3VLForConditionalGeneration.from_pretrained(str(Path(self.model_path)/'text_encoder'),
                local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        self.last_attempts=1;self.last_usage={}
        content=body['messages'][0]['content']
        text=next(item['text'] for item in content if item['type']=='text')
        images=[]
        for item in content:
            if item['type']!='image_url':continue
            url=item['image_url']['url']
            if not url.startswith('data:image/jpeg;base64,'):raise ValueError('local_semantic_requires_inline_jpeg')
            images.append(Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))).convert('RGB'))
        messages=[dict(role='user',content=[dict(type='image') for _ in images]+[dict(type='text',text=text)])]
        prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[prompt],images=images,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            tokens=self.net.generate(**inputs,max_new_tokens=max(32,min(1200,int(body.get('max_tokens',1200)))),do_sample=False)
        tail=tokens[:,inputs['input_ids'].shape[1]:]
        answer=self.processor.batch_decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        self.last_usage=dict(prompt_tokens=int(inputs['input_ids'].shape[1]),completion_tokens=int(tail.shape[1]))
        start=answer.find('{')
        if start<0:raise ValueError('local_semantic_missing_json')
        result,_=json.JSONDecoder().raw_decode(answer[start:])
        self.last_response=result
        return result

    def propose_regions(self,real,image):
        from .local_interior import LocalInteriorLocator
        self.last_attempts=1;self.last_usage={};self.total_requests+=1
        detections=LocalInteriorLocator(real)._detect()
        w,h=image.size
        def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
        def iou(a,b):
            inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
            return inter/max(1,area(a)+area(b)-inter)
        unique=[];rejected=[]
        for d in sorted(detections,key=lambda d:d['confidence'],reverse=True):
            b=d['box']
            if b[2]-b[0]<48 or b[3]-b[1]<96 or area(b)>.65*w*h:
                rejected.append(dict(d,reason='proposal_size'));continue
            # Whole bottom strips misdetected as entrances do not bind one branch.
            if b[1]>=.55*h or (b[1]>=.5*h and b[2]-b[0]>3*(b[3]-b[1])):
                rejected.append(dict(d,reason='bottom_only_or_panoramic_strip_not_entrance'));continue
            if any(iou(b,u['box'])>.80 for u in unique):continue
            unique.append(d)
        regions=[]
        for d in unique:
            outer=d['box']
            # Bind the complete entrance first, including its frame. Never use
            # a nested detector object as the frontier's initial reference.
            inner=outer
            regions.append(dict(kind='opening_interior',opening_box_1000=[round(v/(w if i%2==0 else h)*1000) for i,v in enumerate(outer)],
                box_1000=[round(v/(w if i%2==0 else h)*1000) for i,v in enumerate(inner)],
                evidence=d['phrase'],detector_confidence=d['confidence']))
        self.last_attempts=1;self.last_usage={}
        self.proposal_diagnostic=dict(detections=detections,rejected=rejected,
            source='live_local_grounding_dino',scope='entrance proposals, NOT frontier segmentation')
        self.last_response=dict(regions=regions[:6])
        return self.last_response

    def refine_region(self,real,image,box):
        from .local_interior import opening_core
        from .interior_reference import refine_box
        attempts=[];usage={};calls=0
        for stage,parent in enumerate((opening_core(image,box),box)):
            candidates=[]
            a,b,c,d=parent;span=c-a
            # These are deterministic subregions of ONE entrance, not VLM-invented
            # coordinates. Side fragments can exclude an open door leaf.
            parts=[(a,c),(round(a+.4*span),c),(a,round(c-.4*span)),
                   (round(a+.2*span),round(c-.2*span))]
            # A second pass may rescue room clues lost by edge heuristics, but
            # never offer the whole original doorway again.
            for left,right in (parts if stage==0 else parts[1:]):
                try:
                    crop,_=refine_box(image,(left,b,right,d),
                        dict(width=image.width,height=image.height,column_px=(left+right)/2))
                except ValueError:continue
                if crop not in candidates:candidates.append(crop)
            if not candidates:continue
            result=self._choose_patch(image,candidates)
            calls+=self.last_attempts
            for key,value in self.last_usage.items():
                if isinstance(value,(int,float)):usage[key]=usage.get(key,0)+value
            attempts.append(dict(result,stage='structural_core' if stage==0 else 'side_fragment_fallback'))
            if result['usable']:break
        self.last_attempts=calls;self.last_usage=usage
        if not attempts:return dict(usable=False,reason='no_sufficient_context_fragment',candidates=[])
        return dict(result,attempts=attempts)

    def _choose_patch(self,image,candidates):
        # One local VLM choice among visible patches, instead of free coordinates
        # or accepting a doorway merely because a room is visible somewhere in it.
        sheet=Image.new('RGB',(240*4,300*((len(candidates)+3)//4)),'white')
        draw=ImageDraw.Draw(sheet)
        for i,crop in enumerate(candidates):
            tile=ImageOps.contain(image.crop(crop),(230,270))
            x=(i%4)*240;y=(i//4)*300
            draw.text((x+8,y+5),f'PATCH {i}',fill='black')
            sheet.paste(tile,(x+(240-tile.width)//2,y+25+(270-tile.height)//2))
        buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=95)
        prompt=('This contact sheet shows numbered REAL candidate crops from ONE detected entrance. '
            'Choose the ONE best reference fragment for imagining the INTERIOR of that room, NOT its entrance. '
            'Prefer a crop dominated by visible interior wall/surfaces AND room contents beyond the entrance. '
            'Avoid foreground door leaves, door frames, close furniture, fireplace close-ups and floor-only patches. '
            'An OPEN door leaf is still an occluder. If one side crop removes the door and retains room clues, '
            'choose that side crop rather than the wider doorway. Do not prefer a larger crop just for its size. '
            'A useful small fragment can omit the doorway completely. Use only contents actually visible IN THE PATCH. '
            'Bare flat texture/color alone is insufficient context. Do not borrow an object visible in another tile. '
            'If all patches are occluders, near-object closeups or lack interior context, choose null. '
            'Return JSON {"selected_id":integer patch number or null,"reason":"brief visible evidence"}. '
            'No coordinates, no hidden objects, no search target.')
        result=self._request(dict(messages=[dict(content=[dict(type='text',text=prompt),
            dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()))])]))
        if not isinstance(result,dict) or 'selected_id' not in result:
            raise ValueError('invalid_local_patch_choice_schema')
        index=result['selected_id']
        if index is not None and (type(index) is not int or not 0<=index<len(candidates)):
            raise ValueError('invalid_local_patch_choice')
        return dict(usable=index is not None,box=list(candidates[index]) if index is not None else None,
            content_type='room_context' if index is not None else 'occluder_or_insufficient',
            selected_id=index,reason=result.get('reason',''),candidates=[list(c) for c in candidates])

    def close(self):
        if self.shared:return
        self.net=None;self.processor=None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
