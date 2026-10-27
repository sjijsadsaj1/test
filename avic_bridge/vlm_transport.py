"""Standalone VLM transports: no image-generator modules, ROI policy or prompts."""
import base64
import io
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def model_image(image):
    """Preserve real pixels; neutral padding prevents unsupported thin inputs."""
    from PIL import Image
    image=image.convert('RGB').copy();image.thumbnail((640,480))
    w,h=image.size
    canvas=Image.new('RGB',(max(32,w),max(32,h)),(128,128,128))
    left=(canvas.width-w)//2;top=(canvas.height-h)//2
    canvas.paste(image,(left,top))
    return canvas,dict(content_box=[left,top,left+w,top+h],model_size=list(canvas.size),
                       padding=canvas.size!=image.size,padding_color=[128,128,128])


def image_part(image):
    image,_=model_image(image)
    buf=io.BytesIO();image.save(buf,format='JPEG',quality=90)
    return dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()))


class VLMResponseError(ValueError):
    """A produced response is malformed; distinct from model/input failures."""


def parse_response(text):
    start=text.find('{')
    if start<0:raise VLMResponseError('missing JSON')
    try:return json.JSONDecoder().raw_decode(text[start:])[0]
    except ValueError as exc:raise VLMResponseError('invalid JSON response') from exc


class VLMAPIError(RuntimeError):
    def __init__(self,kind,status=None,code=None):
        self.diagnostic=dict(kind=kind,http_status=status,provider_code=str(code)[:32] if code else None)
        super().__init__(str(self.diagnostic))


class GLMClient:
    def __init__(self):
        self.key=os.environ.get('AVIC_GLM_API_KEY','')
        if not self.key:raise ValueError('GLM API key not configured')
        self.base=os.environ.get('AVIC_GLM_BASE_URL','https://open.bigmodel.cn/api/paas/v4').rstrip('/')
        self.model=os.environ.get('AVIC_GLM_MODEL','glm-4.6v-flash')
        self.last_answer='';self.last_usage={}

    def request(self,content,max_tokens=1200):
        self.last_answer='';self.last_usage={}
        body=dict(model=self.model,messages=[dict(role='user',content=content)],temperature=0,
            thinking=dict(type='disabled'),max_tokens=max_tokens,response_format=dict(type='json_object'))
        req=Request(self.base+'/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        try:
            with urlopen(req,timeout=60) as response:result=json.loads(response.read())
        except HTTPError as exc:
            code=None
            try:code=json.loads(exc.read(8192)).get('error',{}).get('code')
            except (ValueError,TypeError,AttributeError):pass
            raise VLMAPIError('HTTPError',exc.code,code) from None
        except (URLError,TimeoutError) as exc:raise VLMAPIError(type(exc).__name__) from None
        choice=result['choices'][0]
        self.last_answer=choice['message']['content']
        self.last_usage={k:v for k,v in result.get('usage',{}).items()
                         if k in ('prompt_tokens','completion_tokens','total_tokens')}
        if choice.get('finish_reason')=='length':raise VLMResponseError('VLM response truncated')
        return parse_response(self.last_answer)


class LocalVLMClient:
    model='local-qwen3-vl-dictionary'
    def __init__(self):
        # Reuse existing VLM weights, never construct QwenImagePipeline.
        self.root=Path(os.environ.get('AVIC_TEXT_MODEL_DIR',str(Path(__file__).resolve().parents[1]/'.runtime/models/Qwen-Image-2.1-Turbo')))
        self.net=None;self.processor=None;self.last_answer='';self.last_usage={}

    def request(self,content,max_tokens=1200):
        self.last_answer='';self.last_usage={}
        import torch
        from PIL import Image
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        if self.net is None:
            self.processor=AutoProcessor.from_pretrained(self.root/'processor',local_files_only=True)
            self.net=Qwen3VLForConditionalGeneration.from_pretrained(self.root/'text_encoder',
                local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        parts=[];images=[]
        for item in content:
            if item['type']=='text':parts.append(item)
            elif item['type']=='image_url':
                url=item['image_url']['url']
                if not url.startswith('data:image/'):raise ValueError('inline real image required')
                images.append(Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))).convert('RGB'))
                parts.append(dict(type='image'))
            else:raise ValueError('unsupported content type')
        prompt=self.processor.apply_chat_template([dict(role='user',content=parts)],tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[prompt],images=images or None,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():tokens=self.net.generate(**inputs,max_new_tokens=max_tokens,do_sample=False)
        tail=tokens[:,inputs['input_ids'].shape[1]:]
        self.last_answer=self.processor.batch_decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        self.last_usage=dict(prompt_tokens=int(inputs['input_ids'].shape[1]),completion_tokens=int(tail.shape[1]))
        if tail.shape[1]>=max_tokens:raise VLMResponseError('VLM response truncated')
        return parse_response(self.last_answer)


def make_client(provider):
    if provider=='local':return LocalVLMClient()
    if provider=='glm':return GLMClient()
    raise ValueError('provider must be local or glm; no implicit fallback')
