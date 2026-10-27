"""Isolated paper-inspired probes. No simulator ground truth enters prompts."""
import base64
import io
import json
from pathlib import Path
from PIL import Image
from .image_glm_backend import GLMReviewClient


def image_part(value):
    image = Image.open(value) if isinstance(value, (str, Path)) else value
    image = image.convert('RGB')
    image.thumbnail((640, 480))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    return dict(type='image_url', image_url=dict(
        url='data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()))


class LiteratureVLM:
    def __init__(self, backend='glm'):
        self.client = LocalProbeClient() if backend == 'local' else GLMReviewClient()

    def ask(self, content):
        self.last_reply = None
        self.last_reply = self.client._request(dict(model=self.client.model,
            messages=[dict(role='user', content=content)], thinking=dict(type='disabled'),
            temperature=0, max_tokens=600, response_format=dict(type='json_object')))
        return self.last_reply

    def verify_target(self, target, images):
        prompt = ('Check ONLY the actual photographs for the requested target category: ' + target + '. '
            'These are real views, not imagined images. Do not infer presence from room type alone. '
            'A category may have multiple valid instances. For each image return visible=yes/no/uncertain, '
            'a short visual reason, and box_1000=[left,top,right,bottom] when visible=yes. '
            'Use integers 0..1000. Return JSON {"views":[{"index":0,"visible":"yes",'
            '"reason":"...","box_1000":[0,0,1000,1000]}]}. Do not decide navigation success.')
        content = [dict(type='text', text=prompt)]
        for i, img in enumerate(images):
            content.extend([dict(type='text', text=f'Real view index {i}'), image_part(img)])
        result = self.ask(content)
        rows = result.get('views', [])
        if len(rows) != len(images) or {x.get('index') for x in rows} != set(range(len(images))):
            raise ValueError('invalid view identities')
        for row in rows:
            if row.get('visible') not in ('yes', 'no', 'uncertain'):
                raise ValueError('invalid visibility')
            if row['visible'] == 'yes':
                box = row.get('box_1000')
                if (not isinstance(box, list) or len(box) != 4 or
                    any(type(x) is not int or not 0 <= x <= 1000 for x in box) or
                    not (box[0] < box[2] and box[1] < box[3])):
                    raise ValueError('invalid target box')
        return result


    def choose(self, audit, candidates, history=()):
        prompt = ('Choose ONE exploration branch to search for target ' + audit['target']['category'] + '. '
            'The current photograph and history are real observations. Future images are uncertain '
            'room hypotheses, not observations or geometrically accurate camera endpoints. '
            'Use the branch direction and actual visual context; imagined target presence is not proof. '
            'Avoid repeating already inspected space when real history supports that conclusion; '
            'revisiting may still be necessary. Do not choose by list position. '
            'Return JSON {"choice_id":"...","reason":"short evidence-based reason",'
            '"uncertain":true/false}. Use exactly one supplied candidate ID.')
        content = [dict(type='text', text=prompt)]
        for old in history:
            content.extend([dict(type='text', text='Earlier REAL observation; '+json.dumps(old['anchor_pose'])),
                            image_part(old['anchor_path'])])
        content.extend([dict(type='text', text='Current REAL observation; '+json.dumps(audit['anchor_pose'])),
                        image_part(audit['anchor_path'])])
        for c in candidates:
            f = c['source_frontier']
            metadata = dict(choice_id=c['plan_id'], x=f['x'], y=f['y'],
                relative_yaw_rad=f['relative_yaw_rad'], path_length=f['path_length'],
                actions=c['actions'])
            content.extend([dict(type='text', text='Candidate '+json.dumps(metadata)),
                            image_part(c['image_path'])])
        result = self.ask(content)
        if result.get('choice_id') not in {c['plan_id'] for c in candidates}:
            raise ValueError('invalid candidate identity')
        return result


class LocalProbeClient:
    """Reuse existing Qwen3-VL weights; retain ALL interleaved image/ID metadata."""
    model = 'local-qwen3-vl-pilot'

    def __init__(self):
        from .local_semantic_client import LocalSemanticClient
        self.source = LocalSemanticClient()
        self.last_usage = {}

    def _request(self, body):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        source = self.source
        if source.net is None:
            source.processor = AutoProcessor.from_pretrained(str(Path(source.model_path)/'processor'), local_files_only=True)
            source.net = Qwen3VLForConditionalGeneration.from_pretrained(
                str(Path(source.model_path)/'text_encoder'), local_files_only=True,
                torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
        content, images = [], []
        for item in body['messages'][0]['content']:
            if item['type'] == 'text':
                content.append(item)
            elif item['type'] == 'image_url':
                encoded = item['image_url']['url'].split(',', 1)[1]
                images.append(Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB'))
                content.append(dict(type='image'))
        prompt = source.processor.apply_chat_template([dict(role='user', content=content)],
            tokenize=False, add_generation_prompt=True)
        inputs = source.processor(text=[prompt], images=images or None, return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            tokens = source.net.generate(**inputs, max_new_tokens=body['max_tokens'], do_sample=False)
        tail = tokens[:, inputs['input_ids'].shape[1]:]
        answer = source.processor.batch_decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self.last_answer = answer
        self.last_usage = dict(prompt_tokens=int(inputs['input_ids'].shape[1]), completion_tokens=int(tail.shape[1]))
        start = answer.find('{')
        if start < 0:
            raise ValueError('local response contains no JSON')
        result, _ = json.JSONDecoder().raw_decode(answer[start:])
        return result
