"""Target-to-room exploration hints, NOT measured occurrence probabilities.

Never emits detections or success; uses current REAL RGB only. The table and
weights are engineering hypotheses to be tested, not dataset-derived labels.
"""
import json
import math

ROOMS = {
    'sink': ['kitchen', 'bathroom'],
    'toilet': ['bathroom'], 'shower': ['bathroom'], 'bathtub': ['bathroom'],
    'towel': ['bathroom', 'bedroom'],
    'bed': ['bedroom'], 'pillow': ['bedroom', 'living room'], 'cushion': ['bedroom', 'living room'],
    'nightstand': ['bedroom'], 'clothes': ['bedroom', 'laundry room'],
    'couch': ['living room'], 'sofa': ['living room'], 'tv': ['living room', 'bedroom'],
    'table': ['dining room', 'living room', 'office'],
    'counter': ['kitchen', 'bathroom'], 'cabinet': ['kitchen', 'bedroom', 'bathroom'],
    'chest of drawers': ['bedroom'], 'gym equipment': ['gym'],
    'fireplace': ['living room'],
}


def room_candidates(label):
    # Detector aliases are separated by '|'; use the canonical first label.
    return ROOMS.get(label.split('|')[0].strip().lower().replace('_', ' '), [])


def score_with_rooms(rgb, label, base, cosine):
    rooms = room_candidates(label)
    if not rooms: return base
    scores = [float(cosine(rgb, f'Seems like there is a {room} ahead?')) for room in rooms]
    if not math.isfinite(base) or not all(math.isfinite(v) for v in scores):
        raise ValueError('Non-finite room prior score')
    # Same BLIP cosine scale; modest, signed and bounded shift, no fake calibration.
    context=max(scores)
    delta=max(-0.04,min(0.04,0.30*(context-base)))
    result=base+delta
    print('[ROOM_PRIOR] '+json.dumps(dict(label=label, rooms=rooms, room_scores=scores,
          base=base, adjusted=result, delta=delta, calibrated_probability=False)), flush=True)
    return result
