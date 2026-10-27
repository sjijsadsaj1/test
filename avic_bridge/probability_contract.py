"""Shared probability validation and exact-input binding, without model imports."""
import hashlib
import json
import math
import re

def finite(v):
    return type(v) in (int, float) and math.isfinite(v)

def probability(v):
    if not finite(v) or not 0 <= v <= 1:
        raise ValueError('invalid subjective probability')
    return float(v)

def decimal_probability(v):
    """Explicit wire representation, not permissive JSON/error repair."""
    if isinstance(v,str):
        if not re.fullmatch(r'(?:0(?:\.\d+)?|1(?:\.0+)?)',v):
            raise ValueError('invalid decimal probability string')
        v=float(v)
    return probability(v)

def binding(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()
