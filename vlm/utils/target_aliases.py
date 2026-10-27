"""Opt-in detector-only synonyms; never changes task goals or STOP criteria."""
import os

def detector_aliases(label):
    original=[s.strip() for s in label.split('|')]
    mode=os.environ.get('APEXNAV_TARGET_ALIASES','off')
    if mode=='drawer' and original==['nightstand']:
        return ['chest of drawers','dresser','nightstand']
    if mode=='picture' and original==['picture','framed photograph']:
        return ['picture','framed photograph','painting','framed artwork']
    return original

def box_iou(a,b):
    w=max(0.,min(float(a[2]),float(b[2]))-max(float(a[0]),float(b[0])))
    h=max(0.,min(float(a[3]),float(b[3]))-max(float(a[1]),float(b[1])))
    intersection=w*h
    area=lambda r:max(0.,float(r[2])-float(r[0]))*max(0.,float(r[3])-float(r[1]))
    return intersection/max(1e-12,area(a)+area(b)-intersection)
