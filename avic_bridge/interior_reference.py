"""Target-independent directional ROI selection for room imagination.

This is an image-space heuristic, not a measured frontier segmentation or future pose.
"""
import math

PROMPT = (
    'This reference reveals only a small fragment of an indoor space. Imagine what else '
    'might exist in that same room and create a rich, complete interior photograph. '
    'Infer new unseen contents from the visual clues rather than only extending bare walls '
    'or enlarging the fragment. Retain recognizable existing objects and the character '
    'of the space, but freely recompose the view to reveal its interior. '
    'Choose the unseen contents yourself from the image; a plausible guess is enough.'
)


def direction_hint(plan, camera, size):
    anchor = plan['frontier_anchor']; frontier = plan['source_frontier']
    hfov = camera.get('hfov_deg')
    if not isinstance(hfov, (int, float)) or not math.isfinite(hfov) or not 1 < hfov < 179:
        return dict(eligible=False, reason='missing_actual_rgb_hfov')
    pitch = float(camera['pitch_rad'])
    if abs(pitch) > math.radians(15):
        return dict(eligible=False, reason='pitched_view_projection_unsupported')
    dx = frontier['x']-anchor['robot_x']; dy = frontier['y']-anchor['robot_y']
    if math.hypot(dx, dy) < .05:
        return dict(eligible=False, reason='frontier_too_close_for_bearing')
    yaw = math.atan2(math.sin(math.atan2(dy,dx)-anchor['robot_yaw_rad']),
                     math.cos(math.atan2(dy,dx)-anchor['robot_yaw_rad']))
    half = math.radians(hfov)/2
    if abs(yaw) >= half:
        return dict(eligible=False, reason='frontier_outside_current_horizontal_fov', relative_yaw_rad=yaw)
    width,height = size
    x = width/2 - width/(2*math.tan(half))*math.tan(yaw)/math.cos(pitch)
    if not 0 <= x < width:
        return dict(eligible=False, reason='projected_bearing_outside_image')
    return dict(eligible=True, reason='direction_only_not_visibility_proof', column_px=x,
                relative_yaw_rad=yaw, hfov_deg=hfov, width=width, height=height,
                geometric_scope='horizontal bearing in current RGB; no occlusion or traversability proof')


def validate_box(reply, hint):
    if not isinstance(reply,dict) or reply.get('visible') is not True:
        raise ValueError('no_visible_interior_patch')
    box = reply.get('box')
    if not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box):
        raise ValueError('invalid_box_format')
    x1,y1,x2,y2 = box; w,h = hint['width'],hint['height']
    if not (0<=x1<x2<=w and 0<=y1<y2<=h and x2-x1>=24 and y2-y1>=32):
        raise ValueError('invalid_box_bounds_or_size')
    # A small tolerance permits excluding a frame exactly on the projected column.
    if not x1-.05*w <= hint['column_px'] <= x2+.05*w:
        raise ValueError('box_not_in_candidate_direction')
    if (x2-x1)*(y2-y1) > .65*w*h:
        raise ValueError('box_not_local')
    return tuple(box)


def inset_box(box):
    """Use the core of a VLM opening box; heuristic, not door-frame segmentation."""
    x1,y1,x2,y2=box
    mx=min(round((x2-x1)*.25),max(0,(x2-x1-24)//2))
    my=min(round((y2-y1)*.05),max(0,(y2-y1-32)//2))
    return (x1+mx,y1+my,x2-mx,y2-my)


def crop_quality(image):
    """Black pixels are an input-quality warning, NOT a semantic visibility test."""
    import numpy as np
    pixels = np.asarray(image.convert('RGB'))
    return dict(near_black_fraction=float((pixels.max(axis=2) <= 8).mean()),
                width=image.width, height=image.height)


def refine_box(image, detected_box, hint):
    """Largest usable rectangle INSIDE the proposed opening, never another branch.

    Long vertical edges near the proposal sides supply possible frame boundaries;
    otherwise use just a two-pixel inset. Then find a non-black rectangle. This
    cannot identify doors or distinguish dark surfaces from missing renderer pixels.
    """
    import numpy as np
    x1,y1,x2,y2 = detected_box
    quality = dict(policy='content_roi_v2', detected_quality=crop_quality(image.crop(detected_box)),
                   min_width=48,min_height=96,min_area_fraction=.015,max_near_black_fraction=.10,
                   boundary_inset_px=2,scope='RGB quality refinement, NOT semantic segmentation')
    pixels=np.asarray(image.convert('RGB')).astype(np.int16)
    patch=pixels[y1:y2,x1:x2]
    edge_support=(np.max(np.abs(patch[:,1:]-patch[:,:-1]),axis=2)>18).mean(axis=0)
    edges=np.flatnonzero(edge_support>=.35)+x1+1
    left_edges=edges[edges<=x1+.30*(x2-x1)]
    right_edges=edges[edges>=x2-.30*(x2-x1)]
    left=int(left_edges[-1]) if len(left_edges) else x1
    right=int(right_edges[0]) if len(right_edges) else x2
    quality.update(possible_frame_edges=[left if len(left_edges) else None,right if len(right_edges) else None],
                   edge_policy='RGB jump >18 across >=35% rows, outer30% sides only; NOT door segmentation')
    x1,x2=left,right
    x1+=2; y1+=2; x2-=2; y2-=2
    if x2-x1<48 or y2-y1<96:
        raise ValueError('roi_too_small_for_context')
    # Include both exact boundaries. At most 17 positions per dimension.
    xs=np.unique(np.linspace(x1,x2,17).round().astype(int))
    ys=np.unique(np.linspace(y1,y2,17).round().astype(int))
    xa,xb=np.meshgrid(xs,xs,indexing='ij'); ya,yb=np.meshgrid(ys,ys,indexing='ij')
    xvalid=(xb-xa>=48)&(xa-.05*image.width<=hint['column_px'])&(xb+.05*image.width>=hint['column_px'])
    yvalid=yb-ya>=96
    xa,xb=xa[xvalid][:,None],xb[xvalid][:,None]
    ya,yb=ya[yvalid][None,:],yb[yvalid][None,:]
    if not xa.size or not ya.size:raise ValueError('roi_no_directional_context')
    black=(pixels.max(axis=2)<=8).astype(np.int64)
    integral=np.pad(black,((1,0),(1,0))).cumsum(0).cumsum(1)
    area=(xb-xa)*(yb-ya)
    fractions=(integral[yb,xb]-integral[ya,xb]-integral[yb,xa]+integral[ya,xa])/area
    valid=(area>=.015*image.width*image.height)&(fractions<=.10)
    if not valid.any():raise ValueError('roi_no_sufficient_nonblack_context')
    # Area first, black fraction only breaks ties. No target-conditioned rule.
    utility=np.where(valid,area-fractions,-1.)
    i,j=np.unravel_index(utility.argmax(),utility.shape)
    box=tuple(int(v) for v in (xa[i,0],ya[0,j],xb[i,0],yb[0,j]))
    quality.update(final_quality=crop_quality(image.crop(box)),
                   area_fraction=float(area[i,j]/(image.width*image.height)))
    return box,quality
