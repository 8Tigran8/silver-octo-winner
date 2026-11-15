"""Observed-only PFF geometry and a separately constructed event-sequence proxy.

No event/possession field is read by the geometry conversion. References are
provider-event proxies, not manually corrected frame-by-frame possession truth.
"""
from pathlib import Path
import bz2
import collections
import hashlib
import json
from itertools import groupby
import numpy as np
from skillcorner_reader import upsample_observations


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_geometry(tracking,metadata,rosters):
    """Independent period/clock/geometry only; no reference labels in return."""
    meta=json.loads(Path(metadata).read_text())
    if isinstance(meta,list):
        if len(meta)!=1:raise ValueError('One match metadata record required')
        meta=meta[0]
    fps=float(meta['fps']);pitch=np.array([meta['stadium']['pitches'][0]['length'],meta['stadium']['pitches'][0]['width']],float)
    if not 20<=fps<=60 or np.any(pitch<=0):raise ValueError('Invalid source clock/pitch')
    home=str(meta['homeTeam']['id']);away=str(meta['awayTeam']['id'])
    roster=json.loads(Path(rosters).read_text());slots=set()
    for row in roster:
        team=str(row['team']['id']);shirt=row.get('shirtNumber')
        if team not in [home,away]:raise ValueError('Unknown roster team')
        if shirt is not None:slots.add((0 if team==home else 1,int(shirt)))
    slots=sorted(slots);lookup={s:i for i,s in enumerate(slots)}
    first=None;last=None;count=0;clock_error=0.;period_bounds={}
    with bz2.open(tracking,'rt') as stream:
        for line in stream:
            row=json.loads(line);frame=int(row['frameNum']);period=row.get('period')
            if first is None:first=frame
            if last is not None and frame<last:raise ValueError('Non-monotonic source frame order')
            last=frame;count+=1
            clock_error=max(clock_error,abs(float(row['videoTimeMs'])/1000-frame/fps))
            if period not in [None,0,1,2,3,4]:raise ValueError('Unknown source period')
            if period:
                bounds=period_bounds.setdefault(int(period),[frame,frame]);bounds[1]=frame
    if first is None or last is None or clock_error>.002:raise ValueError('Missing frames or source clock disagreement')
    n=last-first+1
    ball=np.full((n,3),np.nan,np.float32);players=np.full((n,len(slots),3),np.nan,np.float32);period=np.zeros(n,np.int8)
    for p,(a,b) in period_bounds.items():
        if np.any(period[a-first:b-first+1]):raise ValueError('Overlapping independent period bounds')
        # Missing tracking rows within a period remain graded missing inputs,
        # rather than being silently excluded as if they were halftime.
        period[a-first:b-first+1]=p
    occupied=np.zeros(n,bool);bad=np.zeros(n,bool);duplicates=0;conflicts=0
    with bz2.open(tracking,'rt') as stream:
        for line in stream:
            row=json.loads(line);index=int(row['frameNum'])-first
            b=np.full(3,np.nan,np.float32);p=np.full((len(slots),3),np.nan,np.float32)
            observed=[v for v in (row.get('balls') or []) if v.get('visibility')=='VISIBLE' and v.get('x') is not None and v.get('y') is not None]
            if len(observed)==1:
                b[:]=[observed[0].get(k) if observed[0].get(k) is not None else np.nan for k in ['x','y','z']]
            used=set()
            for side,key in [(0,'homePlayers'),(1,'awayPlayers')]:
                for value in row.get(key) or []:
                    if value.get('visibility')!='VISIBLE':continue
                    shirt=value.get('jerseyNum')
                    if shirt is None:continue
                    slot=(side,int(shirt))
                    if slot not in lookup:raise ValueError('Observed jersey absent from independent roster')
                    j=lookup[slot]
                    if slot in used:p[j]=np.nan;continue
                    used.add(slot)
                    if value.get('x') is not None and value.get('y') is not None:
                        p[j]=[value['x'],value['y'],value.get('speed') if value.get('speed') is not None else np.nan]
            current=int(row.get('period') or period[index])
            if occupied[index]:
                duplicates+=1
                if not bad[index] and (current!=period[index] or not np.array_equal(b,ball[index],equal_nan=True) or not np.array_equal(p,players[index],equal_nan=True)):
                    # No majority vote or annotation-based choice of coordinates.
                    bad[index]=True;conflicts+=1;ball[index]=np.nan;players[index]=np.nan
                continue
            occupied[index]=True;ball[index]=b;players[index]=p
    times=first/fps+np.arange(int(np.floor((n-1)*25/fps))+1)/25
    positions=np.arange(len(times))*fps/25
    left=np.floor(positions).astype(int);right=np.minimum(left+1,n-1);fraction=positions-left
    periods=period[left].copy();periods[(period[left]!=period[right])&(fraction>1e-8)]=0
    ball25=upsample_observations(ball,positions);players25=upsample_observations(players,positions)
    ball25[:,:2]=ball25[:,:2]/pitch+.5;players25[:,:,:2]=players25[:,:,:2]/pitch+.5
    ball25[periods==0]=np.nan;players25[periods==0]=np.nan
    data={'ball':ball25,'players':players25,'slot_team':np.array([s[0] for s in slots],np.int8),'pitch':pitch,'period':periods,'time':times}
    info={'fps':fps,'source_rows':count,'unique_source_frames':int(occupied.sum()),'duplicate_rows':duplicates,'conflicting_geometry_frames':conflicts,
          'source_clock_max_error_seconds':clock_error,'independent_source_period_bounds':period_bounds,'frames25':len(times),
          'visible_ball_fraction_in_periods':float(np.isfinite(ball25[periods>0,0]).mean()),
          'source_sha256':{'tracking':sha(tracking),'metadata':sha(metadata),'rosters':sha(rosters)}}
    return data,info


def sequence_owners(rows,mode='deliberate_then_first'):
    """Resolve ownership from provider sequences, never from tracked positions."""
    groups=collections.defaultdict(list);stops=[]
    for row in rows:
        game=row['gameEvents'];kind=game['gameEventType'];start=row.get('startTime')
        if start is None:continue
        if kind=='OUT':stops.append((float(start),int(game['period'])))
        # FOUL/SUB/OFF/ON do not themselves establish a ball-state change.
        if kind not in ['OTB','FIRSTKICKOFF','SECONDKICKOFF']:continue
        if row.get('sequence') is None:continue
        groups[int(row['sequence'])].append(row)
    sequences=[];diagnostics={'sequences':len(groups),'ambiguous_sequences':0,'clear_owner_conflicts':0,'clear_first_disagreements':0}
    for sequence,items in groups.items():
        items=sorted(items,key=lambda r:(r['startTime'],r['eventTime']))
        periods={r['gameEvents']['period'] for r in items}
        if len(periods)!=1:raise ValueError('Possession sequence crosses an independent period')
        valid=[r for r in items if isinstance(r['gameEvents'].get('homeTeam'),bool) and r['gameEvents'].get('initialNonEvent') is False]
        first=valid[0]['gameEvents']['homeTeam'] if valid else None
        clear=[r for r in valid if r['possessionEvents'].get('nonEvent') is False and r['possessionEvents'].get('possessionEventType') in ['PA','CR','SH']]
        owners={r['gameEvents']['homeTeam'] for r in clear}
        start=float(items[0]['startTime']);owner=first
        if mode=='deliberate_then_first' and owners:
            if len(owners)>1:
                owner=None;diagnostics['clear_owner_conflicts']+=1
            else:
                owner=next(iter(owners))
                if first is not None and owner!=first:
                    # An ambiguous opening touch is X until this owner's first
                    # annotated action, rather than assigning control backward.
                    diagnostics['clear_first_disagreements']+=1
                    owned=[r for r in valid if r['gameEvents']['homeTeam']==owner]
                    start=float(owned[0]['startTime'])
        elif mode not in ['deliberate_then_first','first_control']:
            raise ValueError('Unknown frozen reference policy')
        label=3 if owner is None else 0 if owner else 1
        diagnostics['ambiguous_sequences']+=label==3
        sequences.append({'sequence':sequence,'source_start':float(items[0]['startTime']),'label_start':start,
                          'period':int(next(iter(periods))),'label':label})
    return sorted(sequences,key=lambda r:r['source_start']),stops,diagnostics


def reference(events,times,period,mode='deliberate_then_first'):
    """H/A possession-sequence proxy, explicit OUT->D, ambiguous source->X."""
    rows=json.loads(Path(events).read_text())
    sequences,stops,info=sequence_owners(rows,mode)
    gt=np.full(len(times),3,np.int8);origin=float(times[0]);actions=[]
    for row in sequences:
        actions.append((row['source_start'],row['period'],3))
        actions.append((row['label_start'],row['period'],row['label']))
    actions += [(t,p,2) for t,p in stops]
    conflicts=0
    for p in sorted(set(period)-{0}):
        indexes=np.flatnonzero(period==p)
        edges=np.diff(np.r_[0,(period==p).astype(np.int8),0]);starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        if len(starts)!=1:raise ValueError('Unexpected split tracking period')
        a,b=int(starts[0]),int(ends[0]);at=collections.defaultdict(list)
        for time,per,label in actions:
            if per!=p:continue
            i=int(np.rint((time-origin)*25))
            if a-1<=i<b:at[max(a,i)].append(label)
        state=3;previous=a
        for i,values in sorted(at.items()):
            gt[previous:i]=state
            # A sequence boundary has an intentional X followed by its owner;
            # incompatible known states at one frame instead stay unknown.
            known={v for v in values if v!=3}
            if len(known)>1:state=3;conflicts+=1
            elif known:state=next(iter(known))
            else:state=3
            previous=i
        gt[previous:b]=state
    valid=period>0;counts={label:int(np.sum(gt[valid]==code)) for code,label in enumerate(['H','A','D','X'])}
    info.update(reference_mode=mode,quantization='nearest25Hzframe from independent source clock',explicit_OUT_markers=len(stops),
                contradictory_same_frame_states=conflicts,period_frames=int(valid.sum()),class_frames=counts,
                known_fraction=float(np.mean(gt[valid]!=3)),events_sha256=sha(events),
                scope='Event-sequence-derived full-task proxy; not corrected frame-by-frame possession truth')
    return gt,info


def tracking_reference(tracking,times,period,fps):
    """Primary kloppy PFF ball-state/owning-team semantics, kept out of features.

    The canonical result additionally resets unknown state at a new period and
    masks contradictory reference updates at a duplicate source frame. An exact
    last-row state-machine shadow is returned for sensitivity diagnostics.
    """
    alive_types={'FIRSTKICKOFF','SECONDKICKOFF','THIRDKICKOFF','FOURTHKICKOFF','OTB'}
    dead_types={'OUT','END'}
    sdk_alive=False;sdk_home=None;strict_alive=None;strict_home=None;last_period=0
    frames=[];source_periods=[];canonical=[];sdk=[];conflicts=0;duplicates=0;types=collections.Counter()
    with bz2.open(tracking,'rt') as stream:
        source=(json.loads(line) for line in stream)
        for frame,group in groupby(source,key=lambda r:int(r['frameNum'])):
            rows=list(group);duplicates+=len(rows)-1
            per={int(r.get('period') or 0) for r in rows}
            if len(per)!=1:raise ValueError('Conflicting source period at one frame')
            p=next(iter(per))
            if p and p!=last_period:
                strict_alive=None;strict_home=None;last_period=p
            states=set();owners=set()
            for row in rows:
                event=row.get('game_event')
                if not event:continue
                kind=event.get('game_event_type');types[kind]+=1
                if kind in dead_types:states.add(False);sdk_alive=False
                elif kind in alive_types:states.add(True);sdk_alive=True
                owner=event.get('home_ball')
                if owner is not None:
                    if not isinstance(owner,bool):raise ValueError('Non-boolean source owning-team flag')
                    owners.add(owner);sdk_home=owner
            if len(states)==1:strict_alive=next(iter(states))
            if len(owners)==1:strict_home=next(iter(owners))
            ambiguous=len(states)>1 or len(owners)>1
            if ambiguous:conflicts+=1
            def label(alive,home):
                if not p:return 3
                if alive is False:return 2
                if alive is True and home is not None:return 0 if home else 1
                return 3
            canonical.append(3 if ambiguous else label(strict_alive,strict_home))
            sdk.append(label(sdk_alive,sdk_home));frames.append(frame);source_periods.append(p)
            if ambiguous:
                # Mask the conflicting frame; subsequent state follows the
                # documented primary loader's row order, never a majority vote.
                strict_alive=sdk_alive;strict_home=sdk_home
    frames=np.asarray(frames,np.int64);source_periods=np.asarray(source_periods,np.int8)
    if np.any(np.diff(frames)<=0):raise ValueError('Non-monotonic unique source frames')
    target=np.floor(times*fps+1e-7).astype(np.int64)
    ix=np.searchsorted(frames,target,side='right')-1
    safe=np.maximum(ix,0);valid=(ix>=0)&(period>0)&(source_periods[safe]==period)
    gt=np.full(len(times),3,np.int8);alternate=gt.copy()
    gt[valid]=np.asarray(canonical,np.int8)[safe[valid]]
    alternate[valid]=np.asarray(sdk,np.int8)[safe[valid]]
    mask=period>0
    info={'policy':'Pinned PySport/kloppy PFF event-state machine; conservative duplicate-conflict mask and independent period resets',
          'reference_type':'Provider event-derived full H/A/D proxy, not corrected frame-by-frame control truth',
          'source_reference_event_types':dict(types),'duplicate_source_rows':duplicates,'conflicting_reference_frames':conflicts,
          'known_fraction':float(np.mean(gt[mask]!=3)),'sdk_shadow_known_fraction':float(np.mean(alternate[mask]!=3)),
          'class_frames':{s:int(np.sum(gt[mask]==k)) for k,s in enumerate(['H','A','D','X'])},
          'period_frames':int(mask.sum()),'sdk_shadow_disagreement_fraction':float(np.mean(gt[mask]!=alternate[mask])),
          'tracking_sha256':sha(tracking),'clock_sampling':'previous source frame at each independent25Hz timestamp; no quality-based alignment'}
    return gt,alternate,info
