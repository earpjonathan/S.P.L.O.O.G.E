#!/usr/bin/env python
"""Cross-clip + intra-clip long-range pair list, generalised to N clips.
COLMAP has no vocab tree here, and sequential_matcher only links neighbours
within a clip, so without this the clips never join into one model."""
import glob, os, sys, itertools, argparse
ap=argparse.ArgumentParser()
ap.add_argument('setname'); ap.add_argument('clips',nargs='+')
ap.add_argument('--stride',type=int,default=8)
ap.add_argument('--out',default='work/cross_pairs.txt')
a=ap.parse_args()
imgs=sorted(os.path.basename(p) for p in glob.glob(f'frames/{a.setname}/*.jpg'))
groups={c:[x for x in imgs if x.startswith(c+'_')] for c in a.clips}
samp={c:g[::a.stride] for c,g in groups.items()}
pairs=[]
for c1,c2 in itertools.combinations(a.clips,2):
    pairs+=[(x,y) for x in samp[c1] for y in samp[c2]]
ncross=len(pairs)
for c in a.clips:
    pairs+=[(x,y) for x,y in itertools.combinations(samp[c],2)]
with open(a.out,'w') as f:
    for x,y in pairs: f.write(f'{x} {y}\n')
for c in a.clips: print(f'  {c}: {len(groups[c]):>5} imgs -> {len(samp[c]):>4} sampled')
print(f'  cross-clip pairs : {ncross:,}')
print(f'  intra-clip loop  : {len(pairs)-ncross:,}')
print(f'  TOTAL            : {len(pairs):,}  -> {a.out}')
