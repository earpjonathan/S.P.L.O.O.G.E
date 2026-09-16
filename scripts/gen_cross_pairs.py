import glob, os, sys, itertools
STRIDE = int(sys.argv[1]) if len(sys.argv)>1 else 8
imgs = sorted(os.path.basename(p) for p in glob.glob('frames/site/*.jpg'))
a=[x for x in imgs if x.startswith('0026_')]
b=[x for x in imgs if x.startswith('0027_')]
sa=a[::STRIDE]; sb=b[::STRIDE]
pairs=[(x,y) for x in sa for y in sb]
# also sparse intra-clip long-range links (loop closure within a clip's orbit)
for s in (sa,sb):
    pairs += [(x,y) for x,y in itertools.combinations(s,2)]
with open('work/cross_pairs.txt','w') as f:
    for x,y in pairs: f.write(f'{x} {y}\n')
print(f'0026: {len(a)} imgs -> {len(sa)} sampled')
print(f'0027: {len(b)} imgs -> {len(sb)} sampled')
print(f'cross-clip pairs : {len(sa)*len(sb):,}')
print(f'intra-clip loop  : {len(pairs)-len(sa)*len(sb):,}')
print(f'TOTAL pairs      : {len(pairs):,}  -> work/cross_pairs.txt')
