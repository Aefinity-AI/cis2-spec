#!/usr/bin/env python3
"""E40 5.3: summarise a `layer_dump` file.

Counts records, distinct tensor names, f32 values, exact zeros and *stored
subnormals* in a dump written by `examples/layer_dump.rs`. Pure stdlib: the
record format is u32 name length, name bytes, u32 element count, then that many
little-endian u32 bit patterns (`tap::emit` writes `x.to_bits()`).

    python3 scripts/e40_dump_stats.py <dump.bin>
"""
import struct,sys,hashlib
from collections import Counter
p=sys.argv[1]
d=open(p,'rb').read()
i=0;n=0;nf=0;names=Counter()
subn=0;zeros=0;mn=None
while i<len(d):
    (ln,)=struct.unpack_from('<I',d,i); i+=4
    name=d[i:i+ln].decode(); i+=ln
    (k,)=struct.unpack_from('<I',d,i); i+=4
    vals=struct.unpack_from('<%dI'%k,d,i); i+=4*k
    n+=1; nf+=k; names[name]+=1
    for b in vals:
        e=(b>>23)&0xFF; m=b&0x7FFFFF
        if e==0:
            if m==0: zeros+=1
            else:
                subn+=1
                if mn is None or m<mn: mn=m
print(f"records={n} distinct-names={len(names)} floats={nf} exact-zeros={zeros} SUBNORMALS-STORED={subn} smallest-subnormal-mantissa={mn}")
