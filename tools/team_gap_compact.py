"""Smaller-receptive-field ownership model initialized from frozen parent blocks."""
from mtah.mixed_temporal import MixedTemporal
SPEC=dict(name='ownership25',dilations=[1,2,4,8,16],context_steps=64)
def from_parent(parent):
    compact=MixedTemporal(SPEC);source=parent.state_dict();mapping={}
    for k in compact.state_dict():
        # Copy input and first five residuals exactly; remap the old final head.
        old='net.9.'+k[len('net.7.'):] if k.startswith('net.7.') else k
        mapping[k]=source[old].clone()
    compact.load_state_dict(mapping);return compact
