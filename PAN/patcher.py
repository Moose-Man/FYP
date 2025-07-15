import pathlib, re
fp = pathlib.Path('models/pan_model.py')
txt = fp.read_text()
# np.float  →  float   |  np.int  →  int   |  np.bool  →  bool
txt = re.sub(r'dtype=np\.float', 'dtype=float', txt)
txt = re.sub(r'dtype=np\.int',   'dtype=int',   txt)
txt = re.sub(r'dtype=np\.bool',  'dtype=bool',  txt)
fp.write_text(txt)
print('✓ patched np.float / np.int / np.bool in pan_model.py')

