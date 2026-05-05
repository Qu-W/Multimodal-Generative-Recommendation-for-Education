import time, sys
print('Testing BGE-M3 load...')
t0 = time.time()
from sentence_transformers import SentenceTransformer
print(f'ST imported in {time.time()-t0:.1f}s')
t1 = time.time()
m = SentenceTransformer('E:/3-Models/bge-m3')
print(f'Model loaded in {time.time()-t1:.1f}s')
t2 = time.time()
v = m.encode(['test sentence'], normalize_embeddings=True)
print(f'Encode in {time.time()-t2:.1f}s, shape={v.shape}')
print('BGE-M3 OK')
