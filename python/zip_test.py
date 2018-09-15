
def test(k):
    for i in range(5):
        yield(i,(i,k[i*2:i*2+2]))

v = list(range(10))
non_none = dict(test(v))
print(non_none)

keys, values = zip(*((k, [v_chunk for v_chunk in v_split]) for k, (_, v_split) in non_none.items()))

print(keys)
print(values)

for shard_tensors in zip(*values):
    print(keys,shard_tensors)