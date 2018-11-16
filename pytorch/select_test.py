import torch




class temp():
    src = torch.rand(1,6,)
    cache = {
            "a" : torch.rand(5,6),
            "b":{"c":torch.rand(5,6)}
            } 
    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 1)
        if self.cache is not None:
            _recursive_map(self.cache)
    def __str__(self):
        print("src:",self.src)
        print("cache:",self.cache['a'])
        print("cache:",self.cache['b']['c'])

        return ""


qq = temp()
select_indices = torch.tensor([1,2,3])

print(qq)
qq.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))


print(qq)