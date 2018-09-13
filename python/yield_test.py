def test(k):
    for i in range(len(k)//2):
        yield(i,(i,k[i*2:i*2+2]))
        print('in',i)

k = list(range(10))

for n,i in enumerate(test(k)):
    print(i)
    print('out',n)
