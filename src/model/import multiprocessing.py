import multiprocessing

def ff(x):
    return x, x**2

if __name__ == "__main__":
    pool_obj = multiprocessing.Pool(3)
    ans = pool_obj.map(ff, [1,2,3])
    print(ans)