def copy_vector(src:list,dest:list,n):
    len_=0
    for i in range(n):
        dest[i] = src[i]
            
        len_+=1
    return len_

def to_ribbon(src, dest, in_,out):
      len_=0
      for row in range(out):
         for elem in range(in_):
             dest[row * in_ + elem] = src[row][elem]
             len_+=1
      return len_

def add_2_vecs_comps(l1, l2, n ):
    res = [0] * n
    for elem in range(n):
        res[elem] = l1[elem] + l2[elem]
        if res[elem] > 1:
            res[elem] = res[elem] 

    return res    
