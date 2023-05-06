#!/usr/bin/env python

"""

Script that visualizes the factorization of integers from 1 to N as an example dataset 
to perform dimensionality reduction on.

Adapted from:
    https://gist.github.com/johnhw/dfc7b8b8519aac530ac97da226c17bd5
    
"""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import expi
import scipy.sparse
import umap
from tqdm import tqdm

from primes import factorization, primesbelow, smallprimes

prime_ix = {p:i for i,p in enumerate(smallprimes)}


## Create sparse binary factor vectors for any number, and assemble into a matrix
## One column for each unique prime factor
## One row for each number, 0=does not have this factor, 1=does have this factor (might be repeated)

def factor_vector_lil(n):
    ## approximate prime counting function (upper bound for the values we are interested in)
    ## gives us the number of rows (dimension of our space)
    d = int(np.ceil(expi(np.log(n))))    
    x = scipy.sparse.lil_matrix((n,d))
    for i in tqdm(range(2,n)): 
        for k,v in factorization(i).items():            
            x[i,prime_ix[k]] = 1
                    
    return x


def main():
    
    n = 1_000_000
    cachepath = Path(f'pts-cache-{n:.0e}.npz')
    
    if not cachepath.is_file():
        # Generate the matrix for 1 million integers
        X = factor_vector_lil(n) 
    
        # embed with UMAP
        embedding = umap.UMAP(metric='cosine', n_epochs=500).fit_transform(X)

        # save for later
        np.savez(cachepath, embedding=embedding)
    else:
        loaded = np.load(cachepath)
        embedding = loaded['embedding']
    
    print(embedding.shape)

    # and save the image
    # s = 0.005
    s = 1
    fig = plt.figure(figsize=(8,8))
    fig.patch.set_facecolor('black')
    plt.scatter(embedding[:,0], embedding[:,1], marker='o', s=s, edgecolor='none',
                c=np.arange(n), cmap="magma")

    plt.axis("off")
    plt.savefig(f"primes_umap_{n:.0e}_pts.png", facecolor='black')


if __name__ == "__main__":
    main()


