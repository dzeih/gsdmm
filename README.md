# GSDMM: Short text clustering

Fork of [rwalk's awesome GSDMM implementation](https://github.com/rwalk/gsdmm), 
following model of [Yin and Wang 2014](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf) for the 
clustering of short text documents. This fork mainly reduces for loops from the original implementation to get some easy victories 
with larger document sets, ignoring possible memory requirement explosion.

As this uses arrays over lists of dictionaries, it can cut execution time as much as
 - 97 % with K=300, 26k docs, doc word counts ranging from 4 to 20
 - With K=300, 2.6M docs, didn't even bother comparing. This runs in ~4 minutes

Take it with a grain of salt, the above "benchmarks" were done with an old i5, ignoring all rules of a good benchmark, such as identical init and similar document distribution.
This may also gain nothing or be even slower than the original with small amounts of clusters and documents. 

## Usage
To use a Movie Group Process to cluster short texts, first initialize a [MovieGroupProcess](gsdmm/mgp.py):
```python
from gsdmm import MovieGroupProcess
mgp = MovieGroupProcess(K=8, alpha=0.1, beta=0.1, n_iters=30)
```
It's important to always choose `K` to be larger than the number of clusters you expect exist in your data, as the algorithm
can never return more than `K` clusters.

To fit the model:
```python
y = mgp.fit(docs, vocab_size)
```
where `docs` is list of documents split into lists of tokens, and `vocab_size` the amount of unique tokens over the set.
