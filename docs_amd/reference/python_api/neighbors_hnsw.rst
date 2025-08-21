HNSW
====

This is a wrapper for hnswlib, to load a CAGRA index as an immutable HNSW index. The loaded HNSW index is only compatible in cuVS, and can be searched using wrapper functions.

.. role:: py(code)
   :language: python
   :class: highlight

Index search parameters
#######################

.. autoapiclass:: cuvs.neighbors.hnsw.hnsw.SearchParams
    :members:

Index
#####

.. autoapiclass:: cuvs.neighbors.hnsw.hnsw.Index
    :members:

Index Conversion
################

.. autoapifunction:: cuvs.neighbors.hnsw.hnsw.from_cagra

Index search
############

.. autoapifunction:: cuvs.neighbors.hnsw.hnsw.search

Index save
##########

.. autoapifunction:: cuvs.neighbors.hnsw.hnsw.save

Index load
##########

.. autoapifunction:: cuvs.neighbors.hnsw.hnsw.load
