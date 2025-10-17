CAGRA
=====

CAGRA is a graph-based nearest neighbors algorithm that was built from the ground up for GPU acceleration. CAGRA demonstrates state-of-the art index build and query performance for both small- and large-batch sized search.

.. role:: py(code)
   :language: python
   :class: highlight

Index build parameters
######################

.. autoapiclass:: cuvs.neighbors.cagra.IndexParams
    :members:

Index search parameters
#######################

.. autoapiclass:: cuvs.neighbors.cagra.SearchParams
    :members:

Index
#####

.. autoapiclass:: cuvs.neighbors.cagra.Index
    :members:

Index build
###########

.. autoapifunction:: cuvs.neighbors.cagra.build

Index search
############

.. autoapifunction:: cuvs.neighbors.cagra.search

Index save
##########

.. autoapifunction:: cuvs.neighbors.cagra.save

Index load
##########

.. autoapifunction:: cuvs.neighbors.cagra.load
