# Embedding Models for Semantic Search

Embedding models convert text, images, or other data into dense numerical vectors that capture semantic meaning. Similar items produce vectors that are close together in the embedding space.

## Bi-Encoder Architecture

Most embedding models use a bi-encoder architecture: the query and each document are encoded independently, producing fixed-size vectors. Similarity is computed via dot product or cosine similarity. This enables pre-computation of document embeddings and fast ANN search at query time.

## Popular Models

Sentence-transformers models like all-MiniLM-L6-v2 (384 dimensions) and all-mpnet-base-v2 (768 dimensions) offer a good balance of quality and speed. Larger models like E5-large or GTE-large provide higher accuracy at the cost of slower encoding.

## Cross-Modal Embeddings

CLIP and SigLIP map both text and images into a shared embedding space, enabling cross-modal search (e.g., text-to-image retrieval). This is the foundation of multimodal search applications.

## Normalisation

Most embedding models produce L2-normalised vectors. When vectors are normalised, inner product is equivalent to cosine similarity, which simplifies the choice of distance metric in the vector index.
