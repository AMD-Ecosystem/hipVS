# The Transformer Architecture

Introduced in the landmark 2017 paper 'Attention Is All You Need' by Vaswani et al., the Transformer replaced recurrence with self-attention and quickly became the dominant architecture for natural language processing, computer vision, and multimodal AI.

## Self-Attention Mechanism

Self-attention computes pairwise interactions between all positions in a sequence. Given queries Q, keys K, and values V (all derived from the input), attention scores are computed as softmax(QK^T / sqrt(d_k)) V. This allows the model to capture long-range dependencies without the vanishing-gradient problems of RNNs.

## Encoder-Decoder Structure

The original Transformer uses an encoder stack (for understanding) and a decoder stack (for generation). Encoder-only models like BERT excel at classification and retrieval; decoder-only models like GPT excel at text generation.

## Scaling Laws

Research by Kaplan et al. (2020) showed that Transformer performance improves predictably with more parameters, more data, and more compute. This insight drove the development of ever-larger language models and established scaling as a key paradigm in AI research.

## Relation to Vector Search

Transformer-based embedding models convert text into dense vectors that capture semantic meaning. These vectors are then indexed and searched using ANN algorithms, forming the retrieval backbone of modern RAG systems.
