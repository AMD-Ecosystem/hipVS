# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Enterprise Research Analyst -- Agentic Multi-Document RAG
powered by hipVS (GPU-accelerated vector search on AMD GPUs).

This demo showcases:
  - Multi-format document ingestion (PDF, Markdown, plain text)
  - Agentic RAG: query decomposition, parallel retrieval, synthesis
  - GPU-accelerated vector indexing via hipVS (CAGRA / IVF-Flat / IVF-PQ / Brute-Force)
  - Live performance dashboard with search latency and throughput metrics
  - Algorithm comparison arena: side-by-side evaluation of all index types
  - Cross-document synthesis with source citations and confidence scores
  - Multi-turn conversation with context-aware follow-ups
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from textwrap import dedent
from typing import Optional

import cupy as cp
import gradio as gr
import numpy as np
from cuvs.common import Resources
from cuvs.neighbors import brute_force, cagra, ivf_flat, ivf_pq
from sentence_transformers import SentenceTransformer

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

ALL_ALGORITHMS = ("cagra", "ivf_flat", "ivf_pq", "brute_force")
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".md", ".txt", ".text", ".rst"}

# ---------------------------------------------------------------------------
# Sample corpus -- built-in articles for quick demos without external data
# ---------------------------------------------------------------------------

SAMPLE_ARTICLES = {
    "vector_search_fundamentals.md": (
        "# Vector Search and Similarity\n\n"
        "Vector search (also called similarity search or nearest-neighbor search) "
        "is the process of finding data points in a high-dimensional space that "
        "are closest to a given query vector. It is the backbone of modern "
        "information-retrieval systems, recommendation engines, and AI "
        "applications.\n\n"
        "## Distance Metrics\n\n"
        "The choice of distance metric determines how closeness is measured. "
        "Common metrics include L2 (Euclidean) distance, cosine similarity, and "
        "inner product. For normalised embeddings, cosine similarity and inner "
        "product are equivalent. Jaccard and Hamming distances are used for "
        "binary or set-valued features.\n\n"
        "## Exact vs. Approximate Search\n\n"
        "Brute-force search computes the distance between the query and every "
        "vector in the database, guaranteeing perfect recall. However, it scales "
        "linearly with dataset size. Approximate nearest-neighbor (ANN) "
        "algorithms trade a small amount of recall for dramatically lower "
        "latency, making billion-scale search practical. Prominent ANN "
        "strategies include graph-based methods (HNSW, CAGRA), inverted-file "
        "indexes (IVF-Flat, IVF-PQ), and tree-based partitioning (Annoy).\n\n"
        "## GPU Acceleration\n\n"
        "GPUs excel at vector search because distance computation is inherently "
        "parallel: each distance can be computed independently. Libraries like "
        "hipVS exploit thousands of GPU cores to achieve sub-millisecond search "
        "latency on datasets with millions of vectors.\n"
    ),
    "gpu_accelerated_computing.md": (
        "# GPU-Accelerated Computing\n\n"
        "Graphics Processing Units (GPUs) were originally designed for rendering "
        "pixels, but their massively parallel architecture makes them ideal for "
        "general-purpose computing. A modern data-center GPU contains thousands "
        "of compute cores, high-bandwidth memory (HBM), and hardware schedulers "
        "that can execute millions of lightweight threads concurrently.\n\n"
        "## SIMT Execution Model\n\n"
        "GPUs use a Single-Instruction, Multiple-Thread (SIMT) model where "
        "groups of threads (called wavefronts on AMD or warps on NVIDIA) execute "
        "the same instruction in lockstep. This is perfect for data-parallel "
        "workloads like matrix multiplication, distance computation, and "
        "embedding generation.\n\n"
        "## Memory Hierarchy\n\n"
        "HBM provides bandwidth exceeding 3 TB/s on the latest AMD MI300X "
        "accelerators, enabling rapid streaming of large vector datasets. "
        "On-chip shared memory (LDS on AMD) allows cooperative data reuse within "
        "a workgroup, which is critical for tiled matrix operations used in "
        "vector search kernels.\n\n"
        "## Applications Beyond Graphics\n\n"
        "Today GPUs power deep-learning training and inference, scientific "
        "simulation, financial modelling, genomics, and -- increasingly -- "
        "vector-database workloads where brute-force or ANN search must run at "
        "interactive latencies over massive datasets.\n"
    ),
    "transformer_architecture.md": (
        "# The Transformer Architecture\n\n"
        "Introduced in the landmark 2017 paper 'Attention Is All You Need' by "
        "Vaswani et al., the Transformer replaced recurrence with self-attention "
        "and quickly became the dominant architecture for natural language "
        "processing, computer vision, and multimodal AI.\n\n"
        "## Self-Attention Mechanism\n\n"
        "Self-attention computes pairwise interactions between all positions in a "
        "sequence. Given queries Q, keys K, and values V (all derived from the "
        "input), attention scores are computed as softmax(QK^T / sqrt(d_k)) V. "
        "This allows the model to capture long-range dependencies without the "
        "vanishing-gradient problems of RNNs.\n\n"
        "## Encoder-Decoder Structure\n\n"
        "The original Transformer uses an encoder stack (for understanding) and "
        "a decoder stack (for generation). Encoder-only models like BERT excel "
        "at classification and retrieval; decoder-only models like GPT excel at "
        "text generation.\n\n"
        "## Scaling Laws\n\n"
        "Research by Kaplan et al. (2020) showed that Transformer performance "
        "improves predictably with more parameters, more data, and more compute. "
        "This insight drove the development of ever-larger language models and "
        "established scaling as a key paradigm in AI research.\n\n"
        "## Relation to Vector Search\n\n"
        "Transformer-based embedding models convert text into dense vectors that "
        "capture semantic meaning. These vectors are then indexed and searched "
        "using ANN algorithms, forming the retrieval backbone of modern RAG "
        "systems.\n"
    ),
    "retrieval_augmented_generation.md": (
        "# Retrieval-Augmented Generation (RAG)\n\n"
        "RAG combines a retrieval system with a generative language model. When "
        "a user asks a question, the system first retrieves relevant passages "
        "from a knowledge base, then feeds those passages to an LLM that "
        "synthesises a grounded, cited answer.\n\n"
        "## Why RAG?\n\n"
        "Pure LLMs suffer from hallucination, knowledge cut-off, and lack of "
        "source attribution. RAG addresses all three: the retriever provides "
        "up-to-date, factual evidence, and the generator can cite its sources. "
        "This makes RAG the preferred architecture for enterprise knowledge "
        "assistants.\n\n"
        "## Agentic RAG\n\n"
        "Advanced RAG systems decompose complex questions into sub-queries, "
        "execute multiple retrieval passes, and reason over the combined "
        "evidence. This agentic approach improves answer quality for "
        "multi-faceted questions that no single retrieval can fully address.\n\n"
        "## The Role of Vector Search\n\n"
        "Vector search is the engine inside every RAG system. The quality of "
        "retrieval directly determines the quality of the generated answer. "
        "GPU-accelerated vector search (like hipVS) enables sub-millisecond "
        "retrieval, which is critical when the agent makes multiple retrieval "
        "calls per question.\n"
    ),
    "amd_instinct_accelerators.md": (
        "# AMD Instinct Accelerators\n\n"
        "AMD Instinct is a family of data-center GPUs designed for AI training, "
        "inference, and high-performance computing. The latest generation, "
        "MI300X, combines the CDNA 3 compute architecture with 192 GB of HBM3 "
        "memory and over 5 TB/s of memory bandwidth.\n\n"
        "## CDNA Architecture\n\n"
        "Unlike AMD's consumer RDNA GPUs, the CDNA architecture is optimised "
        "for matrix and vector compute. It features dedicated Matrix Fused "
        "Multiply-Add (MFMA) units that accelerate GEMM operations central to "
        "deep learning and distance computation.\n\n"
        "## ROCm Software Stack\n\n"
        "ROCm is AMD's open-source GPU-compute platform. It provides HIP (a "
        "portable C++ runtime), math libraries (rocBLAS, hipBLAS), and "
        "deep-learning framework support (PyTorch, TensorFlow). hipVS and "
        "hipRAFT are built on ROCm, bringing GPU-accelerated vector search to "
        "AMD hardware.\n\n"
        "## Multi-GPU Scaling\n\n"
        "AMD Instinct GPUs support high-bandwidth Infinity Fabric links for "
        "multi-GPU communication. hipVS can distribute vector indexes across "
        "multiple GPUs (MG-CAGRA, MG-IVF-Flat, MG-IVF-PQ) to handle "
        "billion-scale datasets that exceed single-GPU memory.\n"
    ),
    "cagra_graph_search.md": (
        "# CAGRA: GPU-Accelerated Graph-Based Search\n\n"
        "CAGRA (Cuda Anns GRAph-based) is a state-of-the-art graph-based "
        "approximate nearest-neighbor algorithm designed from the ground up for "
        "GPU execution. It was introduced by the NVIDIA RAPIDS team and has "
        "been ported to AMD GPUs via hipVS.\n\n"
        "## How CAGRA Works\n\n"
        "CAGRA builds a k-nearest-neighbor graph over the dataset, then prunes "
        "and optimises the graph for efficient traversal. At search time, the "
        "algorithm performs a greedy walk through the graph, visiting neighbours "
        "of neighbours until convergence.\n\n"
        "## GPU-Friendly Design\n\n"
        "Unlike CPU-oriented graph algorithms (e.g., HNSW), CAGRA's traversal "
        "is designed for GPU parallelism. Multiple search queries are batched "
        "and executed simultaneously, saturating GPU compute and memory "
        "bandwidth. This yields orders-of-magnitude speedup over CPU baselines.\n\n"
        "## Build and Search Parameters\n\n"
        "Key build parameters include graph_degree (edges per node) and "
        "intermediate_graph_degree (used during construction for higher "
        "quality). Search parameters like itopk_size control the trade-off "
        "between recall and latency.\n\n"
        "## CAGRA + HNSW Hybrid\n\n"
        "CAGRA indexes can be exported to HNSW format for CPU-based serving, "
        "enabling a workflow where indexes are built on GPU (fast) and served "
        "on CPU (cost-effective, no GPU needed at query time).\n"
    ),
    "ivf_indexing.md": (
        "# Inverted File Indexes (IVF)\n\n"
        "Inverted File (IVF) indexes partition the vector space into clusters "
        "and restrict search to only the most relevant clusters. This "
        "dramatically reduces the number of distance computations needed.\n\n"
        "## IVF-Flat\n\n"
        "IVF-Flat stores full (uncompressed) vectors in each cluster. At search "
        "time, the query is compared to cluster centroids, and the n_probes "
        "closest clusters are exhaustively searched. IVF-Flat offers exact "
        "distance computation within probed clusters, giving high recall at the "
        "cost of higher memory usage.\n\n"
        "## IVF-PQ (Product Quantization)\n\n"
        "IVF-PQ compresses vectors within each cluster using product "
        "quantization, which splits each vector into sub-vectors and encodes "
        "them with a codebook. This reduces memory by 10-50x at the cost of "
        "some recall. It is ideal for very large datasets that must fit in GPU "
        "memory.\n\n"
        "## Tuning IVF Parameters\n\n"
        "The number of IVF lists (n_lists) controls partition granularity. "
        "More lists mean smaller clusters and faster search but potentially "
        "lower recall. n_probes at search time determines how many clusters "
        "are visited -- higher probes improve recall but increase latency.\n\n"
        "## When to Choose IVF\n\n"
        "IVF-Flat is a good default when memory is not constrained and you need "
        "high recall. IVF-PQ is preferred when the dataset is too large for "
        "full-precision storage, or when memory-bandwidth is the bottleneck.\n"
    ),
    "large_language_models.md": (
        "# Large Language Models\n\n"
        "Large Language Models (LLMs) are neural networks with billions of "
        "parameters trained on vast text corpora. They can generate coherent "
        "text, answer questions, summarise documents, write code, and perform "
        "multi-step reasoning.\n\n"
        "## Training and Inference\n\n"
        "LLMs are typically pre-trained with a next-token prediction objective "
        "on trillions of tokens, then fine-tuned (or aligned via RLHF) for "
        "specific use-cases like chat, instruction-following, or tool use. "
        "Inference requires significant compute; quantisation and batching are "
        "used to reduce cost.\n\n"
        "## Limitations\n\n"
        "Despite their capabilities, LLMs can hallucinate (produce plausible "
        "but false claims), lack access to private or recent data, and cannot "
        "reliably cite sources. These limitations motivate the use of "
        "retrieval-augmented generation, where an external knowledge base "
        "provides grounded evidence.\n\n"
        "## LLMs in RAG Pipelines\n\n"
        "In a RAG pipeline, the LLM plays two roles: (1) decomposing complex "
        "queries into focused sub-queries (agentic reasoning), and (2) "
        "synthesising a final answer from retrieved passages. The retriever "
        "(powered by vector search) provides the factual foundation.\n"
    ),
    "embedding_models.md": (
        "# Embedding Models for Semantic Search\n\n"
        "Embedding models convert text, images, or other data into dense "
        "numerical vectors that capture semantic meaning. Similar items produce "
        "vectors that are close together in the embedding space.\n\n"
        "## Bi-Encoder Architecture\n\n"
        "Most embedding models use a bi-encoder architecture: the query and each "
        "document are encoded independently, producing fixed-size vectors. "
        "Similarity is computed via dot product or cosine similarity. This "
        "enables pre-computation of document embeddings and fast ANN search at "
        "query time.\n\n"
        "## Popular Models\n\n"
        "Sentence-transformers models like all-MiniLM-L6-v2 (384 dimensions) "
        "and all-mpnet-base-v2 (768 dimensions) offer a good balance of quality "
        "and speed. Larger models like E5-large or GTE-large provide higher "
        "accuracy at the cost of slower encoding.\n\n"
        "## Cross-Modal Embeddings\n\n"
        "CLIP and SigLIP map both text and images into a shared embedding "
        "space, enabling cross-modal search (e.g., text-to-image retrieval). "
        "This is the foundation of multimodal search applications.\n\n"
        "## Normalisation\n\n"
        "Most embedding models produce L2-normalised vectors. When vectors are "
        "normalised, inner product is equivalent to cosine similarity, which "
        "simplifies the choice of distance metric in the vector index.\n"
    ),
    "rocm_ecosystem.md": (
        "# The ROCm Software Ecosystem\n\n"
        "ROCm (Radeon Open Compute) is AMD's open-source software platform for "
        "GPU computing. It provides compilers, runtime libraries, math "
        "libraries, and framework integrations that enable developers to run "
        "CUDA-like workloads on AMD GPUs.\n\n"
        "## HIP: Portable GPU Programming\n\n"
        "HIP (Heterogeneous-compute Interface for Portability) is a C++ runtime "
        "API that closely mirrors CUDA. Code written in HIP can target both "
        "AMD and NVIDIA GPUs, simplifying porting. hipVS uses HIP to bring "
        "RAPIDS cuVS functionality to AMD hardware.\n\n"
        "## hipRAFT and hipVS\n\n"
        "hipRAFT is a port of NVIDIA's RAFT (Reusable Accelerated Functions and "
        "Tools) library. It provides GPU-accelerated primitives for distance "
        "computation, clustering, and linear algebra. hipVS builds on hipRAFT "
        "to deliver high-performance vector search algorithms (CAGRA, IVF-Flat, "
        "IVF-PQ) on AMD Instinct GPUs.\n\n"
        "## Framework Support\n\n"
        "ROCm supports PyTorch, TensorFlow, and JAX, enabling end-to-end AI "
        "pipelines on AMD hardware. Combined with hipVS for vector search, this "
        "allows building complete RAG applications entirely on AMD GPUs.\n"
    ),
}


def create_sample_corpus(output_dir):
    """Write sample articles to disk for quick demos."""
    if os.path.isdir(output_dir) and any(
        f.endswith((".md", ".txt")) for f in os.listdir(output_dir)
    ):
        print(f"Sample corpus already exists at {output_dir}")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    for filename, content in SAMPLE_ARTICLES.items():
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(content)
    print(
        f"Created sample corpus: {len(SAMPLE_ARTICLES)} articles "
        f"at {output_dir}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text converter using stdlib."""

    def __init__(self):
        super().__init__()
        self._pieces = []

    def handle_data(self, data):
        self._pieces.append(data)

    def get_text(self):
        return " ".join(self._pieces)


def strip_html(html_text):
    s = _HTMLStripper()
    s.feed(html_text)
    return s.get_text()


class DocumentParser:
    """Parse PDF, Markdown, plain-text files and web URLs."""

    @staticmethod
    def parse_file(path):
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return DocumentParser._parse_pdf(path)
        return DocumentParser._parse_text(path)

    @staticmethod
    def _parse_pdf(path):
        if not HAS_PYMUPDF:
            print(
                f"  Skipping PDF '{path}': PyMuPDF not installed "
                f"(pip install pymupdf)"
            )
            return []
        segments = []
        doc = fitz.open(path)
        for page_num, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if text:
                segments.append({
                    "text": text,
                    "source": Path(path).name,
                    "page": page_num,
                })
        doc.close()
        return segments

    @staticmethod
    def _parse_text(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if not text:
            return []
        return [{"text": text, "source": Path(path).name, "page": None}]

    @staticmethod
    def parse_url(url):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "hipVS-ResearchAnalyst/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            text = strip_html(raw).strip()
            if not text:
                return []
            return [{"text": text, "source": url, "page": None}]
        except Exception as e:
            print(f"  Failed to fetch URL '{url}': {e}")
            return []

    @staticmethod
    def parse_directory(directory):
        documents = []
        root = Path(directory)
        if not root.is_dir():
            print(f"Error: '{directory}' is not a directory")
            return []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS and path.is_file():
                docs = DocumentParser.parse_file(str(path))
                documents.extend(docs)
        print(
            f"  Parsed {len(documents)} document segment(s) "
            f"from {directory}"
        )
        return documents


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


class TextChunker:
    """Split document segments into overlapping chunks with metadata."""

    def __init__(self, chunk_size=800, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents):
        chunks = []
        for doc in documents:
            text = doc["text"]
            source = doc["source"]
            page = doc.get("page")
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": source,
                        "page": page,
                        "chunk_id": len(chunks),
                        "start_char": start,
                    })
                if end >= len(text):
                    break
                start = end - self.overlap
        print(
            f"  Chunked into {len(chunks)} passages "
            f"(size={self.chunk_size}, overlap={self.overlap})"
        )
        return chunks


# ---------------------------------------------------------------------------
# Text embedder
# ---------------------------------------------------------------------------


class TextEmbedder:
    """Encode text using sentence-transformers."""

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        print(f"Loading embedding model: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            probe = self.model.encode(["test"], convert_to_numpy=True)
            dim = probe.shape[-1]
        self.dim = dim
        print(f"  Embedding dimension: {self.dim}")

    def encode_batch(self, texts, batch_size=64):
        n = len(texts)
        all_embs = []
        t0 = time.perf_counter()
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            embs = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            all_embs.append(embs)
            done = min(i + batch_size, n)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"\r  Embedding: {done}/{n} ({100 * done / n:.0f}%) "
                f"[{rate:.0f} chunks/s]",
                end="",
                flush=True,
            )
        print()
        return np.vstack(all_embs).astype(np.float32)

    def encode_query(self, text):
        return self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _embedding_cache_key(docs_dir, model_name, chunk_cfg):
    raw = f"{os.path.abspath(docs_dir)}|{model_name}|{chunk_cfg}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_embedding_cache(cache_dir, embeddings, chunks):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(cache_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f)
    print(f"  Saved embedding cache -> {cache_dir}")


def load_embedding_cache(cache_dir, expected):
    emb_path = os.path.join(cache_dir, "embeddings.npy")
    meta_path = os.path.join(cache_dir, "chunks.json")
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        return None
    try:
        embeddings = np.load(emb_path)
        with open(meta_path) as f:
            chunks = json.load(f)
        if len(embeddings) != expected or len(chunks) != expected:
            return None
        print(f"  Loaded {len(embeddings):,} cached embeddings from {cache_dir}")
        return embeddings, chunks
    except Exception:
        return None


# ---------------------------------------------------------------------------
# hipVS vector index
# ---------------------------------------------------------------------------


class HipVSIndex:
    """Multi-algorithm vector index backed by hipVS on AMD GPUs."""

    def __init__(self):
        self.resources = Resources()
        self.indices = {}
        self.build_times = {}
        self.n_vectors = 0
        self.dim = 0

    def available_algorithms(self):
        return list(self.indices.keys())

    def build(self, embeddings, algorithms=ALL_ALGORITHMS, n_lists=100):
        gpu_data = cp.asarray(embeddings, dtype=cp.float32)
        n, dim = embeddings.shape
        self.n_vectors = n
        self.dim = dim

        for algo in algorithms:
            print(
                f"Building hipVS index ({algo}): "
                f"{n:,} vectors, dim={dim} ...",
                flush=True,
            )
            t0 = time.perf_counter()
            try:
                if algo == "cagra":
                    if n < 4:
                        print(f"  Skipping {algo}: too few vectors ({n})")
                        continue
                    gd = min(32, n - 1)
                    igd = min(64, n - 1)
                    params = cagra.IndexParams(
                        metric="inner_product",
                        graph_degree=gd,
                        intermediate_graph_degree=igd,
                    )
                    self.indices[algo] = cagra.build(
                        params, gpu_data, resources=self.resources,
                    )
                elif algo == "ivf_flat":
                    actual_n_lists = min(n_lists, max(1, n // 10))
                    params = ivf_flat.IndexParams(
                        n_lists=actual_n_lists, metric="inner_product",
                    )
                    self.indices[algo] = ivf_flat.build(
                        params, gpu_data, resources=self.resources,
                    )
                elif algo == "ivf_pq":
                    actual_n_lists = min(n_lists, max(1, n // 10))
                    pq_dim = min(dim, max(2, dim // 4))
                    params = ivf_pq.IndexParams(
                        n_lists=actual_n_lists,
                        metric="inner_product",
                        pq_dim=pq_dim,
                        pq_bits=8,
                    )
                    self.indices[algo] = ivf_pq.build(
                        params, gpu_data, resources=self.resources,
                    )
                elif algo == "brute_force":
                    self.indices[algo] = brute_force.build(
                        gpu_data, metric="inner_product",
                        resources=self.resources,
                    )

                self.resources.sync()
                elapsed = time.perf_counter() - t0
                self.build_times[algo] = elapsed
                print(f"  {algo} built in {elapsed:.3f}s", flush=True)
            except Exception as e:
                print(f"  WARNING: {algo} build failed: {e}", flush=True)

    def search(self, query_embedding, top_k=5, algorithm="cagra"):
        """Return (neighbor_ids, distances, latency_ms)."""
        if algorithm not in self.indices:
            return [], [], 0.0

        index = self.indices[algorithm]
        k = min(top_k, self.n_vectors)
        gpu_query = cp.asarray(query_embedding, dtype=cp.float32)

        t0 = time.perf_counter()

        if algorithm == "cagra":
            sp = cagra.SearchParams()
            distances, neighbors = cagra.search(
                sp, index, gpu_query, k, resources=self.resources,
            )
        elif algorithm == "ivf_flat":
            sp = ivf_flat.SearchParams(n_probes=50)
            distances, neighbors = ivf_flat.search(
                sp, index, gpu_query, k=k, resources=self.resources,
            )
        elif algorithm == "ivf_pq":
            n_lists_est = max(1, int(math.sqrt(self.n_vectors)))
            sp = ivf_pq.SearchParams(n_probes=max(1, n_lists_est // 5))
            distances, neighbors = ivf_pq.search(
                sp, index, gpu_query, k, resources=self.resources,
            )
        elif algorithm == "brute_force":
            distances, neighbors = brute_force.search(
                index, gpu_query, k, resources=self.resources,
            )

        self.resources.sync()
        latency_ms = (time.perf_counter() - t0) * 1000

        ids = cp.asnumpy(cp.asarray(neighbors))[0].tolist()
        dists = cp.asnumpy(cp.asarray(distances))[0].tolist()
        return ids, dists, latency_ms


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------


class PerformanceTracker:
    """Collect and summarise search-performance metrics."""

    def __init__(self):
        self.queries = []

    def record(self, algorithm, latency_ms, num_results, num_sub_queries=1):
        self.queries.append({
            "algorithm": algorithm,
            "latency_ms": latency_ms,
            "num_results": num_results,
            "num_sub_queries": num_sub_queries,
            "timestamp": time.time(),
        })

    def avg_latency(self, algorithm=None):
        items = (
            [q for q in self.queries if q["algorithm"] == algorithm]
            if algorithm
            else self.queries
        )
        if not items:
            return 0.0
        return sum(q["latency_ms"] for q in items) / len(items)

    def total_queries(self):
        return len(self.queries)

    def last_latency(self):
        return self.queries[-1]["latency_ms"] if self.queries else 0.0


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class LLMBackend:
    """Abstract base class for LLM inference."""

    def generate(self, prompt):
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    def __init__(self, model="llama3.2:3b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt):
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["response"]


class OpenAIBackend(LLMBackend):
    def __init__(self, model="gpt-4o-mini", api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def generate(self, prompt):
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]


class RetrievalOnlyBackend(LLMBackend):
    """No LLM -- returns empty so the caller formats retrieval results."""

    def generate(self, prompt):
        return ""


# ---------------------------------------------------------------------------
# Research analyst -- agentic RAG engine
# ---------------------------------------------------------------------------


def _parse_sub_queries(response):
    """Best-effort extraction of sub-queries from LLM response."""
    text = response.strip()

    for block in re.split(r"```\w*", text):
        block = block.strip().strip("`")
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q) for q in parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    queries = []
    for line in text.split("\n"):
        cleaned = re.sub(r"^[\d\-\*\.\)\]]+\s*", "", line.strip()).strip("\"'")
        if cleaned and len(cleaned) > 5:
            queries.append(cleaned)
    return queries if queries else [text]


class ResearchAnalyst:
    """Agentic RAG: decompose, multi-retrieve, synthesise with citations."""

    def __init__(
        self, embedder, index, chunks, llm, tracker, top_k_per_query=5,
    ):
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.llm = llm
        self.tracker = tracker
        self.top_k_per_query = top_k_per_query
        self.all_embeddings = None

    @property
    def has_llm(self):
        return not isinstance(self.llm, RetrievalOnlyBackend)

    def decompose_query(self, question):
        if self.has_llm:
            return self._llm_decompose(question)
        return self._heuristic_decompose(question)

    def _llm_decompose(self, question):
        prompt = dedent(f"""\
            You are a research analyst. Break the following complex question
            into 2-4 focused sub-queries that will help retrieve relevant
            information from a technical document collection.

            Question: {question}

            Return ONLY a JSON array of short sub-query strings.
            Example: ["sub-query 1", "sub-query 2", "sub-query 3"]""")
        try:
            response = self.llm.generate(prompt)
            sub_queries = _parse_sub_queries(response)
            if not sub_queries:
                return [question]
            return sub_queries[:5]
        except Exception as e:
            print(f"  Decomposition failed ({e}), using original query")
            return [question]

    def _heuristic_decompose(self, question):
        """Keyword-based decomposition when no LLM is available."""
        q = question.strip().rstrip("?").lower()

        comparisons = [
            "compare", "vs", "versus", "difference", "differ",
            "contrast", "between",
        ]
        is_comparison = any(w in q for w in comparisons)

        clauses = re.split(r"\band\b|\bor\b|,\s*(?:and\s+)?", q)
        clauses = [c.strip() for c in clauses if len(c.strip()) > 10]

        if is_comparison and len(clauses) >= 2:
            return clauses[:4]
        if len(clauses) >= 2:
            return clauses[:4]

        parts = re.split(
            r"\bhow\b|\bwhat\b|\bwhy\b|\bwhen\b|\bwhich\b",
            q, flags=re.IGNORECASE,
        )
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if len(parts) >= 2:
            return parts[:4]

        return [question]

    def retrieve(self, query_text, algorithm, top_k=5):
        """Embed query and search. Return (result_chunks, latency_ms)."""
        query_emb = self.embedder.encode_query(query_text)
        ids, dists, latency_ms = self.index.search(
            query_emb, top_k=top_k, algorithm=algorithm,
        )
        results = []
        for idx, dist in zip(ids, dists):
            if 0 <= idx < len(self.chunks):
                results.append({**self.chunks[idx], "score": float(dist)})
        return results, latency_ms

    def multi_retrieve(self, sub_queries, algorithm):
        """Run all sub-queries, deduplicate results.

        Returns (unique_chunks, per_query_stats, total_latency_ms).
        """
        seen_ids = set()
        unique_chunks = []
        per_query = []
        total_ms = 0.0

        for sq in sub_queries:
            results, ms = self.retrieve(sq, algorithm, self.top_k_per_query)
            total_ms += ms
            new_count = 0
            for r in results:
                cid = r["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    unique_chunks.append(r)
                    new_count += 1
            per_query.append({
                "query": sq,
                "num_chunks": len(results),
                "new_chunks": new_count,
                "latency_ms": ms,
            })

        unique_chunks.sort(key=lambda c: c["score"], reverse=True)
        return unique_chunks, per_query, total_ms

    def synthesise(self, question, sub_queries, chunks):
        if not self.has_llm:
            return ""

        passage_parts = []
        for i, c in enumerate(chunks[:15]):
            page_str = ""
            if c.get("page"):
                page_str = f", p.{c['page']}"
            passage_parts.append(
                f"[{i + 1}] (Source: {c['source']}{page_str}"
                f", score: {c['score']:.3f}):\n\"{c['text'][:600]}\""
            )
        passages = "\n\n".join(passage_parts)

        sub_q_text = "\n".join(
            f"  {i + 1}. {sq}" for i, sq in enumerate(sub_queries)
        )
        prompt = dedent(f"""\
            You are a research analyst. Answer the following question using
            ONLY the retrieved passages below. Structure your answer clearly
            and cite every claim using [N] notation where N is the source
            number.

            Question: {question}

            This question was investigated via these sub-queries:
            {sub_q_text}

            Retrieved passages:
            {passages}

            Rules:
            - Base your answer ONLY on the provided passages.
            - Cite each key claim with [N] notation.
            - If the passages are insufficient, state what you found and
              what is missing.
            - Be thorough but concise.""")
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            return f"*Synthesis failed: {e}*"

    def _unique_sources(self, chunks):
        seen = set()
        sources = []
        for c in chunks:
            if c["source"] not in seen:
                seen.add(c["source"])
                sources.append(c["source"])
        return sources

    def research(self, question, algorithm="cagra"):
        """Full agentic pipeline. Returns dict with response and metrics."""
        t_start = time.perf_counter()

        sub_queries = self.decompose_query(question)

        chunks, per_query_stats, retrieval_ms = self.multi_retrieve(
            sub_queries, algorithm,
        )

        answer = self.synthesise(question, sub_queries, chunks)

        total_ms = (time.perf_counter() - t_start) * 1000
        sources = self._unique_sources(chunks)

        self.tracker.record(
            algorithm=algorithm,
            latency_ms=retrieval_ms,
            num_results=len(chunks),
            num_sub_queries=len(sub_queries),
        )

        response = self._format_response(
            question, sub_queries, per_query_stats,
            chunks, answer, algorithm, retrieval_ms,
        )

        return {
            "response": response,
            "retrieval_ms": retrieval_ms,
            "total_ms": total_ms,
            "num_sub_queries": len(sub_queries),
            "num_chunks": len(chunks),
            "num_sources": len(sources),
            "algorithm": algorithm,
        }

    def _format_response(
        self, question, sub_queries, per_query_stats,
        chunks, answer, algorithm, retrieval_ms,
    ):
        lines = []
        num_sources = len(self._unique_sources(chunks))
        mode_label = "Agentic RAG" if self.has_llm else "Multi-Retrieval"

        # Always show the reasoning trace / retrieval stats
        lines.append(f"**{mode_label} Analysis**\n")
        if len(sub_queries) > 1:
            lines.append(
                f"Decomposed into {len(sub_queries)} sub-queries:\n"
            )
        else:
            lines.append("Retrieval query:\n")
        lines.append("| # | Sub-Query | Chunks | Latency |")
        lines.append("|---|-----------|--------|---------|")
        for i, stat in enumerate(per_query_stats):
            q_display = stat["query"][:60]
            lines.append(
                f"| {i + 1} | {q_display} "
                f"| {stat['num_chunks']} "
                f"| {stat['latency_ms']:.2f}ms |"
            )

        lines.append(
            f"\n**Retrieval**: {len(chunks)} unique passages from "
            f"{num_sources} documents "
            f"in **{retrieval_ms:.2f}ms** "
            f"({algorithm.upper()}, "
            f"{self.index.n_vectors:,} vectors indexed)\n"
        )
        lines.append("---\n")

        # Answer section
        if answer:
            lines.append(f"{answer}\n")
        else:
            lines.append("**Retrieved Passages**\n")
            for i, c in enumerate(chunks[:10]):
                page_info = ""
                if c.get("page"):
                    page_info = f", p.{c['page']}"
                text_preview = c["text"][:300]
                ellipsis = "..." if len(c["text"]) > 300 else ""
                lines.append(
                    f"**[{i + 1}]** *{c['source']}{page_info}* "
                    f"(score: {c['score']:.3f})\n"
                    f"> {text_preview}{ellipsis}\n"
                )

        # Sources
        lines.append("---\n**Sources**\n")
        for i, c in enumerate(chunks[:15]):
            page_info = ""
            if c.get("page"):
                page_info = f", p.{c['page']}"
            excerpt = c["text"][:120].replace("\n", " ")
            lines.append(
                f"- **[{i + 1}]** *{c['source']}{page_info}* "
                f'-- "{excerpt}..."'
            )

        return "\n".join(lines)

    def add_documents(self, file_paths, algorithms=ALL_ALGORITHMS):
        """Parse, chunk, embed new documents and rebuild the index."""
        new_docs = []
        for fp in file_paths:
            new_docs.extend(DocumentParser.parse_file(fp))
        if not new_docs:
            return 0

        chunker = TextChunker()
        new_chunks = chunker.chunk_documents(new_docs)
        offset = len(self.chunks)
        for i, c in enumerate(new_chunks):
            c["chunk_id"] = offset + i

        new_embs = self.embedder.encode_batch([c["text"] for c in new_chunks])
        self.chunks.extend(new_chunks)

        if self.all_embeddings is not None:
            self.all_embeddings = np.vstack([self.all_embeddings, new_embs])
        else:
            self.all_embeddings = new_embs

        self.index.build(self.all_embeddings, algorithms=algorithms)
        return len(new_chunks)

    def warmup(self):
        print("Warming up embedding + hipVS ...", flush=True)
        for algo in self.index.available_algorithms():
            _, _, ms = self.index.search(
                self.embedder.encode_query("warmup"), top_k=3, algorithm=algo,
            )
            print(f"  {algo}: {ms:.2f}ms", flush=True)
        print("Warmup complete.", flush=True)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

APP_CSS = """
.header { text-align: center; margin-bottom: 8px; }
.header h1 { margin-bottom: 2px; }
.header p { color: #666; margin-top: 4px; }
.metric-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 8px; margin-bottom: 12px;
}
.metric-card {
    background: linear-gradient(135deg, #fff5f5 0%, #fff 100%);
    border: 1px solid #fecaca; border-radius: 10px;
    padding: 14px 10px; text-align: center;
}
.metric-value {
    font-size: 22px; font-weight: 700; color: #dc2626;
}
.metric-label {
    font-size: 11px; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;
}
.arena-grid { display: grid; gap: 12px; margin-top: 12px; }
.arena-card {
    border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px;
}
.arena-card h4 { margin: 0 0 8px 0; color: #dc2626; }
.arena-passage {
    font-size: 13px; color: #374151;
    border-left: 3px solid #fecaca; padding-left: 10px; margin: 6px 0;
}
"""

EXAMPLE_QUERIES = [
    "How does CAGRA compare to IVF-based indexes for vector search?",
    "Explain the role of GPU memory bandwidth in vector search performance.",
    "What is retrieval-augmented generation and why does it need fast search?",
    "How do transformers relate to embedding models used in vector search?",
    "Describe AMD's Instinct GPU architecture and the ROCm software stack.",
]


def _build_metrics_html(tracker, index, num_docs):
    avg = tracker.avg_latency()
    last = tracker.last_latency()
    return (
        '<div class="metric-grid">'
        '<div class="metric-card">'
        f'<div class="metric-value">{last:.2f}ms</div>'
        '<div class="metric-label">Last Search</div></div>'
        '<div class="metric-card">'
        f'<div class="metric-value">{avg:.2f}ms</div>'
        '<div class="metric-label">Avg Latency</div></div>'
        '<div class="metric-card">'
        f'<div class="metric-value">{tracker.total_queries()}</div>'
        '<div class="metric-label">Total Queries</div></div>'
        '<div class="metric-card">'
        f'<div class="metric-value">{index.n_vectors:,}</div>'
        '<div class="metric-label">Vectors Indexed</div></div>'
        '<div class="metric-card">'
        f'<div class="metric-value">{num_docs}</div>'
        '<div class="metric-label">Documents</div></div>'
        '<div class="metric-card">'
        f'<div class="metric-value">{index.dim}</div>'
        '<div class="metric-label">Embed Dim</div></div>'
        '</div>'
    )


def _build_arena_html(results_by_algo):
    if not results_by_algo:
        return "<p>No algorithms available.</p>"

    parts = ['<div class="arena-grid">']

    summary_items = []
    for algo, (_, ms) in results_by_algo.items():
        summary_items.append(f"<b>{algo.upper()}</b>: {ms:.2f}ms")
    latency_summary = " &nbsp;|&nbsp; ".join(summary_items)
    parts.append(
        f'<div style="text-align:center; font-size:14px; '
        f'margin-bottom:8px;">{latency_summary}</div>'
    )

    cols = len(results_by_algo)
    parts.append(
        f'<div style="display:grid; grid-template-columns: '
        f'repeat({cols}, 1fr); gap:10px;">'
    )
    for algo, (algo_chunks, ms) in results_by_algo.items():
        parts.append('<div class="arena-card">')
        parts.append(f"<h4>{algo.upper()} &mdash; {ms:.2f}ms</h4>")
        for i, c in enumerate(algo_chunks[:5]):
            score = c.get("score", 0)
            text_preview = c["text"][:150].replace("<", "&lt;")
            parts.append(
                f'<div class="arena-passage">'
                f"<b>[{i + 1}]</b> <i>{c['source']}</i> "
                f"(score: {score:.3f})<br>"
                f"{text_preview}...</div>"
            )
        parts.append("</div>")
    parts.append("</div></div>")
    return "\n".join(parts)


def _corpus_overview_html(analyst):
    source_counts = {}
    for c in analyst.chunks:
        src = c["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    if not source_counts:
        return "<p>No documents loaded.</p>"

    rows = "".join(
        f"<tr><td>{src}</td><td style='text-align:right'>{cnt}</td></tr>"
        for src, cnt in sorted(source_counts.items())
    )
    total_chunks = sum(source_counts.values())
    return (
        '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
        "<thead>"
        '<tr style="border-bottom:2px solid #e5e7eb;">'
        '<th style="text-align:left; padding:6px;">Document</th>'
        '<th style="text-align:right; padding:6px;">Chunks</th>'
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "<tfoot>"
        '<tr style="border-top:2px solid #e5e7eb; font-weight:bold;">'
        f'<td style="padding:6px;">{len(source_counts)} documents</td>'
        f'<td style="text-align:right; padding:6px;">{total_chunks} chunks</td>'
        "</tr></tfoot></table>"
    )


def create_ui(analyst):
    index = analyst.index
    tracker = analyst.tracker
    algos = index.available_algorithms() or ["cagra"]
    num_source_docs = len(set(c["source"] for c in analyst.chunks))

    with gr.Blocks(
        title="hipVS -- Enterprise Research Analyst",
    ) as app:

        gr.HTML(
            '<div class="header">'
            "<h1>Enterprise Research Analyst</h1>"
            "<p>Agentic Multi-Document RAG powered by "
            "<b>hipVS</b> on <b>AMD Instinct</b> GPUs</p>"
            '<p style="font-size:0.8em; color:#999;">'
            "Query decomposition &bull; Parallel retrieval &bull; "
            "Cross-document synthesis &bull; Source citations"
            "</p></div>"
        )

        with gr.Tabs():

            # ---- Tab 1: Research Chat ----
            with gr.TabItem("Research Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Research Analyst",
                            height=500,
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask a research question...",
                                show_label=False,
                                scale=5,
                                container=False,
                            )
                            send_btn = gr.Button(
                                "Send", variant="primary", scale=1,
                            )
                        with gr.Row():
                            algo_dropdown = gr.Dropdown(
                                algos, value=algos[0],
                                label="Algorithm", scale=1,
                            )
                            topk_slider = gr.Slider(
                                3, 20, value=5, step=1,
                                label="Results per sub-query", scale=1,
                            )
                            clear_btn = gr.ClearButton(
                                [chatbot, msg_input],
                                value="Clear", scale=1,
                            )
                        gr.Examples(
                            examples=EXAMPLE_QUERIES,
                            inputs=msg_input,
                            label="Try these questions",
                        )
                    with gr.Column(scale=1):
                        metrics_panel = gr.HTML(
                            _build_metrics_html(tracker, index, num_source_docs),
                        )
                        query_details = gr.Markdown(
                            "*Ask a question to see retrieval details.*"
                        )

            # ---- Tab 2: Algorithm Arena ----
            with gr.TabItem("Algorithm Arena"):
                gr.Markdown(
                    "Run the same query against **all** built algorithms "
                    "and compare results, latency, and relevance side-by-side."
                )
                with gr.Row():
                    arena_query = gr.Textbox(
                        placeholder="Enter a query to compare algorithms...",
                        show_label=False, scale=5, container=False,
                    )
                    arena_btn = gr.Button(
                        "Compare All", variant="primary", scale=1,
                    )
                arena_topk = gr.Slider(
                    3, 15, value=5, step=1,
                    label="Top-K per algorithm",
                )
                arena_results = gr.HTML(
                    '<p style="color:#999; text-align:center;">'
                    "Enter a query and click Compare All.</p>"
                )

            # ---- Tab 3: Knowledge Base ----
            with gr.TabItem("Knowledge Base"):
                gr.Markdown(
                    "Upload additional documents to expand the knowledge base. "
                    "Supported formats: PDF, Markdown, plain text."
                )
                with gr.Row():
                    file_upload = gr.File(
                        file_count="multiple",
                        file_types=[".pdf", ".md", ".txt", ".text", ".rst"],
                        label="Upload Documents",
                        type="filepath",
                        scale=3,
                    )
                    upload_btn = gr.Button(
                        "Add to Knowledge Base",
                        variant="primary", scale=1,
                    )
                upload_status = gr.Markdown("")
                gr.Markdown("### Corpus Overview")
                corpus_info = gr.HTML(_corpus_overview_html(analyst))

        with gr.Accordion("Index Details", open=False):
            build_lines = "\n".join(
                f"  - **{algo}**: {index.build_times.get(algo, 0):.3f}s"
                for algo in algos
            )
            gr.Markdown(
                f"- **Algorithms**: {', '.join(algos)}\n"
                f"- **Total chunks**: {len(analyst.chunks):,}\n"
                f"- **Embedding model**: `{analyst.embedder.model_name}`\n"
                f"- **Embedding dim**: {analyst.embedder.dim}\n"
                f"- **Build times**:\n{build_lines}"
            )

        # ---- Event Handlers ----

        def do_chat(message, history, algorithm, top_k):
            if not message or not message.strip():
                return (
                    history, "",
                    _build_metrics_html(tracker, index, num_source_docs),
                    "*Enter a question.*",
                )
            analyst.top_k_per_query = int(top_k)
            result = analyst.research(message.strip(), algorithm=algorithm)

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result["response"]},
            ]

            detail_md = (
                f"**Last query**: {result['retrieval_ms']:.2f}ms retrieval "
                f"| {result['num_sub_queries']} sub-queries "
                f"| {result['num_chunks']} chunks "
                f"| {result['num_sources']} sources "
                f"| {result['algorithm'].upper()}"
            )
            new_metrics = _build_metrics_html(
                tracker, index, num_source_docs,
            )
            return history, "", new_metrics, detail_md

        def do_arena(query, top_k):
            if not query or not query.strip():
                return "<p>Enter a query first.</p>"
            query = query.strip()
            query_emb = analyst.embedder.encode_query(query)
            results_by_algo = {}
            for algo in index.available_algorithms():
                ids, dists, ms = index.search(
                    query_emb, top_k=int(top_k), algorithm=algo,
                )
                found = []
                for idx, dist in zip(ids, dists):
                    if 0 <= idx < len(analyst.chunks):
                        found.append({
                            **analyst.chunks[idx], "score": float(dist),
                        })
                results_by_algo[algo] = (found, ms)
            return _build_arena_html(results_by_algo)

        def do_upload(files):
            nonlocal num_source_docs
            if not files:
                return "No files selected.", _corpus_overview_html(analyst)

            pdf_files = [f for f in files if f.lower().endswith(".pdf")]
            if pdf_files and not HAS_PYMUPDF:
                return (
                    "**Error**: PDF support requires PyMuPDF. "
                    "Install it with `pip install pymupdf` and restart.",
                    _corpus_overview_html(analyst),
                )

            try:
                n = analyst.add_documents(
                    files, algorithms=tuple(index.available_algorithms()),
                )
            except Exception as e:
                return (
                    f"**Error** processing documents: {e}",
                    _corpus_overview_html(analyst),
                )

            if n == 0:
                names = ", ".join(Path(f).name for f in files)
                return (
                    f"**Warning**: No text could be extracted from: {names}. "
                    f"Check that the files are not empty or corrupted.",
                    _corpus_overview_html(analyst),
                )

            num_source_docs = len(set(c["source"] for c in analyst.chunks))
            return (
                f"Added **{n}** new chunks from {len(files)} file(s). "
                f"Index rebuilt.",
                _corpus_overview_html(analyst),
            )

        send_btn.click(
            do_chat,
            [msg_input, chatbot, algo_dropdown, topk_slider],
            [chatbot, msg_input, metrics_panel, query_details],
        )
        msg_input.submit(
            do_chat,
            [msg_input, chatbot, algo_dropdown, topk_slider],
            [chatbot, msg_input, metrics_panel, query_details],
        )
        arena_btn.click(
            do_arena, [arena_query, arena_topk], [arena_results],
        )
        upload_btn.click(
            do_upload, [file_upload], [upload_status, corpus_info],
        )

    return app


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="hipVS Enterprise Research Analyst -- "
        "Agentic Multi-Document RAG on AMD GPUs",
    )
    parser.add_argument(
        "--docs", type=str, default=None,
        help="Path to a directory of documents (PDF, MD, TXT). "
             "Uses built-in sample corpus if not provided.",
    )
    parser.add_argument(
        "--embed-model", type=str, default="all-MiniLM-L6-v2",
        help="sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800,
        help="Chunk size in characters (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=150,
        help="Chunk overlap in characters (default: 150)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Results per sub-query (default: 5)",
    )
    parser.add_argument(
        "--algorithms", type=str, nargs="+",
        default=list(ALL_ALGORITHMS),
        choices=list(ALL_ALGORITHMS),
        help="Algorithms to build (default: all four)",
    )
    parser.add_argument(
        "--llm", type=str, default="retrieval",
        choices=["ollama", "openai", "retrieval"],
        help="LLM backend: 'ollama' (local), 'openai' (API), or "
             "'retrieval' (no LLM, just show retrieved passages)",
    )
    parser.add_argument("--ollama-model", type=str, default="llama3.2:3b")
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
    )
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-embedding; ignore cached embeddings",
    )
    parser.add_argument(
        "--embed-device", type=str, default="gpu",
        choices=["cpu", "gpu"],
        help="Device for embedding model (default: gpu)",
    )
    parser.add_argument(
        "--port", type=int, default=7863,
        help="Gradio server port (default: 7863)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link",
    )
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("hipVS -- Enterprise Research Analyst", flush=True)
    print("Agentic Multi-Document RAG on AMD GPUs", flush=True)
    print("=" * 60, flush=True)

    # ---- Resolve document directory ----
    if args.docs:
        docs_dir = args.docs
    else:
        docs_dir = create_sample_corpus("./data/sample_corpus")

    # ---- Parse documents ----
    print(f"Loading documents from {docs_dir} ...", flush=True)
    raw_docs = DocumentParser.parse_directory(docs_dir)
    if not raw_docs:
        print(
            f"Error: no documents found in '{docs_dir}'\n"
            f"Supported: {', '.join(sorted(SUPPORTED_DOC_EXTENSIONS))}",
        )
        sys.exit(1)

    # ---- Chunk ----
    chunker = TextChunker(
        chunk_size=args.chunk_size, overlap=args.chunk_overlap,
    )
    chunks = chunker.chunk_documents(raw_docs)
    if not chunks:
        print("Error: no text chunks produced from the documents.")
        sys.exit(1)

    # ---- Embed (cached) ----
    embed_device = "cuda" if args.embed_device == "gpu" else args.embed_device
    embedder = TextEmbedder(model_name=args.embed_model, device=embed_device)

    chunk_cfg = f"{args.chunk_size}_{args.chunk_overlap}"
    cache_key = _embedding_cache_key(docs_dir, args.embed_model, chunk_cfg)
    cache_dir = os.path.join("./data/cache", cache_key)
    cached = (
        None if args.no_cache
        else load_embedding_cache(cache_dir, len(chunks))
    )

    if cached is not None:
        embeddings, chunks = cached[0], cached[1]
    else:
        print(
            f"Embedding {len(chunks):,} chunks with {args.embed_model} ...",
            flush=True,
        )
        t0 = time.perf_counter()
        embeddings = embedder.encode_batch([c["text"] for c in chunks])
        print(
            f"  Done in {time.perf_counter() - t0:.1f}s "
            f"(shape {embeddings.shape})",
            flush=True,
        )
        if not args.no_cache:
            save_embedding_cache(cache_dir, embeddings, chunks)

    # ---- GPU memory ----
    cp.get_default_memory_pool().free_all_blocks()
    mem_free, mem_total = cp.cuda.Device().mem_info
    print(
        f"GPU memory: {mem_free / 1e9:.1f} GB free / "
        f"{mem_total / 1e9:.1f} GB total",
        flush=True,
    )

    # ---- Build hipVS indexes ----
    algos = tuple(args.algorithms)
    print(f"Building indexes: {', '.join(algos)}", flush=True)
    index = HipVSIndex()
    index.build(embeddings, algorithms=algos)

    # ---- LLM backend ----
    if args.llm == "ollama":
        llm = OllamaBackend(
            model=args.ollama_model, base_url=args.ollama_url,
        )
        print(
            f"LLM: Ollama ({args.ollama_model}) at {args.ollama_url}",
            flush=True,
        )
    elif args.llm == "openai":
        llm = OpenAIBackend(model=args.openai_model)
        print(f"LLM: OpenAI ({args.openai_model})", flush=True)
    else:
        llm = RetrievalOnlyBackend()
        print("LLM: Retrieval-only mode (no generation)", flush=True)

    # ---- Assemble analyst ----
    tracker = PerformanceTracker()
    analyst = ResearchAnalyst(
        embedder=embedder,
        index=index,
        chunks=chunks,
        llm=llm,
        tracker=tracker,
        top_k_per_query=args.top_k,
    )
    analyst.all_embeddings = embeddings
    analyst.warmup()

    # ---- Launch ----
    app = create_ui(analyst)
    print("=" * 60, flush=True)
    print(f"Open http://localhost:{args.port}", flush=True)
    print("=" * 60, flush=True)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange"),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
