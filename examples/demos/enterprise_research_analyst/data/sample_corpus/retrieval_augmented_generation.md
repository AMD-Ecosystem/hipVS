# Retrieval-Augmented Generation (RAG)

RAG combines a retrieval system with a generative language model. When a user asks a question, the system first retrieves relevant passages from a knowledge base, then feeds those passages to an LLM that synthesises a grounded, cited answer.

## Why RAG?

Pure LLMs suffer from hallucination, knowledge cut-off, and lack of source attribution. RAG addresses all three: the retriever provides up-to-date, factual evidence, and the generator can cite its sources. This makes RAG the preferred architecture for enterprise knowledge assistants.

## Agentic RAG

Advanced RAG systems decompose complex questions into sub-queries, execute multiple retrieval passes, and reason over the combined evidence. This agentic approach improves answer quality for multi-faceted questions that no single retrieval can fully address.

## The Role of Vector Search

Vector search is the engine inside every RAG system. The quality of retrieval directly determines the quality of the generated answer. GPU-accelerated vector search (like hipVS) enables sub-millisecond retrieval, which is critical when the agent makes multiple retrieval calls per question.
