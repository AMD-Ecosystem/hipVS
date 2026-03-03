# Large Language Models

Large Language Models (LLMs) are neural networks with billions of parameters trained on vast text corpora. They can generate coherent text, answer questions, summarise documents, write code, and perform multi-step reasoning.

## Training and Inference

LLMs are typically pre-trained with a next-token prediction objective on trillions of tokens, then fine-tuned (or aligned via RLHF) for specific use-cases like chat, instruction-following, or tool use. Inference requires significant compute; quantisation and batching are used to reduce cost.

## Limitations

Despite their capabilities, LLMs can hallucinate (produce plausible but false claims), lack access to private or recent data, and cannot reliably cite sources. These limitations motivate the use of retrieval-augmented generation, where an external knowledge base provides grounded evidence.

## LLMs in RAG Pipelines

In a RAG pipeline, the LLM plays two roles: (1) decomposing complex queries into focused sub-queries (agentic reasoning), and (2) synthesising a final answer from retrieved passages. The retriever (powered by vector search) provides the factual foundation.
