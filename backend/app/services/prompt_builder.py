# app/services/prompt_builder.py v1.0.1
"""Responsible for prompt construction used in RAG generation."""
from typing import Optional


def build_rag_prompt(context: str, query: str) -> str:
    """Form an instruction prompt using given context and user query."""
    return f"""You are a precise scientific assistant analyzing technical documents.

INSTRUCTIONS:
1. Answer STRICTLY based on the context below - do not use external knowledge
2. Quote relevant passages when possible to support your answer
3. If information is incomplete, state what's missing
4. If the answer isn't in the context, respond: "Answer not found in the documents."
5. Maintain the question's language in your response

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def build_rag_prompt_advanced(
        context: str,
        query: str,
        metadata: Optional[dict] = None,
) -> str:
    """Enhanced RAG prompt with chain-of-thought reasoning and source tracking."""
    metadata_str = ""
    if metadata:
        metadata_str = "\n".join([
            f"[Source: {m.get('source', 'N/A')}, Chunk: {m.get('chunk_index', '?')}]"
            for m in metadata
    ])
    prompt = f"""... CONTEXT:\n{context}\n\nSOURCES:\n{metadata_str}\n\nQUESTION: ..."""
    
    return f"""You are an expert scientific assistant with strong analytical capabilities.

YOUR TASK:
Answer the user's question using ONLY the provided context. Follow this reasoning process:

STEP 1 - ANALYZE: Identify relevant information in the context
STEP 2 - SYNTHESIZE: Combine information to form a complete answer
STEP 3 - VERIFY: Ensure all claims are supported by the context
STEP 4 - RESPOND: Provide a clear, concise answer

CONTEXT:
{context}
{metadata_str}

QUESTION: {query}

GUIDELINES:
- Use direct quotes with "..." when citing specific facts
- If multiple sources conflict, mention both perspectives
- If information is partial, explain what's present and what's missing
- If no relevant information exists, state: "Answer not found in the documents."
- Match the language of the question
- Be precise - avoid speculation beyond what's written

ANSWER:"""

