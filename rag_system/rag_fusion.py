from typing import List, Tuple
from .llm_interface import LLMInterface
from .vector_store import VectorStore
from langchain.schema import Document as LangchainDocument
import streamlit as st
from config import Config


class RAGFusion:
    """
    RAG Fusion system that generates multiple query variations using an LLM and fuses the retrieval results.
    """

    def __init__(self, llm_interface: LLMInterface, vector_store: VectorStore):
        self.llm_interface = llm_interface
        self.vector_store = vector_store

    def generate_query_variations(self, original_query: str) -> List[str]:
        """
        Generate multiple query variations using the LLM. Fallback to simple variations if LLM is not available.
        """
        if not self.llm_interface.is_available():
            return self._generate_simple_variations(original_query)

        prompt = (
            f'Given the user\'s question: "{original_query}"\n'
            "Generate 5 different ways to ask the same question or search for the same information.\n"
            "Consider synonyms, different phrasings, broader/narrower interpretations, and technical/non-technical language.\n"
            "Return only the variations, one per line, without numbering or explanations.\n"
            "Each variation should be a complete, searchable query."
        )
        try:
            response = self.llm_interface.llm.invoke(prompt)
            variations_text = response.content.strip()
            variations = [
                var.strip() for var in variations_text.split("\n") if var.strip()
            ]
            all_variations = [original_query] + variations
            return all_variations[:6]  # original + 5 variations
        except Exception as e:
            st.warning(
                f"Error generating query variations: {str(e)}. Using fallback method."
            )
            return self._generate_simple_variations(original_query)

    def _generate_simple_variations(self, query: str) -> List[str]:
        """
        Generate simple query variations as fallback when LLM is not available.
        """
        variations = [query]
        import re

        key_terms = re.findall(r"\b\w{4,}\b", query.lower())
        for term in key_terms[:3]:
            if len(term) > 4:
                variations.append(term)
        return variations

    def search_with_fusion(
        self, query: str, k: int = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform RAG Fusion search using multiple query variations.
        """
        if not self.vector_store.is_ready():
            st.error("Vector store is not ready")
            return []
        try:
            query_variations = self.generate_query_variations(query)
            if getattr(Config, "USE_RAG_FUSION_DEBUG", False):
                st.info(
                    f"Generated {len(query_variations)} query variations for: '{query}'"
                )
                for i, variation in enumerate(query_variations):
                    st.write(f"Variation {i+1}: {variation}")
            all_results = []
            for variation in query_variations:
                try:
                    results = self.vector_store.advanced_similarity_search(
                        variation, k=k
                    )
                    all_results.extend(results)
                except Exception as e:
                    st.warning(
                        f"Error searching with variation '{variation}': {str(e)}"
                    )
                    continue
            combined_results = self._combine_and_rank_results(all_results)
            max_chunks = Config.MAX_CHUNKS_TO_RETURN
            return combined_results[:max_chunks]
        except Exception as e:
            st.error(f"Error in RAG Fusion search: {str(e)}")
            return []

    def _combine_and_rank_results(
        self, all_results: List[Tuple[LangchainDocument, float]]
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Combine results from multiple queries and rank them by a fusion score.
        """
        doc_scores = {}
        for doc, score in all_results:
            content_hash = hash(doc.page_content)
            if content_hash not in doc_scores:
                doc_scores[content_hash] = {
                    "doc": doc,
                    "scores": [score],
                    "max_score": score,
                    "count": 1,
                }
            else:
                doc_scores[content_hash]["scores"].append(score)
                doc_scores[content_hash]["max_score"] = max(
                    doc_scores[content_hash]["max_score"], score
                )
                doc_scores[content_hash]["count"] += 1
        fusion_results = []
        for data in doc_scores.values():
            avg_score = sum(data["scores"]) / len(data["scores"])
            frequency_bonus = min(
                data["count"] / len(doc_scores), 0.3
            )  # Cap at 30% bonus
            fusion_score = (
                (data["max_score"] * 0.5) + (avg_score * 0.3) + (frequency_bonus * 0.2)
            )
            fusion_results.append((data["doc"], fusion_score))
        fusion_results.sort(key=lambda x: x[1], reverse=True)
        return fusion_results
