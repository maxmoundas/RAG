import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import Document as LangchainDocument
from langchain.prompts import ChatPromptTemplate
import streamlit as st


class LLMInterface:
    """Handles LLM operations for the RAG system."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM with OpenAI API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY not found in environment variables")
            return

        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key,
            )
            st.success(f"LLM initialized with model: {self.model_name}")
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            self.llm = None

    def generate_answer(
        self, query: str, context_docs: List[LangchainDocument]
    ) -> Dict[str, Any]:
        """Generate an answer based on the query and retrieved context."""
        if not self.llm:
            return {
                "answer": "LLM not initialized. Please check your OpenAI API key.",
                "sources": [],
                "error": True,
            }

        if not context_docs:
            return {
                "answer": "No relevant documents found to answer your question.",
                "sources": [],
                "error": False,
            }

        try:
            # Prepare context from documents
            context = "\n\n".join([doc.page_content for doc in context_docs])

            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a helpful AI assistant that answers questions based on the provided context. 
                Always answer based on the information given in the context. If the context doesn't contain enough 
                information to answer the question, say so. Be concise but thorough in your responses.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:""",
                    ),
                    ("human", "{question}"),
                ]
            )

            # Create chain
            chain = prompt_template | self.llm

            # Generate response
            response = chain.invoke({"context": context, "question": query})

            # Extract sources from context documents
            sources = []
            for doc in context_docs:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)

            return {"answer": response.content, "sources": sources, "error": False}

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "error": True,
            }

    def generate_follow_up_questions(self, query: str, answer: str) -> List[str]:
        """Generate follow-up questions based on the original query and answer."""
        if not self.llm:
            return []

        try:
            prompt = f"""
            Based on the following question and answer, generate 3 relevant follow-up questions that would help explore the topic further.
            
            Original Question: {query}
            Answer: {answer}
            
            Generate 3 follow-up questions (one per line):
            """

            response = self.llm.invoke(prompt)

            # Parse the response to extract questions
            questions = []
            lines = response.content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if (
                    line
                    and not line.startswith("Original Question:")
                    and not line.startswith("Answer:")
                ):
                    # Remove numbering if present
                    if line[0].isdigit() and line[1] in [".", ")", ":"]:
                        line = line[2:].strip()
                    questions.append(line)

            return questions[:3]  # Return max 3 questions

        except Exception as e:
            st.error(f"Error generating follow-up questions: {str(e)}")
            return []

    def is_available(self) -> bool:
        """Check if the LLM is available and properly configured."""
        return self.llm is not None
