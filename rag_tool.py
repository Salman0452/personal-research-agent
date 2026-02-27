import os 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings

load_dotenv()

def load_rag_tool():
    """
    Loads ChromaDB and returns a search function
    that the agent can use as a tool
    """

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    def search_company_documents(query: str) -> str:
        """Search company HR documents and return relevant information"""

        # Get top 4 relevant chunks
        docs = vectorstore.similarity_search(query, k=4)

        if not docs:
            return "No relevant information found in company documents."
        
        # Format results cleanly for the agent to read
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("doc_name", "Unknown")
            page = doc.metadata.get("page", "?")
            results.append(
                f"[Source: {source}, Page{page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(results)
    
    return search_company_documents

# Quick test
if __name__ == "__main__":
    rag_search = load_rag_tool()
    result = rag_search("What is the employee relocation policy?")
    print(result)



