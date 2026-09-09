from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import PubMedRetriever
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ============================================================
# 2. PUBMED RETRIEVER
# ============================================================

pubmed_retriever = PubMedRetriever(
    top_k_results=5,
    doc_content_chars_max=4000
)


# ============================================================
# 3. YOUTUBE
# ============================================================

youtube_url = "https://www.youtube.com/WATCHID"

youtube_loader = YoutubeLoader.from_youtube_url(
    youtube_url,
    add_video_info=False
)

youtube_documents = youtube_loader.load()

print("YouTube transcript loaded.")
print("Number of YouTube documents:", len(youtube_documents))


# ============================================================
# 4. SPLIT YOUTUBE TRANSCRIPT
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

youtube_chunks = text_splitter.split_documents(
    youtube_documents
)

print("Number of YouTube chunks:", len(youtube_chunks))


# ============================================================
# 5. YOUTUBE VECTOR STORE
# ============================================================

embeddings = OpenAIEmbeddings()

youtube_vector_store = FAISS.from_documents(
    youtube_chunks,
    embeddings
)


# ============================================================
# 6. YOUTUBE RETRIEVER
# ============================================================

youtube_retriever = youtube_vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# 7. TAVILY SEARCH
# ============================================================

tavily_retriever = TavilySearch(
    max_results=5
)


# ============================================================
# 8. DUCKDUCKGO SEARCH
# ============================================================

duckduckgo_retriever = DuckDuckGoSearchRun()

# ============================================================
# USER QUERY
# ============================================================

question = input("Enter your search query: ")


# ============================================================
# SEARCH DUCKDUCKGO
# ============================================================

print("\nSearching DuckDuckGo...")

duckduckgo_result = duckduckgo_retriever.invoke(question)

print("\n" + "=" * 70)
print("DUCKDUCKGO SEARCH RESULT")
print("=" * 70)

print(duckduckgo_result)

# ============================================================
# 9. FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    if not docs:
        return "No relevant information was retrieved."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# 10. PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a helpful research assistant.

Answer the user's question ONLY using the provided context.

Do not invent information.

If the provided context is insufficient, say:
"I don't know."

Context:
{context}

Question:
{question}
""",
    input_variables=[
        "context",
        "question"
    ]
)


# ============================================================
# 11. OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# 12. CHOOSE SOURCE
# ============================================================

source = "duckduckgo"

# Available options:
#
# source = "pubmed"
# source = "youtube"
# source = "tavily"
# source = "duckduckgo"


# ============================================================
# 13. QUESTION
# ============================================================

question = "cancer radiation exposure"


# ============================================================
# 14. RETRIEVE FROM SELECTED SOURCE
# ============================================================

if source == "pubmed":

    print("\nSearching PubMed...")

    retrieved_docs = pubmed_retriever.invoke(
        question
    )


elif source == "youtube":

    print("\nSearching YouTube transcript...")

    retrieved_docs = youtube_retriever.invoke(
        question
    )


elif source == "tavily":

    print("\nSearching the internet using Tavily...")

    tavily_results = tavily_retriever.invoke(
        {"query": question}
    )

    # Tavily returns search results rather than
    # LangChain Documents, so convert them.

    retrieved_docs = []

    for result in tavily_results:

        retrieved_docs.append(
            type(
                "Document",
                (),
                {
                    "page_content": (
                        result.get("content", "")
                    ),
                    "metadata": {
                        "title": result.get("title", ""),
                        "url": result.get("url", "")
                    }
                }
            )()
        )


elif source == "duckduckgo":

    print("\nSearching the internet using DuckDuckGo...")

    duckduckgo_result = duckduckgo_retriever.invoke(
        question
    )

    # Convert DuckDuckGo result into a document

    retrieved_docs = [
        type(
            "Document",
            (),
            {
                "page_content": duckduckgo_result,
                "metadata": {
                    "source": "DuckDuckGo"
                }
            }
        )()
    ]


else:

    raise ValueError(
        "source must be "
        "'pubmed', 'youtube', 'tavily', or 'duckduckgo'"
    )


# ============================================================
# 15. SHOW RETRIEVED DOCUMENTS
# ============================================================

print("\n" + "=" * 70)
print("RETRIEVAL RESULT")
print("=" * 70)

print(
    f"Number of documents retrieved: "
    f"{len(retrieved_docs)}"
)


for i, doc in enumerate(retrieved_docs):

    print("\n" + "-" * 70)
    print(f"DOCUMENT {i + 1}")
    print("-" * 70)

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# ============================================================
# 16. FORMAT CONTEXT
# ============================================================

context = format_docs(
    retrieved_docs
)


# ============================================================
# 17. COMPLETE RAG CHAIN
# ============================================================

main_chain = (
    prompt
    | llm
    | parser
)


# ============================================================
# 18. GENERATE ANSWER
# ============================================================

response = main_chain.invoke(
    {
        "context": context,
        "question": question
    }
)


# ============================================================
# 19. FINAL ANSWER
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(response)