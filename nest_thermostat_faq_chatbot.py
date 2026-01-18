import os
import sys
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


PRODUCT_NAME = "Google Nest Thermostat"

SOURCE_URLS = [
    # Change temperature / use thermostat
    "https://support.google.com/googlenest/answer/9243487?hl=en",  # (thermostat controls / adjust temp)
    # Thermostat controls & navigation
    "https://support.google.com/googlenest/answer/9201565?hl=en",  # (use thermostat / menu controls)
    # Schedules
    "https://support.google.com/googlenest/answer/9243489?hl=en",
]


PERSIST_DIR = "./chroma_nest_db"

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)


def load_documents(urls: List[str]):
    loader = WebBaseLoader(urls)
    return loader.load()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_documents(docs)


def build_or_load_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )


def retrieve_docs(retriever, question: str):
    """
    Works with both older and newer retriever interfaces.
    """
    # Newer interface
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)

    # Older interface
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)

    raise RuntimeError("Retriever does not support invoke() or get_relevant_documents().")


def main():
    require_openai_key()

    print(f"\n=== {PRODUCT_NAME} FAQ Chatbot ===")
    print("Loading source documents...")

    docs = load_documents(SOURCE_URLS)
    print(f"Loaded {len(docs)} web documents.")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    vectordb = build_or_load_vectorstore(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    print("Vector store ready.")

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an FAQ chatbot for {product}. "
         "Answer ONLY using the provided context. "
         "If the answer is not in the context, say: "
         "'I’m not sure based on the available product information.' "
         "Be concise and practical. Use steps when helpful."),
        ("human",
         "Question: {question}\n\n"
         "Context:\n{context}")
    ]).partial(product=PRODUCT_NAME)

    print("\nAsk a question (type 'exit' to quit).")

    while True:
        user_q = input("\nYou: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_q:
            continue

        context_docs = retrieve_docs(retriever, user_q)
        context_text = "\n\n---\n\n".join([d.page_content for d in context_docs])

        messages = prompt.format_messages(question=user_q, context=context_text)
        response = llm.invoke(messages)

        print("\nBot:", response.content.strip())

        if context_docs:
            print("\nSources used:")
            for i, d in enumerate(context_docs[:3], start=1):
                print(f"  {i}. {d.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
