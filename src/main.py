from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from agents import QaAgent, SmilesFilterAgent
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # load chat model
    embeddings_provider = OpenAIEmbeddings(model="text-embedding-3-small")
    loader = WebBaseLoader("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8264982/")
    docs = loader.load()  # load parse text

    questions = [
        "What disease are they discussing in the paper?",
        "Summarize the main takeaways from the paper.",
        "Can you list the protein targets they highlight in the paper?",
        "Can you list the small molecule drugs they highlight in the paper?",
    ]
    qa = QaAgent(docs, llm, embeddings_provider, vector_path = "storage/PMC8264982_10.faiss")
    for q in questions:
        print("Question:", q, "\n Answer:", qa.answer(q))

    smiles_filter = SmilesFilterAgent()
    smiles_tokens = smiles_filter.filter(docs[0].page_content)
    print("SMILES tokens:", smiles_tokens)
