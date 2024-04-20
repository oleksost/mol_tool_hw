import pytest
from langchain.llms.fake import FakeListLLM
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from src.mol_tools.tools import SmilesFilter, QaAgent, RetriverTool


def test_qa_agent():

    llm = FakeListLLM(name="fakeGPT", responses=["I am fine, thank you!"])

    embeddings_provider = FakeEmbeddings(size=10)
    loader = WebBaseLoader("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8264982/")
    docs = loader.load()  # load parse text    

    retriever_tool = RetriverTool(
        docs, embeddings_provider, doc_limit=1
    )

    qa = QaAgent(retriever_tool, llm)
    answer = qa.answer("Test question?")

    assert answer == "I am fine, thank you!"


def test_smiles_filter_agent():

    smiles_filter = SmilesFilter()
    smiles_tokens = smiles_filter.filter("blabla bla ADFAD C1=CC=CC=C1")
    assert smiles_tokens == ["C1=CC=CC=C1"]


if __name__ == "__main__":
    pytest.main([__file__])
