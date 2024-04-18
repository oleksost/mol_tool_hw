from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from tools import QaAgent, SmilesFilter, SMILESEnergyPredictionTool
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

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
    qa = QaAgent(
        docs, llm, embeddings_provider, vector_path="storage/PMC8264982_10.faiss"
    )
    for q in questions:
        print("Question:", q, "\n Answer:", qa.answer(q))

    smiles_filter = SmilesFilter()
    smiles_tokens = smiles_filter.filter(docs[0].page_content)
    print("SMILES tokens:", smiles_tokens)

    smiles_temp_predictor = SMILESEnergyPredictionTool(
        model_path="storage/_autosklearn_model.pkl"
    )

    for smiles in smiles_tokens:
        print("SMILES:", smiles, "\n Energy:", smiles_temp_predictor.run(smiles))
