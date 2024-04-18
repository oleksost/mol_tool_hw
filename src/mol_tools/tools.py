import os
import pickle
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import backoff
from rdkit import Chem
import concurrent.futures
from typing import Optional, Type
from autosklearn.regression import AutoSklearnRegressor
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from molfeat.calc import FPCalculator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging


class SMILESEnergyPredictionTool(BaseTool):
    """
    A tool for predicting the hydration free energy of drug molecules based on their SMILES representation.

    Attributes:
        name (str): The name of the tool.
        description (str): A description of the tool and its purpose.
        model (AutoSklearnRegressor): The machine learning model used for prediction.

    Methods:
        __init__(self, model_path: str) -> None: Initializes the tool with a trained model.
        _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str: Runs the tool synchronously.
        _preprocess(self, query: str) -> str: Preprocesses the input query.
        _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str: Runs the tool asynchronously.
    """

    name = "SMILES hydration free energy predictor"
    description = """The calculation of hydration free energy is an important aspect of drug discovery, 
                    as it helps in understanding how a potential drug molecule interacts with water, which is crucial 
                    for its absorption, distribution, metabolism, and excretion (ADME) properties."""
    model: AutoSklearnRegressor

    def __init__(self, model_path: str) -> None:
        """
        Initializes the SMILESEnergyPredictionTool with a trained model.

        Args:
            model_path (str): The path to the trained model file.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with open(model_path, "rb") as f:
            model: AutoSklearnRegressor = pickle.load(f)
        super().__init__(model=model)

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Runs the SMILESEnergyPredictionTool synchronously.

        Args:
            query (str): The input query in SMILES format.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for tool run events.

        Returns:
            str: The predicted hydration free energy.

        """
        input_x = self._preprocess(query)
        return self.model.predict(input_x)[0]

    def _preprocess(self, query: str) -> str:
        """
        Preprocesses the input query.

        Args:
            query (str): The input query in SMILES format.

        Returns:
            str: The preprocessed input.

        """
        calc = FPCalculator("ecfp")
        input_x = pd.DataFrame(calc(query)[None, :])
        input_x.columns = [f"{i}" for i in range(input_x.shape[1])]
        return input_x
    
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """
        Runs the SMILESEnergyPredictionTool asynchronously.

        Args:
            query (str): The input query in SMILES format.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The callback manager for tool run events.

        Returns:
            str: The predicted hydration free energy.

        Raises:
            NotImplementedError: This method does not support asynchronous execution.

        """
        raise NotImplementedError("custom_search does not support async")


class QaAgent:
    """
    Question-Answering Agent that uses a language model and document embeddings for retrieval-based QA.

    Args:
        documents (list): List of documents to be used for retrieval.
        llm (LanguageModel): Language model used for generating responses.
        embeddings_provider (EmbeddingsProvider): Embeddings provider used for document embeddings.
        n_docs (int, optional): Number of documents to be used for retrieval. Defaults to 10.
    """

    def __init__(
        self, documents, llm, embeddings_provider, doc_limit=None, vector_path=None
    ) -> None:
        # check if file exists
        if vector_path and os.path.exists(vector_path):
            self.vector = FAISS.load_local(
                vector_path, embeddings_provider, allow_dangerous_deserialization=True
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(documents)
            if doc_limit:
                documents = documents[:doc_limit]
            self.vector = FAISS.from_documents(documents, embedding=embeddings_provider)
            if vector_path:
                self.vector.save_local(vector_path)
                
        retriever_chain = self.vector.as_retriever()
        
        prompt_template = """Answer the following question based only on the following context:
                            <context>
                            {context}
                            </context>
                            Question: {input}"""
        qa_prompt = ChatPromptTemplate.from_template(prompt_template)
        
        
        # if we wanted to build a chatbot here, we could use MessagesPlaceholder in the qa prompt
        # but we dont realy need to do that here
        # SYSTEM_TEMPLATE = """
        # Answer the user's questions based on the below context. 
        # If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

        # <context>
        # {context}
        # </context>
        # """
        # qa_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             SYSTEM_TEMPLATE,
        #         ),
        #         MessagesPlaceholder(variable_name="input"),
        #     ]
        # )
        qa_chain = create_stuff_documents_chain(
            llm, qa_prompt
        )  # chain the LLM to the prompt
        # could also use langchain.chains.qstion_answering.load_qa_chain
        # it would have a predefined prompt template
        retriever_chain = self.vector.as_retriever()
        self.qa_chain_w_retriver = create_retrieval_chain(retriever_chain, qa_chain)

    def answer(self, question):
        """
        Answer a given question based on the provided context.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer to the question.
        """
        response = self.qa_chain_w_retriver.invoke({"input": question})
        return response["answer"]


class SmilesFilter:
    """A class for filtering SMILES strings."""

    def __init__(self):
        pass

    def _is_valid_smiles(self, token):
        """Check if the token is a valid SMILES string."""
        return token if Chem.MolFromSmiles(token) else None

    def filter(self, text):
        """Filter out invalid SMILES strings from the given text.

        Args:
            text (str): The input text containing SMILES strings.

        Returns:
            list: A list of valid SMILES strings.
        """
        tokens = text.split(" ")
        # remove all tokens that are numeric of only one character
        tokens = [token for token in tokens if not token.isnumeric() and len(token) > 1]        
        smiles_tokens = []
        logging.info(f"Processing {len(tokens)} tokens")

        # Use a multiprocessing pool to process tokens in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Map the function over all tokens and filter out None results
            results = executor.map(self._is_valid_smiles, tokens)
            smiles_tokens = [result for result in results if result is not None]

        return smiles_tokens
