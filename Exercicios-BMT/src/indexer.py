import os
import ast
import pandas as pd
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = f"{SCRIPT_DIR}/.."

import sys
sys.path.append(PROJECT_DIR)

from typing import Text, List
from xml.dom import minidom
from utils.textProcessing import vectorizeText
from utils import log
from src.model import TermDocumentMatrix
from utils.weight import StandardTFIDF

class InvertedListGenerator:
    def __init__(
            self, 
            documentFilePathList: List[Text],
            invertedListFilePath: Text,
            useStemmer: bool = False
        ):
        self.documentFilePathList = documentFilePathList
        self.invertedListFilePath = invertedListFilePath
        self.useStemmer = useStemmer
        self.documentsData = []
        self.logger = log.initLogger("INVERTED_LIST_GENERATOR")

    def parseDocument(self, documentFilePath):
        dataDOM = minidom.parse(documentFilePath)

        data = []
        
        records = dataDOM.getElementsByTagName("RECORD")
        for record in records:
            recordNum = record.getElementsByTagName("RECORDNUM")[0].firstChild.data.strip()
            abstract = record.getElementsByTagName("ABSTRACT")
            extract = record.getElementsByTagName("EXTRACT")
            if abstract != []:
                abstract = abstract[0].firstChild.data
            elif extract != []:
                abstract = extract[0].firstChild.data
            else: 
                abstract = None
            
            data.append([recordNum, abstract])

        return data

    def parseCorpus(self):
        self.documentsData = []
        for documentFilePath in self.documentFilePathList:
            self.documentsData += self.parseDocument(documentFilePath)
        self.documentsData = pd.DataFrame(self.documentsData, columns = ["recordNum", "abstract"])

    def preprocessDocuments(self):
        self.documentsData = self.documentsData.dropna()
        self.documentsData["abstract"] = self.documentsData["abstract"].apply(
            lambda text: vectorizeText(text, self.useStemmer)
        )

    def generateInvertedList(self):
        self.documentsData = self.documentsData.explode("abstract")
        self.documentsData = self.documentsData.groupby("abstract").agg(lambda group: list(group)).reset_index()
        self.documentsData.columns = ["term", "documentIDList"]
        self.documentsData = self.documentsData.sort_values("term")
        self.documentsData = self.documentsData.dropna()

    def storeInvertedList(self):
        self.documentsData.to_csv(self.invertedListFilePath, index = False, sep = ";")

    def _run(self):
        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Loading documents",
            onFinishMessage = "Documents were loaded with success",
            onErrorMessage = "Error while loading documents",
            func = self.parseCorpus
        )
        self.logger.info(f"Total Documents: {self.documentsData.shape[0]}")

        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Preprocessing documents",
            onFinishMessage = "Documents were preprocessed with success",
            onErrorMessage = "Error while preprocessing documents",
            func = self.preprocessDocuments
        )

        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Generating inverted list",
            onFinishMessage = "Inverted list was generated with success",
            onErrorMessage = "Error while generating inverted list",
            func = self.generateInvertedList
        )
        self.logger.info(f"Total Terms: {self.documentsData.shape[0]}")

        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Storing inverted list",
            onFinishMessage = "Inverted list was stored with success",
            onErrorMessage = "Error while storing inverted list",
            func = self.storeInvertedList
        )

    def run(self):
        log.executeModule(self.logger, self._run)

class Indexer:
    def __init__(
        self, 
        invertedListFilePath: Text,
        indexesFilePath: Text
    ):
        self.invertedListFilePath = invertedListFilePath
        self.indexesFilePath = indexesFilePath
        self.logger = log.initLogger("INDEXER")

    def processInvertedList(self) -> pd.DataFrame:
        # Generating Statistics
        invertedList = pd.read_csv(self.invertedListFilePath, sep = ";").dropna()
        invertedList.documentIDList = invertedList.documentIDList.apply(ast.literal_eval)

        invertedList["documentCount"] = invertedList.documentIDList.apply(len)
        invertedList.documentIDList = invertedList.documentIDList.apply(
            lambda documents: pd.DataFrame(
                data = pd.Series(documents).value_counts().to_dict().items(),
                columns = ["documentID", "termCount"]
            ).set_index("documentID")
        )
        invertedList = invertedList.set_index("term")
        
        # Preprocessing the terms
        ## Filtering terms with only letters
        invertedList = invertedList[pd.Series(invertedList.index).apply(lambda term: term.isalpha()).values]

        ## Filtering terms with 2 or more letters
        invertedList = invertedList[(pd.Series(invertedList.index).apply(len) >= 2).values]

        ## Uppercasing terms
        invertedList.index = pd.Series(invertedList.index).apply(str.upper)

        return invertedList
    
    def createTermDocumentMatrix(self, invertedList):
        termDocumentMatrix = TermDocumentMatrix(invertedList = invertedList, weightCalculator = StandardTFIDF)
        pickle.dump(termDocumentMatrix, open(self.indexesFilePath, "wb"))

    def _run(self):
        processedInvertedList = log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Loading and processing inverted list",
            onFinishMessage = "Inverted list was loaded and processed with success",
            onErrorMessage = "Error while loading and processing inverted list",
            func = self.processInvertedList
        )
        self.logger.info(f"Total Terms: {processedInvertedList.shape[0]}")

        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Generating and storing model",
            onFinishMessage = "Model was generated and stored with success",
            onErrorMessage = "Error while generating and storing model",
            func = self.createTermDocumentMatrix,
            invertedList = processedInvertedList
        )
        
    def run(self):
        log.executeModule(self.logger, self._run)