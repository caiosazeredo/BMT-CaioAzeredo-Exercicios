import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = f"{SCRIPT_DIR}/.."

import sys
sys.path.append(PROJECT_DIR)

from typing import Text
from xml.dom import minidom
from utils.textProcessing import textPreprocessingFunc
from utils import log

class QueryProcessor:
    def __init__(
            self, 
            queriesFilePath: Text,
            processedQueriesFilePath: Text,
            expectedResultsFilePath: Text
        ):
        self.queriesFilePath = queriesFilePath
        self.processedQueriesFilePath = processedQueriesFilePath
        self.expectedResultsFilePath = expectedResultsFilePath
        self.queries = None
        self.logger = log.initLogger("QUERY_PROCESSOR")
    
    def parseQueries(self):
        dataDOM = minidom.parse(self.queriesFilePath)

        columns = ["queryNumber", "queryText", "queryResults"]
        data = []

        queries = dataDOM.getElementsByTagName("QUERY")
        for query in queries:
            queryNumber = query.getElementsByTagName("QueryNumber")[0].firstChild.data
            queryText = query.getElementsByTagName("QueryText")[0].firstChild.data
            queryResults = [
                {
                    "resultDoc": resultItem.firstChild.data,
                    "resultScores": resultItem.getAttribute("score") 
                } for resultItem in query.getElementsByTagName("Item")
            ]
            data.append([queryNumber, queryText, queryResults])

        self.queries = pd.DataFrame(data = data, columns = columns)

    
    def preprocessQueries(self):
        self.queries.loc[:, "queryText"] = self.queries.loc[:, "queryText"].apply(textPreprocessingFunc)

    def storeQueries(self):
        dataToStore = self.queries.loc[:, ["queryNumber", "queryText"]]
        dataToStore.to_csv(self.processedQueriesFilePath, index = False, sep = ";")

    def storeExpectedResults(self):
        dataToStore = self.queries.explode("queryResults")
        dataToStore[["docNumber", "docVotes"]] = dataToStore.loc[:,"queryResults"].apply(pd.Series)
        dataToStore["docVotes"] = dataToStore.loc[:, "docVotes"].apply(lambda scores: len(scores.replace("0","")))
        dataToStore = dataToStore.loc[:,["queryNumber", "docNumber", "docVotes"]]
        dataToStore.to_csv(self.expectedResultsFilePath, index = False, sep = ";")

    def _run(self):
        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Loading queries and expected results",
            onFinishMessage = "Queries and expected results were loaded with success",
            onErrorMessage = "Error while loading queries",
            func = self.parseQueries
        )
        self.logger.info(f"Total Queries: {self.queries.shape[0]}")

        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Preprocessing queries",
            onFinishMessage = "Queries were preprocessed with success",
            onErrorMessage = "Error while preprocessing queries",
            func = self.preprocessQueries
        )

        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Storing preprocessed queries",
            onFinishMessage = "Preprocessed queries were stored with success",
            onErrorMessage = "Error while storing preprocessed queries",
            func = self.storeQueries
        )
        self.logger.info(f"Total stored queries: {self.queries.shape[0]}")
        
        log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Storing expected results",
            onFinishMessage = "Expected results were stored with success",
            onErrorMessage = "Error while storing expected results",
            func = self.storeExpectedResults
        )

    def run(self):
        log.executeModule(self.logger, self._run)