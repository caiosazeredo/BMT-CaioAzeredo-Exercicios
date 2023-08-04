import os
SCRIPT_DIR = "/home/cainafpereira/UFRJ/COS738-BMT/exercises/vectorSpaceModel/src"
PROJECT_DIR = os.path.normpath(f"{SCRIPT_DIR}/..")

import sys
sys.path.append(PROJECT_DIR)

import pickle
import numpy as np
import pandas as pd
from typing import Text, List
from tqdm import tqdm
from utils.textProcessing import vectorizeText
from utils import log

class Searcher:
    def __init__(
        self,
        modelFilePath: Text, 
        queriesFilePath: Text,
        resultsFilePath: Text,
        useStemmer: bool = False
    ) -> None:
        self.modelFilePath = modelFilePath
        self.queriesFilePath = queriesFilePath
        self.resultsFilePath = resultsFilePath
        self.useStemmer = useStemmer
        self.model = None
        self.queries = None
        self.logger = log.initLogger("SEARCHER")

    def loadModel(self):
        model = pickle.load(open(self.modelFilePath, "rb"))
        return model
    
    def loadQueries(self):
        queries = pd.read_csv(self.queriesFilePath, sep = ";")
        return queries  
    
    def searchFromQuery(self, query: Text, limit = None, simThreshold = None):
        if (limit is not None) and simThreshold is not None:
            raise ValueError("limit and simThreshold can not be set at the same time.")
        queryTerms = vectorizeText(query, self.useStemmer)
        queryTerms = pd.Series(queryTerms).apply(str.upper).unique()
        queryTerms = self.model.filterQueryTerms(queryTerms)
        similarities = []
        for queryTerm in queryTerms:
            documentIDs = self.model.filterDocumentsByQueryTerms(queryTerms)
            for documentID in documentIDs:
                weightQueryTermInDocumentID = self.model.getWeight(documentID, queryTerm, normalized = True)
                similarities.append([documentID, weightQueryTermInDocumentID])
        similarities = pd.DataFrame(data = similarities, columns = ["documentID", "similarity"])
        similarities = similarities.groupby("documentID").sum().reset_index()
        similarities = similarities.sort_values("similarity", ascending = False).reset_index(drop = True)
        similarities["rank"] = similarities.index + 1
        if limit:
            similarities = similarities.iloc[:limit]
        if simThreshold:
            similarities = similarities[similarities.similarity >= simThreshold]
        return similarities

    def runQueries(self, limit = None, simThreshold = None):
        results = []
        for i in tqdm(self.queries.index, desc = "Running queries..."):
            row = self.queries.loc[i]
            query = row.queryText
            number = row.queryNumber
            queryResults = self.searchFromQuery(query, limit = limit, simThreshold = simThreshold)
            queryResults["queryNumber"] = number
            queryResults = queryResults[["queryNumber", "rank", "documentID", "similarity"]]
            results.append(queryResults)
        results = pd.concat(results)
        return results

    def _run(self):
        self.model = log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Loading model",
            onFinishMessage = "Model was loaded with success",
            onErrorMessage = "Error while loading model",
            func = self.loadModel
        )

        self.queries = log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Loading queries",
            onFinishMessage = "Queries were loaded with success",
            onErrorMessage = "Error while loading queries",
            func = self.loadQueries
        )
        self.logger.info(f"Total Queries: {self.queries.shape[0]}")

        results = log.executeFunction(
            logger = self.logger, 
            onStartMessage = "Running queries",
            onFinishMessage = "All queries were executed with success",
            onErrorMessage = "Error while running queries",
            func = self.runQueries,
        )

        self.logger.info("Storing results")
        results.to_csv(self.resultsFilePath, index = False, sep = ";")
        self.logger.info("Results were stored with success")

    def run(self):
        log.executeModule(self.logger, self._run)