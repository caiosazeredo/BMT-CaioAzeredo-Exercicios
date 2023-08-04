import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = f"{SCRIPT_DIR}/.."

import sys
sys.path.append(PROJECT_DIR)

import pandas as pd
from utils import log
from utils import metrics
from typing import Text, List, Dict
from matplotlib import pyplot as plt

class ResultsComparison:
    def __init__(
        self, 
        relevantFilePath: Text, 
        retrievedList: List[Dict], # Each dict element should have the following keys: name and filepath
        storeDir: Text = None
    ):
        self.relevant = pd.read_csv(relevantFilePath, sep = ";").rename(
            columns = {
                "queryNumber": "queryNumber", 
                "docNumber": "documentID", 
                "docVotes": "relevance"
            }
        )
        self.retrievedList = [
            {
                "name": retrieved["name"], 
                "data": pd.read_csv(retrieved["filepath"], sep = ";").drop("rank", axis = 1).rename(
                    columns = {
                        "queryNumber": "queryNumber", 
                        "documentID": "documentID", 
                        "similarity": "score"
                    }
                )
            } for retrieved in retrievedList
        ]
        self.storeDir = storeDir
        self.logger = log.initLogger("EVALUATOR")

    def elevenPoints(self, limit = None):
        self.logger.info(f"Generating eleven points plot (limit = {limit})")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            df, fig = metrics.plotElevenPoints(data, relevant, limit = limit)

            df.to_csv(f"{self.storeDir}/11points-{name}-1.csv", index = False, sep = ";")
            fig.savefig(f"{self.storeDir}/11points-{name}-2.png")
        self.logger.info("Eleven points plot generated and stored with success")

    def f1(self, limit = None):
        self.logger.info(f"Generating F1 score (limit = {limit})")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            df = metrics.getMetricScore(data, relevant, scoreFuncs = [metrics._f1Score], limit = limit)

            filename = f"f1-{name}-1.csv" if limit is None else f"f1At{limit}-{name}-1.csv"
            df.to_csv(f"{self.storeDir}/{filename}", index = False, sep = ";")
        self.logger.info("F1 score generated with success")

    def precision(self, limit = None):
        self.logger.info(f"Generating precision score (limit = {limit})")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            df = metrics.getMetricScore(data, relevant, scoreFuncs = [metrics._precisionScore], limit = limit)

            filename = f"precision-{name}-1.csv" if limit is None else f"precisionAt{limit}-{name}-1.csv"
            df.to_csv(f"{self.storeDir}/{filename}", index = False, sep = ";")
        self.logger.info("Precision score generated with success")

    def recall(self, limit = None):
        self.logger.info(f"Generating recall score (limit = {limit})")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            df = metrics.getMetricScore(data, relevant, scoreFuncs = [metrics._recallScore], limit = limit)

            filename = f"recall-{name}-1.csv" if limit is None else f"recallAt{limit}-{name}-1.csv"
            df.to_csv(f"{self.storeDir}/{filename}", index = False, sep = ";")
        self.logger.info("Recall score generated with success")

    def rPrecisionHistogram(self, firstRetrievedName, secondRetrievedName):
        self.logger.info(f"Generating R-Precision between '{firstRetrievedName}' and '{secondRetrievedName}'")
        names = [retrieved["name"] for retrieved in self.retrievedList]
        if firstRetrievedName not in names or secondRetrievedName not in names:
            raise ValueError(f"R-Precision can only be calculated for the following data: {', '.join(names)}")
        
        firstRetrievedData = self.retrievedList[names.index(firstRetrievedName)]["data"]
        secondRetrievedData = self.retrievedList[names.index(secondRetrievedName)]["data"]

        df, fig = metrics.rPrecisionHistogram(firstRetrievedData, secondRetrievedData, self.relevant)
        plt.ylabel(f"R-Precision {firstRetrievedName}/{secondRetrievedName}")

        filename = f"rPrecision-{firstRetrievedName}-{secondRetrievedName}"

        df.to_csv(f"{self.storeDir}/{filename}-1.csv", index = False, sep = ";")
        fig.savefig(f"{self.storeDir}/{filename}-2.png")
        self.logger.info("R-Precision generated with success")

    def meanAveragePrecision(self):
        self.logger.info(f"Generating MAP score")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            mapScore = metrics.meanAveragePrecision(data, relevant)

            filename = f"map-{name}-1.csv"
            with open(f"{self.storeDir}/{filename}", "w") as f:
                f.write(f"mapScore\n{mapScore}")
        self.logger.info("MAP generated with success")

    def meanReciprocalRank(self, limit = None):
        self.logger.info(f"Generating MRR score")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            mrrScore = metrics.meanReciprocalRank(data, relevant, limit = limit)

            filename = f"mrr-{name}-1.csv" if limit is None else f"mrrAt{limit}-{name}-1.csv"
            with open(f"{self.storeDir}/{filename}", "w") as f:
                f.write(f"mrrScore\n{mrrScore}")
        self.logger.info("MRR generated with success")

    def discountedCumulativeGain(self, limit = None):
        self.logger.info(f"Generating DCG score (limit = {limit})")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            df, fig = metrics.discountedCumulativeGain(data, relevant, limit = limit)
            df = pd.Series(df)

            filename = f"dcg-{name}" if limit is None else f"dcgAt{limit}-{name}"
            df.to_csv(f"{self.storeDir}/{filename}-1.csv", index = False, sep = ";")
            fig.savefig(f"{self.storeDir}/{filename}-2.png")
        self.logger.info("DCG generated with success")

    def normalizedDiscountedCumulativeGain(self, limit = None):
        self.logger.info(f"Generating NDCG score (limit = {limit})")
        for retrieved in self.retrievedList:
            name = retrieved["name"]
            data = retrieved["data"]
            relevant = self.relevant

            df = metrics.normalizedDiscountedCumulativeGain(data, relevant, limit = limit)
            df = pd.Series(df)

            filename = f"ndcg-{name}" if limit is None else f"ndcgAt{limit}-{name}"
            df.to_csv(f"{self.storeDir}/{filename}-1.csv", index = False, sep = ";")
        self.logger.info("NDCG generated with success")