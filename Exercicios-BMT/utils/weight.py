import numpy as np
from abc import ABC, abstractmethod

class WeightCalculator(ABC):
    def __init__(self, invertedList):
        self.invertedList = invertedList
        self.documentIDs = self.getDocumentIDs()

    def getDocumentIDs(self):
        documentIDs = self.invertedList.documentIDList.apply(
            lambda document: list(document.index)
        ).explode().unique()
        return set(documentIDs)

    def getTermCountInDocument(self, documentID, term):
        termCount = self.invertedList.loc[term].documentIDList.loc[documentID].termCount
        return termCount

    def getDocumentCountForTerm(self, term):
        documentCount = self.invertedList.loc[term].documentCount
        return documentCount

    def calculateDocumentWeightLengths(self):
        documentWeights = {}
        for term in self.invertedList.index:
            documents = self.invertedList.loc[term].documentIDList
            for documentID in documents.index:
                weight = self.getWeight(documentID, term, normalized = False)
                if documentID in documentWeights.keys():
                    documentWeights[documentID] += weight**2
                else:
                    documentWeights[documentID] = weight**2
        documentWeights = {documentID: np.sqrt(sumSquaredWeights) for documentID, sumSquaredWeights in documentWeights.items()}
        return documentWeights

    @abstractmethod
    def weightFunction(self, documentID, term):
        pass

    def getWeight(self, documentID, term, normalized = False):
        if term not in self.invertedList.index:
            raise Exception(f"Invalid term: the term {term} does not exist.")
        if documentID not in self.documentIDs:
            raise Exception(f"Invalid document ID: the document {documentID} does not exist.")
        try:
            weight = self.weightFunction(documentID, term)
            if normalized:
                weight = weight/self.documentWeightLengths[documentID]
            return weight
        except:
            return 0

class StandardTFIDF(WeightCalculator):
    def __init__(self, invertedList):
        super(StandardTFIDF, self).__init__(invertedList)
        self.totalDocuments = self.calculateNumberOfDocuments()
        self.maxTermCount = self.calculateMaxTermCount()
        self.documentWeightLengths = self.calculateDocumentWeightLengths()

    def weightFunction(self, documentID, term):
        termCount = self.getTermCountInDocument(documentID, term)
        documentCount = self.getDocumentCountForTerm(term)
        maxTermCount = self.maxTermCount
        totalDocuments = self.totalDocuments

        tf = termCount/maxTermCount
        idf = np.log(totalDocuments/documentCount)

        weight = tf*idf

        return weight

    def calculateNumberOfDocuments(self):
        documentsIDs = self.invertedList.documentIDList.apply(lambda document: document.index).explode().unique()
        totalDocuments = len(documentsIDs)
        return totalDocuments
    
    def calculateMaxTermCount(self):
        maxTermCount = self.invertedList.documentIDList.apply(lambda document: list(document.termCount)).explode().max()
        return maxTermCount