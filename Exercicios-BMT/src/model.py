from utils.weight import WeightCalculator, StandardTFIDF
from typing import List, Dict

class TermDocumentMatrix:
    def __init__(self, invertedList: List[Dict], weightCalculator: WeightCalculator = StandardTFIDF):
        self.invertedList = invertedList
        self.weightCalculator = weightCalculator(invertedList)

    def getWeight(self, documentID, term, normalized = False):
        weight = self.weightCalculator.getWeight(documentID, term, normalized)
        return weight

    def filterQueryTerms(self, queryTerms) -> List:
        queryTerms = set(queryTerms)
        vocabulary = set(self.invertedList.index)
        return list(queryTerms.intersection(vocabulary))

    def filterDocumentsByQueryTerms(self, queryTerms) -> List:
        documentIDs = self.invertedList.loc[queryTerms].documentIDList.apply(lambda document: list(document.index))
        documentIDs = documentIDs.explode().unique()
        return documentIDs