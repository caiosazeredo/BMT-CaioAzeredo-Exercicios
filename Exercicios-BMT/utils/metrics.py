import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Union, List

def filterRetrievedDocs(retrievedDocs: Union[pd.DataFrame, List], limit = None, threshold = None):
    if limit is not None and threshold is not None:
        raise ValueError("Either limit or threshold should be None.")

    allQueryNumbers = retrievedDocs[["queryNumber"]].drop_duplicates()

    if limit is not None:
        if limit <= 0:
            raise ValueError("Limit should be greater than zero.")

        else:
            if type(retrievedDocs) is list:
                retrievedDocs = retrievedDocs[:limit]
            else:
                tmpDF = retrievedDocs.copy(deep = True)
                tmpDF["keep"] = False
                tmpDF["keep"] = tmpDF[["queryNumber", "keep"]].groupby("queryNumber").transform(
                    lambda group: [i < limit for i in range(group.shape[0])]
                )
                retrievedDocs = tmpDF[tmpDF["keep"]].drop("keep", axis = 1)
    
    if threshold is not None:
        if type(retrievedDocs) is list:
            raise ValueError("It is not possible to filter retrieved docs by threshold because the scores are not provided.")
        retrievedDocs = retrievedDocs[retrievedDocs["score"] >= threshold]

    retrievedDocs = retrievedDocs[["queryNumber", "documentID"]]
    retrievedDocs = retrievedDocs.groupby("queryNumber").agg({
        "documentID": lambda group: list(group)
    }).reset_index()
    retrievedDocs = pd.merge(allQueryNumbers, retrievedDocs, on = "queryNumber", how = "left")
    retrievedDocs["documentID"] = retrievedDocs.documentID.apply(lambda documents: documents if type(documents) == list else [])
    
    return retrievedDocs

def _precisionScore(queryDocsDF):
    retrieved = set(queryDocsDF.retrievedDoc)
    relevant = set(queryDocsDF.relevantDoc)
    retrievedAndRelevant = retrieved.intersection(relevant)
    try:
        precision = len(retrievedAndRelevant)/len(retrieved)
    except:
        precision = np.nan
    return precision

def _recallScore(queryDocsDF):
    retrieved = set(queryDocsDF.retrievedDoc)
    relevant = set(queryDocsDF.relevantDoc)
    retrievedAndRelevant = retrieved.intersection(relevant)
    try:
        recall = len(retrievedAndRelevant)/len(relevant)
    except:
        recall = np.nan
    return recall

def _f1Score(queryDocsDF):
    precision = _precisionScore(queryDocsDF)
    recall = _recallScore(queryDocsDF)
    try: 
        f1 = (2*precision*recall)/(precision + recall)
    except:
        f1 = np.nan
    return f1

def _rPrecisionScore(queryDocsDF):
    relevant = set(queryDocsDF.relevantDoc)
    rel = len(relevant)

    retrieved = set(queryDocsDF.retrievedDoc[:rel])
    retrievedAndRelevant = retrieved.intersection(relevant)
    
    rPrecision = len(retrievedAndRelevant)/rel
    return rPrecision

def _meanAveragePrecisionScore(queryDocsDF):
    retrieved = queryDocsDF.retrievedDoc
    retrievedSet = set(retrieved)
    relevant = queryDocsDF.relevantDoc
    relevantRanks = [retrieved.index(doc) + 1 if doc in retrievedSet else 0 for doc in relevant]
    precisionAtK = []
    for k in relevantRanks:
        df = queryDocsDF.copy(deep = True)
        df["retrievedDoc"] = df["retrievedDoc"][:k]
        df["relevantDoc"] = df["relevantDoc"][:k]
        precisionAtK.append(_precisionScore(df))
    meanAveragePrecision = np.nanmean(precisionAtK)
    return meanAveragePrecision

def _meanReciprocalRankScore(queryDocsDF):
    retrieved = queryDocsDF.retrievedDoc
    relevantSet = set(queryDocsDF.relevantDoc)
    retrievedAndRelevant = [documentID for documentID in retrieved if documentID in relevantSet]
    
    if len(retrievedAndRelevant) == 0:
        RR = 0
    else:
        firstRelevantDocumentID = retrievedAndRelevant[0]
        firstRelevantRank = retrieved.index(firstRelevantDocumentID) + 1
        RR = 1/firstRelevantRank
    return RR

def _discountedCumulativeGainScore(queryDocsDF):
    retrieved = queryDocsDF.retrievedDoc
    relevant = {k:v for k, v in zip(queryDocsDF.relevantDoc, queryDocsDF.relevantDocRelevance)}
    discountedCumulativeGain = [relevant.get(retrieved[0],0)]
    for i, documentID in enumerate(retrieved[1:], start = 2):
        discountFactor = 1/np.log2(i)
        discountedCumulativeGain.append(discountedCumulativeGain[-1] + relevant.get(documentID, 0)*discountFactor)
    return discountedCumulativeGain

def getMetricScore(
        retrievedDocs: pd.DataFrame, 
        relevantDocs: pd.DataFrame, 
        scoreFuncs = [_precisionScore, _recallScore, _f1Score], 
        queryNumber: int = None, 
        limit = None, 
        threshold = None
    ):
    retrievedDocs = retrievedDocs if queryNumber is None else retrievedDocs[retrievedDocs.queryNumber == queryNumber]       
    retrievedDocs = filterRetrievedDocs(retrievedDocs, limit = limit, threshold = threshold)[["queryNumber", "documentID"]]

    relevantDocs = relevantDocs.groupby("queryNumber").agg({
        "documentID": lambda group: list(group),
        "relevance": lambda group: list(group)
    }).reset_index()
    
    queriesDocs = pd.merge(retrievedDocs, relevantDocs, how = "inner", on = "queryNumber")
    queriesDocs.columns = ["queryNumber", "retrievedDoc", "relevantDoc", "relevantDocRelevance"]

    for scoreFunc in scoreFuncs:
        queriesDocs[scoreFunc.__name__] = queriesDocs.apply(lambda row: scoreFunc(row), axis = 1)
    return queriesDocs.drop(["retrievedDoc", "relevantDoc", "relevantDocRelevance"], axis = 1)

def plotElevenPoints(retrievedDF: pd.DataFrame, relevantDF: pd.DataFrame, limit = None, threshold = None):
    scoresDF = getMetricScore(
        retrievedDF, 
        relevantDF, 
        scoreFuncs = [_precisionScore, _recallScore],
        limit = limit,
        threshold = threshold
    ).sort_values("_recallScore")
    scoresDF._recallScore = scoresDF._recallScore.apply(lambda recall: f"{recall:.2f}")
    scoresDF._precisionScore = scoresDF._precisionScore.apply(lambda precision: f"{precision:.2f}")

    recallPrecisionTab = pd.crosstab(scoresDF._recallScore, scoresDF.queryNumber)
    for row, recall in enumerate(recallPrecisionTab.index):
        for column, queryNumber in enumerate(recallPrecisionTab.columns):
            try:
                precision = scoresDF[(scoresDF.queryNumber == queryNumber) & (scoresDF._recallScore == recall)].iloc[0]["_precisionScore"]
                recallPrecisionTab.iloc[row, column] = float(precision)
            except:
                recallPrecisionTab.iloc[row, column] = float(recallPrecisionTab.iloc[row, column])

    elevenPointsRecall = [f"{i/10:.2f}" for i in range(0,11)]
    interpolatedPrecisionColumns = recallPrecisionTab.columns
    elevenPointsDF = pd.DataFrame(data = {"R (%)": elevenPointsRecall})
    elevenPointsDF[interpolatedPrecisionColumns] = 0
    elevenPointsDF = elevenPointsDF.set_index("R (%)")

    for row, recall in enumerate(elevenPointsDF.index):
        for column, queryNumber in enumerate(elevenPointsDF.columns):
            interpolatedPrecision = recallPrecisionTab[recallPrecisionTab.index >= recall][queryNumber].max()
            elevenPointsDF.iloc[row, column] = interpolatedPrecision

    elevenPointsDF["averagePrecision"] = elevenPointsDF.apply(lambda row: f"{row.mean():.2f}", axis = 1)
    elevenPointsDF = elevenPointsDF.reset_index()[["R (%)", "averagePrecision"]]
    elevenPointsDF.columns = ["Recall (%)", "Precision (%)"]
    if list(elevenPointsDF["Precision (%)"].value_counts().index) == ["nan"]:
        raise Exception("All queries retrieved 0 documents.")
    elevenPointsDF = elevenPointsDF.apply(lambda column: [int(np.round(float(value)*100)) if value != "nan" else 0 for value in column])

    fig = sns.lineplot(data = elevenPointsDF, x = "Recall (%)", y = "Precision (%)")
    plt.ylim([0,100])
    plt.xlim([0,100])

    return elevenPointsDF, fig.get_figure()

def rPrecisionHistogram(resultsA, resultsB, expectedResults):
    rPrecisionA = getMetricScore(resultsA, expectedResults, scoreFuncs = [_rPrecisionScore])
    rPrecisionB = getMetricScore(resultsB, expectedResults, scoreFuncs = [_rPrecisionScore])
    rPrecision = pd.merge(rPrecisionA, rPrecisionB, on = "queryNumber", how = "inner")
    rPrecision["delta"] = rPrecision._rPrecisionScore_x - rPrecision._rPrecisionScore_y
    rPrecision = rPrecision[["queryNumber", "delta"]]
    rPrecision.columns = ["Query Number", "R-Precision A/B" ]
    fig = plt.figure(figsize = (14,5))
    fig = sns.barplot(data = rPrecision, x = "Query Number", y = "R-Precision A/B")
    plt.ylim([-1.05, 1.05])
    plt.xticks(rotation = 90)
    return rPrecision, fig.get_figure()

def meanAveragePrecision(retrieved, relevant):
    mapQueries = getMetricScore(retrieved, relevant, scoreFuncs = [_meanAveragePrecisionScore])
    mapSystem = mapQueries._meanAveragePrecisionScore.mean()
    return mapSystem

def meanReciprocalRank(retrieved, relevant, limit = 10):
    queriesRR = getMetricScore(retrieved, relevant, scoreFuncs = [_meanReciprocalRankScore], limit = limit)
    return queriesRR._meanReciprocalRankScore.mean()

def discountedCumulativeGain(retrieved, relevant, limit = 20, returnPlot = True):
    score = getMetricScore(retrieved, relevant, scoreFuncs = [_discountedCumulativeGainScore], limit = limit)
    maxRetrievedDocs = score._discountedCumulativeGainScore.apply(len).max()
    scoreVectors = []
    for i, vector in enumerate(score._discountedCumulativeGainScore):
        scoreVectors.append([])
        for j in range(maxRetrievedDocs):
            try:
                value = vector[j]
            except:
                value = 0
            scoreVectors[i].append(value)
    scoreVectors = np.array(scoreVectors)
    score = scoreVectors.mean(axis = 0)
    if returnPlot:
        fig = plt.figure(figsize = (8,6))
        fig = sns.lineplot(x = range(1,len(score)+1), y = score)
        plt.xlabel("Rank")
        plt.ylabel("Average Discounted Cumulative Gain")
        return score, fig.get_figure()
    else:
        return score
    
def normalizedDiscountedCumulativeGain(retrieved, relevant, limit = 20):
    relevances = relevant.set_index(["queryNumber", "documentID"]).to_dict()['relevance']

    idealRetrieval = retrieved.copy(deep = True)
    idealRetrieval["relevance"] = pd.Series([relevances.get(pair[1:],0) for pair in idealRetrieval[["queryNumber", "documentID"]].itertuples()])
    idealRetrieval = idealRetrieval.sort_values(["queryNumber", "relevance"], ascending = [True, False]).drop("relevance", axis = 1)
    idealRetrievalScore = discountedCumulativeGain(idealRetrieval, relevant, limit = limit, returnPlot = False)

    retrievedScore = discountedCumulativeGain(retrieved, relevant, limit = limit, returnPlot = False)
    
    score = retrievedScore / idealRetrievalScore
    return score