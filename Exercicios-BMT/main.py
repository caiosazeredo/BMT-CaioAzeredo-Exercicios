import os
WORKDIR = os.path.dirname(os.path.abspath(__file__))

import argparse
import sys
sys.path.append(WORKDIR)

from utils.cfg import QueryProcessorConfig, InvertedListGeneratorConfig, IndexerConfig, SearcherConfig, EvaluatorConfig
from utils import log
from src.queryProcessor import QueryProcessor
from src.indexer import InvertedListGenerator, Indexer
from src.searcher import Searcher
from src.evaluation import ResultsComparison

def search():
    # Init Loggers
    settingsLogger = log.initLogger("SETTINGS")

    # Loading Settings
    QUERY_PROCESSOR_CFG_FILEPATH = os.path.normpath(f"{WORKDIR}/PC.CFG")
    INVERTED_LIST_CFG_FILEPATH = os.path.normpath(f"{WORKDIR}/GLI.CFG")
    INDEXER_CFG_FILEPATH = os.path.normpath(f"{WORKDIR}/INDEX.CFG")
    SEARCHER_CFG_FILEPATH = os.path.normpath(f"{WORKDIR}/BUSCA.CFG")

    queryProcessorCFG = log.executeFunction(
        logger = settingsLogger,
        onStartMessage = "Loading query processor settings",
        onFinishMessage = "Query processor settings were loaded with success",
        logResults = True,
        func = QueryProcessorConfig(configPath = QUERY_PROCESSOR_CFG_FILEPATH).loadConfig
    )
    
    invertedListCFG = log.executeFunction(
        logger = settingsLogger,
        onStartMessage = "Loading inverted list generator settings",
        onFinishMessage = "Inverted list generator settings were loaded with success",
        logResults = True,
        func = InvertedListGeneratorConfig(configPath = INVERTED_LIST_CFG_FILEPATH).loadConfig
    )
    
    indexerCFG = log.executeFunction(
        logger = settingsLogger,
        onStartMessage = "Loading indexer settings",
        onFinishMessage = "Indexer settings were loaded with success",
        logResults = True,
        func = IndexerConfig(configPath = INDEXER_CFG_FILEPATH).loadConfig
    )

    searcherCFG = log.executeFunction(
        logger = settingsLogger,
        onStartMessage = "Loading searcher settings",
        onFinishMessage = "Searcher settings were loaded with success",
        logResults = True,
        func = SearcherConfig(configPath = SEARCHER_CFG_FILEPATH).loadConfig
    )

    # Query Processor
    queriesFilePath = os.path.abspath(queryProcessorCFG["LEIA"])
    processedQueriesFilePath = os.path.abspath(queryProcessorCFG["CONSULTAS"])
    expectedResultsFilePath = os.path.abspath(queryProcessorCFG["ESPERADOS"])

    os.makedirs(os.path.dirname(processedQueriesFilePath), exist_ok = True)
    os.makedirs(os.path.dirname(expectedResultsFilePath), exist_ok = True)

    queryProcessor = QueryProcessor(
        queriesFilePath = queriesFilePath,
        processedQueriesFilePath = processedQueriesFilePath, 
        expectedResultsFilePath = expectedResultsFilePath
    )

    # Inverted List   
    documentFilePathList = [os.path.abspath(path) for path in invertedListCFG["LEIA"]]
    invertedListFilePath = os.path.abspath(invertedListCFG["ESCREVA"])
    useStemmer = invertedListCFG["STEMMER"]

    os.makedirs(os.path.dirname(invertedListFilePath), exist_ok = True)

    invertedListGenerator = InvertedListGenerator(
        documentFilePathList = documentFilePathList,
        invertedListFilePath = invertedListFilePath,
        useStemmer = useStemmer
    )

    ## Indexer  
    invertedListFilePath = os.path.abspath(indexerCFG["LEIA"])
    indexesFilePath = os.path.abspath(indexerCFG["ESCREVA"])

    os.makedirs(os.path.dirname(indexesFilePath), exist_ok = True)

    indexer = Indexer(
        invertedListFilePath = invertedListFilePath,
        indexesFilePath = indexesFilePath
    )

    ## Searcher   
    modelFilePath = os.path.abspath(searcherCFG["MODELO"])
    queriesFilePath = os.path.abspath(searcherCFG["CONSULTAS"])
    resultsFileDir, resultsFile = os.path.split(os.path.abspath(searcherCFG["RESULTADOS"]))
    resultsFileName, resultsFileExt = os.path.splitext(resultsFile)
    resultsFileName += f"-{'STEMMER' if useStemmer else 'NOSTEMMER'}"
    resultsFilePath = f"{resultsFileDir}/{resultsFileName}{resultsFileExt}"

    os.makedirs(os.path.dirname(resultsFilePath), exist_ok = True)

    searcher = Searcher(
        modelFilePath = modelFilePath, 
        queriesFilePath = queriesFilePath,
        resultsFilePath = resultsFilePath,
        useStemmer = useStemmer
    )

    # Putting all together
    queryProcessor.run()
    invertedListGenerator.run()
    indexer.run()
    searcher.run()

def eval():
    # Init Loggers
    settingsLogger = log.initLogger("SETTINGS")

    # Loading Settings
    EVAL_CONFIG_FILEPATH = os.path.normpath(f"{WORKDIR}/AVALIA.CFG")

    evalCFG = log.executeFunction(
        logger = settingsLogger,
        onStartMessage = "Loading evaluator settings",
        onFinishMessage = "Evaluator settings were loaded with success",
        logResults = True,
        func = EvaluatorConfig(configPath = EVAL_CONFIG_FILEPATH).loadConfig
    )

    # Evaluator
    resultsFilePathList = [os.path.abspath(path) for path in evalCFG["RESULTADOS"]]
    resultsNameList = evalCFG["NOME"]
    resultsList = [{"name": name, "filepath": filepath} for name, filepath in zip(resultsNameList, resultsFilePathList)]
    expectedResultsFilePath = os.path.abspath(evalCFG["ESPERADOS"])
    storeDir = os.path.abspath(evalCFG["ESCREVA_DIRETORIO"])

    os.makedirs(storeDir, exist_ok = True)

    evaluator = ResultsComparison(
        relevantFilePath = expectedResultsFilePath,
        retrievedList = resultsList,
        storeDir = storeDir
    )

    evaluator.elevenPoints(limit = 10)
    evaluator.f1(limit = 10)
    evaluator.precision(limit = 5)
    evaluator.precision(limit = 10)
    # TODO: Generate R-Precision for other pairs of result datasets
    evaluator.rPrecisionHistogram(resultsList[0]["name"], resultsList[1]["name"])
    evaluator.meanAveragePrecision()
    evaluator.meanReciprocalRank(limit = 10)
    evaluator.discountedCumulativeGain(limit = 10)
    evaluator.normalizedDiscountedCumulativeGain(limit = 10)

if __name__ == "__main__":
    # Logger
    logger = log.initLogger("MAIN")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help = "Execution mode ('search' or 'eval')", dest = "mode", default = "search")
    args = parser.parse_args()

    executionMode = args.mode

    # Search
    if executionMode == "search":
        log.executeFunction(
            logger, 
            onStartMessage = "Welcome! The system has been started in search mode",
            onFinishMessage = "All done! The system has been finished", 
            onErrorMessage = "An error was found while executing the system",
            func = search
        )

    # Evaluation
    elif executionMode == "eval":
        log.executeFunction(
            logger, 
            onStartMessage = "Welcome! The system has been started in evaluation mode",
            onFinishMessage = "All done! The system has been finished", 
            onErrorMessage = "An error was found while executing the system",
            func = eval
        )
    
    else:
        raise ValueError("Mode should be either 'search' or 'eval'.")