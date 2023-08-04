import logging
import pandas as pd
from time import time
from typing import Text

def initLogger(name: Text):
    logging.basicConfig(
        level = logging.DEBUG,
        format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    logger = logging.getLogger(name)
    return logger

def executeFunction(
    
        logger: logging.Logger, 
        onStartMessage: Text = "Starting execution...", 
        onFinishMessage: Text = "Execution has finished with success...", 
        onErrorMessage: Text = "An error was found while executing the function.", 
        logResults: bool = False,
        func = None, 
        **kwargs
    ):
    if func is None: 
        raise ValueError("A function is expected.")
    logger.info(onStartMessage)
    startTime = time()
    try:
        results = func(**kwargs)
    except Exception as e:
        logger.error(f"{onErrorMessage}")
        raise e
    finishTime = time()
    elapsedTime = finishTime - startTime
    logger.info(f"{onFinishMessage} (Elapsed Time: {elapsedTime:.2f}s)")
    if logResults:
        logger.info("Results: " + str(results))
    return results

def executeModule(logger: logging.Logger, moduleFunction, **kwargs):
    logger.info("Starting module")
    startTime = time()
    moduleFunction(**kwargs)
    finishTime = time()
    elapsedTime = finishTime - startTime
    logger.info(f"Module has been executed with success (Elapsed Time: {elapsedTime:.2f}s)")