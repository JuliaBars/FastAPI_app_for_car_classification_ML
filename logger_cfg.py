config = {
    "handlers": [
        {"sink": "log.log", 
         "format": "<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
         'rotation': "1 MB", 'level': "DEBUG", 'compression': "zip"},
    ],
}


# logger.remove()
# logger.add(
#     colorize=True,
#     format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
#     level=10,
# )
# logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")
