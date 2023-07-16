import json

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from routers import check, classify, try_yolo


# region FastAPI SetUp

app = FastAPI(
    title="Car models Classification FastAPI",
    description="""Obtain object value out of image
                    and return image with label""",
    version="2023.7.16",
)


origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def save_openapi_json():
    ''' This function is used to save the OpenAPI documentation
    data of the FastAPI application to a JSON file.
    The purpose of saving the OpenAPI documentation data is to
    which can be used for documentation purposes or
    to generate client libraries. '''
    openapi_data = app.openapi()
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# endregion


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


app.include_router(check.router)
app.include_router(classify.router)
app.include_router(try_yolo.router)
