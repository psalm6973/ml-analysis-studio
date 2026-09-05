from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
from io import StringIO

from utils.analysis import analyze_dataset
from utils.model_selection import (
    suggest_targets,
    determine_problem_type,
    get_available_models
)

from utils.model_training import train_model


app = FastAPI(title="ML Analysis Studio")


# Serve CSS and JavaScript
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Request format for problem type detection
class ProblemRequest(BaseModel):
    csv_text: str
    target_column: str


# Request format for model training
class TrainingRequest(BaseModel):
    csv_text: str
    target_column: str
    model_name: str


# Home page
@app.get("/")
def home():

    return FileResponse(
        "templates/index.html"
    )


# Health check
@app.get("/health")
def health():

    return {
        "status": "ML Analysis Studio is running"
    }


# =========================================================
# ANALYZE DATASET
# =========================================================

@app.post("/analyze")
def analyze_csv(
    csv_text: str = Body(
        ...,
        media_type="text/plain"
    )
):

    try:

        df = pd.read_csv(
            StringIO(csv_text)
        )

        if df.empty:

            raise ValueError(
                "The CSV file is empty."
            )

        analysis = analyze_dataset(df)

        targets = suggest_targets(df)

        return {
            "analysis": analysis,
            "suggested_targets": targets
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze CSV: {str(e)}"
        )


# =========================================================
# DETERMINE PROBLEM TYPE
# =========================================================

@app.post("/problem-type")
def problem_type(
    request: ProblemRequest
):

    try:

        df = pd.read_csv(
            StringIO(request.csv_text)
        )

        if request.target_column not in df.columns:

            raise ValueError(
                f"Target column '{request.target_column}' "
                "was not found."
            )

        problem = determine_problem_type(
            df,
            request.target_column
        )

        models = get_available_models(
            problem
        )

        return {
            "problem_type": problem,
            "models": models
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not determine problem type: {str(e)}"
        )


# =========================================================
# TRAIN MODEL
# =========================================================

@app.post("/train")
def train(
    request: TrainingRequest
):

    try:

        # Convert CSV into DataFrame
        df = pd.read_csv(
            StringIO(request.csv_text)
        )

        # Check target
        if request.target_column not in df.columns:

            raise ValueError(
                f"Target column '{request.target_column}' "
                "was not found."
            )

        # Train the selected model
        result = train_model(
            df,
            request.target_column,
            request.model_name
        )

        # Return only information that
        # can safely be sent to the browser
        return {
            "problem_type": result["problem_type"],
            "target": result["target"],
            "model": result["model"],
            "metrics": result["metrics"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not train model: {str(e)}"
        )

