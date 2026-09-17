/* =========================================================
   ML ANALYSIS STUDIO
   Frontend Controller
========================================================= */


/* =========================================================
   STATE
========================================================= */

let csvText = "";
let targetSuggestions = [];
let selectedTarget = "";
let selectedModel = "";

let currentProblemType = "";
let currentFeatureSchema = [];

let trainedModelData = null;

const pages = [
    "landingPage",
    "uploadPage",
    "overviewPage",
    "targetPage",
    "modelPage",
    "resultsPage",
    "predictionPage"
];


/* =========================================================
   ELEMENTS
========================================================= */

const landingPage = document.getElementById("landingPage");
const uploadPage = document.getElementById("uploadPage");
const overviewPage = document.getElementById("overviewPage");
const targetPage = document.getElementById("targetPage");
const modelPage = document.getElementById("modelPage");
const resultsPage = document.getElementById("resultsPage");
const predictionPage = document.getElementById("predictionPage");

const csvFile = document.getElementById("csvFile");
const selectedFile = document.getElementById("selectedFile");

const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");

const rowCount = document.getElementById("rowCount");
const columnCount = document.getElementById("columnCount");
const missingCount = document.getElementById("missingCount");

const columnInfo = document.getElementById("columnInfo");
const columnSummary = document.getElementById("columnSummary");

const targetSuggestionsBox =
    document.getElementById("targetSuggestions");

const targetColumn =
    document.getElementById("targetColumn");

const problemType =
    document.getElementById("problemType");

const problemDescription =
    document.getElementById("problemDescription");

const modelOptions =
    document.getElementById("modelOptions");

const resultTarget =
    document.getElementById("resultTarget");

const resultProblem =
    document.getElementById("resultProblem");

const resultModel =
    document.getElementById("resultModel");

const reliabilityCard =
    document.getElementById("reliabilityCard");

const reliabilityStatus =
    document.getElementById("reliabilityStatus");

const metrics =
    document.getElementById("metrics");

const importantFeatures =
    document.getElementById("importantFeatures");

const predictionForm =
    document.getElementById("predictionForm");

const predictionDescription =
    document.getElementById("predictionDescription");

const predictionError =
    document.getElementById("predictionError");

const predictionResult =
    document.getElementById("predictionResult");

const predictionValue =
    document.getElementById("predictionValue");

const predictionTargetLabel =
    document.getElementById("predictionTargetLabel");

const predictionContinueButton =
    document.getElementById("predictionContinueButton");

const newPredictionButton =
    document.getElementById("newPredictionButton");


/* =========================================================
   PAGE NAVIGATION
========================================================= */

function showPage(pageId) {

    pages.forEach(id => {

        const page = document.getElementById(id);

        if (!page) {
            return;
        }

        page.classList.remove("active-page");
    });

    const selectedPage = document.getElementById(pageId);

    if (selectedPage) {
        selectedPage.classList.add("active-page");
    }

    updateStepIndicator(pageId);

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


/* =========================================================
   STEP INDICATOR
========================================================= */

function updateStepIndicator(pageId) {

    const stepMap = {
        landingPage: 0,
        uploadPage: 0,
        overviewPage: 1,
        targetPage: 2,
        modelPage: 3,
        resultsPage: 4,
        predictionPage: 5
    };

    const currentStep = stepMap[pageId] ?? 0;

    const dots =
        document.querySelectorAll(".step-dot");

    dots.forEach((dot, index) => {

        if (index <= currentStep) {
            dot.classList.add("active");
        } else {
            dot.classList.remove("active");
        }

    });
}


/* =========================================================
   GENERAL HELPERS
========================================================= */

function showGlobalError(message) {

    if (!errorBox) {
        return;
    }

    errorBox.textContent = message;
    errorBox.style.display = "block";

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


function hideGlobalError() {

    if (!errorBox) {
        return;
    }

    errorBox.style.display = "none";
    errorBox.textContent = "";
}


function showPredictionError(message) {

    predictionError.textContent = message;
    predictionError.style.display = "block";

    predictionResult.style.display = "none";

    predictionError.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


function hidePredictionError() {

    predictionError.textContent = "";
    predictionError.style.display = "none";
}


function setLoading(show) {

    loading.style.display =
        show ? "flex" : "none";
}


function escapeHtml(value) {

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


/* =========================================================
   LANDING
========================================================= */

document
    .getElementById("getStartedButton")
    .addEventListener("click", () => {

        hideGlobalError();

        showPage("uploadPage");
    });


/* =========================================================
   FILE SELECTION
========================================================= */

csvFile.addEventListener("change", () => {

    if (!csvFile.files.length) {

        selectedFile.textContent =
            "No file selected";

        return;
    }

    const file = csvFile.files[0];

    selectedFile.textContent =
        `${file.name} selected`;
});

/* =========================================================
   DRAG & DROP CSV UPLOAD
========================================================= */

const uploadCard =
    document.querySelector(".upload-card");

if (uploadCard) {

    uploadCard.addEventListener(
        "dragover",
        (event) => {

            event.preventDefault();

            uploadCard.classList.add(
                "drag-over"
            );
        }
    );


    uploadCard.addEventListener(
        "dragenter",
        (event) => {

            event.preventDefault();

            uploadCard.classList.add(
                "drag-over"
            );
        }
    );


    uploadCard.addEventListener(
        "dragleave",
        () => {

            uploadCard.classList.remove(
                "drag-over"
            );
        }
    );


    uploadCard.addEventListener(
        "drop",
        (event) => {

            event.preventDefault();

            uploadCard.classList.remove(
                "drag-over"
            );


            const files =
                event.dataTransfer.files;


            if (!files.length) {
                return;
            }


            const file =
                files[0];


            if (
                !file.name
                    .toLowerCase()
                    .endsWith(".csv")
            ) {

                showGlobalError(
                    "Please upload a CSV file."
                );

                return;
            }


            /*
             * Put the dropped file into
             * the existing file input.
             */

            const dataTransfer =
                new DataTransfer();

            dataTransfer.items.add(file);

            csvFile.files =
                dataTransfer.files;


            selectedFile.textContent =
                `${file.name} selected`;
        }
    );
}


/* =========================================================
   ANALYZE DATASET
========================================================= */

document
    .getElementById("analyzeButton")
    .addEventListener("click", async () => {

        hideGlobalError();

        if (!csvFile.files.length) {

            showGlobalError(
                "Please select a CSV file first."
            );

            return;
        }

        const file = csvFile.files[0];

        if (!file.name.toLowerCase().endsWith(".csv")) {

            showGlobalError(
                "Please upload a CSV file."
            );

            return;
        }

        try {

            setLoading(true);

            csvText = await file.text();

            const response = await fetch("/analyze", {
                method: "POST",

                headers: {
                    "Content-Type": "text/plain"
                },

                body: csvText
            });

            const data = await response.json();

            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Could not analyze the dataset."
                );
            }

            displayAnalysis(data);

            hideGlobalError();

            showPage("overviewPage");

        } catch (error) {

            showGlobalError(
                error.message ||
                "Something went wrong while analyzing the dataset."
            );

        } finally {

            setLoading(false);
        }

    });


/* =========================================================
   DISPLAY DATASET ANALYSIS
========================================================= */

function displayAnalysis(data) {

    const analysis = data.analysis || {};

    const rows = analysis.rows || 0;
    const columns = analysis.columns || 0;

    rowCount.textContent =
        rows.toLocaleString();

    columnCount.textContent =
        columns.toLocaleString();


    /* Calculate total missing values */

    let totalMissing = 0;

    const info = analysis.column_info || {};

    Object.values(info).forEach(column => {

        totalMissing +=
            Number(column.missing_values || 0);

    });

    missingCount.textContent =
        totalMissing.toLocaleString();


    /* Column summary */

    columnSummary.textContent =
        `${columns} columns`;


    /* Column table */

    columnInfo.innerHTML = "";


    const header = document.createElement("div");

    header.className =
        "column-row header";

    header.innerHTML = `
        <span>Column</span>
        <span>Type</span>
        <span>Missing</span>
        <span>Unique</span>
    `;

    columnInfo.appendChild(header);


    Object.entries(info).forEach(
        ([name, details]) => {

            const row =
                document.createElement("div");

            row.className =
                "column-row";

            row.innerHTML = `
                <span>${escapeHtml(name)}</span>
                <span>${escapeHtml(details.dtype)}</span>
                <span>${details.missing_values}</span>
                <span>${details.unique_values}</span>
            `;

            columnInfo.appendChild(row);
        }
    );


    /* Target suggestions */

    targetSuggestions =
        data.suggested_targets || [];

    buildTargetSelection(
        targetSuggestions,
        Object.keys(info)
    );
}


/* =========================================================
   TARGET SELECTION
========================================================= */

function buildTargetSelection(
    suggestions,
    allColumns
) {

    targetSuggestionsBox.innerHTML = "";
    targetColumn.innerHTML = "";

    selectedTarget = "";


    /* Default option */

    const defaultOption =
        document.createElement("option");

    defaultOption.value = "";
    defaultOption.textContent =
        "Select target column";

    targetColumn.appendChild(
        defaultOption
    );


    /* Manual dropdown */

    allColumns.forEach(column => {

        const option =
            document.createElement("option");

        option.value = column;
        option.textContent = column;

        targetColumn.appendChild(option);
    });


    /* Suggested cards */

    suggestions.forEach(
        (column, index) => {

            const card =
                document.createElement("div");

            card.className =
                "target-option";

            card.dataset.target =
                column;

            const label =
                index === 0
                    ? "Recommended target"
                    : "Suggested target";

            card.innerHTML = `
                <strong>
                    ${escapeHtml(column)}
                </strong>

                <small>
                    ${label}
                </small>
            `;

            card.addEventListener(
                "click",
                () => {

                    selectTarget(column);
                }
            );

            targetSuggestionsBox.appendChild(card);
        }
    );


    /* Automatically select first recommendation */

    if (suggestions.length > 0) {

        selectTarget(suggestions[0]);
    }
}


function selectTarget(column) {

    selectedTarget = column;

    targetColumn.value = column;


    const cards =
        document.querySelectorAll(
            ".target-option"
        );

    cards.forEach(card => {

        if (card.dataset.target === column) {
            card.classList.add("selected");
        } else {
            card.classList.remove("selected");
        }

    });
}


targetColumn.addEventListener(
    "change",
    () => {

        if (!targetColumn.value) {
            selectedTarget = "";
            return;
        }

        selectTarget(
            targetColumn.value
        );
    }
);


/* =========================================================
   OVERVIEW → TARGET
========================================================= */

document
    .getElementById("overviewContinueButton")
    .addEventListener("click", () => {

        if (!selectedTarget) {

            showGlobalError(
                "Please select a target column."
            );

            return;
        }

        hideGlobalError();

        showPage("targetPage");
    });


/* =========================================================
   TARGET → PROBLEM TYPE
========================================================= */

document
    .getElementById("targetContinueButton")
    .addEventListener("click", async () => {

        if (!selectedTarget) {

            showGlobalError(
                "Please select a target column."
            );

            return;
        }

        hideGlobalError();

        try {

            setLoading(true);

            const response = await fetch(
                "/problem-type",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        csv_text: csvText,
                        target_column:
                            selectedTarget
                    })
                }
            );

            const data =
                await response.json();

            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Could not determine the problem type."
                );
            }

            currentProblemType =
                data.problem_type;

            displayProblemType(
                data.problem_type
            );

            displayModelOptions(
                data.models || []
            );

            showPage("modelPage");

        } catch (error) {

            showGlobalError(
                error.message ||
                "Could not determine problem type."
            );

        } finally {

            setLoading(false);
        }

    });


/* =========================================================
   PROBLEM TYPE
========================================================= */

function displayProblemType(type) {

    problemType.className =
        "problem-type " + type;

    problemType.textContent =
        type === "classification"
            ? "Classification"
            : "Regression";


    if (type === "classification") {

        problemDescription.textContent =
            "The model will predict a category or class.";

    } else {

        problemDescription.textContent =
            "The model will predict a numerical value.";

    }
}


/* =========================================================
   MODEL OPTIONS
========================================================= */

function displayModelOptions(models) {

    modelOptions.innerHTML = "";

    selectedModel = "";


    models.forEach(model => {

        const card =
            document.createElement("div");

        card.className =
            "model-option";

        card.dataset.value =
            model.value;


        const description =
            getModelDescription(
                model.value
            );


        card.innerHTML = `
            <h4>
                ${escapeHtml(model.name)}
            </h4>

            <p>
                ${description}
            </p>
        `;


        card.addEventListener(
            "click",
            () => {

                selectModel(
                    model.value
                );
            }
        );


        modelOptions.appendChild(card);
    });


    /* Select first model automatically */

    if (models.length > 0) {

        selectModel(
            models[0].value
        );
    }
}


function selectModel(value) {

    selectedModel = value;

    const cards =
        document.querySelectorAll(
            ".model-option"
        );

    cards.forEach(card => {

        if (card.dataset.value === value) {
            card.classList.add("selected");
        } else {
            card.classList.remove("selected");
        }

    });
}


function getModelDescription(model) {

    const descriptions = {

        logistic_regression:
            "A fast and interpretable model for classification.",

        knn:
            "Classifies data using nearby examples.",

        svm:
            "Finds an effective boundary between classes.",

        decision_tree:
            "Makes predictions using a tree of decisions.",

        linear_regression:
            "Predicts numerical values using a fitted line."
    };

    return descriptions[model] ||
        "Machine learning model.";
}


/* =========================================================
   TRAIN MODEL
========================================================= */

document
    .getElementById("trainButton")
    .addEventListener("click", async () => {

        if (!selectedModel) {

            showGlobalError(
                "Please select a model first."
            );

            return;
        }


        hideGlobalError();

        try {

            setLoading(true);


            const response = await fetch(
                "/train",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        csv_text: csvText,
                        target_column:
                            selectedTarget,
                        model_name:
                            selectedModel
                    })
                }
            );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Could not train the model."
                );
            }


            trainedModelData = data;

            displayModelResults(data);

            currentFeatureSchema =
                data.feature_schema || [];

            createPredictionForm(
                currentFeatureSchema
            );

            showPage("resultsPage");

        } catch (error) {

            showGlobalError(
                error.message ||
                "Something went wrong while training the model."
            );

        } finally {

            setLoading(false);
        }

    });


/* =========================================================
   MODEL RESULTS
========================================================= */

function displayModelResults(data) {

    resultTarget.textContent =
        data.target || "-";

    resultProblem.textContent =
        capitalize(
            data.problem_type || "-"
        );

    resultModel.textContent =
        formatModelName(
            data.model || "-"
        );


    /* Reliability */

    const reliable =
        data.reliability === "reliable";

    reliabilityCard.classList.toggle(
        "needs-improvement",
        !reliable
    );


    const icon =
        reliabilityCard.querySelector(
            ".reliability-icon"
        );


    if (reliable) {

        icon.textContent = "✓";

        reliabilityStatus.textContent =
            "Reliable Model";

    } else {

        icon.textContent = "!";

        reliabilityStatus.textContent =
            "Needs Improvement";

    }


    /* Metrics */

    metrics.innerHTML = "";

    const metricData =
        data.metrics || {};


    Object.entries(metricData).forEach(
        ([name, value]) => {

            const card =
                document.createElement("div");

            card.className =
                "metric-card";

            card.innerHTML = `
                <span>
                    ${escapeHtml(
                        formatMetricName(name)
                    )}
                </span>

                <strong>
                    ${formatMetricValue(value)}
                </strong>
            `;

            metrics.appendChild(card);
        }
    );


    /* Features */

    displayFeatures(
        data.features || []
    );


    /* Prediction is available even when the model
       needs improvement. The user is warned instead.
    */

    predictionContinueButton.disabled = false;

    predictionContinueButton.innerHTML =
        `Make a Prediction <span>→</span>`;

    predictionContinueButton.style.opacity =
        "1";

   if (reliable) {

    predictionDescription.textContent =
        "These are the features selected by your trained model. Enter values within the ranges shown below.";

} else {

    predictionDescription.textContent =
        "⚠ This model needs improvement. Predictions may have limited accuracy. Enter values within the ranges shown below.";
}
}


/* =========================================================
   DISPLAY IMPORTANT FEATURES
========================================================= */

function displayFeatures(features) {

    importantFeatures.innerHTML = "";


    if (!features.length) {

        importantFeatures.innerHTML =
            `<p style="color:#71798b;">
                No feature information available.
            </p>`;

        return;
    }


    features.forEach(
        (feature, index) => {

            const row =
                document.createElement("div");

            row.className =
                "feature-row";


            row.innerHTML = `
                <div class="feature-name">
                    ${index + 1}. ${escapeHtml(feature)}
                </div>

                <div class="feature-bar">
                    <div
                        class="feature-bar-fill"
                        style="
                            width:${100 - (index * 12)}%;
                        "
                    ></div>
                </div>
            `;


            importantFeatures.appendChild(row);
        }
    );
}


/* =========================================================
   METRIC HELPERS
========================================================= */

function formatMetricName(name) {

    const names = {

        accuracy: "Accuracy",
        precision: "Precision",
        recall: "Recall",
        f1: "F1 Score",

        mae: "MAE",
        mse: "MSE",
        rmse: "RMSE",
        r2: "R²"
    };

    return names[name] ||
        name;
}


function formatMetricValue(value) {

    const number =
        Number(value);

    if (Number.isNaN(number)) {
        return escapeHtml(value);
    }


    if (number >= 0 && number <= 1) {

        return `${(
            number * 100
        ).toFixed(2)}%`;

    }


    return number.toFixed(4);
}


/* =========================================================
   RESULTS → PREDICTION
========================================================= */

predictionContinueButton.addEventListener(
    "click",
    () => {

        if (!trainedModelData) {
            return;
        }

        hidePredictionError();

        predictionResult.style.display =
            "none";

        showPage("predictionPage");
    }
);


function createPredictionForm(schema) {

    predictionForm.innerHTML = "";


    if (!schema.length) {

        predictionForm.innerHTML = `
            <p style="color:#71798b;">
                No prediction features available.
            </p>
        `;

        return;
    }


    predictionDescription.textContent =
        "These are the features selected by your trained model. Enter values within the ranges shown below.";


    schema.forEach(feature => {

        const field =
            document.createElement("div");

        field.className =
            "prediction-field";


        /* =================================================
           FEATURE TITLE
        ================================================== */

        const title =
            document.createElement("div");

        title.className =
            "prediction-field-title";


        const label =
            document.createElement("label");

        label.textContent =
            formatFeatureName(feature.name);


        title.appendChild(label);

        field.appendChild(title);


        /* =================================================
           FEATURE DESCRIPTION
        ================================================== */

        const help =
            document.createElement("div");

        help.className =
            "feature-help";

        help.textContent =
            getFeatureDescription(feature);


        field.appendChild(help);


        /* =================================================
           DROPDOWN
        ================================================== */

        if (
            feature.input_type ===
            "select"
        ) {

            const select =
                document.createElement("select");

            select.dataset.feature =
                feature.name;


            const placeholder =
                document.createElement(
                    "option"
                );

            placeholder.value = "";

            placeholder.textContent =
                `Select ${formatFeatureName(feature.name)}`;

            select.appendChild(
                placeholder
            );


            (feature.options || [])
                .forEach(option => {

                    const optionElement =
                        document.createElement(
                            "option"
                        );

                    optionElement.value =
                        option;

                    optionElement.textContent =
                        option;

                    select.appendChild(
                        optionElement
                    );
                });


            field.appendChild(select);

        }


        /* =================================================
           NUMBER INPUT
        ================================================== */

        else {

            const range =
                document.createElement("div");

            range.className =
                "feature-range";


            if (
                feature.min !== undefined &&
                feature.max !== undefined
            ) {

                range.textContent =
                    `Range: ${formatFeatureNumber(feature.min)} – ${formatFeatureNumber(feature.max)}`;

            } else {

                range.textContent =
                    "Enter a numerical value from your dataset.";
            }


            field.appendChild(range);


            if (
                feature.default !== undefined
            ) {

                const suggested =
                    document.createElement("div");

                suggested.className =
                    "feature-default";

                suggested.textContent =
                    `Suggested starting value: ${formatFeatureNumber(feature.default)}`;

                field.appendChild(
                    suggested
                );
            }


            const input =
                document.createElement(
                    "input"
                );

            input.type = "number";

            input.dataset.feature =
                feature.name;

            input.step = "any";


            if (feature.min !== undefined) {

                input.min =
                    feature.min;
            }


            if (feature.max !== undefined) {

                input.max =
                    feature.max;
            }


            if (
                feature.default !== undefined
            ) {

                input.value =
                    feature.default;
            }


            field.appendChild(input);
        }


        predictionForm.appendChild(field);
    });
}

/* =========================================================
   FEATURE NAME FORMATTER
========================================================= */

function formatFeatureName(name) {

    return String(name)
        .replace(/_/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/\b\w/g, letter =>
            letter.toUpperCase()
        );
}


/* =========================================================
   FEATURE DESCRIPTION
========================================================= */

function getFeatureDescription(feature) {

    if (
        feature.input_type ===
        "select"
    ) {

        return "Choose one of the values found in your dataset.";
    }


    return "Numerical feature from your uploaded dataset.";
}


/* =========================================================
   NUMBER FORMATTER
========================================================= */

function formatFeatureNumber(value) {

    const number =
        Number(value);


    if (Number.isNaN(number)) {
        return String(value);
    }


    if (
        Number.isInteger(number)
    ) {

        return number.toString();
    }


    return number.toFixed(4);
}

/* =========================================================
   PREDICT
========================================================= */

document
    .getElementById("predictButton")
    .addEventListener("click", async () => {

        hidePredictionError();

        predictionResult.style.display =
            "none";


        const features = {};


        const fields =
            predictionForm.querySelectorAll(
                "[data-feature]"
            );


        for (const field of fields) {

            const name =
                field.dataset.feature;

            const rawValue =
                field.value;


            if (
                rawValue === null ||
                rawValue === ""
            ) {

                showPredictionError(
                    `Please enter a value for ${name}.`
                );

                return;
            }


            /* NUMBER */

            if (
                field.tagName ===
                "INPUT"
            ) {

                const numberValue =
                    Number(rawValue);


                if (
                    Number.isNaN(numberValue)
                ) {

                    showPredictionError(
                        `${name} must be a valid number.`
                    );

                    return;
                }


                const min =
                    field.min !== ""
                        ? Number(field.min)
                        : null;

                const max =
                    field.max !== ""
                        ? Number(field.max)
                        : null;


                if (
                    min !== null &&
                    numberValue < min
                ) {

                    showPredictionError(
                        `${name} must be at least ${min}.`
                    );

                    return;
                }


                if (
                    max !== null &&
                    numberValue > max
                ) {

                    showPredictionError(
                        `${name} must be at most ${max}.`
                    );

                    return;
                }


                features[name] =
                    numberValue;

            }


            /* SELECT */

            else {

                features[name] =
                    rawValue;
            }
        }


        try {

            setLoading(true);


            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            features: features
                        })
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Could not make prediction."
                );
            }


            displayPrediction(
                data
            );


        } catch (error) {

            showPredictionError(
                error.message ||
                "Something went wrong while making the prediction."
            );

        } finally {

            setLoading(false);
        }

    });


/* =========================================================
   DISPLAY PREDICTION
========================================================= */

function displayPrediction(data) {

    let value =
        data.prediction;


    /* Titanic */

    if (
        data.target ===
        "Survived"
    ) {

        value =
            Number(value) === 1
                ? "Survived"
                : "Not Survived";
    }


    predictionValue.textContent =
        value;

    predictionTargetLabel.textContent =
        `Predicted ${data.target}`;


    predictionResult.style.display =
        "block";


    predictionResult.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


/* =========================================================
   NEW PREDICTION
========================================================= */

newPredictionButton.addEventListener(
    "click",
    () => {

        hidePredictionError();

        predictionResult.style.display =
            "none";

        predictionForm
            .querySelectorAll(
                "input, select"
            )
            .forEach(field => {

                if (
                    field.tagName ===
                    "SELECT"
                ) {

                    field.selectedIndex =
                        0;

                } else {

                    field.value = "";
                }
            });

        createPredictionForm(
            currentFeatureSchema
        );
    }
);


/* =========================================================
   TEXT HELPERS
========================================================= */

function capitalize(text) {

    if (!text) {
        return "";
    }

    return text.charAt(0).toUpperCase() +
        text.slice(1);
}


function formatModelName(name) {

    return name
        .replace(/_/g, " ")
        .replace(/\b\w/g, letter =>
            letter.toUpperCase()
        );
}


/* =========================================================
   INITIAL STATE
========================================================= */

showPage("landingPage");