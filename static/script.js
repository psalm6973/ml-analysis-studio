const fileInput =
    document.getElementById("fileInput");

const analyzeButton =
    document.getElementById("analyzeButton");

const loading =
    document.getElementById("loading");

const error =
    document.getElementById("error");

const overviewSection =
    document.getElementById("overviewSection");

const targetSection =
    document.getElementById("targetSection");

const problemSection =
    document.getElementById("problemSection");

const modelSection =
    document.getElementById("modelSection");

const resultsSection =
    document.getElementById("resultsSection");

const rowCount =
    document.getElementById("rowCount");

const columnCount =
    document.getElementById("columnCount");

const columnInfo =
    document.getElementById("columnInfo");

const targetSuggestions =
    document.getElementById("targetSuggestions");

const targetSelect =
    document.getElementById("targetSelect");

const continueButton =
    document.getElementById("continueButton");

const problemType =
    document.getElementById("problemType");

const problemDescription =
    document.getElementById("problemDescription");

const modelSelect =
    document.getElementById("modelSelect");

const trainButton =
    document.getElementById("trainButton");


/* =========================
   ANALYZE DATASET
========================= */

analyzeButton.addEventListener(
    "click",
    async function () {

        const file =
            fileInput.files[0];

        if (!file) {

            showError(
                "Please select a CSV file first."
            );

            return;
        }

        loading.style.display = "block";
        error.style.display = "none";

        try {

            const csvText =
                await file.text();

            const response =
                await fetch(
                    "/analyze",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "text/plain"
                        },

                        body: csvText
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {

                throw new Error(
                    data.detail || "Analysis failed."
                );
            }

            displayAnalysis(data);

        } catch (err) {

            showError(err.message);

        } finally {

            loading.style.display = "none";
        }
    }
);


/* =========================
   DISPLAY ANALYSIS
========================= */

function displayAnalysis(data) {

    overviewSection.style.display = "block";
    targetSection.style.display = "block";

    problemSection.style.display = "none";
    modelSection.style.display = "none";
    resultsSection.style.display = "none";

    const analysis =
        data.analysis;

    rowCount.textContent =
        analysis.rows;

    columnCount.textContent =
        analysis.columns;


    /* COLUMN INFORMATION */

    columnInfo.innerHTML = "";

    for (
        const [column, info]
        of Object.entries(
            analysis.column_info
        )
    ) {

        const div =
            document.createElement("div");

        div.className =
            "column-box";

        div.innerHTML = `
            <strong>${column}</strong>

            <p>
                Type: ${info.dtype}
            </p>

            <p>
                Missing Values:
                ${info.missing_values}
            </p>

            <p>
                Unique Values:
                ${info.unique_values}
            </p>
        `;

        columnInfo.appendChild(div);
    }


    /* TARGETS */

    targetSuggestions.innerHTML = "";

    targetSelect.innerHTML = `
        <option value="">
            -- Select Target --
        </option>
    `;

    const targets =
        data.suggested_targets;

    if (
        targets &&
        targets.length > 0
    ) {

        targets.forEach(
            function (target) {

                const p =
                    document.createElement(
                        "p"
                    );

                p.textContent =
                    "• " + target;

                targetSuggestions
                    .appendChild(p);

                const option =
                    document.createElement(
                        "option"
                    );

                option.value =
                    target;

                option.textContent =
                    target;

                targetSelect
                    .appendChild(option);
            }
        );

    } else {

        targetSuggestions.textContent =
            "No suitable target found.";
    }

    continueButton.disabled = true;
}


/* =========================
   TARGET SELECTION
========================= */

targetSelect.addEventListener(
    "change",
    function () {

        continueButton.disabled =
            targetSelect.value === "";
    }
);


/* =========================
   DETERMINE PROBLEM TYPE
========================= */

continueButton.addEventListener(
    "click",
    async function () {

        const file =
            fileInput.files[0];

        const target =
            targetSelect.value;

        if (!file || !target) {

            showError(
                "Please select a target column."
            );

            return;
        }

        loading.style.display = "block";
        error.style.display = "none";

        try {

            const csvText =
                await file.text();

            const response =
                await fetch(
                    "/problem-type",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            csv_text: csvText,
                            target_column: target
                        })
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Could not determine problem type."
                );
            }

            showProblemType(data);

        } catch (err) {

            showError(err.message);

        } finally {

            loading.style.display = "none";
        }
    }
);


/* =========================
   SHOW PROBLEM TYPE
========================= */

function showProblemType(data) {

    problemSection.style.display = "block";
    modelSection.style.display = "block";
    resultsSection.style.display = "none";


    if (
        data.problem_type ===
        "classification"
    ) {

        problemType.innerHTML = `
            <div class="problem-type classification">
                🟦 Classification
            </div>
        `;

        problemDescription.textContent =
            "Your target contains categories or classes. " +
            "Choose a classification algorithm.";

    } else {

        problemType.innerHTML = `
            <div class="problem-type regression">
                🟩 Regression
            </div>
        `;

        problemDescription.textContent =
            "Your target contains continuous numerical values. " +
            "Linear Regression is available.";
    }


    /* MODELS */

    modelSelect.innerHTML = `
        <option value="">
            -- Select Model --
        </option>
    `;

    data.models.forEach(
        function (model) {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                model.value;

            option.textContent =
                model.name;

            modelSelect.appendChild(
                option
            );
        }
    );

    modelSection.scrollIntoView({
        behavior: "smooth"
    });
}


/* =========================
   TRAIN MODEL
========================= */

trainButton.addEventListener(
    "click",
    async function () {

        const file =
            fileInput.files[0];

        const target =
            targetSelect.value;

        const model =
            modelSelect.value;


        if (!file) {

            showError(
                "Please select a CSV file."
            );

            return;
        }


        if (!target) {

            showError(
                "Please select a target column."
            );

            return;
        }


        if (!model) {

            showError(
                "Please select a model."
            );

            return;
        }


        loading.style.display = "block";
        error.style.display = "none";


        try {

            const csvText =
                await file.text();


            const response =
                await fetch(
                    "/train",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({

                            csv_text:
                                csvText,

                            target_column:
                                target,

                            model_name:
                                model
                        })
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Model training failed."
                );
            }


            displayModelResults(data);


        } catch (err) {

            showError(err.message);

        } finally {

            loading.style.display = "none";
        }
    }
);


/* =========================
   DISPLAY MODEL RESULTS
========================= */

function displayModelResults(data) {

    resultsSection.style.display =
        "block";


    const modelResults =
        document.getElementById(
            "modelResults"
        );


    let html = `

        <p>
            Target:
            <strong>
                ${data.target}
            </strong>
        </p>

        <p>
            Problem Type:
            <strong>
                ${data.problem_type}
            </strong>
        </p>

        <p>
            Model:
            <strong>
                ${data.model}
            </strong>
        </p>

        <h3>
            Model Performance
        </h3>

    `;


    /* CLASSIFICATION */

    if (
        data.problem_type ===
        "classification"
    ) {

        html += `

            <p>
                Accuracy:
                <strong>
                    ${data.metrics.accuracy}%
                </strong>
            </p>

            <p>
                Precision:
                <strong>
                    ${data.metrics.precision}%
                </strong>
            </p>

            <p>
                Recall:
                <strong>
                    ${data.metrics.recall}%
                </strong>
            </p>

            <p>
                F1 Score:
                <strong>
                    ${data.metrics.f1_score}%
                </strong>
            </p>

        `;
    }


    /* REGRESSION */

    else {

        html += `

            <p>
                MAE:
                <strong>
                    ${data.metrics.mae}
                </strong>
            </p>

            <p>
                MSE:
                <strong>
                    ${data.metrics.mse}
                </strong>
            </p>

            <p>
                RMSE:
                <strong>
                    ${data.metrics.rmse}
                </strong>
            </p>

            <p>
                R² Score:
                <strong>
                    ${data.metrics.r2_score}
                </strong>
            </p>

        `;
    }


    modelResults.innerHTML =
        html;


    resultsSection.scrollIntoView({
        behavior: "smooth"
    });
}


/* =========================
   ERROR
========================= */

function showError(message) {

    error.textContent =
        "Error: " + message;

    error.style.display =
        "block";
}

