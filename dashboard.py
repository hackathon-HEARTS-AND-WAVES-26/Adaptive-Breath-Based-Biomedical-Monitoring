# ============================================================
# ADAPTIVE BIOMEDICAL MONITORING - DEMO DASHBOARD
# ============================================================
#
# This dashboard demonstrates two independent adaptive agents:
#
# 1. SENSOR SELECTION AGENT
#    Decides whether another sensor is required.
#
# 2. TEMPORAL SAMPLING AGENT
#    Decides how aggressively the physiological signal
#    should be sampled.
#
# IMPORTANT:
# The e-nose and BIDMC datasets are NOT synchronized patient data.
# Therefore this dashboard intentionally keeps their indices separate.
# They are combined only at the level of the demonstration step.
#
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd


from flask import Flask, jsonify, render_template_string


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\adaptive_biomedical_ml"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

# Sensor environment
SENSOR_DATA = os.path.join(
    DATA_DIR,
    "enose_noisy_environment.csv"
)

# Temporal waveform data
TEMPORAL_DATA = os.path.join(
    DATA_DIR,
    "bidmc_waveform_windows.csv"
)

# Trained models
SENSOR_MODEL = os.path.join(
    DATA_DIR,
    "adaptive_policy_model.pkl"
)

TEMPORAL_MODEL = os.path.join(
    DATA_DIR,
    "temporal_sampling_policy.pkl"
)


# ============================================================
# PRODUCT SENSOR MAPPING
# ============================================================
#
# IMPORTANT:
# S1-S8 originate from the e-nose dataset.
# They are mapped here only for conceptual demonstration
# of the proposed biomedical sensing architecture.
#
# This is NOT claiming that the e-nose dataset contains
# these exact biomedical sensors.
# ============================================================

SENSOR_MAP = {
    "S1": "SDP810 Differential Pressure",
    "S2": "BME688 Environmental / Gas",
    "S3": "CO₂ Sensor",
    "S4": "SGP40 VOC Sensor",
    "S5": "ECG",
    "S6": "SDP810 Verification",
    "S7": "BME688 Verification",
    "S8": "CO₂ Verification",
}


# ============================================================
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("ADAPTIVE BIOMEDICAL MONITORING DASHBOARD")
print("============================================================")
print()


# ------------------------------------------------------------
# Sensor dataset
# ------------------------------------------------------------

if not os.path.exists(SENSOR_DATA):
    raise FileNotFoundError(
        f"\nSensor dataset not found:\n{SENSOR_DATA}"
    )

sensor_df = pd.read_csv(SENSOR_DATA)

print(
    "Sensor dataset:",
    sensor_df.shape
)


# ------------------------------------------------------------
# Temporal dataset
# ------------------------------------------------------------

if not os.path.exists(TEMPORAL_DATA):
    raise FileNotFoundError(
        f"\nTemporal dataset not found:\n{TEMPORAL_DATA}"
    )

temporal_df = pd.read_csv(TEMPORAL_DATA)

print(
    "Temporal dataset:",
    temporal_df.shape
)


# ============================================================
# LOAD MODELS
# ============================================================

# ------------------------------------------------------------
# Sensor model
# ------------------------------------------------------------

if not os.path.exists(SENSOR_MODEL):
    raise FileNotFoundError(
        f"\nSensor model not found:\n{SENSOR_MODEL}"
    )

sensor_model = joblib.load(SENSOR_MODEL)

print(
    "Sensor model loaded:",
    type(sensor_model).__name__
)


# ------------------------------------------------------------
# Temporal model
# ------------------------------------------------------------

if not os.path.exists(TEMPORAL_MODEL):
    raise FileNotFoundError(
        f"\nTemporal model not found:\n{TEMPORAL_MODEL}"
    )

temporal_model = joblib.load(TEMPORAL_MODEL)

print(
    "Temporal model loaded:",
    type(temporal_model).__name__
)


# ============================================================
# MODEL FEATURE NAMES
# ============================================================

def get_model_features(model):
    """
    Safely obtain feature names from sklearn models.
    """

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):

        for _, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


sensor_features = get_model_features(
    sensor_model
)

temporal_features = get_model_features(
    temporal_model
)


print()
print("Sensor model features:")

if sensor_features is not None:

    print(
        sensor_features
    )

else:

    print(
        "Feature names unavailable"
    )


print()
print("Temporal model features:")

if temporal_features is not None:

    print(
        temporal_features
    )

else:

    print(
        "Feature names unavailable"
    )


# ============================================================
# PREPARE SENSOR INPUT
# ============================================================

def prepare_sensor_input(row):
    """
    Prepare exactly the columns expected by the trained
    sensor policy model.
    """

    if sensor_features is None:

        raise RuntimeError(
            "Sensor model does not expose feature_names_in_. "
            "The model must be retrained with named dataframe columns."
        )

    missing = [
        col
        for col in sensor_features
        if col not in row.index
    ]

    if missing:

        raise RuntimeError(
            "Sensor dataset is missing model features:\n"
            + "\n".join(missing)
        )

    data = pd.DataFrame(
        [
            row[sensor_features].values
        ],
        columns=sensor_features
    )

    return data


# ============================================================
# PREPARE TEMPORAL INPUT
# ============================================================

def prepare_temporal_input(row):
    """
    Prepare exactly the columns expected by the temporal
    sampling policy model.
    """

    if temporal_features is None:

        raise RuntimeError(
            "Temporal model does not expose feature_names_in_. "
            "The model must be retrained with named dataframe columns."
        )

    missing = [
        col
        for col in temporal_features
        if col not in row.index
    ]

    if missing:

        raise RuntimeError(
            "Temporal dataset is missing model features:\n"
            + "\n".join(missing)
        )

    data = pd.DataFrame(
        [
            row[temporal_features].values
        ],
        columns=temporal_features
    )

    return data


# ============================================================
# RUN SENSOR MODEL
# ============================================================

print()
print("Running sensor policy predictions...")


sensor_predictions = []

sensor_confidences = []

sensor_reliabilities = []


for idx in range(
    len(sensor_df)
):

    row = sensor_df.iloc[idx]

    X = prepare_sensor_input(
        row
    )

    prediction = sensor_model.predict(
        X
    )[0]

    sensor_predictions.append(
        str(prediction)
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = None

    if hasattr(
        sensor_model,
        "predict_proba"
    ):

        try:

            probabilities = sensor_model.predict_proba(
                X
            )[0]

            confidence = float(
                np.max(probabilities)
            )

        except Exception:

            confidence = None

    sensor_confidences.append(
        confidence
    )

    # --------------------------------------------------------
    # Average reliability
    # --------------------------------------------------------

    reliability_values = []

    for col in sensor_df.columns:

        if (
            "reliability" in col.lower()
            and pd.notna(row[col])
        ):

            try:

                reliability_values.append(
                    float(row[col])
                )

            except Exception:

                pass

    if reliability_values:

        sensor_reliabilities.append(
            float(
                np.mean(
                    reliability_values
                )
            )
        )

    else:

        sensor_reliabilities.append(
            None
        )


sensor_predictions = np.array(
    sensor_predictions
)

print(
    "Sensor predictions:",
    len(sensor_predictions)
)


# ============================================================
# RUN TEMPORAL MODEL
# ============================================================

print()
print("Running temporal sampling predictions...")


temporal_predictions = []

temporal_confidences = []


for idx in range(
    len(temporal_df)
):

    row = temporal_df.iloc[idx]

    X = prepare_temporal_input(
        row
    )

    prediction = temporal_model.predict(
        X
    )[0]

    temporal_predictions.append(
        str(prediction)
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = None

    if hasattr(
        temporal_model,
        "predict_proba"
    ):

        try:

            probabilities = temporal_model.predict_proba(
                X
            )[0]

            confidence = float(
                np.max(probabilities)
            )

        except Exception:

            confidence = None

    temporal_confidences.append(
        confidence
    )


temporal_predictions = np.array(
    temporal_predictions
)

print(
    "Temporal predictions:",
    len(temporal_predictions)
)


# ============================================================
# DATASET SIZES
# ============================================================

SENSOR_COUNT = len(
    sensor_predictions
)

TEMPORAL_COUNT = len(
    temporal_predictions
)


print()
print("============================================================")
print("DATASET SIZES")
print("============================================================")
print(
    "Sensor rows  :",
    SENSOR_COUNT
)
print(
    "Temporal rows:",
    TEMPORAL_COUNT
)
print(
    "These datasets are kept independent."
)
print()


# ============================================================
# BUILD SAFE DEMO REPLAY
# ============================================================
#
# VERY IMPORTANT:
#
# DO NOT do:
#
#     temporal_predictions[i]
#
# using an index selected from sensor_predictions.
#
# The two datasets have different sizes.
#
# Instead:
#
#     sensor_index
#     temporal_index
#
# are maintained independently.
# ============================================================


def representative_sensor_indices():

    selected = []

    actions = [
        "STOP",
        "ACTIVATE_S2",
        "ACTIVATE_S3",
        "ACTIVATE_S4",
        "ACTIVATE_S5",
        "ACTIVATE_S6",
        "ACTIVATE_S7",
        "ACTIVATE_S8",
    ]

    for action in actions:

        matches = np.where(
            sensor_predictions == action
        )[0]

        if len(matches) == 0:
            continue

        if action == "STOP":

            selected.extend(
                matches[:12].tolist()
            )

        else:

            selected.extend(
                matches[:8].tolist()
            )

    return selected


def representative_temporal_indices():

    selected = []

    actions = [
        "LOW_RATE",
        "MEDIUM_RATE",
        "HIGH_RATE",
    ]

    for action in actions:

        matches = np.where(
            temporal_predictions == action
        )[0]

        if len(matches) == 0:
            continue

        selected.extend(
            matches[:15].tolist()
        )

    return selected


sensor_demo_indices = (
    representative_sensor_indices()
)

temporal_demo_indices = (
    representative_temporal_indices()
)


# ------------------------------------------------------------
# If some action is missing, use first rows
# ------------------------------------------------------------

if len(sensor_demo_indices) == 0:

    sensor_demo_indices = list(
        range(
            min(
                SENSOR_COUNT,
                50
            )
        )
    )


if len(temporal_demo_indices) == 0:

    temporal_demo_indices = list(
        range(
            min(
                TEMPORAL_COUNT,
                50
            )
        )
    )


# ------------------------------------------------------------
# Remove duplicates
# ------------------------------------------------------------

sensor_demo_indices = list(
    dict.fromkeys(
        int(i)
        for i in sensor_demo_indices
        if 0 <= int(i) < SENSOR_COUNT
    )
)


temporal_demo_indices = list(
    dict.fromkeys(
        int(i)
        for i in temporal_demo_indices
        if 0 <= int(i) < TEMPORAL_COUNT
    )
)


# ============================================================
# CREATE DEMO STEPS
# ============================================================
#
# Each demonstration step has:
#
#   one sensor-model row
#   one temporal-model row
#
# They are NOT assumed to represent the same patient/window.
#
# This is a software architecture demonstration.
# ============================================================

DEMO_LENGTH = max(
    len(sensor_demo_indices),
    len(temporal_demo_indices)
)


if DEMO_LENGTH == 0:

    raise RuntimeError(
        "Could not create demonstration states."
    )


demo_steps = []


for step in range(
    DEMO_LENGTH
):

    sensor_index = sensor_demo_indices[
        step % len(sensor_demo_indices)
    ]

    temporal_index = temporal_demo_indices[
        step % len(temporal_demo_indices)
    ]

    demo_steps.append(
        {
            "step": step,
            "sensor_index": sensor_index,
            "temporal_index": temporal_index,
        }
    )


print()
print("============================================================")
print("DEMO REPLAY")
print("============================================================")
print(
    "Demo steps:",
    len(demo_steps)
)
print(
    "Sensor demo indices:",
    len(sensor_demo_indices)
)
print(
    "Temporal demo indices:",
    len(temporal_demo_indices)
)
print()


for item in demo_steps[:20]:

    s = item["sensor_index"]
    t = item["temporal_index"]

    print(
        "Step",
        item["step"],
        "→ Sensor row",
        s,
        ":",
        sensor_predictions[s],
        "| Temporal row",
        t,
        ":",
        temporal_predictions[t]
    )


print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sensor_display_name(
    action
):

    if action == "STOP":

        return "No additional sensor"

    if action.startswith(
        "ACTIVATE_"
    ):

        sensor_id = action.replace(
            "ACTIVATE_",
            ""
        )

        return SENSOR_MAP.get(
            sensor_id,
            sensor_id
        )

    return str(action)


def sampling_rate(
    action
):

    if action == "LOW_RATE":

        return "LOW"

    if action == "MEDIUM_RATE":

        return "MEDIUM"

    if action == "HIGH_RATE":

        return "HIGH"

    return "UNKNOWN"


def sampling_interval(
    action
):

    if action == "LOW_RATE":

        return "10 s"

    if action == "MEDIUM_RATE":

        return "5 s"

    if action == "HIGH_RATE":

        return "2 s"

    return "-"


def sensing_cost(
    sensor_action,
    sampling_action
):

    # Normalized demonstration cost.
    #
    # This is NOT actual battery energy.
    #
    # Sensor activation:
    # STOP = 1
    # Additional sensor = 2
    #
    # Sampling:
    # LOW = 1
    # MEDIUM = 2
    # HIGH = 4

    if sensor_action == "STOP":

        sensor_cost = 1

    else:

        sensor_cost = 2

    sampling_cost = {
        "LOW_RATE": 1,
        "MEDIUM_RATE": 2,
        "HIGH_RATE": 4,
    }.get(
        sampling_action,
        1
    )

    return sensor_cost * sampling_cost


def physiological_state(
    sampling_action
):

    if sampling_action == "LOW_RATE":

        return "STABLE"

    if sampling_action == "MEDIUM_RATE":

        return "CHANGING"

    if sampling_action == "HIGH_RATE":

        return "HIGH VARIABILITY"

    return "UNKNOWN"


def safe_float(
    value
):

    if value is None:

        return None

    try:

        value = float(value)

        if np.isnan(value):

            return None

        return value

    except Exception:

        return None


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__
)


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
Adaptive Biomedical Monitoring
</title>


<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        #0b1220;

    color:
        #f5f7fa;

}


.header {

    padding: 24px 40px;

    background:
        #111827;

    border-bottom:
        1px solid #273449;

}


.header h1 {

    margin: 0;

    font-size: 28px;

}


.header p {

    margin-top: 8px;

    color:
        #9ca3af;

}


.container {

    padding: 25px 40px;

}


.grid {

    display: grid;

    grid-template-columns:
        repeat(
            4,
            1fr
        );

    gap: 18px;

}


.card {

    background:
        #111827;

    border:
        1px solid #273449;

    border-radius:
        14px;

    padding:
        20px;

}


.card-title {

    font-size: 13px;

    color:
        #9ca3af;

    text-transform:
        uppercase;

    letter-spacing:
        1px;

}


.value {

    margin-top:
        10px;

    font-size:
        24px;

    font-weight:
        bold;

}


.main-grid {

    margin-top:
        20px;

    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        20px;

}


.panel {

    background:
        #111827;

    border:
        1px solid #273449;

    border-radius:
        14px;

    padding:
        22px;

}


.panel h2 {

    margin-top:
        0;

    font-size:
        20px;

}


.agent-box {

    padding:
        18px;

    margin-top:
        14px;

    background:
        #0b1220;

    border-radius:
        10px;

    border:
        1px solid #273449;

}


.agent-label {

    color:
        #9ca3af;

    font-size:
        12px;

    text-transform:
        uppercase;

}


.agent-value {

    margin-top:
        7px;

    font-size:
        20px;

    font-weight:
        bold;

}


button {

    margin-top:
        20px;

    padding:
        12px 22px;

    border:
        none;

    border-radius:
        8px;

    background:
        #2563eb;

    color:
        white;

    font-size:
        15px;

    cursor:
        pointer;

}


button:hover {

    background:
        #1d4ed8;

}


.chart-container {

    height:
        330px;

}


.sensor-list {

    margin-top:
        15px;

}


.sensor-row {

    display:
        flex;

    justify-content:
        space-between;

    padding:
        10px 0;

    border-bottom:
        1px solid #273449;

}


.sensor-row:last-child {

    border-bottom:
        none;

}


.status {

    display:
        inline-block;

    padding:
        6px 10px;

    border-radius:
        20px;

    background:
        #1f2937;

}


.small {

    color:
        #9ca3af;

    font-size:
        13px;

    line-height:
        1.5;

}


.footer {

    margin-top:
        20px;

    color:
        #6b7280;

    font-size:
        12px;

}


@media (
    max-width: 1000px
) {

    .grid {

        grid-template-columns:
            repeat(
                2,
                1fr
            );

    }

    .main-grid {

        grid-template-columns:
            1fr;

    }

}


</style>

</head>


<body>


<div class="header">

    <h1>
        Adaptive Biomedical Monitoring
    </h1>

    <p>
        AI-driven sensing and sampling policy demonstration
    </p>

</div>


<div class="container">


    <div class="grid">


        <div class="card">

            <div class="card-title">
                Physiological State
            </div>

            <div
                class="value"
                id="state"
            >
                --
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Sampling Rate
            </div>

            <div
                class="value"
                id="sampling"
            >
                --
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Active Sensor
            </div>

            <div
                class="value"
                id="sensor"
            >
                --
            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Normalized Cost
            </div>

            <div
                class="value"
                id="cost"
            >
                --
            </div>

        </div>


    </div>



    <div class="main-grid">


        <div class="panel">

            <h2>
                Sensor Selection Agent
            </h2>


            <div class="agent-box">

                <div class="agent-label">
                    Next Agent Action
                </div>

                <div
                    class="agent-value"
                    id="next-action"
                >
                    --
                </div>

            </div>


            <div class="agent-box">

                <div class="agent-label">
                    Sensor Policy Confidence
                </div>

                <div
                    class="agent-value"
                    id="sensor-confidence"
                >
                    --
                </div>

            </div>


            <div class="agent-box">

                <div class="agent-label">
                    Sensor Reliability
                </div>

                <div
                    class="agent-value"
                    id="reliability"
                >
                    --
                </div>

            </div>


            <div class="agent-box">

                <div class="agent-label">
                    Decision
                </div>

                <div
                    class="agent-value"
                    id="combined"
                >
                    --
                </div>

            </div>

        </div>



        <div class="panel">

            <h2>
                Temporal Sampling Agent
            </h2>


            <div class="agent-box">

                <div class="agent-label">
                    Sampling Decision
                </div>

                <div
                    class="agent-value"
                    id="sampling-action"
                >
                    --
                </div>

            </div>


            <div class="agent-box">

                <div class="agent-label">
                    Sampling Confidence
                </div>

                <div
                    class="agent-value"
                    id="temporal-confidence"
                >
                    --
                </div>

            </div>


            <div class="agent-box">

                <div class="agent-label">
                    Sampling Interval
                </div>

                <div
                    class="agent-value"
                    id="interval"
                >
                    --
                </div>

            </div>


            <div class="agent-box">

                <div class="agent-label">
                    Agent Principle
                </div>

                <div class="small">

                    Stable physiological state →
                    reduce sampling.

                    <br><br>

                    Changing state →
                    increase monitoring.

                    <br><br>

                    High variability →
                    monitor aggressively.

                </div>

            </div>

        </div>


    </div>



    <div class="panel" style="margin-top:20px">

        <h2>
            Adaptive Monitoring Timeline
        </h2>

        <div class="chart-container">

            <canvas id="chart"></canvas>

        </div>

    </div>



    <div class="panel" style="margin-top:20px">

        <h2>
            Current Sensor Configuration
        </h2>

        <div
            class="sensor-list"
            id="sensor-list"
        >
        </div>

    </div>


    <button onclick="resetDemo()">
        Restart Demonstration
    </button>


    <div class="footer">

        Software demonstration using independent
        sensor-policy and temporal-policy datasets.
        The datasets are not synchronized patient recordings.

    </div>


</div>


<script>


let chart;


let chartLabels = [];

let chartCosts = [];



function createChart() {

    const ctx =
        document
        .getElementById(
            "chart"
        )
        .getContext(
            "2d"
        );


    chart =
        new Chart(
            ctx,
            {

                type:
                    "line",

                data:
                    {

                        labels:
                            chartLabels,

                        datasets:
                            [
                                {

                                    label:
                                        "Normalized Sensing Cost",

                                    data:
                                        chartCosts,

                                    tension:
                                        0.3

                                }

                            ]

                    },

                options:
                    {

                        responsive:
                            true,

                        maintainAspectRatio:
                            false,

                        scales:
                            {

                                y:
                                    {

                                        beginAtZero:
                                            true

                                    }

                            }

                    }

            }
        );

}



function updateChart(
    step,
    cost
) {

    chartLabels.push(
        step
    );

    chartCosts.push(
        cost
    );


    if (
        chartLabels.length > 30
    ) {

        chartLabels.shift();

        chartCosts.shift();

    }


    chart.update();

}



async function updateDashboard() {

    try {

        const response =
            await fetch(
                "/state"
            );


        const data =
            await response.json();


        document
            .getElementById(
                "state"
            )
            .innerText =
            data.state;


        document
            .getElementById(
                "sampling"
            )
            .innerText =
            data.sampling_rate;


        document
            .getElementById(
                "sensor"
            )
            .innerText =
            data.active_sensor;


        document
            .getElementById(
                "cost"
            )
            .innerText =
            data.cost;


        document
            .getElementById(
                "next-action"
            )
            .innerText =
            data.next_agent_action;


        document
            .getElementById(
                "sensor-confidence"
            )
            .innerText =
            data.sensor_confidence;


        document
            .getElementById(
                "reliability"
            )
            .innerText =
            data.sensor_reliability;


        document
            .getElementById(
                "combined"
            )
            .innerText =
            data.combined_decision;


        document
            .getElementById(
                "sampling-action"
            )
            .innerText =
            data.sampling_action;


        document
            .getElementById(
                "temporal-confidence"
            )
            .innerText =
            data.temporal_confidence;


        document
            .getElementById(
                "interval"
            )
            .innerText =
            data.interval;


        updateChart(
            data.step,
            data.cost
        );


        const list =
            document
            .getElementById(
                "sensor-list"
            );


        list.innerHTML = "";


        data.sensor_configuration
            .forEach(
                function(sensor) {

                    const row =
                        document.createElement(
                            "div"
                        );

                    row.className =
                        "sensor-row";


                    const name =
                        document.createElement(
                            "span"
                        );

                    name.innerText =
                        sensor.name;


                    const status =
                        document.createElement(
                            "span"
                        );

                    status.className =
                        "status";

                    status.innerText =
                        sensor.status;


                    row.appendChild(
                        name
                    );

                    row.appendChild(
                        status
                    );


                    list.appendChild(
                        row
                    );

                }
            );


    }

    catch(error) {

        console.error(
            error
        );

    }

}



async function resetDemo() {

    await fetch(
        "/reset",
        {
            method:
                "POST"
        }
    );


    chartLabels.length = 0;

    chartCosts.length = 0;

    chart.update();


    updateDashboard();

}



createChart();

updateDashboard();


setInterval(
    updateDashboard,
    2500
);


</script>


</body>

</html>
"""


# ============================================================
# DEMO POINTER
# ============================================================

current_step = 0


# ============================================================
# BUILD CURRENT STATE
# ============================================================

def build_state():

    global current_step


    item = demo_steps[
        current_step
        % len(demo_steps)
    ]


    sensor_index = item[
        "sensor_index"
    ]

    temporal_index = item[
        "temporal_index"
    ]


    # --------------------------------------------------------
    # Sensor decision
    # --------------------------------------------------------

    sensor_action = (
        sensor_predictions[
            sensor_index
        ]
    )


    sensor_name = sensor_display_name(
        sensor_action
    )


    sensor_confidence = (
        sensor_confidences[
            sensor_index
        ]
    )


    if sensor_confidence is None:

        sensor_confidence_text = "N/A"

    else:

        sensor_confidence_text = (
            f"{sensor_confidence * 100:.1f}%"
        )


    reliability = (
        sensor_reliabilities[
            sensor_index
        ]
    )


    if reliability is None:

        reliability_text = "N/A"

    else:

        reliability_text = (
            f"{reliability * 100:.1f}%"
        )


    # --------------------------------------------------------
    # Temporal decision
    # --------------------------------------------------------

    temporal_action = (
        temporal_predictions[
            temporal_index
        ]
    )


    temporal_confidence = (
        temporal_confidences[
            temporal_index
        ]
    )


    if temporal_confidence is None:

        temporal_confidence_text = "N/A"

    else:

        temporal_confidence_text = (
            f"{temporal_confidence * 100:.1f}%"
        )


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state = physiological_state(
        temporal_action
    )


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    rate = sampling_rate(
        temporal_action
    )


    interval = sampling_interval(
        temporal_action
    )


    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    cost = sensing_cost(
        sensor_action,
        temporal_action
    )


    # --------------------------------------------------------
    # Combined decision
    # --------------------------------------------------------

    combined = (
        sensor_action
        + " + "
        + temporal_action
    )


    # --------------------------------------------------------
    # Sensor configuration
    # --------------------------------------------------------

    configuration = []


    for sensor_id, sensor_name_full in SENSOR_MAP.items():

        if sensor_action == "STOP":

            status = "Standby"

        else:

            requested_sensor = (
                sensor_action.replace(
                    "ACTIVATE_",
                    ""
                )
            )

            if sensor_id == requested_sensor:

                status = "ACTIVE"

            else:

                status = "Standby"


        configuration.append(
            {
                "id": sensor_id,
                "name": sensor_name_full,
                "status": status,
            }
        )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = {

        "step":
            current_step,

        "sensor_index":
            sensor_index,

        "temporal_index":
            temporal_index,

        "state":
            state,

        "sampling_rate":
            rate,

        "sampling_action":
            temporal_action,

        "interval":
            interval,

        "next_agent_action":
            sensor_name,

        "sensor_action":
            sensor_action,

        "active_sensor":
            sensor_name,

        "sensor_confidence":
            sensor_confidence_text,

        "temporal_confidence":
            temporal_confidence_text,

        "sensor_reliability":
            reliability_text,

        "combined_decision":
            combined,

        "cost":
            cost,

        "sensor_configuration":
            configuration,

    }


    # Move to next step for next request
    current_step += 1


    return result


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


# ------------------------------------------------------------
# State API
# ------------------------------------------------------------

@app.route("/state")
def state():

    return jsonify(
        build_state()
    )


# ------------------------------------------------------------
# Reset
# ------------------------------------------------------------

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    global current_step

    current_step = 0

    return jsonify(
        {
            "status": "reset"
        }
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("============================================================")
    print("DASHBOARD READY")
    print("============================================================")
    print()
    print(
        "Open:"
    )
    print(
        "http://127.0.0.1:5005"
    )
    print()
    print(
        "Press CTRL+C to stop."
    )
    print()

    app.run(
        host="0.0.0.0",
        port=5005,
        debug=False,
        use_reloader=False
    )
