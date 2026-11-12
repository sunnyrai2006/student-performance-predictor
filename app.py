
import streamlit as st
import pandas as pd
import requests

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

API_URL = "http://127.0.0.1:5000/predict"

# ---------------- HEADER ----------------

st.title("🎓 AI-Based Student Performance Predictor")

st.caption(
    "Predict student academic performance using "
    "Machine Learning and Artificial Neural Networks."
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Settings")

selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    [
        "ANN",
        "Logistic Regression",
        "Random Forest",
        "Support Vector Machine",
        "Compare All Models"
    ]
)

# ---------------- INPUT FORM ----------------

st.subheader("Enter Student Information")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age", 10, 100, 21
        )

        attendance = st.number_input(
            "Attendance (%)", 0.0, 100.0, 85.0
        )

        study_hours = st.number_input(
            "Study Hours Per Week", 0.0, 168.0, 20.0
        )

        previous_score = st.number_input(
            "Previous Score", 0.0, 100.0, 75.0
        )

        assignment = st.number_input(
            "Assignment Completion (%)", 0.0, 100.0, 90.0
        )

    with col2:

        internal_score = st.number_input(
            "Internal Assessment Score", 0.0, 100.0, 78.0
        )

        participation = st.number_input(
            "Participation Score", 0.0, 10.0, 7.0
        )

        sleep = st.number_input(
            "Sleep Hours Per Day", 0.0, 24.0, 7.0
        )

        extracurricular = st.number_input(
            "Extracurricular Hours Per Week", 0.0, 168.0, 5.0
        )

        failed_subjects = st.number_input(
            "Failed Subjects", 0, 50, 0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Performance",
        use_container_width=True
    )

# ---------------- DISPLAY RESULT ----------------

def display_result(model_name, result):

    prediction = result["prediction"]
    confidence = result.get("confidence")
    probabilities = result.get("probabilities")

    st.markdown(f"### {model_name}")

    if prediction.lower() == "high":
        st.success(f"Performance Category: {prediction}")

    elif prediction.lower() == "medium":
        st.warning(f"Performance Category: {prediction}")

    else:
        st.error(f"Performance Category: {prediction}")

    if confidence is not None:
        st.metric(
            "Prediction Confidence",
            f"{confidence}%"
        )

    if probabilities:

        probability_df = pd.DataFrame({
            "Performance": list(probabilities.keys()),
            "Probability (%)": [
                value * 100
                for value in probabilities.values()
            ]
        })

        st.write("Class Probabilities")

        st.bar_chart(
            probability_df.set_index("Performance")
        )

# ---------------- API PREDICTION ----------------

if submitted:

    input_data = {
        "model": selected_model,
        "age": age,
        "attendance_pct": attendance,
        "study_hours_per_week": study_hours,
        "previous_score": previous_score,
        "assignment_completion_pct": assignment,
        "internal_assessment_score": internal_score,
        "participation_score": participation,
        "sleep_hours_per_day": sleep,
        "extracurricular_hours_per_week": extracurricular,
        "failed_subjects": failed_subjects
    }

    st.divider()
    st.subheader("Prediction Results")

    try:

        with st.spinner("Analyzing student information..."):

            response = requests.post(
                API_URL,
                json=input_data,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            if selected_model == "Compare All Models":

                results = result["results"]

                st.success("All models completed their predictions.")

                for model_name, model_result in results.items():

                    with st.expander(
                        model_name,
                        expanded=True
                    ):

                        display_result(
                            model_name,
                            model_result
                        )

            else:

                display_result(
                    selected_model,
                    result
                )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            try:
                st.json(response.json())
            except ValueError:
                st.write(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to Flask API. "
            "Please start api.py first."
        )

    except requests.exceptions.Timeout:

        st.error("API request timed out.")

    except Exception as e:

        st.error(f"Unexpected error: {e}")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI-Based Student Performance Predictor | "
    "Machine Learning + Artificial Neural Network"
)

st.caption(
    "This application provides educational predictions "
    "based on model outputs. It is not a definitive assessment "
    "of a student's ability or future performance."
)
