import streamlit as st
from ultralytics import YOLO
from PIL import Image
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px

import sys
import os

# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# --------------------------------------------------

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from database import create_database, save_detection, get_detections


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RoadVision AI",
    page_icon="🚧",
    layout="wide"
)


# --------------------------------------------------
# CREATE DATABASE
# --------------------------------------------------

create_database()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "detection_data" not in st.session_state:
    st.session_state.detection_data = None

if "detected_image" not in st.session_state:
    st.session_state.detected_image = None

if "database_saved" not in st.session_state:
    st.session_state.database_saved = False

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🚧 RoadVision AI")

st.sidebar.write(
    "Intelligent Road Damage Detection "
    "and Mapping System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Detection",
        "📊 Dashboard",
        "🗺️ Damage Map",
        "📋 Detection History"
    ]
)
# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🚧 RoadVision AI")

st.write(
    "Intelligent Road Damage Detection and Mapping System"
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
    
st.header("📷 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    @st.cache_resource
    def load_model():
        return YOLO("models/road_damage_best.pt")


    model = load_model()

    st.success(
        "Road damage detection model loaded successfully!"
    )


    # --------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Road Image")

    st.image(
        image,
        caption="Uploaded Road Image",
        width="stretch"
    )


    # --------------------------------------------------
    # DETECT BUTTON
    # --------------------------------------------------

    if st.button("🔍 Detect Road Damage"):

        # Reset database saved status
        st.session_state.database_saved = False

        with st.spinner(
            "Analyzing road image..."
        ):

            results = model(
                image,
                conf=0.25
            )


        detection_data = []
        detected_image = None


        # --------------------------------------------------
        # PROCESS YOLO RESULTS
        # --------------------------------------------------

        for result in results:

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                detected_image = result.plot()


                for box in result.boxes:

                    # --------------------------------------
                    # CLASS
                    # --------------------------------------

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = model.names[
                        class_id
                    ]


                    # --------------------------------------
                    # CONFIDENCE
                    # --------------------------------------

                    confidence = float(
                        box.conf[0]
                    )


                    # --------------------------------------
                    # BOUNDING BOX
                    # --------------------------------------

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    box_width = x2 - x1
                    box_height = y2 - y1


                    # --------------------------------------
                    # AREA
                    # --------------------------------------

                    damage_area = (
                        box_width * box_height
                    )

                    image_width, image_height = (
                        image.size
                    )

                    image_area = (
                        image_width * image_height
                    )

                    area_percentage = (
                        damage_area / image_area
                    ) * 100


                    # --------------------------------------
                    # SEVERITY
                    # --------------------------------------

                    if area_percentage >= 15:

                        severity = "High"

                    elif area_percentage >= 5:

                        severity = "Medium"

                    else:

                        severity = "Low"


                    # --------------------------------------
                    # STORE RESULT
                    # --------------------------------------

                    detection_data.append({

                        "Damage Type":
                            class_name,

                        "Confidence":
                            confidence,

                        "Severity":
                            severity,

                        "Affected Area (%)":
                            area_percentage

                    })


        # --------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------

        st.session_state.detection_data = (
            detection_data
        )

        st.session_state.detected_image = (
            detected_image
        )


        if detection_data:

            st.success(
                f"{len(detection_data)} road damage(s) detected."
            )

        else:

            st.warning(
                "No road damage detected in this image."
            )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if st.session_state.detection_data:

    df = pd.DataFrame(
        st.session_state.detection_data
    )


    # --------------------------------------------------
    # DETECTION RESULT
    # --------------------------------------------------

    st.subheader("🔍 Detection Result")


    if (
        st.session_state.detected_image
        is not None
    ):

        st.image(
            st.session_state.detected_image,
            caption="Detected Road Damage",
            width="stretch"
        )


    # --------------------------------------------------
    # DAMAGE DETAILS
    # --------------------------------------------------

    st.subheader("Damage Details")


    for _, row in df.iterrows():

        st.write(
            f"**Damage Type:** "
            f"{row['Damage Type']}"
        )

        st.write(
            f"**Confidence:** "
            f"{row['Confidence']:.2%}"
        )


        if row["Severity"] == "High":

            severity_display = "High 🔴"

        elif row["Severity"] == "Medium":

            severity_display = "Medium 🟠"

        else:

            severity_display = "Low 🟢"


        st.write(
            f"**Estimated Severity:** "
            f"{severity_display}"
        )

        st.write(
            f"**Affected Area:** "
            f"{row['Affected Area (%)']:.2f}%"
        )

        st.divider()


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    st.subheader("📊 Detection Summary")


    total_damage = len(df)

    average_confidence = (
        df["Confidence"].mean()
    )

    average_area = (
        df["Affected Area (%)"].mean()
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Damage",
            total_damage
        )


    with col2:

        st.metric(
            "Average Confidence",
            f"{average_confidence:.2%}"
        )


    with col3:

        st.metric(
            "Average Affected Area",
            f"{average_area:.2f}%"
        )


    # --------------------------------------------------
    # DAMAGE TYPE CHART
    # --------------------------------------------------

    st.subheader(
        "📈 Damage Type Distribution"
    )


    damage_counts = (
        df["Damage Type"]
        .value_counts()
        .reset_index()
    )

    damage_counts.columns = [
        "Damage Type",
        "Count"
    ]


    fig_damage = px.bar(
        damage_counts,
        x="Damage Type",
        y="Count",
        title="Detected Road Damage Types"
    )


    st.plotly_chart(
        fig_damage,
        use_container_width=True
    )


    # --------------------------------------------------
    # SEVERITY CHART
    # --------------------------------------------------

    st.subheader(
        "📊 Severity Distribution"
    )


    severity_counts = (
        df["Severity"]
        .value_counts()
        .reset_index()
    )

    severity_counts.columns = [
        "Severity",
        "Count"
    ]


    fig_severity = px.pie(
        severity_counts,
        names="Severity",
        values="Count",
        title="Road Damage Severity"
    )


    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )


    # --------------------------------------------------
    # TABLE
    # --------------------------------------------------

    st.subheader(
        "📋 Detection Details Table"
    )


    display_df = df.copy()


    display_df["Confidence (%)"] = (
        display_df["Confidence"] * 100
    ).round(2)


    display_df["Affected Area (%)"] = (
        display_df["Affected Area (%)"]
        .round(2)
    )


    display_df = display_df.drop(
        columns=["Confidence"]
    )


    st.dataframe(
        display_df,
        use_container_width=True
    )


    # --------------------------------------------------
    # LOCATION
    # --------------------------------------------------

    st.subheader("📍 Road Location")


    st.write(
        "Enter the approximate road location manually."
    )


    col1, col2 = st.columns(2)


    with col1:

        latitude = st.number_input(
            "Latitude",
            value=11.0168,
            format="%.6f"
        )


    with col2:

        longitude = st.number_input(
            "Longitude",
            value=76.9558,
            format="%.6f"
        )


    st.write(
        f"**Selected Location:** "
        f"{latitude:.6f}, "
        f"{longitude:.6f}"
    )


    # --------------------------------------------------
    # SAVE TO DATABASE
    # --------------------------------------------------

    st.subheader("💾 Save Detection")


    if st.button("💾 Save Detection to Database"):

        if not st.session_state.database_saved:

            for detection in st.session_state.detection_data:

                save_detection(

                    damage_type=
                        detection["Damage Type"],

                    confidence=
                        detection["Confidence"],

                    severity=
                        detection["Severity"],

                    affected_area=
                        detection["Affected Area (%)"],

                    latitude=latitude,

                    longitude=longitude

                )


            st.session_state.database_saved = True


            st.success(
                "Detection and location saved successfully!"
            )

        else:

            st.info(
                "This detection has already been saved."
            )


    # --------------------------------------------------
    # MAP
    # --------------------------------------------------

    st.subheader(
        "🗺️ Road Damage Location Map"
    )


    road_map = folium.Map(

        location=[
            latitude,
            longitude
        ],

        zoom_start=14

    )


    folium.Marker(

        [
            latitude,
            longitude
        ],

        popup=(

            f"Road Damage Detected<br>"
            f"Damage Count: {total_damage}"

        ),

        tooltip="Road Damage Location"

    ).add_to(road_map)


    st_folium(

        road_map,

        width=1000,

        height=500

    )


    # --------------------------------------------------
    # CSV REPORT
    # --------------------------------------------------

    st.subheader(
        "📄 Road Damage Report"
    )


    report_df = df.copy()


    report_df["Confidence (%)"] = (
        report_df["Confidence"] * 100
    ).round(2)


    report_df["Affected Area (%)"] = (
        report_df["Affected Area (%)"]
        .round(2)
    )


    report_df = report_df.drop(
        columns=["Confidence"]
    )


    report_df["Latitude"] = latitude

    report_df["Longitude"] = longitude


    report_csv = report_df.to_csv(
        index=False
    )


    st.download_button(

        label="⬇️ Download Detection Report",

        data=report_csv,

        file_name="road_damage_report.csv",

        mime="text/csv"

    )


# --------------------------------------------------
# BEFORE DETECTION
# --------------------------------------------------

elif uploaded_file is not None:

    st.info(
        "Click 'Detect Road Damage' "
        "to analyze the uploaded image."
    )
    # --------------------------------------------------
# DETECTION HISTORY
# --------------------------------------------------
st.header("📊 RoadVision AI Dashboard")

# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

history = get_detections()

if history:

    history_df = pd.DataFrame(
        history,
        columns=[
            "ID",
            "Damage Type",
            "Confidence",
            "Severity",
            "Affected Area (%)",
            "Latitude",
            "Longitude",
            "Detected At"
        ]
    )

    total_detections = len(history_df)

    total_potholes = (
        history_df["Damage Type"]
        .str.lower()
        .eq("pothole")
        .sum()
    )

    total_cracks = (
        history_df["Damage Type"]
        .str.lower()
        .str.contains("crack")
        .sum()
    )

    high_severity = (
        history_df["Severity"]
        .eq("High")
        .sum()
    )

    average_confidence = (
        history_df["Confidence"]
        .mean()
    )

    # ----------------------------------------------
    # METRIC CARDS
    # ----------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Detections",
            total_detections
        )

    with col2:
        st.metric(
            "Potholes",
            total_potholes
        )

    with col3:
        st.metric(
            "Cracks",
            total_cracks
        )

    with col4:
        st.metric(
            "High Severity",
            high_severity
        )

    with col5:
        st.metric(
            "Avg Confidence",
            f"{average_confidence:.2%}"
        )
    # --------------------------------------------------
    # DASHBOARD CHARTS
    # --------------------------------------------------

    st.subheader("📈 Damage Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        damage_chart = (
            history_df["Damage Type"]
            .value_counts()
            .reset_index()
        )

        damage_chart.columns = [
            "Damage Type",
            "Count"
        ]

        fig_damage_history = px.bar(
            damage_chart,
            x="Damage Type",
            y="Count",
            title="Damage Type Distribution"
        )

        st.plotly_chart(
            fig_damage_history,
            use_container_width=True
        )

    with chart_col2:

        severity_chart = (
            history_df["Severity"]
            .value_counts()
            .reset_index()
        )

        severity_chart.columns = [
            "Severity",
            "Count"
        ]

        fig_severity_history = px.pie(
            severity_chart,
            names="Severity",
            values="Count",
            title="Severity Distribution"
        )

        st.plotly_chart(
            fig_severity_history,
            use_container_width=True
        )
        
else:

    st.info(
        "Dashboard data will appear after saving detections."
    )
   # --------------------------------------------------
# HISTORICAL DAMAGE MAP
# --------------------------------------------------

st.subheader("🗺️ Historical Damage Map")

history_map_data = get_detections()

if history_map_data:

    map_df = pd.DataFrame(
        history_map_data,
        columns=[
            "ID",
            "Damage Type",
            "Confidence",
            "Severity",
            "Affected Area (%)",
            "Latitude",
            "Longitude",
            "Detected At"
        ]
    )

    valid_locations = map_df.dropna(
        subset=["Latitude", "Longitude"]
    )

    if not valid_locations.empty:

        center_lat = valid_locations["Latitude"].mean()
        center_lon = valid_locations["Longitude"].mean()

        history_map = folium.Map(
            location=[
                center_lat,
                center_lon
            ],
            zoom_start=11
        )

        for _, row in valid_locations.iterrows():

            popup_text = (
                f"<b>Damage:</b> {row['Damage Type']}<br>"
                f"<b>Severity:</b> {row['Severity']}<br>"
                f"<b>Confidence:</b> "
                f"{row['Confidence']:.2%}<br>"
                f"<b>Location:</b> "
                f"{row['Latitude']:.6f}, "
                f"{row['Longitude']:.6f}"
            )

            folium.Marker(
                location=[
                    row["Latitude"],
                    row["Longitude"]
                ],
                popup=popup_text,
                tooltip=row["Damage Type"]
            ).add_to(history_map)

        st_folium(
            history_map,
            width=1000,
            height=500
        )

    else:
        st.info(
            "No valid location data available."
        )

else:
    st.info(
        "Save detections with latitude and longitude "
        "to display them on the historical map."
    )
# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "📊 Dashboard":

    st.header("📊 RoadVision AI Dashboard")

    history = get_detections()

    if history:

        history_df = pd.DataFrame(
            history,
            columns=[
                "ID",
                "Damage Type",
                "Confidence",
                "Severity",
                "Affected Area (%)",
                "Latitude",
                "Longitude",
                "Detected At"
            ]
        )

        total_detections = len(history_df)

        total_potholes = (
            history_df["Damage Type"]
            .str.lower()
            .eq("pothole")
            .sum()
        )

        total_cracks = (
            history_df["Damage Type"]
            .str.lower()
            .str.contains("crack")
            .sum()
        )

        high_severity = (
            history_df["Severity"]
            .eq("High")
            .sum()
        )

        average_confidence = (
            history_df["Confidence"]
            .mean()
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Detections", total_detections)

        with col2:
            st.metric("Potholes", total_potholes)

        with col3:
            st.metric("Cracks", total_cracks)

        with col4:
            st.metric("High Severity", high_severity)

        with col5:
            st.metric(
                "Avg Confidence",
                f"{average_confidence:.2%}"
            )

        st.subheader("📈 Damage Analytics")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            damage_chart = (
                history_df["Damage Type"]
                .value_counts()
                .reset_index()
            )

            damage_chart.columns = [
                "Damage Type",
                "Count"
            ]

            fig_damage_history = px.bar(
                damage_chart,
                x="Damage Type",
                y="Count",
                title="Damage Type Distribution"
            )

            st.plotly_chart(
                fig_damage_history,
                use_container_width=True
            )

        with chart_col2:

            severity_chart = (
                history_df["Severity"]
                .value_counts()
                .reset_index()
            )

            severity_chart.columns = [
                "Severity",
                "Count"
            ]

            fig_severity_history = px.pie(
                severity_chart,
                names="Severity",
                values="Count",
                title="Severity Distribution"
            )

            st.plotly_chart(
                fig_severity_history,
                use_container_width=True
            )

    else:

        st.info(
            "Dashboard data will appear after saving detections."
        )

# --------------------------------------------------
# DETECTION HISTORY
# --------------------------------------------------

st.subheader("📋 Detection History")

history = get_detections()

if history:

    history_df = pd.DataFrame(
        history,
        columns=[
            "ID",
            "Damage Type",
            "Confidence",
            "Severity",
            "Affected Area (%)",
            "Latitude",
            "Longitude",
            "Detected At"
        ]
    )

    # Format confidence
    history_df["Confidence"] = (
        history_df["Confidence"] * 100
    ).round(2)

    history_df = history_df.rename(
        columns={
            "Confidence": "Confidence (%)"
        }
    )

    history_df["Affected Area (%)"] = (
        history_df["Affected Area (%)"]
        .round(2)
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No detection history available yet."
    )