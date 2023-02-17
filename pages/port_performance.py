import os
import yaml

import streamlit as st
import altair as alt
import pandas as pd

from sklearn.feature_selection import f_regression

from utils import process_schedule_data, restrict_by_coverage, load_excel_data


st.set_page_config(page_title="Port Performance", page_icon="⚓")

st.markdown("# Port Dwell")
st.sidebar.header("Port Performance")
st.write(
    """We visualize the relationship between average wait time (schedule data)
    and port dwell time (port performance data) and compute an F-statistic p-value
    as a preliminary check for linear regression on wait time."""
)

# DATA

config = yaml.safe_load(open('config.yml', 'r'))

data_path = config["data_path"]


# SCHEDULE

# read in reliability schedule data
schedule_file_path = os.path.join(
    data_path,
    config["schedule"]["filename"]
)
schedule_data = pd.read_csv(schedule_file_path)

rel_df_nona = process_schedule_data(schedule_data)
rel_df_nona = restrict_by_coverage(rel_df_nona)

# exclude rows with port code USORF from rel_df since it's missing
rel_df_no_orf = rel_df_nona[~rel_df_nona.POD.isin(["USORF"])]


# PORT PERFORMANCE
port_data = load_excel_data(
    config,
    "port_call"
)

port_call_df = port_data

show_port_call = st.checkbox("Show port call data")
if show_port_call:
    st.write(port_call_df)

# ALIGN PORT DATA WITH SCHEDULE
# create new column seaport_code
# for port_call_df and rel_df
# eliminating ambiguous port codes
seaport_code_map= {"CNSHG": "CNSHA", "CNTNJ": "CNTXG", "CNQIN": "CNTAO"}

# add seaport_code column to port data
port_call_df.loc[:, "seaport_code"] = port_call_df["UNLOCODE"].apply(
    lambda x: seaport_code_map[x] if x in seaport_code_map else x
)

# do the same for rel_df
rel_df_no_orf.loc[:, "seaport_code"] = rel_df_no_orf["POD"]

# compute average hours per call
agg_cols = ["seaport_code", "Month", "Year"]
target_cols = ["Total_Calls", "Port_Hours", "Anchorage_Hours"]

# sum up calls, port/anchorage hours
# and aggregate by port, month, and year
port_hours_avg = port_call_df[target_cols + agg_cols].groupby(
    agg_cols
).sum().reset_index()

# average port hours by port, month
port_hours_avg.loc[:, "Avg_Port_Hours(by_call)"] = port_hours_avg[
    "Port_Hours"
] / port_hours_avg["Total_Calls"]

# average anchorage hours by port, month
port_hours_avg.loc[:, "Avg_Anchorage_Hours(by_call)"] = port_hours_avg[
    "Anchorage_Hours"
] / port_hours_avg["Total_Calls"]

port_hours_avg_2022 = port_hours_avg[port_hours_avg["Year"]==2022]

# merge avg hours
rel_df_no_orf_pt_hrs = rel_df_no_orf.merge(
    port_hours_avg_2022,
    left_on=["Calendary_Year", "Month(int)", "seaport_code"],
    right_on=["Year", "Month", "seaport_code"]
)


with st.sidebar:
    POD_options = tuple(
        rel_df_no_orf_pt_hrs["POD"].unique()
    )

    POD_option = st.selectbox(
        'POD: ',
        POD_options)

    # TODO: include in a diff. page
    # carrier_options = tuple(rel_df_no_orf_pt_hrs[
    #     rel_df_no_orf_pt_hrs["POD"]==POD_option
    # ]["Carrier"].unique()
    # )

    # carrier_option = st.selectbox(
    #     'Carrier: ',
    #     carrier_options
    # )

    plot = st.button("Plot")


if plot:

    pod_mask = rel_df_no_orf_pt_hrs["POD"]==POD_option

    # TODO: include in a different page
    # carrier_mask = rel_df_no_orf_pt_hrs["Carrier"]==carrier_option
    # source = rel_df_no_orf_pt_hrs[
    #     pod_mask &
    #     carrier_mask
    # ]

    source = rel_df_no_orf_pt_hrs[
        pod_mask
    ]

    # TODO: implement drop down menu for target labels
    # compute p-value
    label = "Avg_WaitTime_POD_Days"
    predictor_label = "Avg_Port_Hours(by_call)"  #"Avg_Anchorage_Hours"
    target = source[label]

    predictor_format_label = ""

    if predictor_label == "Avg_Anchorage_Hours(by_call)":
        predictor_format_label = "Anchorage"
    else:
        predictor_format_label = "Service"  # Dhaval and Jiahao found service hours include anchorage time

    predictors = source[[predictor_label]]

    p_value = round(f_regression(predictors, target)[1][0], 8)

    # TODO: include in a different page
    # base = alt.Chart(source, title=f"Wait/{predictor_format_label} Time at port {POD_option} for carrier {carrier_option}\n(p-value={p_value})").encode(
    #     alt.X('month(Date):T', axis=alt.Axis(title=None))
    # )

    base = alt.Chart(source, title=f"Wait/{predictor_format_label} Time at port {POD_option} \n(p-value={p_value})").encode(
        alt.X('month(Date):T', axis=alt.Axis(title=None))
    )

    transittime = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
        alt.Y(f'average({label})',
            axis=alt.Axis(title='Avg. Wait Time (days)', titleColor='#5276A7'))
    )

    anchoragetime = base.mark_line(stroke='green', interpolate='monotone').encode(
        alt.Y(f'average({predictor_label})',
            axis=alt.Axis(title=f'Avg. {predictor_format_label} Time (hours)', titleColor='green'))
    )

    c = alt.layer(transittime, anchoragetime).resolve_scale(
        y = 'independent'
    )

    st.altair_chart(c, use_container_width=True)
