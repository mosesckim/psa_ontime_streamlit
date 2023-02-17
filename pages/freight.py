import os
import yaml

import streamlit as st
import altair as alt
import pandas as pd

from sklearn.feature_selection import f_regression

from utils import process_schedule_data, restrict_by_coverage, load_excel_data


st.set_page_config(page_title="Freight", page_icon="🚢")

st.markdown("# Carrier Rate")
st.sidebar.header("Freight")
st.write(
    """We visualize the relationship between average transit time and freight rate
    for the Asia-Europe trade from February 2022 through August 2022 and compute an
    F-statistic p-value as a preliminary check for linear regression on
    average transit time."""
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


# CARRIER RATE
carrier_data = load_excel_data(
    config,
    "xeneta"
)


show_freight_data = st.checkbox("Show freight data")
if show_freight_data:
    st.write(carrier_data)


freight_df = carrier_data

# restrict to 2022
freight_df.loc[:, "year"] = freight_df["Date"].apply(
    lambda x: int(x[:4])
)

freight_df.loc[:, "month"] = freight_df["Date"].apply(
    lambda x: int(x[5:7])
)

freight_df_2022 = freight_df[freight_df["year"]==2022]

# now group by month and carrier
freight_df_2022_month_carr = freight_df_2022.groupby(
    ["Carrier Name", "month"]
).mean().reset_index()[["Carrier Name", "month", "Carrier Average"]]

# write map dictionary
freight_carrier_map = {
    'American President Line': 'APL',
    'China Ocean Shipping Group': 'COSCO SHIPPING',
    'Evergreen': 'EVERGREEN',
    'Hapag Lloyd': 'HAPAG-LLOYD',
    'Maersk Line': 'MAERSK',
    'Mediterranean Shipping Company': 'MSC',
    'Orient Overseas Container Line': 'OOCL',
    'Yang Ming Lines': 'YANG MING',
    'Hamburg Süd': 'HAMBURG SÜD',
    'HYUNDAI Merchant Marine': 'HMM',
    'Ocean Network Express (ONE)': 'ONE'
}


# create new column
freight_df_2022_month_carr.loc[:, "Carrier"] = freight_df_2022_month_carr[
    "Carrier Name"
].apply(lambda x: freight_carrier_map[x] if x in freight_carrier_map else x)

freight_df_2022_month_carr.drop("Carrier Name", inplace=True, axis=1)

# ALIGN

# first restrict schedule data to asia-europe trade routes
rel_df_no_orf = rel_df_no_orf[
    rel_df_no_orf["Trade"]=="Asia-Europe"
]

# merge freight
rel_df_no_orf_freight = rel_df_no_orf.merge(
    freight_df_2022_month_carr,
    left_on=["Carrier", "Month(int)"],
    right_on=["Carrier", "month"]
)


with st.sidebar:

    carrier_options = tuple(
        rel_df_no_orf_freight["Carrier"].unique()
    )

    carrier_option = st.selectbox(
        'Carrier: ',
        carrier_options)

    plot = st.button("Plot")


if plot:

    # average transit time across all carriers
    carrier_mask = rel_df_no_orf_freight["Carrier"]==carrier_option
    source = rel_df_no_orf_freight[
        carrier_mask
    ]

    # compute p-value
    target = source["Avg_TTDays"]
    predictors = source[["Carrier Average"]]

    p_value = round(f_regression(predictors, target)[1][0], 2)

    # transit time across all carriers
    base = alt.Chart(source, title=f"Transit Time vs. Carrier Rate for carrier {carrier_option} \n(p-value={p_value})").encode(
        alt.X('month(Date):T', axis=alt.Axis(title=None))
    )

    transittime = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
        alt.Y('average(Avg_TTDays)',
            axis=alt.Axis(title='Avg. Transit Time (days)', titleColor='#5276A7'))
    )

    anchoragetime = base.mark_line(stroke='green', interpolate='monotone').encode(
        alt.Y('average(Carrier Average)',
            axis=alt.Axis(title='Avg. Carrier Rate', titleColor='green'))
    )

    c = alt.layer(transittime, anchoragetime).resolve_scale(
        y = 'independent'
    )

    st.altair_chart(c, use_container_width=True)
