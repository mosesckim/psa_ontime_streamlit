import datetime
import yaml
import os

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from utils import split_data, process_schedule_data, restrict_by_coverage, get_carr_serv_mask, \
    get_reg_train_test, compute_train_val_mae, plot_feature_importance, get_feat_imp_df
from baseline import BaselineModel




@st.experimental_memo(ttl=600)
def load_excel_data(config: dict, data_name: str):
    """Load excel data corresp. to data name

    Args:
        config (dict): config dict consisting of data and eval params
        data_name (str): string representing data name (e.g. port call or retail sales)

    Returns:
        pd.DataFrame: dataframe corresponding to data_name
    """

    filename = config[data_name]["filename"]
    sheetname = config[data_name]["sheet"]

    data_dir = config["data_path"]

    path = os.path.join(data_dir, filename)
    data = pd.read_excel(path, sheet_name=sheetname)

    return data



st.set_page_config(
    page_title="PSA-ONTIME: Schedule",
    page_icon="📅",
)


# TITLE
# TODO: alternative?
st.title("PSA-ONTIME")

# INTRODUCTION
st.subheader("Introduction")

st.write("We train a baseline (aggregate) model on shipping schedule data by \
    carrier and service and evaluate it by choosing a time horizon \
    (June by default). In order to gauge model performance, we compute an MAPE \
    (or mean average percentage error), where percentage error is given as below:"
)

st.latex(r'''
            \text{percent error} = \frac{\text{pred} - \text{actual}}{\text{actual}}
''')

st.write("For completeness, we include predictions and an additional metric (i.e. MAE or mean \
    absolute error). To see how models perform on delayed transit times, we compute both\
    metrics on prediction results with negative percent error.")


# DATA

config = yaml.safe_load(open('config.yml', 'r'))

data_path = config["data_path"]


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
port_data = load_excel_data(config, "port_call")
port_call_df = port_data


# CRUDE OIL
co_filename = "world_bank_crude_oil_spot_price.xlsx"
co_data = load_excel_data(config, "crude_oil")

# map dates to first of month
co_data.loc[:, "Date"] = co_data["Date"].apply(
    lambda x: x.replace(day=1)
)
# change column names
co_data.columns = ["Date", "crude_oil"]

# process port call data
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



# schedule + retail

# reliability POL mapping -> retail_sales country/region
rel_port_map = {
    'AEAUH': 'Agg Middle East & Africa',
    'AEJEA': 'Agg Middle East & Africa',
    'BEANR': 'Belgium',
    'BRRIG': 'Brazil',
    'CNNGB': 'China',
    'CNSHA': 'China',
    'CNSHK': 'China',
    'CNTAO': 'China',
    'CNYTN': 'China',
    'COCTG': 'Colombia',
    'DEHAM': 'Denmark',
    'ESBCN': 'Spain',
    'ESVLC': 'Spain',
    'GBLGP': 'U.K.',
    'GRPIR': 'Greece',
    'HKHKG': 'Hong Kong',
    'JPUKB': 'Japan',
    'KRPUS': 'South Korea',
    'LKCMB': 'Agg Asia Pacific',
    'MAPTM': 'Agg Middle East & Africa',
    'MXZLO': 'Mexico',
    'MYPKG': 'Agg Asia Pacific',
    'MYTPP': 'Agg Asia Pacific',
    'NLRTM': 'Netherlands',
    'NZAKL': 'Agg Asia Pacific',
    'PAMIT': 'Agg Latin America',
    'SAJED': 'Agg Middle East & Africa',
    'SAJUB': 'Agg Middle East & Africa',
    'SGSIN': 'Singapore',
    'THLCH': 'Thailand',
    'TWKHH': 'Taiwan',
    'USBAL': 'U.S.',
    'USCHS': 'U.S.',
    'USHOU': 'U.S.',
    'USILM': 'U.S.',
    'USLAX': 'U.S.',
    'USLGB': 'U.S.',
    'USMOB': 'U.S.',
    'USMSY': 'U.S.',
    'USNYC': 'U.S.',
    'USORF': 'U.S.',
    'USSAV': 'U.S.',
    'USTIW': 'U.S.'
}

rel_df_nona.loc[:, "region"] = rel_df_nona["POL"].apply(
    lambda x: rel_port_map[x]
)


# RETAIL SALES
sales_df = load_excel_data(
    config,
    "sales"
)

# process retail sales data
new_cols = [col.strip() for col in sales_df.columns]
sales_df.columns = new_cols

sales_df.loc[:, "month"] = sales_df["MonthYear"].apply(
    lambda x: int(x.split("/")[0])
)

sales_df.loc[:, "year"] = sales_df["MonthYear"].apply(
    lambda x: int(x.split("/")[1])
)

sales_df.loc[:, "date"] = sales_df["MonthYear"].apply(
    lambda x: datetime.datetime.strptime(
        x, "%m/%Y"
    )
)

# create offset date column
# sales_df.loc[:, "date(offset)"] = sales_df['date'] + pd.DateOffset(months=1)

sales_df.loc[:, "date(offset)"] = sales_df['date']

# create a retail sales map given date and country/region
# date, country/region -> retail sales index
regions = [
    'Agg North America', 'U.S.', 'Canada', 'Mexico',
    'Agg Western Europe', 'Austria', 'Belgium', 'Cyprus', 'Denmark',
    'Euro Area', 'Finland', 'France', 'Germany', 'Greece', 'Iceland',
    'Ireland', 'Italy', 'Luxembourg', 'Netherlands', 'Norway', 'Portugal',
    'Spain', 'Sweden', 'Switzerland', 'U.K.', 'Agg Asia Pacific',
    'Australia', 'China', 'Hong Kong', 'Indonesia', 'Japan', 'Kazakhstan',
    'Macau', 'Singapore', 'South Korea', 'Taiwan', 'Thailand', 'Vietnam',
    'Agg Eastern Europe', 'Bulgaria', 'Croatia', 'Czech Republic',
    'Estonia', 'Hungary', 'Latvia', 'Lithuania', 'Poland', 'Romania',
    'Russia', 'Serbia', 'Slovenia', 'Turkey', 'Agg Latin America',
    'Argentina', 'Brazil', 'Chile', 'Colombia', 'Agg Middle East & Africa',
    'Israel', 'South Africa'
]


date_region_sales = {}
for region in regions:
    region_dict = dict(
        zip(
            sales_df["date(offset)"],
            sales_df[region]
        )
    )

    date_region_sales[region] = region_dict


# calculate max date to avoid index error
max_date = sales_df["date(offset)"].max()

# finally, create new columns
# iterate over rows
rel_df_nona.loc[:, "retail_sales"] = rel_df_nona.apply(
    lambda x: date_region_sales[x["region"]][x["Date"]] if x["Date"] <= max_date else None, axis=1
)


# CPI
cpi_df = load_excel_data(
    config,
    "cpi"
)

cpi_df.columns = [
    col.strip() for col in cpi_df.columns
]

# print("cpi columns: ", cpi_df.columns)

cpi_df.columns = ['MonthYear', 'Agg North America', 'U.S.', 'Canada', 'Mexico',
       'Agg Western Europe', 'Austria', 'Belgium', 'Cyprus', 'Denmark',
       'Euro Area', 'Finland', 'France', 'Germany', 'Greece', 'Iceland',
       'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands', 'Norway',
       'Portugal', 'Spain', 'Sweden', 'Switzerland', 'U.K.',
       'Agg Asia Pacific', 'Australia', 'China', 'India*', 'Indonesia',
       'Japan', 'Philippines', 'Singapore', 'South Korea', 'Taiwan',
       'Thailand', 'Agg Latin America', 'Argentina', 'Brazil',
       'Chile', 'Colombia', 'Peru', 'Agg Eastern Europe', 'Bulgaria',
       'Croatia', 'Czech Republic', 'Estonia', 'Hungary', 'Latvia',
       'Lithuania', 'Poland', 'Romania', 'Russia', 'Serbia', 'Slovakia',
       'Slovenia', 'Turkey', 'Agg Middle East & Africa', 'Egypt', 'Iraq',
       'Israel', 'South Africa']

cpi_df.loc[:, "date"] = cpi_df["MonthYear"].apply(
    lambda x: datetime.datetime.strptime(
        x, "%m/%Y"
    )
)

cpi_df.loc[:, "date(offset)"] = cpi_df['date']

regions_cpi = ['Agg North America', 'U.S.', 'Canada', 'Mexico',
       'Agg Western Europe', 'Austria', 'Belgium', 'Cyprus', 'Denmark',
       'Euro Area', 'Finland', 'France', 'Germany', 'Greece', 'Iceland',
       'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands', 'Norway',
       'Portugal', 'Spain', 'Sweden', 'Switzerland', 'U.K.',
       'Agg Asia Pacific', 'Australia', 'China', 'India*', 'Indonesia',
       'Japan', 'Philippines', 'Singapore', 'South Korea', 'Taiwan',
       'Thailand', 'Agg Latin America', 'Argentina', 'Brazil',
       'Chile', 'Colombia', 'Peru', 'Agg Eastern Europe', 'Bulgaria',
       'Croatia', 'Czech Republic', 'Estonia', 'Hungary', 'Latvia',
       'Lithuania', 'Poland', 'Romania', 'Russia', 'Serbia', 'Slovakia',
       'Slovenia', 'Turkey', 'Agg Middle East & Africa', 'Egypt', 'Iraq',
       'Israel', 'South Africa']


date_region_cpi = {}
for region in regions_cpi:
    region_dict = dict(
        zip(
            cpi_df["date(offset)"],
            cpi_df[region]
        )
    )

    date_region_cpi[region] = region_dict


# calculate max date to avoid index error
max_date = cpi_df["date(offset)"].max()

rel_df_nona.loc[:, "cpi"] = rel_df_nona.apply(
    lambda x: date_region_cpi[x["region"]][x["Date"]] if x["Date"] <= max_date else None, axis=1
)


# add delta parameter
# merge with crude oil data

rel_df_nona = rel_df_nona.merge(
    co_data,
    on=["Date"]
)


with st.sidebar:

    restrict_trade = st.checkbox("Select Trade")

    if restrict_trade:
        trade_options = list(rel_df_nona["Trade"].unique())
        trade_option = st.selectbox(
            'Trade: ',
            trade_options
        )

        rel_df_nona = rel_df_nona[
            rel_df_nona["Trade"]==trade_option
        ]

        rel_df_no_orf_pt_hrs = rel_df_no_orf_pt_hrs[
            rel_df_no_orf_pt_hrs["Trade"]==trade_option
        ]


    label = st.selectbox(
            'Label: ',
            ("Avg_TTDays", "Avg_WaitTime_POD_Days")) #"OnTime_Reliability"))

    # time horizon for train split
    split_month = st.slider('Time Horizon (month)', 3, 10, 10)

    include_reg = st.checkbox("Linear Regression")

    overall_pred = st.button("Predict (overall)")


# split date

# baseline
datetime_split = datetime.datetime(2022, split_month, 1)
train_df, val_res = split_data(rel_df_nona, datetime_split, label=label)

# since we only have port call data up to august we restrict val_res
if include_reg:

    val_res = val_res[val_res["Date"] < datetime.datetime(2022, 9, 1)]

    split_month = min(8, split_month)
    datetime_split = datetime.datetime(2022, split_month, 1)

    # linear regression split (port hours)
    print("macro features: ", config["macro_features"])
    train_X_rg, train_y_rg, val_X_rg, val_y_rg = get_reg_train_test(
        rel_df_no_orf_pt_hrs,
        datetime_split,
        label=label,
        macro=config["macro_features"]
    )

    # linear regression split (retail)
    train_X_rg_ret, train_y_rg_ret, val_X_rg_ret, val_y_rg_ret = get_reg_train_test(
        rel_df_nona, #rel_df_sales,
        datetime_split,
        label=label,
        use_retail=True,
        macro=config["macro_features"]
    )


    # TODO: include in pytest
    # print("rel_df_nona shape: ", rel_df_nona.shape)

    # print("rel_df_nona max date: ", rel_df_nona.Date.max())
    # print("rel_df_nona min date: ", rel_df_nona.Date.min())

    # print("val sales shape: ", val_X_rg_ret.shape)


    # # compute common rows

    # print("val res shape: ", val_res.shape)



val_X = val_res[["Carrier", "Service", "POD", "POL"]]
val_y = val_res[label]


with st.sidebar:
    carrier_options = tuple(
        val_X["Carrier"].unique()
    )

    carrier_option = st.selectbox(
        'Carrier: ',
        carrier_options)

    service_options = tuple(val_X[
        val_X["Carrier"]==carrier_option
    ]["Service"].unique()
    )

    service_option = st.selectbox(
        'Service: ',
        service_options)

    partial_pred = st.button("Predict (Carrier, Service)")


train_df_filtered = train_df.copy()
val_X_filtered = val_X.copy()
val_y_filtered = val_y.copy()


if partial_pred:
    # train
    train_mask = get_carr_serv_mask(train_df_filtered, carrier_option, service_option)
    train_df_filtered = train_df_filtered[train_mask]
    # val
    val_mask = get_carr_serv_mask(val_X_filtered, carrier_option, service_option)
    val_X_filtered = val_X_filtered[val_mask]
    val_y_filtered = val_y_filtered[val_mask]


if val_X_filtered.shape[0] == 0 or train_df_filtered.shape[0] == 0:
    st.error('Insufficient data, pease choose another split', icon="🚨")

eval_lin_reg = False

if partial_pred or overall_pred:
    # instantiate baseline model
    base_model = BaselineModel(train_df_filtered, label=label)
    preds = []
    preds_std = []
    with st.spinner("Computing predictions..."):
        for ind, row in val_X_filtered.iterrows():
            pred, pred_std = base_model.predict(*row)

            preds.append(pred)
            preds_std.append(pred_std)



    preds_array = np.array(preds)
    preds_std_array = np.array(preds_std)

    nonzero_mask = val_y_filtered != 0
    nonzero_mask = nonzero_mask.reset_index()[label]


    if sum(nonzero_mask) != 0:

        preds = pd.Series(preds)[nonzero_mask]
        preds_std = pd.Series(preds_std)[nonzero_mask]

        val_y_filtered = val_y_filtered.reset_index()[label]
        val_y_filtered = val_y_filtered[nonzero_mask]

        val_X_filtered = val_X_filtered.reset_index().drop("index", axis=1)
        val_X_filtered = val_X_filtered[nonzero_mask]

        preds_array = np.array(preds)
        preds_std_array = np.array(preds_std)

        val_gt = val_y_filtered.values

        baseline_mae = mean_absolute_error(val_gt, preds_array)
        baseline_mape = mean_absolute_percentage_error(val_gt, preds_array)

        # calculate underestimates mape
        diff = preds_array - val_gt
        mask = diff < 0

        if sum(mask) != 0:
            preds_array_under = preds_array[mask]
            val_y_values_under = val_gt[mask]
            mae_under = mean_absolute_error(preds_array_under, val_y_values_under)
            mape_under = mean_absolute_percentage_error(val_y_values_under, preds_array_under)
            mae_under = round(mae_under, 3)
            mape_under = round(mape_under, 3)
        else:
            mae_under = "NA"
            mape_under = "NA"

        st.subheader("Predictions")
        df_preds = val_X_filtered.copy()
        df_preds.loc[:, "actual"] = val_y_filtered
        df_preds.loc[:, "pred"] = preds_array
        df_preds.loc[:, "error"] = preds_array - val_y_filtered
        df_preds.loc[:, "perc_error"] = (preds - val_y_filtered) / val_y_filtered
        st.write(df_preds)

        st.write(f"baseline val shape: {df_preds.shape}")

        if overall_pred:
            st.subheader("Error Analysis")

            # filter by error and error percentage
            perc_error_thresh = 0.50
            error_thresh = 4.0

            abs_errors = np.abs(df_preds["error"].values)
            abs_perc_errors = np.abs(df_preds["perc_error"].values)

            df_preds.loc[:, "abs_perc_error"] = 100 * abs_perc_errors

            df_preds.loc[:, "abs_error"] = abs_errors

            pred_outliers = df_preds[
                (abs_perc_errors > perc_error_thresh) &
                (abs_errors > error_thresh)
            ][
                [
                    "Carrier",
                    "Service",
                    "POL",
                    "POD",
                    "actual",
                    "pred",
                    "error",
                    "perc_error",
                    "abs_perc_error",
                    "abs_error"

                ]
            ]

            if pred_outliers.shape[0] != 0:


                st.write("""The scatter plot below shows predictions with an absolute percentage error
                            greater than 50 percent and absolute error greater than 4 days.
                        """)

                pred_scatter = alt.Chart(pred_outliers).mark_circle(size=60).encode(
                    x='actual',
                    y='pred',
                    color='Carrier',
                    size='abs_error',
                    tooltip=['POL', 'POD', 'actual', 'pred', 'perc_error']
                ).interactive()

                limit_thresh = 81 if label=="Avg_TTDays" else 21

                line = pd.DataFrame(
                    {
                        'actual': range(0, limit_thresh),
                        'pred': range(0, limit_thresh)
                    }
                )

                line_plot = alt.Chart(line).mark_line(color='red').encode(
                    x='actual',
                    y='pred'
                )

                st.altair_chart(pred_scatter + line_plot, use_container_width=True)

            else:
                st.error("No outliers found!")


            if include_reg:

                try:
                    # evaluate linear regression
                    linreg = LinearRegression()
                    val_mae_rg, val_mape_rg, val_mae_over_rg, val_mape_over_rg, result_df_rg = compute_train_val_mae(
                        linreg,
                        train_X_rg,
                        val_X_rg,
                        train_y_rg,
                        val_y_rg,
                        calc_mape=True,
                        label=label
                    )

                    linreg = LinearRegression()  # I am no too sure if we need to instantiate twice
                    val_mae_rg_ret, val_mape_rg_ret, val_mae_over_rg_ret, val_mape_over_rg_ret, result_df_rg_ret = compute_train_val_mae(
                        linreg,
                        train_X_rg_ret,
                        val_X_rg_ret,
                        train_y_rg_ret,
                        val_y_rg_ret,
                        calc_mape=True,
                        label=label
                    )

                    # random forests
                    rf = RandomForestRegressor()
                    val_mae_rf_ret, val_mape_rf_ret, val_mae_over_rf_ret, val_mape_over_rf_ret, result_df_rf_ret, rf = compute_train_val_mae(
                        rf,
                        train_X_rg_ret,
                        val_X_rg_ret,
                        train_y_rg_ret,
                        val_y_rg_ret,
                        calc_mape=True,
                        label=label,
                        model_type="rf"
                    )

                    eval_lin_reg = True

                except:
                    st.error("Not enough data. Choose a different split")

        # percentage correct within window
        # window = 5
        # pred_interval_acc = np.mean(np.abs(df_preds["error"]) <= window)
        # st.write(f"Prediction Interval ({window} day(s)): {pred_interval_acc}")

        # # negative errors
        # errors = df_preds["error"]
        # errors_neg = errors[errors < 0]

        # pred_interval_acc = np.mean(np.abs(errors_neg) <= window)
        # st.write(f"Prediction Interval ({window} day(s), negative error): {pred_interval_acc}")

        # prediction interval accuracy
        no_std = 2
        abs_errors = np.abs(df_preds["error"].values)
        # print("len(preds_std_array)", len(preds_std_array))
        # print("len(abs_errors)", len(abs_errors))


        pred_interval_acc = np.mean(abs_errors < no_std * preds_std_array)

        # st.write(f"Accuracy within (within {no_std} standard deviation): {pred_interval_acc}")
        # st.metric("Accuracy (95\% CI)", round(pred_interval_acc, 2))


        st.subheader("Metrics")

        st.markdown("#### Baseline")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", round(baseline_mae,3))
        col2.metric("MAPE", round(baseline_mape,3))
        col3.metric("MAE (delays)", mae_under)
        col4.metric("MAPE (delays)", mape_under)
        col5.metric("Accuracy (95\% CI)", round(pred_interval_acc, 2))

        if eval_lin_reg:
            st.markdown("#### Linear Regression")
            st.markdown("##### Port Hours")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MAE", round(val_mae_rg, 3))
            col2.metric("MAPE", round(val_mape_rg, 3))
            col3.metric("MAE (delays)", round(val_mae_over_rg, 3))
            col4.metric("MAPE (delays)", round(val_mape_over_rg, 3))
            col5.metric("Accuracy (95\% CI)", "NA")

            # show_reg_pred = st.checkbox("Show predictions (port hours)")
            # if show_reg_pred: st.write(result_df_rg)
            # st.write(result_df_rg)

            st.write(f"port val shape {result_df_rg.shape}")


            st.markdown("##### Retail Sales")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MAE", round(val_mae_rg_ret, 3))
            col2.metric("MAPE", round(val_mape_rg_ret, 3))
            col3.metric("MAE (delays)", round(val_mae_over_rg_ret, 3))
            col4.metric("MAPE (delays)", round(val_mape_over_rg_ret, 3))
            col5.metric("Accuracy (95\% CI)", "NA")

            # show_reg_ret_pred = st.checkbox("Show predictions (retail sales)")
            # if show_reg_ret_pred: st.write(result_df_rg_ret)
            # st.write(result_df_rg_ret)
            st.write(f"ret val shape {result_df_rg_ret.shape}")

            st.markdown("#### Random Forests ")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MAE", round(val_mae_rf_ret, 3))
            col2.metric("MAPE", round(val_mape_rf_ret, 3))
            col3.metric("MAE (delays)", round(val_mae_over_rf_ret, 3))
            col4.metric("MAPE (delays)", round(val_mape_over_rf_ret, 3))
            col5.metric("Accuracy (95\% CI)", "NA")


            # plot feature importance
            source = get_feat_imp_df(rf.feature_importances_, rf.feature_names_in_, topn=5)


            feat_impt_chart = alt.Chart(source).mark_bar().encode(
                x="importance:Q",
                y=alt.Y('feature:O', sort='-x')
                # y="feature:O"
            )

            st.altair_chart(feat_impt_chart)

    else:
        st.error('All expected labels are zero', icon="🚨")
