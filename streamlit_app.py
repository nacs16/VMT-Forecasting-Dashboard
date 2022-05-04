import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from vmt_utils import anomaly_detect_V1, fix_vmt_anomalies, clean_series, standard_rescale, run_VAR_forecast, score_forecast, invert_diff
from statsmodels.tsa.api import VAR

DATA_DIR = "data/"
LABEL_TO_FILENAME = {"National VMT": "Arity_national_VMT_20220413.csv", "State VMT": "Arity_state_VMT_20220413.csv", "County VMT": "Arity_county_VMT_20220413.csv"}

### Functions ###

def load_vmt(filename):
    return pd.read_csv(filename, parse_dates=["tripStartDate"], index_col="tripStartDate")

def plot_vmt(vmt_series):
    fig, ax = plt.subplots(figsize=(10,7))

    ax.plot(vmt_series, linewidth=2.5)
    ax.set_title("VMT Plot")
    ax.set_ylabel("Miles Driven")

    return fig

def plot_multiple_series(df):
    """
    series_dict is a dictionary where keys are the labels and values are the series themselves
    """
    fig, ax = plt.subplots(figsize=(10,7))

    ax.plot(df, linewidth=2.5)
    ax.set_title("VMT Plot")
    ax.set_ylabel("Scaled Metrics")

    return fig

def run_forecast_pressed(df):
    st.session_state["forecasted_series"], st.session_state["forecasted_accuracy_metrics"] = run_VAR_forecast(df, st.session_state["uploaded_file_metric"])
    st.session_state["forecast_done"] = True

def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

### Web App Content Begins ###
st.set_page_config(page_title="VMT Dashboard", layout="wide")
if "preloaded_vmt" not in st.session_state:
    with st.spinner('Preparing Dashboard...'):
        st.session_state["preloaded_vmt"] = True
        st.session_state["National VMT df"] = load_vmt(DATA_DIR+LABEL_TO_FILENAME["National VMT"])
        st.session_state["bad_dates_method1"], _, _ = anomaly_detect_V1(st.session_state["National VMT df"]["total_mileage_estimate"], printing=False)
        st.session_state["State VMT df"] = load_vmt(DATA_DIR+LABEL_TO_FILENAME["State VMT"])
        st.session_state["County VMT df"] = load_vmt(DATA_DIR+LABEL_TO_FILENAME["County VMT"])
        st.session_state["uploaded_file_ready"] = False

##########################################################################################################################################
############################################## VMT File ##################################################################################
##########################################################################################################################################
with st.sidebar:
    st.title("Vehicle Miles Traveled")
    col1, col2, col3 = st.columns(3)

    with col1:
        nat_button = st.button("National", on_click=style_button_row, kwargs={'clicked_button_ix': 1, 'n_buttons': 3})
        if nat_button:
            st.session_state["VMT_FILE"] = "National VMT"
    with col2:
        state_button = st.button("State", on_click=style_button_row, kwargs={'clicked_button_ix': 2, 'n_buttons': 3})
        if state_button:
            st.session_state["VMT_FILE"] = "State VMT"
    with col3:
        county_button = st.button("County", on_click=style_button_row, kwargs={'clicked_button_ix': 3, 'n_buttons': 3})
        if county_button:
            st.session_state["VMT_FILE"] = "County VMT"

if "VMT_FILE" in st.session_state:
    style_button_row(list(LABEL_TO_FILENAME.keys()).index(st.session_state["VMT_FILE"])+1, 3)

if "VMT_FILE" in st.session_state:
    st.session_state["vmt_df"] = st.session_state[st.session_state["VMT_FILE"]+" df"].copy()
    
    with st.sidebar:
        with st.expander("VMT Options"):
            if st.session_state["VMT_FILE"] in ("National VMT"):
                metric_selection = st.selectbox('Select Metric', ("total_mileage_estimate", "total_active_drivers_estimate"))

            if st.session_state["VMT_FILE"] in ("State VMT", "County VMT"):
            
                metric_selection = st.selectbox('Select Metric', ("total_mileage_estimate", "total_resident_mileage_estimate", "total_active_drivers_estimate"))
                state_selection = st.selectbox('Select State', set(pd.Series(st.session_state["vmt_df"].state_abbr.unique()).sort_values()))
                vmt_filter = (st.session_state["vmt_df"].state_abbr == state_selection)

            if st.session_state["VMT_FILE"] in ("County VMT") and state_selection:
                county_selection = st.selectbox('Select County', set(st.session_state["vmt_df"][st.session_state["vmt_df"].state_abbr == state_selection].county_name.unique()))
                vmt_filter = (st.session_state["vmt_df"].state_abbr == state_selection) & (st.session_state["vmt_df"].county_name == county_selection)


with st.sidebar:
    with st.expander("Pre-Processing Options"):
        apply_anomaly_fix = st.checkbox('Fix Anomalies')
        apply_rolling_average = st.select_slider('Rolling Window', [1,7,14,21,28], value=7)



##########################################################################################################################################
############################################## User's Uploaded File ######################################################################
##########################################################################################################################################


def new_file_uploaded_callback():
    st.session_state["uploaded_index_set"] = False
    st.session_state["load_uploaded_file"] = True
    st.session_state["uploaded_file_ready"] = False
    st.session_state["forecast_done"] = False

def uploaded_file_ready_clicked():
    try:
        with st.spinner("Performing data cleaning steps on your input file..."):
            st.session_state["uploaded_df"] = pd.to_numeric(st.session_state["uploaded_df"][st.session_state["uploaded_file_filter"]][st.session_state["uploaded_file_metric"]])
            st.session_state["uploaded_df"] = clean_series(st.session_state["uploaded_df"])
            st.session_state["uploaded_file_ready"] = True
    except:
        st.error('Failed Processing Your File')


with st.sidebar:
    
    with st.expander("Combine VMT with your own data"):
    
        # https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
        uploaded_file = st.file_uploader("Upload your own .csv file", on_change=new_file_uploaded_callback)
        if uploaded_file is not None and uploaded_file.name[-4:] == ".csv":
            
            if st.session_state["uploaded_file_ready"] is False:
                if st.session_state["load_uploaded_file"]:
                    with st.spinner('Reading Your File...'):
                        st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)
                        
                        for c in st.session_state["uploaded_df"].columns:
                            try:
                                pd.to_datetime(st.session_state["uploaded_df"][c].iloc[0:5])
                                st.session_state["uploaded_df"][c] = pd.to_datetime(st.session_state["uploaded_df"][c])
                                st.session_state["uploaded_df"] = st.session_state["uploaded_df"].set_index(st.session_state["uploaded_df"][c])
                                st.session_state["uploaded_index_set"] = True
                                break
                            except:
                                continue
                        st.session_state["load_uploaded_file"] = False
                
                if st.session_state["uploaded_index_set"] is False:
                    uploaded_index_col_selection = st.selectbox("Select Date Column", list(st.session_state["uploaded_df"].columns))
                    if uploaded_index_col_selection:
                        try:
                            st.session_state["uploaded_df"][uploaded_index_col_selection] = pd.to_datetime(st.session_state["uploaded_df"][uploaded_index_col_selection])
                            st.session_state["uploaded_df"].set_index(st.session_state["uploaded_df"][uploaded_index_col_selection])
                            st.session_state["uploaded_index_set"] = True
                        except:
                            st.error('Failed to process {} as the date column'.format(uploaded_index_col_selection))
                #st.write(st.session_state["uploaded_df"].sample(5))

                uploaded_file_metric = st.selectbox('Select Target Metric', [c for c in list(st.session_state["uploaded_df"].apply(lambda s: len(s.unique())).sort_values(ascending=False).index) if c != st.session_state["uploaded_df"].index.name])

                uploaded_file_filtering_options = st.multiselect('Select Filtering Columns', st.session_state["uploaded_df"].drop(columns=[uploaded_file_metric]).select_dtypes(include=['object']).columns, [])
                uploaded_file_filters = {}
                uploaded_file_filter = ([True] * len(st.session_state["uploaded_df"]))
                if len(uploaded_file_filtering_options) > 0:
                    for i in range(len(uploaded_file_filtering_options)):
                        lab = uploaded_file_filtering_options[i]
                        uploaded_file_filters[lab] = st.selectbox('Select '+lab, set(st.session_state["uploaded_df"][uploaded_file_filter][lab].unique()))
                        uploaded_file_filter = uploaded_file_filter & (st.session_state["uploaded_df"][lab] == uploaded_file_filters[lab])
                st.session_state["uploaded_file_filter"] = uploaded_file_filter
                st.session_state["uploaded_file_metric"] = uploaded_file_metric
                uploaded_file_ready_button = st.button("My File is Ready!", on_click=uploaded_file_ready_clicked)
            else:
                st.success('Your data is ready to pair with VMT!')


if "vmt_df" in st.session_state:
    


    if st.session_state["VMT_FILE"] == "National VMT":
        plot_this = st.session_state["vmt_df"][metric_selection]
    else:
        plot_this = st.session_state["vmt_df"][vmt_filter][metric_selection]

    if apply_anomaly_fix:
        plot_this = fix_vmt_anomalies(plot_this, st.session_state["bad_dates_method1"])
    
    plot_this = plot_this.rolling(apply_rolling_average).mean()

    if st.session_state["uploaded_file_ready"]:
        plot_this = standard_rescale(plot_this)
        plot_this.name = metric_selection
        plot_this2 = st.session_state["uploaded_df"].rolling(apply_rolling_average).mean()
        plot_this2 = standard_rescale(plot_this2)
        plot_this2.name = st.session_state["uploaded_file_metric"]
        plot_this = pd.DataFrame(plot_this, index=plot_this.index).join(plot_this2, how="inner").dropna()
        
        #st.pyplot(plot_multiple_series(plot_this))
       
        layout = go.Layout(autosize=False,width=1000,height=600)

        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=plot_this.index, y=plot_this[metric_selection], mode='lines', name=metric_selection, connectgaps=True))
        fig.add_trace(go.Scatter(x=plot_this.index, y=plot_this[st.session_state["uploaded_file_metric"]], mode='lines', name=st.session_state["uploaded_file_metric"], connectgaps=True))
        
        fig.update_layout(
            title={'text': "VMT and {}".format(st.session_state["uploaded_file_metric"]), 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 26}},
            xaxis_title="Date",
            yaxis_title="Scaled Metrics")

        if st.session_state["forecast_done"]:
            fig.add_trace(go.Scatter(x=st.session_state["forecasted_series"].index, y=st.session_state["forecasted_series"], mode='lines', name="{} Forecast".format(st.session_state["uploaded_file_metric"]), connectgaps=True))
        
        st.plotly_chart(fig, use_container_width=True)


    else:
        layout = go.Layout(autosize=False,width=1000,height=600)
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(x=plot_this.index, y=plot_this, mode='lines', name=metric_selection, connectgaps=True))
        
        fig.update_layout(
            #title="VMT {}".format(metric_selection),
            title = {'text': "VMT {}".format( " ".join( w.capitalize() for w in metric_selection.split("_"))), 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 26}},
            xaxis_title={'text': 'Date', 'font': {'size': 16}},
            yaxis_title={'text':metric_selection.replace("_", " "), 'font': {'size': 24}})
        
        st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    if st.session_state["uploaded_file_ready"] and st.session_state["forecast_done"] is False:
        st.session_state["plot_this_df"] = plot_this
        run_forecast_button = st.button("Run Forecast!", on_click=run_forecast_pressed, args=[plot_this])
