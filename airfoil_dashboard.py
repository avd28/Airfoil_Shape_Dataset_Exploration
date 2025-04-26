import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Airfoil Performance Dashboard", layout="wide")

# Connect to the database
@st.cache_data
def load_data():
    conn = sqlite3.connect('airfoil_data.db')
    df = pd.read_sql_query("""
        SELECT airfoilName, coefficientLift, coefficientDrag, coefficientMoment, 
               reynoldsNumber, alpha 
        FROM airfoils
    """, conn)
    conn.close()
    return df

# Load the data
df = load_data()

# Get unique airfoil names
airfoil_names = sorted(df['airfoilName'].unique())

# Add a title to the top of the page
st.title("Airfoil Performance Dashboard")

# Main page tabs for mode selection
main_tabs = st.tabs(["Individual Characteristics", "Airfoil Comparison"])

with main_tabs[0]:
    selected_airfoil = st.selectbox("Select Airfoil", airfoil_names, key="individual_airfoil")
    airfoil_data = df[df['airfoilName'] == selected_airfoil]

    # Performance statistics in the sidebar for individual airfoil
    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        st.markdown("<div style='font-size:1.1em; font-weight:700; color:#f66; margin-bottom:0.2em;'>" + selected_airfoil + "</div>", unsafe_allow_html=True)
        max_lift_idx = airfoil_data['coefficientLift'].idxmax()
        min_drag_idx = airfoil_data['coefficientDrag'].idxmin()
        max_lift = airfoil_data.loc[max_lift_idx, 'coefficientLift']
        max_lift_re = airfoil_data.loc[max_lift_idx, 'reynoldsNumber']
        max_lift_alpha = airfoil_data.loc[max_lift_idx, 'alpha']
        min_drag = airfoil_data.loc[min_drag_idx, 'coefficientDrag']
        min_drag_re = airfoil_data.loc[min_drag_idx, 'reynoldsNumber']
        min_drag_alpha = airfoil_data.loc[min_drag_idx, 'alpha']
        st.markdown(f"""
        <div style='margin-bottom:1.2em;'>
          <div style='font-size:1em; font-weight:600;'>Maximum Lift Coefficient</div>
          <div style='font-size:1.5em; font-weight:700; color:#fff;'>{max_lift:.3f}</div>
          <div style='font-size:0.95em; color:#aaa;'>Obtained at <b>Re = {max_lift_re}</b>, <b>α = {max_lift_alpha}</b></div>
          <div style='font-size:1em; font-weight:600; margin-top:0.5em;'>Minimum Drag Coefficient</div>
          <div style='font-size:1.5em; font-weight:700; color:#fff;'>{min_drag:.3f}</div>
          <div style='font-size:0.95em; color:#aaa;'>Obtained at <b>Re = {min_drag_re}</b>, <b>α = {min_drag_alpha}</b></div>
        </div>
        """, unsafe_allow_html=True)

    # Individual Characteristics Tab Content
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lift vs Drag (by Reynolds Number)")
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        for reynolds in sorted(airfoil_data['reynoldsNumber'].unique()):
            mask = airfoil_data['reynoldsNumber'] == reynolds
            fig1.add_trace(
                go.Scatter(
                    x=airfoil_data[mask]['coefficientDrag'],
                    y=airfoil_data[mask]['coefficientLift'],
                    mode='lines+markers',
                    name=f'Re={reynolds}',
                    text=[f'α={alpha}°' for alpha in airfoil_data[mask]['alpha']],
                    hovertemplate="Drag: %{x:.3f}<br>Lift: %{y:.3f}<br>α: %{text}<extra></extra>"
                ),
                secondary_y=False,
            )
        fig1.update_layout(
            title_text=f"Lift vs Drag for {selected_airfoil}",
            xaxis_title="Coefficient of Drag (Cd)",
            yaxis_title="Coefficient of Lift (Cl)",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Lift vs Angle of Attack")
        fig_lift = go.Figure()
        for reynolds in sorted(airfoil_data['reynoldsNumber'].unique()):
            mask = airfoil_data['reynoldsNumber'] == reynolds
            fig_lift.add_trace(
                go.Scatter(
                    x=airfoil_data[mask]['alpha'],
                    y=airfoil_data[mask]['coefficientLift'],
                    mode='lines+markers',
                    name=f'Lift (Re={reynolds})',
                    hovertemplate="α: %{x}°<br>Lift: %{y:.3f}<extra></extra>"
                )
            )
        fig_lift.update_layout(
            title_text=f"Lift vs Angle of Attack for {selected_airfoil}",
            xaxis_title="Angle of Attack (α)",
            yaxis_title="Coefficient of Lift (Cl)",
            showlegend=True,
            height=400,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        st.plotly_chart(fig_lift, use_container_width=True)
        st.subheader("Drag vs Angle of Attack")
        fig_drag = go.Figure()
        for reynolds in sorted(airfoil_data['reynoldsNumber'].unique()):
            mask = airfoil_data['reynoldsNumber'] == reynolds
            fig_drag.add_trace(
                go.Scatter(
                    x=airfoil_data[mask]['alpha'],
                    y=airfoil_data[mask]['coefficientDrag'],
                    mode='lines+markers',
                    name=f'Drag (Re={reynolds})',
                    hovertemplate="α: %{x}°<br>Drag: %{y:.3f}<extra></extra>"
                )
            )
        fig_drag.update_layout(
            title_text=f"Drag vs Angle of Attack for {selected_airfoil}",
            xaxis_title="Angle of Attack (α)",
            yaxis_title="Coefficient of Drag (Cd)",
            showlegend=True,
            height=400,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        st.plotly_chart(fig_drag, use_container_width=True)

with main_tabs[1]:
    st.markdown("<div style='font-size:1.3em; font-weight:700; margin-bottom:0.5em;'>Airfoil Comparison</div>", unsafe_allow_html=True)
    panel_col, plot_col = st.columns([1, 3])
    with panel_col:
        compare_airfoils = st.multiselect("Select up to 5 Airfoils", airfoil_names, max_selections=5, key="compare_airfoils")
        chart_type = st.selectbox("Chart Type", ["Lift vs Drag", "Lift vs Alpha", "Drag vs Alpha"], key="compare_chart_type")
        if compare_airfoils:
            compare_df = df[df['airfoilName'].isin(compare_airfoils)]
            available_re = sorted(compare_df['reynoldsNumber'].unique())
            available_alpha = sorted(compare_df['alpha'].unique())
            selected_re = st.selectbox("Select Reynolds Number", available_re, key="compare_re")
            # Show stats block for each selected airfoil
            for airfoil in compare_airfoils:
                airfoil_data = df[df['airfoilName'] == airfoil]
                max_lift_idx = airfoil_data['coefficientLift'].idxmax()
                min_drag_idx = airfoil_data['coefficientDrag'].idxmin()
                max_lift = airfoil_data.loc[max_lift_idx, 'coefficientLift']
                max_lift_re = airfoil_data.loc[max_lift_idx, 'reynoldsNumber']
                max_lift_alpha = airfoil_data.loc[max_lift_idx, 'alpha']
                min_drag = airfoil_data.loc[min_drag_idx, 'coefficientDrag']
                min_drag_re = airfoil_data.loc[min_drag_idx, 'reynoldsNumber']
                min_drag_alpha = airfoil_data.loc[min_drag_idx, 'alpha']
                st.markdown(f"""
                <div style='margin-bottom:1.2em; border-bottom:1px solid #444; padding-bottom:0.5em;'>
                  <div style='font-size:1.1em; font-weight:700; color:#f66; margin-bottom:0.2em;'>{airfoil}</div>
                  <div style='font-size:1em; font-weight:600;'>Maximum Lift Coefficient</div>
                  <div style='font-size:1.5em; font-weight:700; color:#fff;'>{max_lift:.3f}</div>
                  <div style='font-size:0.95em; color:#aaa;'>Obtained at <b>Re = {max_lift_re}</b>, <b>α = {max_lift_alpha}</b></div>
                  <div style='font-size:1em; font-weight:600; margin-top:0.5em;'>Minimum Drag Coefficient</div>
                  <div style='font-size:1.5em; font-weight:700; color:#fff;'>{min_drag:.3f}</div>
                  <div style='font-size:0.95em; color:#aaa;'>Obtained at <b>Re = {min_drag_re}</b>, <b>α = {min_drag_alpha}</b></div>
                </div>
                """, unsafe_allow_html=True)
    with plot_col:
        if 'compare_airfoils' in locals() and compare_airfoils:
            st.header("Airfoil Comparison")
            if chart_type == "Lift vs Drag":
                fig = go.Figure()
                for airfoil in compare_airfoils:
                    subdf = df[(df['airfoilName'] == airfoil) & (df['reynoldsNumber'] == selected_re)]
                    fig.add_trace(go.Scatter(
                        x=subdf['coefficientDrag'],
                        y=subdf['coefficientLift'],
                        mode='lines+markers',
                        name=airfoil
                    ))
                fig.update_layout(
                    title=f"Lift vs Drag at Re={selected_re}",
                    xaxis_title="Coefficient of Drag (Cd)",
                    yaxis_title="Coefficient of Lift (Cl)",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Lift vs Alpha":
                fig = go.Figure()
                for airfoil in compare_airfoils:
                    subdf = df[(df['airfoilName'] == airfoil) & (df['reynoldsNumber'] == selected_re)]
                    fig.add_trace(go.Scatter(
                        x=subdf['alpha'],
                        y=subdf['coefficientLift'],
                        mode='lines+markers',
                        name=airfoil
                    ))
                fig.update_layout(
                    title=f"Lift vs Angle of Attack at Re={selected_re}",
                    xaxis_title="Angle of Attack (α)",
                    yaxis_title="Coefficient of Lift (Cl)",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Drag vs Alpha":
                fig = go.Figure()
                for airfoil in compare_airfoils:
                    subdf = df[(df['airfoilName'] == airfoil) & (df['reynoldsNumber'] == selected_re)]
                    fig.add_trace(go.Scatter(
                        x=subdf['alpha'],
                        y=subdf['coefficientDrag'],
                        mode='lines+markers',
                        name=airfoil
                    ))
                fig.update_layout(
                    title=f"Drag vs Angle of Attack at Re={selected_re}",
                    xaxis_title="Angle of Attack (α)",
                    yaxis_title="Coefficient of Drag (Cd)",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)