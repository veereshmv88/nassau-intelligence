"""
Interactive Plotly Chart Components for Nassau Intelligence.
Styled with the enterprise glassmorphism aesthetic:
emerald greens, mint, teal, soft translucent backdrops, and Swiss modernist typography.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.config import FACTORIES, REGION_COORDINATES, THEME


def get_base_layout() -> Dict[str, Any]:
    """Base layout styling for all Plotly charts."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=THEME["text_main"], size=12),
        margin=dict(l=40, r=30, t=40, b=40),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11, color=THEME["text_secondary"])
        )
    )


def create_factory_network_map(df: pd.DataFrame = None) -> go.Figure:
    """
    US Map showing 5 manufacturing plants, regional demand centers,
    and active shipping supply routes.
    """
    fig = go.Figure()
    
    # 1. Plot Shipping Route Lines
    route_pairs = [
        ("Sugar Shack", "Interior", 60),
        ("Sugar Shack", "Pacific", 25),
        ("Sugar Shack", "Atlantic", 40),
        ("Secret Factory", "Interior", 85),
        ("Secret Factory", "Atlantic", 70),
        ("Lot's O' Nuts", "Pacific", 90),
        ("Lot's O' Nuts", "Gulf", 30),
        ("Wicked Choccy's", "Gulf", 85),
        ("Wicked Choccy's", "Atlantic", 75),
        ("The Other Factory", "Gulf", 65),
        ("The Other Factory", "Interior", 45),
    ]
    
    for f_name, r_name, vol in route_pairs:
        f_lat = FACTORIES[f_name]["lat"]
        f_lon = FACTORIES[f_name]["lon"]
        r_lat, r_lon = REGION_COORDINATES[r_name]
        
        # Opacity based on volume
        alpha = min(0.65, max(0.20, vol / 120.0))
        width = 1.2 if vol < 40 else (2.0 if vol < 75 else 2.8)
        
        fig.add_trace(go.Scattergeo(
            locationmode='USA-states',
            lon=[f_lon, r_lon],
            lat=[f_lat, r_lat],
            mode='lines',
            line=dict(width=width, color=f"rgba(8, 127, 91, {alpha})"),
            hoverinfo='none',
            showlegend=False
        ))
        
    # 2. Add Factory Markers
    f_lats = [info["lat"] for info in FACTORIES.values()]
    f_lons = [info["lon"] for info in FACTORIES.values()]
    f_names = list(FACTORIES.keys())
    f_hover = [
        f"<b>{name}</b><br>{info['city']}, {info['state']}<br><i>{info['specialty']}</i>"
        for name, info in FACTORIES.items()
    ]
    
    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lon=f_lons,
        lat=f_lats,
        mode='markers+text',
        text=[f"  {n}" for n in f_names],
        textposition="top right",
        textfont=dict(size=11, color=THEME["text_main"], family="Inter"),
        marker=dict(
            size=14,
            color=THEME["primary_green"],
            line=dict(width=3, color='#FFFFFF'),
            symbol='circle'
        ),
        hoverinfo='text',
        hovertext=f_hover,
        name='Factory Location'
    ))
    
    # 3. Add Regional Demand Hubs
    r_lats = [coord[0] for coord in REGION_COORDINATES.values()]
    r_lons = [coord[1] for coord in REGION_COORDINATES.values()]
    r_names = [f"Region: {r}" for r in REGION_COORDINATES.keys()]
    
    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lon=r_lons,
        lat=r_lats,
        mode='markers',
        marker=dict(
            size=10,
            color=THEME["teal_accent"],
            line=dict(width=2, color='#FFFFFF'),
            opacity=0.85
        ),
        hoverinfo='text',
        hovertext=r_names,
        name='Regional Hub'
    ))
    
    fig.update_layout(
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='#EAF1EB',
            subunitcolor='rgba(255,255,255,0.85)',
            countrycolor='rgba(255,255,255,0.95)',
            countrywidth=1.5,
            subunitwidth=1.0,
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=5, b=5),
        height=300,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='left',
            x=0.02,
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='rgba(255,255,255,0.7)',
            borderwidth=1,
            font=dict(size=10)
        )
    )
    return fig


def create_lead_time_regional_bar(df: pd.DataFrame = None) -> go.Figure:
    """
    Lead-Time Analysis Bar Chart comparing Current vs Optimized configurations by region.
    Matches the reference dashboard layout and color scheme.
    """
    regions = ['West', 'Midwest', 'South', 'Northeast']
    current_leads = [5.8, 5.4, 5.1, 5.6]
    optimized_leads = [3.6, 3.1, 3.0, 3.4]
    
    fig = go.Figure()
    
    # Baseline Bar
    fig.add_trace(go.Bar(
        x=regions,
        y=current_leads,
        name='Current Assignment',
        marker=dict(color='#8DA69B', opacity=0.75, line=dict(width=1, color='rgba(255,255,255,0.8)')),
        width=0.32
    ))
    
    # Optimized Bar
    fig.add_trace(go.Bar(
        x=regions,
        y=optimized_leads,
        name='Optimized Recommendation',
        marker=dict(color=THEME["primary_green"], opacity=0.9, line=dict(width=1, color='rgba(255,255,255,0.8)')),
        width=0.32
    ))
    
    # Midwest Callout annotation matching reference dashboard
    fig.add_annotation(
        x='Midwest',
        y=5.4,
        text="<b>Midwest</b><br>Current: 5.4 days<br>Optimized: 3.1 days<br><span style='color:#16865B'>↓ 42.6%</span>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=THEME["deep_green"],
        ax=0,
        ay=-55,
        bgcolor="rgba(255, 255, 255, 0.90)",
        bordercolor=THEME["mint_accent"],
        borderwidth=1.5,
        borderpad=6,
        font=dict(size=11, color=THEME["text_main"])
    )
    
    base_layout = get_base_layout()
    fig.update_layout(
        **base_layout,
        barmode='group',
        bargap=0.25,
        bargroupgap=0.08,
        yaxis=dict(
            title=dict(text="Avg Lead Time (Days)", font=dict(size=11, color=THEME["text_secondary"])),
            tickfont=dict(size=11),
            gridcolor='rgba(0,0,0,0.06)',
            range=[0, 8]
        ),
        xaxis=dict(
            tickfont=dict(size=12, color=THEME["text_main"]),
            gridcolor='rgba(0,0,0,0.02)'
        ),
        height=260
    )
    return fig


def create_profit_vs_lead_time_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: Profit Margin (%) vs Lead Time (Days) with Optimal Zone highlight.
    """
    fig = go.Figure()
    
    colors = {
        "Chocolate": "#E9A23B",
        "Sugar": "#FF8FAB",
        "Other": "#5AB9AD"
    }
    
    # Sample or aggregate to maintain crisp interactive rendering
    sample_df = df.sample(min(450, len(df)), random_state=42).copy()
    
    for div, color in colors.items():
        subset = sample_df[sample_df['Division'] == div]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset['lead_time_days'] + np.random.normal(0, 0.15, len(subset)),
                y=subset['Profit_Margin_Pct'],
                mode='markers',
                name=div,
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.5, color='#FFFFFF')
                ),
                text=subset['Product Name'],
                hovertemplate="<b>%{text}</b><br>Lead Time: %{x:.1f} days<br>Profit Margin: %{y:.1f}%<extra></extra>"
            ))
            
    # Highlight Optimal Zone Box (0-4 days lead time, 15-30% profit margin)
    fig.add_shape(
        type="rect",
        x0=1.5, y0=12, x1=5.0, y1=28,
        line=dict(color=THEME["primary_green"], width=1.5, dash="dash"),
        fillcolor="rgba(167, 232, 200, 0.18)"
    )
    
    fig.add_annotation(
        x=3.25,
        y=25.5,
        text="<b>Optimal Zone</b><br>Low Lead Time<br>High Profit",
        showarrow=False,
        font=dict(size=10, color=THEME["deep_green"]),
        align="center",
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor=THEME["mint_accent"],
        borderwidth=1,
        borderpad=4
    )
    
    base_layout = get_base_layout()
    fig.update_layout(
        **base_layout,
        xaxis=dict(
            title=dict(text="Avg Lead Time (Days)", font=dict(size=11, color=THEME["text_secondary"])),
            gridcolor='rgba(0,0,0,0.06)',
            range=[0, 10]
        ),
        yaxis=dict(
            title=dict(text="Profit Margin (%)", font=dict(size=11, color=THEME["text_secondary"])),
            gridcolor='rgba(0,0,0,0.06)',
            range=[0, 35]
        ),
        height=260
    )
    return fig


def create_feature_importance_chart(feat_imp: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart for model feature importance/coefficients."""
    items = list(feat_imp.items())[:12]
    # Reverse so top feature appears at the top
    items.reverse()
    
    names = [i[0].replace('cat__', '').replace('num__', '') for i in items]
    values = [i[1] for i in items]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(
            color=values,
            colorscale=[[0, '#A7E8C8'], [1, '#087F5B']],
            line=dict(color='rgba(255,255,255,0.8)', width=1)
        )
    ))
    
    base_layout = get_base_layout()
    fig.update_layout(
        **base_layout,
        title=dict(text="<b>Feature Coefficients / Permutation Impact</b>", font=dict(size=14)),
        xaxis=dict(title="Relative Influence", gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(tickfont=dict(size=11)),
        height=340
    )
    return fig


def create_actual_vs_predicted_chart(actuals: List[float], predictions: List[float]) -> go.Figure:
    """Actual vs Predicted scatter plot with perfect prediction line."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actuals,
        y=predictions,
        mode='markers',
        marker=dict(
            size=6,
            color=THEME["teal_accent"],
            opacity=0.6,
            line=dict(width=0.5, color='#FFFFFF')
        ),
        name='Predictions'
    ))
    
    # 45-degree reference line
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color=THEME["primary_green"], width=2, dash='dash'),
        name='Perfect Fit'
    ))
    
    base_layout = get_base_layout()
    fig.update_layout(
        **base_layout,
        title=dict(text="<b>Actual vs Predicted Lead Time (Days)</b>", font=dict(size=14)),
        xaxis=dict(title="Actual Lead Time (Days)", gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(title="Predicted Lead Time (Days)", gridcolor='rgba(0,0,0,0.06)'),
        height=320
    )
    return fig


def create_lead_time_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Distribution of lead times segmented by shipping mode."""
    fig = px.histogram(
        df,
        x="lead_time_days",
        color="Ship Mode",
        barmode="overlay",
        nbins=12,
        color_discrete_sequence=[THEME["primary_green"], THEME["teal_accent"], THEME["amber_warning"], THEME["lavender_accent"]],
        opacity=0.75
    )
    base_layout = get_base_layout()
    fig.update_layout(
        **base_layout,
        title=dict(text="<b>Lead Time Distribution by Shipping Mode</b>", font=dict(size=14)),
        xaxis=dict(title="Lead Time (Days)", gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(title="Order Count", gridcolor='rgba(0,0,0,0.06)'),
        height=300
    )
    return fig
