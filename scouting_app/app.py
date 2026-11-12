import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="T20 Scouting", layout="wide")

# Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    h1, h2, h3 { color: #f0f2f6; }
    .stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🏏 T20 Scouting: batters, bowlers, line and length")
st.markdown("Metrics: **RAA (Runs Above Average)** | **True SR** | **WAA (Wicket Advancement Average)**")

@st.cache_data(ttl=3600) # Cache for 1 hour, or clear manually
def load_data():
    import os
    import time
    # Load relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "processed_data.parquet")
    
    # Adding a stat check to ensure cache invalidation if file changes
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        st.error(f"Data file not found at {file_path}. Please run 'raar_model.py'.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Year
years = sorted(df['year'].unique(), reverse=True)
year_options = ['All'] + list(years)
selected_years = st.sidebar.multiselect("Select Year(s)", year_options, default=[years[0]])

# Resolve Year Filter
if 'All' in selected_years:
    filter_years = years
else:
    filter_years = selected_years

# Competition
competitions = sorted(df['competition'].unique())
comp_options = ['All'] + list(competitions)
default_comp = ['IPL'] if 'IPL' in competitions else comp_options[1:2] # Default to first non-All if IPL missing
selected_comps = st.sidebar.multiselect("Select Competition", comp_options, default=default_comp)

# Resolve Comp Filter
if 'All' in selected_comps:
    filter_comps = competitions
else:
    filter_comps = selected_comps

# Phase Filter
phases = ['PP', 'MO', 'Slog', 'Death']
selected_phases = st.sidebar.multiselect("Select Phase", phases, default=phases)

# Apply Filter
filtered_df = df[
    (df['year'].isin(filter_years)) & 
    (df['competition'].isin(filter_comps)) & 
    (df['phase'].isin(selected_phases))
]
st.sidebar.caption(f"Data Points: {len(filtered_df):,}")

# Header Helper
year_str = "All" if 'All' in selected_years else ', '.join(map(str, selected_years))

# --- Tabs ---
tab_bat, tab_bowl, tab_player, tab_ll = st.tabs(["Batting Leaderboard", "Bowling Leaderboard", "Player Analysis", "Line & Length Matrix"])

# Helper: Color scales
def highlighting_raa(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}'

with tab_bat:
    st.header(f"Top Batters ({year_str})")
    st.markdown("Sorted by **RAA** (Runs Above Average).")

    min_balls = st.slider("Min Balls Faced", 30, 500, 100, key="bat_min_balls")
    
    # Aggregation
    bat_grouped = filtered_df.groupby('bat').agg({
        'batruns': 'sum',
        'ball': 'count', # Balls Faced
        'RAA_bat': 'sum',
        'WAA_bat': 'sum',
        'is_wicket': 'sum',
        'True_SR': 'mean',
        'batruns': ['sum', lambda x: (x == 0).sum(), lambda x: (x.isin([4, 6])).sum()] 
    }).reset_index()
    
    # Flatten MultiIndex
    bat_grouped.columns = ['bat', 'batruns', 'dots', 'boundaries', 'ball', 'RAA_bat', 'WAA_bat', 'is_wicket', 'True_SR']
    
    # Re-map batruns (it might have gotten messed up by multi-agg on same col? No, actually dictionary agg with dup keys overrides. 
    # Use tuple list or specific named agg for safety)
    bat_grouped = filtered_df.groupby('bat').agg(
        batruns=('batruns', 'sum'),
        ball=('ball', 'count'),
        RAA_bat=('RAA_bat', 'sum'),
        WAA_bat=('WAA_bat', 'sum'),
        is_wicket=('is_wicket', 'sum'),
        True_SR=('True_SR', 'mean'),
        dots=('batruns', lambda x: (x == 0).sum()),
        boundaries=('batruns', lambda x: (x.isin([4, 6])).sum())
    ).reset_index()
    
    # Calculate Standard Metrics
    bat_grouped['SR'] = (bat_grouped['batruns'] / bat_grouped['ball'] * 100).round(2)
    bat_grouped['Avg'] = np.where(bat_grouped['is_wicket'] > 0, bat_grouped['batruns'] / bat_grouped['is_wicket'], bat_grouped['batruns']).round(2)
    
    # Filter
    bat_grouped = bat_grouped[bat_grouped['ball'] >= min_balls]
    
    # Rename/Round
    bat_grouped['RAA'] = bat_grouped['RAA_bat'].round(2)
    bat_grouped['Outs'] = bat_grouped['is_wicket'].astype(int)
    
    # New Metrics
    bat_grouped['Dot%'] = (bat_grouped['dots'] / bat_grouped['ball'] * 100).round(1)
    # BpB (Avoid div/0)
    bat_grouped['BpB'] = np.where(bat_grouped['boundaries'] > 0, bat_grouped['ball'] / bat_grouped['boundaries'], np.nan)
    bat_grouped['BpB'] = bat_grouped['BpB'].round(1)

    # RAA per Ball
    bat_grouped['RAA/Ball'] = (bat_grouped['RAA_bat'] / bat_grouped['ball']).round(2)

    # WAA per 100 Balls
    # Sum(WAA_bat) = Total Wickets Preserved
    # Rate = (Total / Balls) * 100
    bat_grouped['WAA'] = (bat_grouped['WAA_bat'] / bat_grouped['ball'] * 100).round(2)
    
    bat_grouped['True SR'] = bat_grouped['True_SR'].round(2)
    
    display_cols = ['bat', 'batruns', 'ball', 'SR', 'Dot%', 'BpB', 'True SR', 'Avg', 'Outs', 'RAA', 'RAA/Ball']
    
    st.dataframe(
        bat_grouped[display_cols].sort_values('RAA', ascending=False).style
        .format({
            'SR': "{:.2f}", 
            'True SR': "{:.2f}", 
            'Avg': "{:.2f}", 
            'RAA': "{:.2f}",
            'RAA/Ball': "{:.2f}",
            'Dot%': "{:.1f}%",
            'BpB': "{:.1f}"
        })
        .background_gradient(subset=['RAA', 'True SR'], cmap='Greens')
        .background_gradient(subset=['Dot%', 'BpB'], cmap='RdYlGn_r'), # Low = Green (Good)
        use_container_width=True,
        hide_index=True
    )

with tab_bowl:
    st.header(f"Top Bowlers ({year_str})")
    st.markdown("Sorted by **RAA** (Runs Saved). Positive = Good.")
    
    min_balls_bowl = st.slider("Min Balls Bowled", 30, 500, 100, key="bowl_min_balls")
    
    bowl_grouped = filtered_df.groupby('bowl').agg({
        'bowl_runs_conceded': 'sum',
        'ball': 'count',
        'RAA_bowl': 'sum',
        'WAA_bowl': 'sum',
        'is_wicket': 'sum',
        'True_SR': 'mean' # Inverse impact
    }).reset_index()
    
    bowl_grouped['Econ'] = (bowl_grouped['bowl_runs_conceded'] / (bowl_grouped['ball']/6)).round(2)
    bowl_grouped = bowl_grouped[bowl_grouped['ball'] >= min_balls_bowl]
    
    bowl_grouped['RAA'] = bowl_grouped['RAA_bowl'].round(2)
    
    # RAA per Ball
    bowl_grouped['RAA/Ball'] = (bowl_grouped['RAA_bowl'] / bowl_grouped['ball']).round(2)
    
    # WAA per 100 Balls (Wickets Taken Above Average per 100)
    bowl_grouped['WAA'] = (bowl_grouped['WAA_bowl'] / bowl_grouped['ball'] * 100).round(2)
    
    display_cols_b = ['bowl', 'bowl_runs_conceded', 'ball', 'is_wicket', 'Econ', 'WAA', 'RAA', 'RAA/Ball']
    
    st.dataframe(
        bowl_grouped[display_cols_b].sort_values('RAA', ascending=False).style
        .format({
            'Econ': "{:.2f}",
            'WAA': "{:.2f}",
            'RAA': "{:.2f}",
            'RAA/Ball': "{:.2f}"
        })
        .background_gradient(subset=['RAA', 'WAA'], cmap='Blues'),
        use_container_width=True,
        hide_index=True
    )

with tab_player:
    st.header("Player Inspector")
    
    player_list = sorted(filtered_df['bat'].unique())
    selected_player = st.selectbox("Select Player", player_list)
    
    p_df = filtered_df[filtered_df['bat'] == selected_player]
    
    # --- Helper: Scoring Rose Chart ---
    def generate_scoring_rose(df_valid, player_name, plot_title=None, group_by='Length'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Filter valid coordinates
        df_valid = df_valid[(df_valid['wagonX'] > 0) & (df_valid['wagonY'] > 0)].copy()
        
        if len(df_valid) == 0:
            st.info("No wagon wheel data available for this player.")
            return None

        # Calculate Angles
        df_valid['dx'] = df_valid['wagonX'] - 200
        df_valid['dy'] = 200 - df_valid['wagonY']
        df_valid['theta'] = np.arctan2(df_valid['dy'], df_valid['dx'])
        df_valid['theta_norm'] = df_valid['theta'].apply(lambda t: t if t >= 0 else t + 2*np.pi)
        
        # --- Dynamic Grouping Logic ---
        category_col = 'Category'
        colors = {}
        order = []
        
        if group_by == 'Length':
            def map_length(l):
                if pd.isna(l): return "Unknown"
                l = str(l).upper()
                if l in ['SHORT']: return 'Short'
                elif l in ['SHORT_OF_A_GOOD_LENGTH']: return 'Back of Length'
                elif l == 'GOOD_LENGTH': return 'Good Length'
                elif l == 'FULL': return 'Full'
                elif l == 'FULL_TOSS': return 'Full Toss'
                elif l == 'YORKER': return 'Yorker'
                return 'Unknown'
            df_valid[category_col] = df_valid['length'].apply(map_length)
            df_valid = df_valid[df_valid[category_col] != 'Unknown']
            order = ['Short', 'Good', 'Full']
            colors = {'Short': '#9B59B6', 'Good': '#2980B9', 'Full': '#27AE60'}
            
        elif group_by == 'Line':
            # Simplified Line Mapping
            def map_line(l):
                if pd.isna(l): return "Unknown"
                l = str(l).lower()
                if 'wide' in l: return 'Wide'
                if 'outside' in l: return 'Outside Off'
                if 'stumps' in l or 'middle' in l: return 'Stumps'
                if 'leg' in l: return 'Leg Side'
                return 'Other'
            df_valid[category_col] = df_valid['line'].apply(map_line)
            order = ['Wide', 'Outside Off', 'Stumps', 'Leg Side', 'Other']
            # Palette: Blue -> Purple -> Orange -> Red
            colors = {'Wide': '#3498DB', 'Outside Off': '#9B59B6', 'Stumps': '#F39C12', 'Leg Side': '#E74C3C', 'Other': '#95A5A6'}
            
        elif group_by == 'Bowler Kind':
            df_valid[category_col] = df_valid['bowl_kind'].fillna('Unknown')
            unique_kinds = sorted(df_valid[category_col].unique())
            order = unique_kinds
            # Simple Palette
            colors = {k: c for k, c in zip(unique_kinds, sns.color_palette("muted", len(unique_kinds)).as_hex())}
            
        elif group_by == 'Bowler Style':
             df_valid[category_col] = df_valid['bowl_style'].fillna('Unknown')
             # Keep Top 5 styles, others to 'Other' to avoid clutter
             top_styles = df_valid[category_col].value_counts().nlargest(5).index.tolist()
             df_valid[category_col] = df_valid[category_col].apply(lambda x: x if x in top_styles else 'Other')
             order = sorted(df_valid[category_col].unique())
             colors = {k: c for k, c in zip(order, sns.color_palette("bright", len(order)).as_hex())}
        
        else:
             st.error(f"Unknown grouping: {group_by}")
             return None

        if len(df_valid) == 0:
            st.info(f"No valid data for grouping: {group_by}")
            return None

        # Binning
        n_bins = 16 
        bins = np.linspace(0.0, 2 * np.pi, n_bins + 1)
        df_valid['bin'] = np.digitize(df_valid['theta_norm'], bins) - 1
        df_valid['bin'] = df_valid['bin'].clip(0, n_bins - 1)
        
        # Aggregation
        agg = df_valid.groupby(['bin', category_col])['batruns'].sum().unstack(fill_value=0)
        agg = agg.reindex(range(n_bins), fill_value=0)
        
        # Ensure all columns exist
        for cat in order:
            if cat not in agg.columns:
                agg[cat] = 0
        agg = agg[order] # Reorder cols

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        bg_color = 'white'
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        width = 2 * np.pi / n_bins
        theta = bins[:-1]
        bottom = np.full(n_bins, 2)
        
        for cat in order:
            if cat in agg.columns:
                values = agg[cat].values
                color = colors.get(cat, '#7F8C8D') # Default Grey
                ax.bar(theta, values, width=width, bottom=bottom, color=color, label=cat, edgecolor='white', linewidth=0.8, align='edge', alpha=0.95)
                bottom += values
            
        ax.grid(color='#E0E0E0', linestyle=':', linewidth=1.0)
        ax.spines['polar'].set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        # Labels
        texts = [
            (np.deg2rad(112.5), "Third Man"), (np.deg2rad(67.5), "Fine Leg"),
            (np.deg2rad(157.5), "Point"), (np.deg2rad(22.5), "Square Leg"),
            (np.deg2rad(202.5), "Cover"), (np.deg2rad(337.5), "Mid Wicket"),
            (np.deg2rad(247.5), "Long Off"), (np.deg2rad(292.5), "Long On")
        ]
        
        max_val = bottom.max()
        if max_val == 2: max_val = 10 

        for angle, text in texts:
            dist_mult = 1.25 if text in ["Square Leg", "Mid Wicket"] else 1.15
            dist = max_val * dist_mult
            ax.text(angle, dist, text, ha='center', va='center', fontsize=10, color='#2C3E50', weight='bold')

        legend = plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), title=group_by, frameon=False)
        plt.setp(legend.get_title(), color='#2C3E50', weight='bold')
        
        final_title = plot_title if plot_title else "Scoring Distribution"
        plt.title(final_title, y=1.08, fontsize=14, color='#2C3E50', weight='bold')
        
        return fig

    # --- Layout: 1. Metrics, 2. Bowler Types, 3. Phase & Rose ---
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", p_df['batruns'].sum())
    col1.metric("Total RAA", round(p_df['RAA_bat'].sum(), 2))
    col2.metric("Overall SR", round(p_df['batruns'].sum() / len(p_df) * 100, 2))
    col2.metric("Avg True SR", round(p_df['True_SR'].mean(), 2))
    
    # --- Advanced Metrics (Bowler Breakdown) ---
    st.subheader("Performance vs Bowler Types")
    
    # 1. By Bowler Kind
    st.markdown("**By Bowler Kind (Pace/Spin)**")
    if 'bowl_kind' in p_df.columns:
        kind_stats = p_df.groupby('bowl_kind').agg({
            'batruns': 'sum', 'ball': 'count', 'RAA_bat': 'sum', 'True_SR': 'mean',
            'is_wicket': 'sum' # Outs
        }).reset_index()
        
        kind_stats['SR'] = (kind_stats['batruns'] / kind_stats['ball'] * 100).round(2)
        kind_stats['Avg'] = np.where(kind_stats['is_wicket'] > 0, kind_stats['batruns'] / kind_stats['is_wicket'], kind_stats['batruns']).round(2)
        kind_stats = kind_stats.rename(columns={'is_wicket': 'Outs', 'RAA_bat': 'RAA', 'bowl_kind': 'Type'})
        
        st.dataframe(
            kind_stats[['Type', 'batruns', 'ball', 'Outs', 'Avg', 'SR', 'True_SR', 'RAA']].style
            .format({'SR': "{:.2f}", 'True_SR': "{:.2f}", 'RAA': "{:.2f}", 'Avg': "{:.2f}"}),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Bowler Kind data not available.")

    # 2. By Bowler Style
    st.markdown("**By Bowler Style**")
    if 'bowl_style' in p_df.columns:
        style_stats = p_df.groupby('bowl_style').agg({
            'batruns': 'sum', 'ball': 'count', 'RAA_bat': 'sum', 'True_SR': 'mean',
            'is_wicket': 'sum'
        }).reset_index()
        
        style_stats['SR'] = (style_stats['batruns'] / style_stats['ball'] * 100).round(2)
        style_stats['Avg'] = np.where(style_stats['is_wicket'] > 0, style_stats['batruns'] / style_stats['is_wicket'], style_stats['batruns']).round(2)
        style_stats = style_stats.rename(columns={'is_wicket': 'Outs', 'RAA_bat': 'RAA', 'bowl_style': 'Style'})
        
        st.dataframe(
            style_stats[['Style', 'batruns', 'ball', 'Outs', 'Avg', 'SR', 'True_SR', 'RAA']].style
            .format({'SR': "{:.2f}", 'True_SR': "{:.2f}", 'RAA': "{:.2f}", 'Avg': "{:.2f}"}),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Bowler Style data not available.") 
    
    # --- Performance by Phase & Rose Chart ---
    # --- Performance by Phase ---
    st.markdown("---")
    st.subheader("Performance by Phase")
    
    phase_stats = p_df.groupby('phase').agg({
        'batruns': 'sum',
        'ball': 'count',
        'RAA_bat': 'sum',
        'True_SR': 'mean'
    }).reset_index()
    
    phase_stats['SR'] = (phase_stats['batruns'] / phase_stats['ball'] * 100).round(2)
    phase_stats['RAA'] = phase_stats['RAA_bat'].round(2)
    
    # Custom Order
    phase_order = ['PP', 'MO', 'Slog', 'Death']
    phase_stats['phase'] = pd.Categorical(phase_stats['phase'], categories=phase_order, ordered=True)
    phase_stats = phase_stats.sort_values('phase')
    
    st.dataframe(
        phase_stats[['phase', 'batruns', 'ball', 'SR', 'True_SR', 'RAA']].style
        .format({
            'SR': "{:.2f}",
            'True_SR': "{:.2f}",
            'RAA': "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --- Detailed Scoring Analysis (Rose Chart) ---
    st.markdown("---")
    st.subheader("Detailed Scoring Analysis")
    st.caption("Filter specific delivery types to see scoring patterns.")
    
    if 'wagonX' in p_df.columns:
        # Filters Row
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        # 1. Bowler Kind
        b_kinds = sorted(p_df['bowl_kind'].dropna().unique())
        sel_kinds = col_f1.multiselect("Bowler Kind", b_kinds, default=b_kinds)
        
        # 2. Bowler Style
        b_styles = sorted(p_df['bowl_style'].dropna().unique())
        sel_styles = col_f2.multiselect("Bowler Style", b_styles, default=b_styles)
        
        # 3. Line
        lines = sorted(p_df['line'].dropna().unique())
        sel_lines = col_f3.multiselect("Line", lines, default=lines)
        
        # 4. Length
        lengths = sorted(p_df['length'].dropna().unique())
        sel_lengths = col_f4.multiselect("Length", lengths, default=lengths)
        
        st.markdown("---")
        # Color By Selector
        group_options = ['Length', 'Line', 'Bowler Kind', 'Bowler Style']
        group_by = st.radio("Color Distribution By:", group_options, index=0, horizontal=True)
        
        # Filter Data for Chart
        rose_df = p_df[
            (p_df['bowl_kind'].isin(sel_kinds)) &
            (p_df['bowl_style'].isin(sel_styles)) &
            (p_df['line'].isin(sel_lines)) &
            (p_df['length'].isin(sel_lengths))
        ]
        
        # Dynamic Title Construction
        title_parts = []
        if len(sel_kinds) < len(b_kinds): title_parts.append(f"vs {'/'.join(sel_kinds)}")
        if len(sel_lines) < len(lines): title_parts.append(f"Line: {'/'.join(sel_lines)}")
        if len(sel_lengths) < len(lengths): title_parts.append(f"Length: {'/'.join(sel_lengths)}")
        
        dynamic_title = "Scoring Distribution"
        if title_parts:
            dynamic_title += f"\n({', '.join(title_parts)})"
        
        st.markdown(f"**Balls Analyzed:** {len(rose_df)}")
        
        if not rose_df.empty:
            fig = generate_scoring_rose(rose_df, selected_player, plot_title=dynamic_title, group_by=group_by)
            if fig:
                st.pyplot(fig)
        else:
            st.warning("No data matches the selected filters.")
            
    else:
        st.info("Wagon Wheel data missing for Rose Chart.")

# --- Line & Length Analysis Tab ---
with tab_ll:
    st.header("Line & Length Analysis")
    st.caption("Analyzing Success % (Wickets) vs Efficiency (Economy) for different deliveries based on active filters.")
    
    # Data Prep
    ll_df = filtered_df.copy()
    
    # Optional Local Filters
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        focus_bowler = st.selectbox("Focus on Bowler (Optional)", [None] + sorted(ll_df['bowl'].unique().tolist()))
    
    if focus_bowler:
        ll_df = ll_df[ll_df['bowl'] == focus_bowler]
        
    # Validation: Ensure Line/Length exist
    if 'line' in ll_df.columns and 'length' in ll_df.columns and not ll_df.empty:
        # Group by Line & Length
        # Metrics:
        # Success % = (Wickets / Balls) * 100
        # Economy = (Runs / Balls) * 6
        
        ll_stats = ll_df.groupby(['line', 'length']).agg({
            'is_wicket': 'sum',
            'bowl_runs_conceded': 'sum',
            'ball': 'count'
        }).reset_index()
        
        # Min Sample Size (avoid noise)
        ll_stats = ll_stats[ll_stats['ball'] >= 5] 
        
        if not ll_stats.empty:
            ll_stats['Success %'] = (ll_stats['is_wicket'] / ll_stats['ball'] * 100).round(2)
            ll_stats['Economy'] = (ll_stats['bowl_runs_conceded'] / ll_stats['ball'] * 6).round(2)
            
            # 1. Success Matrix & Economy
            import plotly.express as px
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.subheader("🎯 Wicket Probability (Success %)")
                if len(ll_stats) > 0:
                    succ_matrix = ll_stats.pivot(index='length', columns='line', values='Success %')
                    fig_succ = px.imshow(
                        succ_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        color_continuous_scale="Reds",
                        title="Success % (Higher is Better)"
                    )
                    st.plotly_chart(fig_succ, use_container_width=True)
                
            with col_m2:
                st.subheader("💰 Economy Rate")
                if len(ll_stats) > 0:
                    econ_matrix = ll_stats.pivot(index='length', columns='line', values='Economy')
                    fig_econ = px.imshow(
                        econ_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        color_continuous_scale="RdYlGn_r", # Red is High (Bad), Green is Low (Good)
                        title="Economy Rate (Lower is Better)"
                    )
                    st.plotly_chart(fig_econ, use_container_width=True)
                
            # 2. Trade-off Scatter (Quadrant Chart)
            st.subheader("⚖️ Risk vs Reward (Quadrant Analysis)")
            st.markdown("Use this to instantly spot the **Golden Deliveries**.")
            
            ll_stats['Delivery'] = ll_stats['length'] + " - " + ll_stats['line']
            
            # Create Figure
            fig_trade = px.scatter(
                ll_stats,
                x="Economy",
                y="Success %",
                size="ball",
                color="Economy", # Color by Economy (Red=High)
                color_continuous_scale="RdYlGn_r", # Green=Low Econ
                hover_name="Delivery",
                hover_data=["ball", "is_wicket", "bowl_runs_conceded"],
                title="Line & Length Efficiency",
            )
            
            # --- Add Zones (Quadrants) ---
            # Define Boundaries (Approximate for T20 Death)
            # Good Econ: < 9. Good Success: > 15%?
            # Let's use dynamic means or static benchmarks.
            # Static T20 Death Benchmarks:
            # Great Econ: < 8. Good Success: > 20%
            
            # Green Zone (Top Left): Econ < 8.5, Success > 15
            fig_trade.add_shape(type="rect",
                x0=0, y0=15, x1=8.5, y1=100,
                fillcolor="green", opacity=0.1, layer="below", line_width=0,
            )
            fig_trade.add_annotation(x=4.25, y=90, text="GOLDEN ZONE\n(Wickets + Savings)", showarrow=False, font=dict(color="green", size=14, weight="bold"))

            # Red Zone (Bottom Right): Econ > 10.5, Success < 10
            fig_trade.add_shape(type="rect",
                x0=10.5, y0=0, x1=20, y1=10,
                fillcolor="red", opacity=0.1, layer="below", line_width=0,
            )
            fig_trade.add_annotation(x=15, y=5, text="DANGER ZONE\n(Expensive, No Wickets)", showarrow=False, font=dict(color="red", size=12))
            
            # Update Layout
            fig_trade.update_layout(
                xaxis_title="Economy Rate (Runs/Over) ⬅️ Lower is Better",
                yaxis_title="Success % (Wickets/Ball) ⬆️ Higher is Better",
                showlegend=False
            )
            
            # Fix Axis Ranges to look good
            max_econ = ll_stats['Economy'].max() + 1
            max_succ = ll_stats['Success %'].max() + 5
            fig_trade.update_xaxes(range=[0, max(14, max_econ)])
            fig_trade.update_yaxes(range=[0, max(30, max_succ)])
            
            st.plotly_chart(fig_trade, use_container_width=True)
            
            # 3. Top Deliveries Table
            st.subheader("🏆 Best Deliveries Ranked")
            st.caption("Ranked by a composite 'Impact Score' (High Wickets + Low Economy).")
            
            # Simple Heuristic Score: Success% * 2 - Economy
            ll_stats['Impact Score'] = (ll_stats['Success %'] * 2) - ll_stats['Economy']
            
            top_deliveries = ll_stats.sort_values('Impact Score', ascending=False).head(5)
            st.table(top_deliveries[['Delivery', 'ball', 'Success %', 'Economy', 'is_wicket']].style.format({
                'Success %': "{:.1f}%",
                'Economy': "{:.2f}",
            }))
        else:
            st.info("Not enough data points (min 5) for selected filters.")
            
    else:
        st.error("Line/Length data missing or empty selection.") 
