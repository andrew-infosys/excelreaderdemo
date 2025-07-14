"""
Visualization components for Excel Reading Agent
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Any
import streamlit as st
from config import CHART_CONFIG

class DataVisualizer:
    """
    Advanced data visualization capabilities
    """
    
    def __init__(self):
        self.color_schemes = CHART_CONFIG["color_schemes"]
        self.numeric_charts = CHART_CONFIG["numeric_chart_types"]
        self.categorical_charts = CHART_CONFIG["categorical_chart_types"]
    
    def create_chart(self, df: pd.DataFrame, chart_type: str, column: str, 
                    color_scheme: str = "Default", **kwargs) -> Optional[go.Figure]:
        """
        Create a chart based on the specified type and parameters
        """
        try:
            if chart_type in self.numeric_charts:
                return self._create_numeric_chart(df, chart_type, column, color_scheme, **kwargs)
            elif chart_type in self.categorical_charts:
                return self._create_categorical_chart(df, chart_type, column, color_scheme, **kwargs)
            else:
                return None
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    def _create_numeric_chart(self, df: pd.DataFrame, chart_type: str, column: str, 
                             color_scheme: str, **kwargs) -> go.Figure:
        """Create charts for numeric data"""
        color_map = self.color_schemes.get(color_scheme)
        
        if chart_type == "Histogram":
            fig = px.histogram(
                df, x=column, 
                title=f"Distribution of {column}",
                color_discrete_sequence=px.colors.qualitative.Set3 if not color_map else None
            )
            if kwargs.get("show_stats", False):
                fig.add_vline(
                    x=df[column].mean(), 
                    line_dash="dash",
                    annotation_text=f"Mean: {df[column].mean():.2f}"
                )
        
        elif chart_type == "Box Plot":
            fig = px.box(df, y=column, title=f"Box Plot of {column}")
        
        elif chart_type == "Violin Plot":
            fig = px.violin(df, y=column, title=f"Violin Plot of {column}")
        
        elif chart_type == "Strip Plot":
            fig = px.strip(df, y=column, title=f"Strip Plot of {column}")
        
        elif chart_type == "Line Chart":
            fig = px.line(df, y=column, title=f"Line Chart of {column}")
            if len(df) > 1000:
                fig.update_layout(
                    title=f"Line Chart of {column} (Showing trend for {len(df)} points)"
                )
        
        elif chart_type == "Density Plot":
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[column], 
                histnorm='probability density',
                name='Density', 
                opacity=0.7
            ))
            fig.update_layout(title=f"Density Plot of {column}")
        
        elif chart_type == "Q-Q Plot":
            from scipy import stats
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(df)))
            sample_quantiles = np.sort(df[column].dropna())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles[:len(sample_quantiles)],
                y=sample_quantiles,
                mode='markers',
                name='Data points'
            ))
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles[:len(sample_quantiles)],
                y=theoretical_quantiles[:len(sample_quantiles)],
                mode='lines',
                name='Reference line',
                line=dict(dash='dash')
            ))
            fig.update_layout(
                title=f"Q-Q Plot of {column}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
        
        return fig
    
    def _create_categorical_chart(self, df: pd.DataFrame, chart_type: str, column: str, 
                                 color_scheme: str, **kwargs) -> go.Figure:
        """Create charts for categorical data"""
        value_counts = df[column].value_counts()
        
        if chart_type == "Bar Chart":
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f"Distribution of {column}",
                labels={'x': column, 'y': 'Count'}
            )
        
        elif chart_type == "Pie Chart":
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {column}"
            )
        
        elif chart_type == "Donut Chart":
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {column}",
                hole=0.4
            )
        
        elif chart_type == "Treemap":
            fig = px.treemap(
                names=value_counts.index,
                values=value_counts.values,
                title=f"Treemap of {column}"
            )
        
        elif chart_type == "Sunburst":
            # For sunburst, we need hierarchical data
            # Simplify by grouping small categories
            top_categories = value_counts.head(8)
            others_count = value_counts.tail(-8).sum() if len(value_counts) > 8 else 0
            
            if others_count > 0:
                plot_data = pd.concat([top_categories, pd.Series({'Others': others_count})])
            else:
                plot_data = top_categories
            
            fig = px.sunburst(
                names=plot_data.index,
                values=plot_data.values,
                title=f"Sunburst Chart of {column}"
            )
        
        elif chart_type == "Count Plot":
            fig = px.histogram(
                df, x=column,
                title=f"Count Plot of {column}"
            )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=True
        )
        
        fig.update_layout(
            title="Correlation Matrix of Numeric Variables",
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: str = None, size_col: str = None) -> go.Figure:
        """Create interactive scatter plot"""
        scatter_kwargs = {
            'data_frame': df,
            'x': x_col,
            'y': y_col,
            'title': f"{y_col} vs {x_col}"
        }
        
        if color_col and color_col in df.columns:
            scatter_kwargs['color'] = color_col
            scatter_kwargs['title'] += f" (colored by {color_col})"
        
        if size_col and size_col in df.columns:
            scatter_kwargs['size'] = size_col
            scatter_kwargs['title'] += f" (sized by {size_col})"
        
        fig = px.scatter(**scatter_kwargs)
        
        # Add trend line if both columns are numeric
        if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
            fig = px.scatter(**scatter_kwargs, trendline="ols")
        
        return fig
    
    def create_time_series_chart(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """Create time series visualization"""
        # Ensure date column is datetime
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        
        # Sort by date
        df_temp = df_temp.sort_values(date_col)
        
        fig = px.line(
            df_temp, 
            x=date_col, 
            y=value_col,
            title=f"{value_col} Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode='x unified'
        )
        
        return fig
    
    def create_multi_column_comparison(self, df: pd.DataFrame, columns: List[str]) -> go.Figure:
        """Create comparison chart for multiple columns"""
        if len(columns) < 2:
            return None
        
        fig = go.Figure()
        
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    boxpoints='outliers'
                ))
        
        fig.update_layout(
            title="Multi-Column Comparison",
            yaxis_title="Values",
            showlegend=True
        )
        
        return fig
    
    def create_distribution_comparison(self, df: pd.DataFrame, columns: List[str]) -> go.Figure:
        """Create overlaid distribution plots"""
        fig = go.Figure()
        
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                fig.add_trace(go.Histogram(
                    x=df[col],
                    name=col,
                    opacity=0.7,
                    nbinsx=30
                ))
        
        fig.update_layout(
            title="Distribution Comparison",
            xaxis_title="Values",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        return fig

class VisualizationInterface:
    """
    Streamlit interface for data visualization
    """
    
    def __init__(self):
        self.visualizer = DataVisualizer()
    
    def render_visualization_section(self, df: pd.DataFrame):
        """Render the complete visualization section"""
        if df is None or df.empty:
            st.warning("Please upload data to create visualizations.")
            return
        
        st.divider()
        st.subheader("📈 Interactive Visualizations")
        
        # Get column information
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = self._detect_date_columns(df)
        all_cols = df.columns.tolist()
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs([
            "📊 Single Column", 
            "📈 Two Columns", 
            "🔍 Distribution", 
            "📋 Summary Charts", 
            "⏱️ Time Series", 
            "🎯 Advanced Analytics"
        ])
        
        with viz_tabs[0]:
            self._render_single_column_viz(df, numeric_cols, categorical_cols, all_cols)
        
        with viz_tabs[1]:
            self._render_two_column_viz(df, numeric_cols, categorical_cols, all_cols)
        
        with viz_tabs[2]:
            self._render_distribution_viz(df, numeric_cols)
        
        with viz_tabs[3]:
            self._render_summary_charts(df, numeric_cols, categorical_cols)
        
        with viz_tabs[4]:
            self._render_time_series_viz(df, date_cols, numeric_cols)
        
        with viz_tabs[5]:
            self._render_advanced_analytics(df, numeric_cols)
    
    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect potential date columns"""
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(10), errors='raise')
                date_cols.append(col)
            except:
                continue
        return date_cols
    
    def _render_single_column_viz(self, df: pd.DataFrame, numeric_cols: List[str], 
                                 categorical_cols: List[str], all_cols: List[str]):
        """Render single column visualization tab"""
        st.markdown("**📊 Single Column Analysis**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_col = st.selectbox("Select Column to Visualize", all_cols, key="single_col")
        
        with col2:
            if selected_col in numeric_cols:
                chart_type = st.selectbox("Chart Type", self.visualizer.numeric_charts, key="single_chart")
            else:
                chart_type = st.selectbox("Chart Type", self.visualizer.categorical_charts, key="single_chart")
        
        with col3:
            st.markdown("**Options:**")
            show_stats = st.checkbox("Show Statistics", key="single_stats")
            color_scheme = st.selectbox("Color Scheme", list(self.visualizer.color_schemes.keys()), key="single_color")
        
        if st.button("Generate Chart", key="single_gen"):
            with st.spinner("Creating visualization..."):
                fig = self.visualizer.create_chart(
                    df, chart_type, selected_col, color_scheme, show_stats=show_stats
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to create chart. Please try different settings.")
    
    def _render_two_column_viz(self, df: pd.DataFrame, numeric_cols: List[str], 
                              categorical_cols: List[str], all_cols: List[str]):
        """Render two column visualization tab"""
        st.markdown("**📈 Two Column Analysis**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("Select X Column", all_cols, key="x_col")
        
        with col2:
            y_col = st.selectbox("Select Y Column", all_cols, key="y_col")
        
        with col3:
            color_col = st.selectbox("Color by (optional)", ["None"] + all_cols, key="color_col")
            color_col = None if color_col == "None" else color_col
        
        if st.button("Create Scatter Plot", key="scatter_gen"):
            if x_col != y_col:
                with st.spinner("Creating scatter plot..."):
                    fig = self.visualizer.create_scatter_plot(df, x_col, y_col, color_col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please select different columns for X and Y axes.")
    
    def _render_distribution_viz(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Render distribution visualization tab"""
        st.markdown("**🔍 Distribution Analysis**")
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for distribution analysis.")
            return
        
        selected_cols = st.multiselect(
            "Select columns to compare", 
            numeric_cols, 
            default=numeric_cols[:min(3, len(numeric_cols))],
            key="dist_cols"
        )
        
        if len(selected_cols) >= 1:
            viz_type = st.radio(
                "Visualization Type",
                ["Box Plot Comparison", "Distribution Overlay", "Individual Histograms"],
                key="dist_type"
            )
            
            if st.button("Generate Distribution Chart", key="dist_gen"):
                with st.spinner("Creating distribution visualization..."):
                    if viz_type == "Box Plot Comparison":
                        fig = self.visualizer.create_multi_column_comparison(df, selected_cols)
                    elif viz_type == "Distribution Overlay":
                        fig = self.visualizer.create_distribution_comparison(df, selected_cols)
                    else:  # Individual Histograms
                        for col in selected_cols:
                            fig = self.visualizer.create_chart(df, "Histogram", col)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        return
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_summary_charts(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
        """Render summary charts tab"""
        st.markdown("**📋 Summary Charts**")
        
        if st.button("Generate Correlation Heatmap", key="corr_gen"):
            if len(numeric_cols) >= 2:
                with st.spinner("Creating correlation heatmap..."):
                    fig = self.visualizer.create_correlation_heatmap(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        st.divider()
        
        if categorical_cols:
            st.markdown("**Category Distribution Summary**")
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                with st.expander(f"Distribution of {col}"):
                    fig = self.visualizer.create_chart(df, "Bar Chart", col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_time_series_viz(self, df: pd.DataFrame, date_cols: List[str], numeric_cols: List[str]):
        """Render time series visualization tab"""
        st.markdown("**⏱️ Time Series Analysis**")
        
        if not date_cols:
            st.warning("No date columns detected in your data.")
            return
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for time series analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Select Date Column", date_cols, key="ts_date")
        
        with col2:
            value_col = st.selectbox("Select Value Column", numeric_cols, key="ts_value")
        
        if st.button("Create Time Series Chart", key="ts_gen"):
            with st.spinner("Creating time series chart..."):
                try:
                    fig = self.visualizer.create_time_series_chart(df, date_col, value_col)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating time series chart: {str(e)}")
    
    def _render_advanced_analytics(self, df: pd.DataFrame, numeric_cols: List[str]):
        """Render advanced analytics tab"""
        st.markdown("**🎯 Advanced Analytics**")
        
        if len(numeric_cols) < 2:
            st.warning("Advanced analytics require at least 2 numeric columns.")
            return
        
        # Statistical summary
        if st.button("Generate Statistical Summary", key="stats_summary"):
            st.subheader("📊 Detailed Statistics")
            stats_df = df[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
            
            # Additional statistics
            st.subheader("📈 Additional Metrics")
            additional_stats = {}
            for col in numeric_cols:
                additional_stats[col] = {
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis(),
                    'CV (%)': (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0
                }
            
            additional_df = pd.DataFrame(additional_stats).T
            st.dataframe(additional_df, use_container_width=True) 