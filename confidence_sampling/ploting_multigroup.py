import wandb
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. Configuration & Colors ---
MY_COLORS = {
    'lrdy_04': "#05D216",
    'rand': "#EE9D1B",
    'lrdy_04_adj': "#4CC7A8",
    'geosketch': "#0A3DCA",
    'el2n': "#A538A2",
    'aum': "#825050"
}

def get_config_val(config, key_path):
    """Safely retrieves nested config values."""
    keys = key_path.split('.')
    val = config
    try:
        for k in keys:
            val = val.get(k, {})
        if isinstance(val, dict) and not val:
            return "N/A"
        return val
    except:
        return "N/A"

# --- 2. STEP 1: Fetch Raw Data (Network Intensive) ---
def fetch_raw_runs_data(entity, project, group_ids, metrics_list):
    """
    Fetches raw run objects and their history for the specified metrics.
    Returns a list of dictionaries (one per run) and a metadata dict.
    """
    api = wandb.Api()
    raw_runs_data = []
    metadata = {} 
    
    print(f"--- Fetching raw data for {len(group_ids)} groups ---")
    
    for gid in group_ids:
        # Fetch runs in this group
        runs = api.runs(f"{entity}/{project}", filters={"group": gid})
        
        if not runs:
            print(f"Warning: No runs found for group {gid}")
            continue

        # Capture metadata from the first valid run in the group
        if not metadata:
            first_run = runs[0]
            params = first_run.config.get('actual_train_params', {})
            metadata = {
                'test_size': params.get('test_size', 'N/A'),
                'n_classes': params.get('n_classes', 'N/A'),
                'train_test_settings': first_run.config.get('train_test_settings', 'N/A')
            }
        
        # Loop through runs and fetch history ONCE for all metrics
        for run in runs:
            # We fetch all requested metrics in a single API call
            try:
                # keys=metrics_list makes it efficient
                history = run.history(keys=metrics_list)
                
                # If the run has data, store it
                if not history.empty:
                    raw_runs_data.append({
                        "run_name": run.name,
                        "group_id": gid,
                        "config": run.config,
                        "history": history # This is a pandas DF
                    })
            except Exception as e:
                print(f"Error fetching history for run {run.name}: {e}")

    print(f"-> Successfully fetched data for {len(raw_runs_data)} runs.")
    return raw_runs_data, metadata


# --- 3. STEP 2: Construct DataFrame (Logic Intensive) ---
def construct_plotting_df(raw_runs_data, x_axis_key, k_last_epochs=20):
    """
    Takes raw data and builds a clean DataFrame for plotting.
    """
    all_data = []
    
    for item in raw_runs_data:
        history_df = item['history']
        config = item['config']
        
        # 1. Get X-Axis Value (e.g. Sample Size)
        x_val = get_config_val(config, x_axis_key)
        
        # 2. Slice the last K epochs
        # We assume the history df is already sorted by step/epoch
        if len(history_df) > k_last_epochs:
            subset_df = history_df.tail(k_last_epochs).copy()
        else:
            subset_df = history_df.copy()
            
        # 3. Add Metadata Columns to this subset
        subset_df['Group'] = item['group_id']
        subset_df['Method'] = item['run_name'] # Or parse method from config if needed
        subset_df[x_axis_key] = x_val
        
        all_data.append(subset_df)

    # Combine all
    if not all_data:
        return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by X-axis if possible
    try:
        df = df.sort_values(by=x_axis_key)
    except:
        pass
        
    return df


# --- 4. STEP 3: Plotting ---
def create_comparison_box(df, x_axis_key, y_metric, metadata, k_epochs):
    """
    Plots a specific metric (y_metric) from the DataFrame.
    """
    if df.empty:
        print("No data to plot.")
        return None

    # Ensure X-axis is string/categorical
    df[x_axis_key] = df[x_axis_key].astype(str)
    
    title_text = (
        f"<b>{y_metric} vs {x_axis_key}</b><br>"
        f"<span style='font-size: 12px;'>Last {k_epochs} epochs aggregated</span><br>"
        f"<span style='font-size: 10px; color: gray;'>Test Size: {metadata.get('test_size')} | Classes: {metadata.get('n_classes')} | Set: {metadata.get('train_test_settings')}</span>"
    )

    fig = px.box(
        df,
        x=x_axis_key,
        y=y_metric,  # <--- We now pass the specific column name we want to plot
        color="Method",
        color_discrete_map=MY_COLORS,
        title=title_text,
        template="plotly_white",
        category_orders={"Method": ["lrdy_04", "lrdy_04_adj", 'rand',
                                    'aum', 'el2n', 'geosketch']}
    )

    fig.update_layout(
        xaxis_title=x_axis_key,
        yaxis_title=y_metric,
        legend_title="Method",
        hovermode="x unified",
        boxmode='group',
        boxgap=0.1,
        boxgroupgap=0,
        yaxis=dict(gridcolor='lightgrey', gridwidth=1, dtick=0.05)
    )
    fig.update_traces(opacity=0.8)

    return fig

def upload_to_wandb(fig, project, run_name):
    with wandb.init(project=project, name=run_name, job_type="report_visuals"):
        wandb.log({"comparison_box": fig})
        print(f"✅ Plot uploaded to project '{project}'.")

# --- MAIN USAGE ---
if __name__ == "__main__":
    ENTITY = "your_entity"
    PROJECT = "your_project"
    
    # 1. Define Groups & Metrics
    GROUP_IDS = [
        # "2025-12-30_09-28-45"
    ]
    METRICS_TO_FETCH = ["f1_macro", "acc", "f1_weighted"] # <--- Fetch multiple!
    X_AXIS_KEY = "actual_train_params.sample_size" 

    # 2. STEP 1: Fetch (Slow, runs once)
    raw_data, meta = fetch_raw_runs_data(ENTITY, PROJECT, GROUP_IDS, METRICS_TO_FETCH)
    
    # 3. STEP 2: Process (Fast, run multiple times if you change k_epochs)
    df = construct_plotting_df(raw_data, X_AXIS_KEY, k_last_epochs=20)
    
    # 4. STEP 3: Plot (Can plot different metrics from same df)
    if not df.empty:
        # Plot 1: Accuracy
        fig_acc = create_comparison_box(df, X_AXIS_KEY, "acc", meta, k_epochs=20)
        # fig_acc.show()
        
        # Plot 2: F1 Macro
        fig_f1 = create_comparison_box(df, X_AXIS_KEY, "f1_macro", meta, k_epochs=20)
        # fig_f1.show()
        
        upload_to_wandb(fig_acc, PROJECT, f"box_plot_acc")