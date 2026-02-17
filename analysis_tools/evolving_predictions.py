import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ---------------------------------------------------------------
# CONFIG — add/remove prediction files here
# ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

HISTORY_CSV = os.path.join(HERE, 'history.csv')

PREDICTION_FILES = [
    os.path.join(HERE, '1.txt'),
    os.path.join(HERE, '2.txt'),
    # os.path.join(HERE, '48hr_2026-02-18.txt'),
    # os.path.join(HERE, '48hr_2026-02-19.txt'),
]

# Output images will be saved alongside this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TIMEZONE   = 'Australia/Sydney'
# ---------------------------------------------------------------


def load_history(path):
    df = pd.read_csv(path)
    df['last_changed'] = pd.to_datetime(df['last_changed'], utc=True)
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df = df.dropna(subset=['state'])
    df['time'] = df['last_changed'].dt.tz_convert(TIMEZONE)
    return df


def load_predictions(path):
    times, values = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('- time:'):
                t = line.replace('- time:', '').strip().strip("'")
                times.append(pd.to_datetime(t))
            elif line.startswith('value:'):
                values.append(float(line.replace('value:', '').strip()))
    df = pd.DataFrame({'time': times, 'value': values})
    # Timestamps already contain offset info (e.g. +11:00) — don't force utc=True
    df['time'] = pd.to_datetime(df['time']).dt.tz_convert(TIMEZONE)
    return df


def plot_prediction(history_df, pred_df, label, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(history_df['time'], history_df['state'],
            color='steelblue', linewidth=0.8, alpha=0.7, label='Historical (actual)')
    ax.plot(pred_df['time'], pred_df['value'],
            color='tomato', linewidth=2, linestyle='--',
            marker='o', markersize=4, label=f'48h Prediction ({label})')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=45, ha='right')

    ax.set_xlabel(f'Time ({TIMEZONE})')
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Plant Consumed Power — Historical vs 48h Prediction\n{label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path}')


# --- Main ---
history = load_history(HISTORY_CSV)

for pred_file in PREDICTION_FILES:
    pred_df = load_predictions(pred_file)
    # Use the filename (without extension) as the identifier
    identifier = os.path.splitext(os.path.basename(pred_file))[0]
    out_path = os.path.join(OUTPUT_DIR, f'power_plot_{identifier}.png')
    plot_prediction(history, pred_df, label=identifier, out_path=out_path)