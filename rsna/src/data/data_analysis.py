"""Print class frequencies and patient counts from the prepared manifest."""
import pandas as pd


def analyze_dataset(config):
    frame = pd.read_csv(config.TRAIN_CSV, dtype={'patient': str})
    print(f'Training slices: {len(frame)}; patients: {frame.patient.nunique()}')
    print('Positive fraction by class:')
    print(frame[config.CLASS_NAMES].mean().to_string())
    print('Slices per validation fold:')
    print(frame.groupby('fold').size().to_string())
