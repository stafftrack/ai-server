import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv
from supa import parse_data, update

def pred_repair_date():
    parse_data()

    # Load the CSV file into a DataFrame
    df = pd.read_csv('output.csv')

    # Convert the 'DateTime' column to a datetime object
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # Adjust the figsize as needed (width, height)

    # Smooth the data using a rolling mean
    window_size = 100  # Adjust the window size as needed
    HQ_state_avg_duration = 0
    AZ_state_avg_duration = 0
    HQ_steady_start_date = None
    AZ_steady_start_date = None
    for zone, zone_data in df.groupby('Zone'):
        smoothed_data = zone_data['ToolScanTime'].rolling(window=window_size).mean()

        # Calculate the slope for smoothed data
        smoothed_slope = np.gradient(smoothed_data)

        # Convert the smoothed slope to a Pandas Series
        smoothed_slope_series = pd.Series(smoothed_slope)

        # Apply smoothing to the slope
        slope_smoothing_window = 20  # Adjust the window size for slope smoothing
        smoothed_slope_smoothed = smoothed_slope_series.rolling(window=slope_smoothing_window).mean()

        # Calculate the slope for smoothed-slope
        second_smoothed_slope = np.gradient(smoothed_slope_smoothed)
        
        # Convert the smoothed slope to a Pandas Series
        second_smoothed_slope_series = pd.Series(second_smoothed_slope)
        
        # Apply smoothing to the slope
        second_slope_smoothing_window = 10  # Adjust the window size for slope smoothing
        second_smoothed_slope_smoothed = second_smoothed_slope_series.rolling(window=second_slope_smoothing_window).mean()

        # FSM
        cs, ns = 'break_in', 'break_in'
        if zone == 'HQ':
            slope_thres = 0.002
            sec_slope_thres = 0.0002
            steady_thres = 0.0005
        else:
            slope_thres = 0.003
            sec_slope_thres = 0.0001
            steady_thres = 0.0005
        break_in_points = []
        steady_points = []
        accelerate_points = []
        repair_points = []
        duration = 0
        steady_start = None
        accelerate_start = None
        state_durations = {
            'steady': [],
            'accelerate': []
        }
        for i in range(0, len(smoothed_data)):
            cur_slope = smoothed_slope_smoothed[i]
            cur_sec_slope = second_smoothed_slope_smoothed[i]
            if cs == 'break_in':
                break_in_points.append(i)
                if duration < 50:
                    duration += 1
                else:
                    duration = 0
                    ns = 'break_in'
                    if cur_slope < steady_thres:
                        ns = 'steady'
                        steady_start = zone_data['DateTime'].iloc[i]
                        if zone == 'AZ':
                            AZ_steady_start_date = steady_start
                        if zone == 'HQ':
                            HQ_steady_start_date = steady_start
                    else:
                        ns = 'break_in'
            elif cs == 'steady':
                steady_points.append(i)
                if cur_slope > slope_thres:
                    ns = 'accelerate'
                    accelerate_start = zone_data['DateTime'].iloc[i]
                    state_durations['steady'].append((steady_start, zone_data['DateTime'].iloc[i]))
                else:
                    ns = 'steady'
            elif cs == 'accelerate':
                accelerate_points.append(i)
                if cur_sec_slope < -sec_slope_thres:
                    ns = 'repair'
                    state_durations['accelerate'].append((accelerate_start, zone_data['DateTime'].iloc[i]))
                else:
                    ns = 'accelerate'
            elif cs == 'repair':
                repair_points.append(i)
                if cur_sec_slope > sec_slope_thres:
                    ns = 'break_in'
                else:
                    ns = 'repair'
            cs = ns
        # Calculate the duration for "steady" and "accelerate" states
        if zone == 'HQ':
            HQ_state_avg_duration = 0
            for state in ['steady', 'accelerate']:
                if state_durations[state]:
                    durations = [(end - start).total_seconds() / (60 * 60 * 24) for start, end in state_durations[state]]
                    state_average_duration = np.mean(durations)
                    HQ_state_avg_duration += state_average_duration
        elif zone == 'AZ':
            AZ_state_avg_duration = 0
            for state in ['steady', 'accelerate']:
                if state_durations[state]:
                    durations = [(end - start).total_seconds() / (60 * 60 * 24) for start, end in state_durations[state]]
                    state_average_duration = np.mean(durations)
                    AZ_state_avg_duration += state_average_duration

        # Plot smoothed data
        ax1.plot(zone_data['DateTime'], smoothed_data, label=f'Smoothed - {zone}')

        # Highlight points where the FSM state changes
        ax1.scatter(zone_data['DateTime'].iloc[break_in_points], smoothed_data.iloc[break_in_points], c='green', label='Break In')
        ax1.scatter(zone_data['DateTime'].iloc[steady_points], smoothed_data.iloc[steady_points], c='blue', label='Steady')
        ax1.scatter(zone_data['DateTime'].iloc[accelerate_points], smoothed_data.iloc[accelerate_points], c='red', label='Accelerate')
        ax1.scatter(zone_data['DateTime'].iloc[repair_points], smoothed_data.iloc[repair_points], c='orange', label='Repair')

        # Plot smoothed and smoothed-slope data
        ax2.plot(zone_data['DateTime'], smoothed_slope_smoothed, label=f'Smoothed Slope - {zone}')

    HQ_next_repairs_date = HQ_steady_start_date + datetime.timedelta(days=HQ_state_avg_duration)
    print(f'HQ_next_repairs_date: {HQ_next_repairs_date}')
    if AZ_state_avg_duration > 0:
        AZ_next_repairs_date = AZ_steady_start_date + datetime.timedelta(days=AZ_state_avg_duration)
    else:
        AZ_next_repairs_date = AZ_steady_start_date + datetime.timedelta(days=HQ_state_avg_duration*1.5)
    print(f'AZ_next_repairs_date: {AZ_next_repairs_date}')


    # Customize the first plot (Smoothed Data)
    ax1.set_xlabel('Date and Time')
    ax1.set_ylabel('Smoothed ToolScanTime')
    ax1.set_title('Smoothed ToolScanTime by Zone')
    ax1.legend()

    # Customize the second plot (Smoothed Slope Data)
    ax2.set_xlabel('Date and Time')
    ax2.set_ylabel('Smoothed Slope')
    ax2.set_title('Smoothed Slope of ToolScanTime by Zone')
    ax2.legend()

    # Adjust spacing between the subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

if __name__ == '__main__':
    pred_repair_date()