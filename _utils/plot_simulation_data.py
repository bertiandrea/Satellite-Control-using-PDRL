import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

folder_path = 'C:/Users/andre/Downloads'
smoothing_percentage = 10  

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    window = max(1, int(len(df) * smoothing_percentage / 100))
    df['Smoothed'] = df['Value'].rolling(window, center=True, min_periods=1).mean()
    
    plt.figure()
    plt.plot(df['Step'], df['Value'], linestyle='-', alpha=0.3, color='blue', label='Raw Value')
    plt.plot(df['Step'], df['Smoothed'], linestyle='-', alpha=1.0, color='blue', label=f'Smoothed ({smoothing_percentage}%)')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f"Value over Step — {os.path.basename(csv_file)}")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(
        folder_path,
        os.path.splitext(os.path.basename(csv_file))[0] + '_step_plot.png'
    )
    plt.savefig(output_path)
    plt.close()
