import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import signal
from scipy.signal import stft

def timeseries_preprocess(df_metadata):
    # Substract time series of views
    df_timeseries_views = df_metadata[['categories', 'display_id', 'view_count', 'upload_date']].copy()
    
    # Filter rows with 'Gaming' category and create a copy
    df_timeseries_Gaming = df_timeseries_views[df_timeseries_views['categories'].isin(['Gaming'])].copy()

    # Transform 'upload_date' to datetime type
    df_timeseries_Gaming['upload_date'] = pd.to_datetime(df_timeseries_Gaming['upload_date'])

    # Transform date to period with 5 categories
    df_timeseries_Gaming['Day'] = df_timeseries_Gaming['upload_date'].dt.to_period('D')
    df_timeseries_Gaming['Week'] = df_timeseries_Gaming['upload_date'].dt.to_period('W')
    df_timeseries_Gaming['Month'] = df_timeseries_Gaming['upload_date'].dt.to_period('M')
    df_timeseries_Gaming['Season'] = df_timeseries_Gaming['upload_date'].dt.to_period('Q')
    df_timeseries_Gaming['Year'] = df_timeseries_Gaming['upload_date'].dt.to_period('Y')

    # Choose Gaming for example, and group it by 'Day'
    df_timeseries_Gaming_views = df_timeseries_Gaming[['view_count', 'Day']].groupby('Day').agg(
        view_mean=('view_count', 'mean'),
        view_sum=('view_count', 'sum')
    ).reset_index()

    # Transform 'Day' to timestamp for plotting
    df_timeseries_Gaming_views['Day'] = df_timeseries_Gaming_views['Day'].dt.to_timestamp()

    return df_timeseries_Gaming_views

# Visualize the mean and sum of views over time
def timeseries_time_visualization(df_timeseries_Gaming_views):
    def df_timeseries_plot_time_domain(df_timeseries_Gaming_views, *args):
        view_label = ['View Mean']
        plt.figure(figsize=(10, 5))
        i = 0
        # for i in range(2):  
        # plt.subplot(2, 1, i + 1, )  
        plt.plot(df_timeseries_Gaming_views['Day'], df_timeseries_Gaming_views.iloc[:, i+1], label=view_label[i], marker='o', markersize=4)
        if len(args) > 0:
            df_timeseries_Gaming_views_filtered = args[0]
            plt.plot(df_timeseries_Gaming_views_filtered['Day'], df_timeseries_Gaming_views_filtered.iloc[:, i+1], label=f"smoothed {view_label[i]}", linestyle='-', color='red', markersize=4)
        plt.yscale('log')  
        plt.xlabel("Date")
        plt.ylabel("Views (log)")
        plt.title(f"{view_label[i]} of Gaming Category in Time Domain")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        
    # df_timeseries_plot_time_domain(df_timeseries_Gaming_views)

    # Cut the last season's data and plot
    # df_timeseries_plot_time_domain(df_timeseries_Gaming_views.iloc[4100:])

    # Drop the last 2 points
    df_timeseries_Gaming_views = df_timeseries_Gaming_views.iloc[:4232]

    def signal_FIR(df_timeseries_Gaming_views, column):
        signal_no_filtered = df_timeseries_Gaming_views.loc[:, column]
        filter_order = 51  # total length of FIR
        cutoff_frequency = 0.05  # between 0 and 1
        fir_filter = signal.firwin(filter_order, cutoff_frequency)
        signal_filtered = signal.lfilter(fir_filter, 1.0, signal_no_filtered) # lfilter has cut the tails
        return signal_filtered

    df_timeseries_Gaming_views_filtered = pd.DataFrame({'Day':df_timeseries_Gaming_views['Day'], 
                                                        'view_mean':signal_FIR(df_timeseries_Gaming_views, 'view_mean'),
                                                        'view_sum':signal_FIR(df_timeseries_Gaming_views, 'view_sum')})

    df_timeseries_plot_time_domain(df_timeseries_Gaming_views, df_timeseries_Gaming_views_filtered)

    return df_timeseries_Gaming_views, df_timeseries_Gaming_views_filtered


# FFT frequency visualization
def timeseries_freq_visualization(df_timeseries_Gaming_views):
    def signal_fft(signal, sampling_rate, *args):
        if len(args) > 0:
            DC_flag = args[0]
            # If true, minus mean to reduce 0-frequency (DC) components
            if DC_flag == True:
                signal = signal - np.mean(signal)
        fft_result = np.fft.fft(signal)
        n = len(signal)
        # Obtain frequency and magnitude
        frequencies = np.fft.fftfreq(n, d=sampling_rate)
        magnitude = np.abs(fft_result)
        # Because the signal is real-valued, its spectrum is symmetrical.
        # Choose positive side 
        positive_frequencies = frequencies[:n // 2]
        positive_magnitude = magnitude[:n // 2]
        return fft_result, positive_magnitude, positive_frequencies

    # Plot frequency spectrum
    def plot_fft(positive_magnitude, positive_frequencies, *args):
        plt.figure(figsize=(12, 6))
        if len(args) > 0:
            min_freq_percentage = args[0]
            max_freq_percentage = args[1]
            min_cut_point = np.floor(min_freq_percentage*len(positive_magnitude)).astype(int)
            max_cut_point = np.floor(max_freq_percentage*len(positive_magnitude)).astype(int)
            plt.plot(positive_frequencies[min_cut_point:max_cut_point], positive_magnitude[min_cut_point:max_cut_point])
            plt.title('FFT of Smoothed Time Signal')
        else:
            plt.plot(positive_frequencies, positive_magnitude)
            plt.title('Mean View of Gaming Category in Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid()
        # plt.show()

    # Take view_mean for example
    # Note that the signal has not been transformed by fft.
    # fft, magn, freq = signal_fft(df_timeseries_Gaming_views['view_mean'], 1)
    # ax = plot_fft(magn, freq)

    # Substract DC component
    fft, magn, freq = signal_fft(df_timeseries_Gaming_views['view_mean'], 1, True)
    plot_fft(magn, freq)

def timeseries_stft(df_timeseries_Gaming_views):
    def signal_stft(signal):
        signal = signal - np.mean(signal)
        # fs is sampling frequency, and nperseg is the width of the Window
        f, t, S = stft(signal, fs=1.0, nperseg=256)
        magnitude_spectrum = np.abs(S)
        # Use log scale to show more clear spectrum
        log_magnitude_spectrum = np.log1p(magnitude_spectrum)
        return f, t, magnitude_spectrum, log_magnitude_spectrum

    def plot_spectrum(f, t, magnitude_spectrum):
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, magnitude_spectrum, shading='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title('Magnitude Spectrum (STFT) of View Mean')
        plt.xlabel('Time (day)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    f, t, magnitude_spectrum, log_magnitude_spectrum = signal_stft(df_timeseries_Gaming_views['view_mean'])
    plot_spectrum(f, t[:len(t)-1], log_magnitude_spectrum[:,:len(t)-1])

def timeseries_cut_freq_visualization(df_timeseries_Gaming_views):
    df_timeseries_Gaming_views_cut = df_timeseries_Gaming_views[900:len(df_timeseries_Gaming_views)].copy()
    def signal_fft(signal, sampling_rate, *args):
        if len(args) > 0:
            DC_flag = args[0]
            # If true, minus mean to reduce 0-frequency (DC) components
            if DC_flag == True:
                signal = signal - np.mean(signal)
        fft_result = np.fft.fft(signal)
        n = len(signal)
        # Obtain frequency and magnitude
        frequencies = np.fft.fftfreq(n, d=sampling_rate)
        magnitude = np.abs(fft_result)
        # Because the signal is real-valued, its spectrum is symmetrical.
        # Choose positive side 
        positive_frequencies = frequencies[:n // 2]
        positive_magnitude = magnitude[:n // 2]
        return fft_result, positive_magnitude, positive_frequencies

    # Plot frequency spectrum
    def plot_fft(positive_magnitude, positive_frequencies, *args):
        plt.figure(figsize=(12, 6))
        if len(args) > 0:
            min_freq_percentage = args[0]
            max_freq_percentage = args[1]
            min_cut_point = np.floor(min_freq_percentage*len(positive_magnitude)).astype(int)
            max_cut_point = np.floor(max_freq_percentage*len(positive_magnitude)).astype(int)
            plt.plot(positive_frequencies[min_cut_point:max_cut_point], positive_magnitude[min_cut_point:max_cut_point])
            plt.title('FFT of Smoothed Time Signal')
        else:
            plt.plot(positive_frequencies, positive_magnitude)
            plt.title('Mean View of Gaming Category in Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid()

    fft, magn, freq = signal_fft(df_timeseries_Gaming_views_cut['view_mean'], 1, True)
    plot_fft(magn, freq)

    # Detact peak
    peak = [x > 1e7 for x in magn]
    peak_index = np.where(peak) 
    print(f"Magnitude peak is at: {freq[peak_index]} Hz")