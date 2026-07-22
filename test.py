import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from config import SPHConfig
from initialization import initialize_particles
from simulation import run_simulation


def main():
    # Define the dx values you want to test
    dx_values = np.linspace(0.07, 0.02, num=6)

    # List to store the rows of data
    execution_data = []

    print(f"{'dx value':<10} | {'Execution Time (seconds)'}")
    print("-" * 35)

    for dx in dx_values:
        # 1. Initialize config and update dx
        config = SPHConfig()
        config.dx = dx

        # 2. Record the start time
        start_time = time.time()

        # 3. Run the setup and simulation
        pos, vel, m, h = initialize_particles(config)
        frames = run_simulation(config, pos, vel, m, h)

        # 4. Record the end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time

        # 5. Log the results in memory
        execution_data.append([dx, duration])
        print(f"{dx:<10.4f} | {duration:.4f}")

    # 6. Save the data to a CSV file
    filename = "simulation_times_1.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["dx", "execution_time_seconds"])
        # Write the data rows
        writer.writerows(execution_data)

    print(f"\nData successfully saved to {filename}")

    return dx_values, [row[1] for row in execution_data]


if __name__ == "__main__":
    dx_vals, times = main()
