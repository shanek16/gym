import numpy as np

class UAVCircleCounter:
    def __init__(self):
        self.tolerance_d = 3
        self.target_d = 10
        self.initial_beta = None
        self.total_beta_change = 0
        self.last_beta = None
        self.circles_completed = 0

    def update(self, beta, r):
        if self.initial_beta is None:
            # Initialize the initial beta value
            self.initial_beta = beta
            self.last_beta = beta

        # Calculate the change in beta
        beta_change = self._calculate_beta_change(self.last_beta, beta)
        self.total_beta_change += beta_change

        # Update last_beta
        self.last_beta = beta

        # Check if the UAV's r is within the tolerance range from the target distance
        if abs(self.target_d - r) > self.tolerance_d:
            # If it's out of range, reset the total_beta_change
            self.total_beta_change = 0
        elif abs(self.total_beta_change) >= 2 * np.pi:
            # If it's within range and a full circle (2*pi radians) is completed
            self.circles_completed += 1
            # Reset the total_beta_change for the next circle
            self.total_beta_change = 0

        return self.circles_completed

    def _calculate_beta_change(self, last_beta, current_beta):
        # Calculate the shortest distance between the two angles
        delta_beta = current_beta - last_beta
        delta_beta = (delta_beta + np.pi) % (2 * np.pi) - np.pi
        return delta_beta

# Example usage:
# counter = UAVCircleCounter()
# while some_condition:
#     beta, r = get_beta_and_r()  # Replace with actual method to get current values
#     circles = counter.update(beta, r)
#     print(f"Circles completed: {circles}")
