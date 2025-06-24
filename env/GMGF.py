# uncertain_model.py
# Implements the G-MGF (GDOP-Assisted Manifold Gradient Filtering) algorithm
# based on the paper: "Cooperative Positioning Algorithm Based on Manifold Gradient Filtering in UAV-WSN"
# by Song, Zhang, Yu, and Tang (IEEE SENSORS JOURNAL, VOL. 24, NO. 8, 15 APRIL 2024).

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from typing import Optional

# Matplotlib configuration for displaying Chinese characters in plots
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']
mpl.rcParams['axes.unicode_minus'] = False


class ManifoldFilterModel:
    """
    Implements a sensor position uncertainty model using the Manifold Gradient Filtering (MGF) algorithm
    as described in the reference paper.

    This model replaces a standard Extended Kalman Filter (EKF) with an approach based on
    information geometry. Each measurement updates the sensor's estimated position and covariance
    by calculating the natural gradient on a Riemannian manifold, leading to potentially faster
    and more stable convergence.

    Key concepts from the paper implemented here:
    - State: 2D position vector `x` of the sensor.
    - Covariance: 2x2 matrix `P` representing the uncertainty.
    - Measurement function `h(x)`: The distance from the UAV to the sensor's estimated position.
    - Fisher Information Matrix `G(x)`: Acts as the metric for the manifold (Eq. 36).
      It combines information from the prior estimate and the new measurement.
    - Natural Gradient `T(x)`: The "steepest descent" direction on the manifold (Eq. 37).
    - Update Rule: A gradient descent step using the natural gradient (Eq. 38).
    - Covariance Update: The new covariance is the inverse of the Fisher Information Matrix (Table II).
    """

    def __init__(self, num_sensors: int, confidence_level: float = 0.99):
        """
        Initializes the Manifold Gradient Filter model.

        Args:
            num_sensors: The number of sensors to track.
            confidence_level: The confidence level for calculating uncertainty ellipses.
        """
        self.num_sensors = num_sensors
        self.confidence_level = confidence_level

        # State vector (estimated positions) [x_k]
        self.estimated_positions = np.zeros((num_sensors, 2))
        # State covariance matrices [P_k]
        self.covariance_matrices = np.array([np.eye(2) * 100 ** 2 for _ in range(num_sensors)])
        # Radius of the uncertainty ellipse for quick checks
        self.uncertainty_radii = np.full(num_sensors, 100.0)

        # Process noise Q, representing model uncertainty for static sensors (e.g., slight drift)
        # This is added during the prediction step to prevent the filter from becoming overconfident.
        q_val = 1e-5
        self.process_noise_q = np.eye(2) * q_val

    def initialize_states(self, true_positions: np.ndarray, initial_radius: float, np_random: np.random.Generator):
        """Initializes sensor states with some random offset from their true positions."""
        self.estimated_positions = np.array([
            true_pos + np_random.uniform(-initial_radius / 2, initial_radius / 2, 2)
            for true_pos in true_positions
        ])
        self.covariance_matrices = np.array([np.eye(2) * initial_radius ** 2 for _ in range(self.num_sensors)])
        self.uncertainty_radii = np.full(self.num_sensors, initial_radius)

    def g_mgf_update(self, sensor_id: int, drone_position: np.ndarray,
                     measured_distance: float, measurement_variance: float, step_size_eta: float = 1.0):
        """
        Performs a single G-MGF (GDOP-Assisted Manifold Gradient Filtering) update step.
        This is the core implementation of the algorithm from the paper.

        Args:
            sensor_id: The ID of the sensor to update.
            drone_position: The 2D position of the UAV making the measurement.
            measured_distance: The measured range from the UAV to the sensor.
            measurement_variance: The variance (R) of the ranging measurement noise.
            step_size_eta: The iteration control parameter η (from Eq. 38 and Table II).
                         In the G-MGF algorithm, this is dynamically set, e.g., η = GDOP_k / GDOP_sup.
                         A value of 1.0 corresponds to a standard MGF step.
        """
        # --- 1. State Prediction (as per Table II, Step 1) ---
        # For a static sensor model, the predicted state is the previous state.
        # The predicted covariance includes the addition of process noise.
        x_prior = self.estimated_positions[sensor_id]
        P_prior = self.covariance_matrices[sensor_id] + self.process_noise_q

        # Scalar measurement noise variance R
        R = measurement_variance
        if R < 1e-9: R = 1e-9  # Avoid division by zero

        # --- 2. Calculate Components for Natural Gradient (Table II, Step 3) ---

        # a. Measurement function h(x) and its Jacobian H = ∇x h(x)
        diff_vec = x_prior - drone_position
        estimated_dist = np.linalg.norm(diff_vec)
        if estimated_dist < 1e-6:
            return  # Avoid numerical instability if drone is on top of estimated position

        # H is the Jacobian of the distance function w.r.t sensor position
        H = (diff_vec / estimated_dist).reshape(1, 2)  # Shape (1, 2)

        # b. Measurement residual (error) e_z = h(x) - z
        # Paper uses h(x) - z; this sign convention is important for the gradient direction.
        e_z = estimated_dist - measured_distance

        # c. State residual e_x = x - μ
        # In this single-step filter formulation, the gradient is evaluated at the prior estimate (x=μ),
        # so the state residual e_x = x_prior - x_prior = 0. The update is driven by measurement error.

        # d. Euclidean Gradient ∇L (from Eq. 35)
        # ∇L = H^T * R^-1 * e_z + P_prior^-1 * e_x. With e_x=0, it simplifies.
        grad_L = H.T * (1 / R) * e_z  # Shape (2, 1)

        # e. Fisher Information Matrix (FIM) G(x) (from Eq. 36)
        # G(x) = H^T * R^-1 * H + P_prior^-1
        # This matrix represents the total precision (inverse of covariance) from both the
        # measurement and the prior belief. It is the metric of the Riemannian manifold.
        try:
            P_prior_inv = np.linalg.inv(P_prior)
        except np.linalg.LinAlgError:
            P_prior_inv = np.linalg.pinv(P_prior)  # Use pseudo-inverse for stability

        G = (H.T * (1 / R)) @ H + P_prior_inv  # Shape (2, 2)

        # --- 3. Perform the MGF Update ---

        # a. Calculate the Natural Gradient T(x) = G(x)^-1 * ∇L (from Eq. 37)
        # The inverse of the FIM, G_inv, reshapes the Euclidean gradient into the
        # steepest descent direction on the curved information manifold.
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            # If G is singular, the update is not well-defined. Skip this step.
            return

        natural_grad = G_inv @ grad_L  # Shape (2, 1)

        # b. Update State Estimate using the Manifold Gradient (from Eq. 38)
        # x_post = x_prior - η * T(x)
        x_post = x_prior - step_size_eta * natural_grad.flatten()

        # c. Update Covariance Matrix (from Table II, Σ_k+1 = G_k^-1)
        # The posterior covariance is simply the inverse of the Fisher Information Matrix.
        # This elegantly captures the new uncertainty after fusing the information.
        P_post = G_inv

        # --- 4. Save Updated State and Covariance ---
        self.estimated_positions[sensor_id] = x_post
        self.covariance_matrices[sensor_id] = P_post
        self.uncertainty_radii[sensor_id] = self.calculate_confidence_radius(P_post)

    def calculate_confidence_radius(self, covariance_matrix: np.ndarray) -> float:
        """Calculates the radius of the confidence ellipse's major axis."""
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        chi2_val = chi2.ppf(self.confidence_level, df=2)  # 2 degrees of freedom for 2D position
        return np.sqrt(chi2_val * max_eigenvalue)

    def get_confidence_ellipse(self, sensor_id: int, n_points: int = 100) -> np.ndarray:
        """Generates points on the boundary of the confidence ellipse for plotting."""
        center = self.estimated_positions[sensor_id]
        cov = self.covariance_matrices[sensor_id]

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        chi2_val = chi2.ppf(self.confidence_level, df=2)

        # Get the angle of the largest eigenvector
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        if angle < 0:
            angle += 2 * np.pi

        # Get the lengths of the semi-axes
        a, b = np.sqrt(chi2_val * eigenvalues)

        t = np.linspace(0, 2 * np.pi, n_points)
        ellipse_x = a * np.cos(t)
        ellipse_y = b * np.sin(t)

        # Rotate the ellipse
        R_mat = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        ellipse_points = np.dot(R_mat, np.vstack([ellipse_x, ellipse_y]))

        # Translate to the center
        return ellipse_points.T + center

    def is_converged(self, threshold: float = 1.0) -> bool:
        """Checks if all sensor uncertainties are below a given threshold."""
        return np.all(self.uncertainty_radii < threshold)

    def visualize_uncertainty(self, sensor_id: int, true_position: Optional[np.ndarray] = None):
        """Visualizes the uncertainty of a single sensor."""
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        est_pos = self.estimated_positions[sensor_id]
        ax.scatter(est_pos[0], est_pos[1], c='blue', s=100, marker='x', label='估计位置 (MGF)')

        ellipse_points = self.get_confidence_ellipse(sensor_id)
        ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'b-', alpha=0.7, label='99% 置信椭圆')

        if true_position is not None:
            ax.scatter(true_position[0], true_position[1], c='red', s=100, marker='*', label='真实位置', zorder=5)

        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.set_title(f'传感器 {sensor_id} 位置不确定性 (MGF 算法)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis('equal')
        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # --- Simulation Setup ---
    num_sensors = 1
    true_sensor_pos = np.array([500.0, 500.0])

    # Instantiate the model
    model = ManifoldFilterModel(num_sensors=num_sensors)

    # Initialize the sensor's state with a large initial uncertainty
    initial_pos_estimate = true_sensor_pos + np.array([30, -30])
    model.estimated_positions[0] = initial_pos_estimate
    initial_covariance = np.eye(2) * 50 ** 2  # Large initial uncertainty (std dev = 50m)
    model.covariance_matrices[0] = initial_covariance
    model.uncertainty_radii[0] = model.calculate_confidence_radius(initial_covariance)

    print("--- 初始状态 ---")
    print(f"真实位置: {true_sensor_pos}")
    print(f"初始估计位置: {model.estimated_positions[0]}")
    print(f"初始不确定性半径: {model.uncertainty_radii[0]:.2f}m")

    # Visualize initial state
    model.visualize_uncertainty(0, true_sensor_pos)

    # --- Simulate UAV Measurements ---
    # A set of UAV positions providing good geometric diversity
    uav_positions = [
        np.array([500, 0]),
        np.array([1000, 500]),
        np.array([500, 1000]),
        np.array([0, 500]),
        np.array([800, 800])
    ]

    ranging_noise_std = 2.0  # Standard deviation of ranging error (e.g., 2 meters)
    measurement_variance = ranging_noise_std ** 2

    print("\n--- 开始 MGF 更新 ---")
    for i, uav_pos in enumerate(uav_positions):
        # Simulate measurement
        true_dist = np.linalg.norm(uav_pos - true_sensor_pos)
        measured_dist = true_dist + np.random.normal(0, ranging_noise_std)

        # In a full G-MGF implementation, η would be calculated based on GDOP.
        # For this example, we use η=1.0 for a standard MGF update.
        step_size_eta = 1.0

        # Perform the update
        model.g_mgf_update(0, uav_pos, measured_dist, measurement_variance, step_size_eta)

        print(f"\n--- 更新 #{i + 1} (无人机位置: {uav_pos}) ---")
        print(f"估计位置: {model.estimated_positions[0]}")
        print(f"不确定性半径: {model.uncertainty_radii[0]:.2f}m")

    print("\n--- 最终结果 ---")
    final_error = np.linalg.norm(model.estimated_positions[0] - true_sensor_pos)
    print(f"最终估计位置: {model.estimated_positions[0]}")
    print(f"最终不确定性半径: {model.uncertainty_radii[0]:.2f}m")
    print(f"最终定位误差: {final_error:.2f}m")

    # Visualize final state
    model.visualize_uncertainty(0, true_sensor_pos)