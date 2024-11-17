import numpy as np
import matplotlib.pyplot as plt

def govardovskii_template(wavelengths, lambda_max, A1_proportion=100):
    """
    Implements Govardovskii's (2000) visual pigment template with A1/A2 chromophore mixing.
    
    Parameters:
    wavelengths : array-like
        Wavelengths at which to evaluate the template (nm)
    lambda_max : float
        Peak wavelength of the visual pigment (nm)
    A1_proportion : float
        Percentage of A1 chromophore (0-100), default 100 (pure A1)
        
    Returns:
    tuple: (wavelengths, sensitivities)
        Arrays of wavelengths and corresponding normalized sensitivity values
    """
    wavelengths = np.asarray(wavelengths)
    x = lambda_max / wavelengths
    
    # A1 template parameters
    a1 = 0.8795 + 0.0459 * np.exp(-((lambda_max - 300)**2) / 11940)
    lambda_beta_A1 = 189 + 0.315 * lambda_max
    beta_bandwidth_A1 = -40.5 + 0.195 * lambda_max
    
    # A1 alpha and beta bands
    alpha_A1 = 1 / (np.exp(69.7 * (a1 - x)) + 
                    np.exp(28 * (0.922 - x)) + 
                    np.exp(-14.9 * (1.104 - x)) + 
                    0.674)
    beta_A1 = 0.26 * np.exp(-((wavelengths - lambda_beta_A1) / 
                              beta_bandwidth_A1)**2)
    sensitivity_A1 = alpha_A1 + beta_A1
    
    # A2 template parameters
    a2 = 0.875 + 0.0268 * np.exp((lambda_max - 665) / 40.7)
    A2_peak = 62.7 + 1.834 * np.exp((lambda_max - 625) / 54.2)
    lambda_beta_A2 = 216.7 + 0.287 * lambda_max
    beta_bandwidth_A2 = 317 - 1.149 * lambda_max + 0.00124 * (lambda_max**2)
    
    # A2 alpha and beta bands
    alpha_A2 = 1 / (np.exp(A2_peak * (a2 - x)) + 
                    np.exp(20.85 * (0.9101 - x)) + 
                    np.exp(-10.37 * (1.1123 - x)) + 
                    0.5343)
    beta_A2 = 0.26 * np.exp(-((wavelengths - lambda_beta_A2) / 
                              beta_bandwidth_A2)**2)
    sensitivity_A2 = alpha_A2 + beta_A2
    
    # Combine A1 and A2 templates according to proportion
    A1_weight = A1_proportion / 100
    A2_weight = 1 - A1_weight
    total_sensitivity = (A1_weight * sensitivity_A1 + 
                        A2_weight * sensitivity_A2)
    
    # Normalize
    normalized_sensitivity = total_sensitivity / np.max(total_sensitivity)
    
    return wavelengths, normalized_sensitivity

def plot_visual_pigments(wavelength_range, lambda_maxs, A1_proportions=None, title=None):
    """
    Plot multiple visual pigment sensitivity curves with varying A1/A2 proportions.
    
    Parameters:
    wavelength_range : tuple
        (min_wavelength, max_wavelength) in nm
    lambda_maxs : list
        List of peak wavelengths to plot
    A1_proportions : list, optional
        List of A1 proportions (0-100) for each lambda_max
    title : str, optional
        Plot title
    """
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
    
    if A1_proportions is None:
        A1_proportions = [100] * len(lambda_maxs)
    
    plt.figure(figsize=(10, 6))
    
    for lambda_max, A1_prop in zip(lambda_maxs, A1_proportions):
        _, sensitivity = govardovskii_template(wavelengths, lambda_max, A1_prop)
        label = f'λmax={lambda_max}nm, A1={A1_prop}%'
        plt.plot(wavelengths, sensitivity, label=label)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Sensitivity')
    plt.title(title or 'Visual Pigment Templates')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.1)
    
    return plt

# Example usage
if __name__ == "__main__":
    # Define parameters
    wavelength_range = (300, 800)
    lambda_maxs = [400,400,400,525,525,525, 650, 650,650]  # Same peak wavelength
    A1_proportions = [100, 50, 0,100, 50, 0,100, 50, 0,]  # Different A1/A2 mixtures
    
    # Create and show plot
    plot_visual_pigments(
        wavelength_range, 
        lambda_maxs, 
        A1_proportions,
        "Govardovskii Template: A1/A2 Chromophore Mixing"
    )
    plt.show()