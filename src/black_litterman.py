import numpy as np
import pandas as pd

def is_pos_def(matrix: np.ndarray, threshold: float = 1e-8) -> bool:
    """
    Vérifie si une matrice est définie positive en testant ses valeurs propres.
    Inspiré des vérifications de stabilité de Riskfolio-Lib.
    """
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues > threshold)
    except np.linalg.LinAlgError:
        return False

def fix_cov_matrix(matrix: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Corrige une matrice de covariance pour forcer sa définition positive 
    via l'algorithme d'Eigenvalue Clipping (méthode 'clipped').
    """
    # Symétrisation forcée pour éviter les erreurs d'arrondi
    matrix = (matrix + matrix.T) / 2.0
    
    # Décomposition spectrale
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Clipping des valeurs propres sous le seuil
    eigenvalues_clipped = np.maximum(eigenvalues, threshold)
    
    # Reconstitution de la matrice
    fixed_matrix = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
    
    # Symétrisation finale
    return (fixed_matrix + fixed_matrix.T) / 2.0

def compute_black_litterman(
    mu_hist: np.ndarray, 
    cov_hist: np.ndarray, 
    w_mkt: np.ndarray, 
    P: np.ndarray, 
    Q: np.ndarray, 
    tau: float = 0.05, 
    rf: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les rendements et la matrice de covariance a posteriori de Black-Litterman.
    """
    mu_hist = np.asarray(mu_hist).flatten()
    w_mkt = np.asarray(w_mkt).flatten()
    
    # 1. Calcul du coefficient d'aversion au risque implicite du marché (delta)
    market_variance = w_mkt.T @ cov_hist @ w_mkt
    delta = (mu_hist.T @ w_mkt - rf) / market_variance
    
    # 2. Rendements d'équilibre implicites (Pi)
    Pi = delta * (cov_hist @ w_mkt)
    
    # 3. Modélisation de l'incertitude des vues (Omega) via He-Litterman
    # Diagonale de P * (tau * Sigma) * P.T
    tau_cov = tau * cov_hist
    Omega = np.diag(np.diag(P @ tau_cov @ P.T))
    
    # 4. Inversions matricielles sécurisées (Pseudo-inverse pour la stabilité)
    tau_cov_inv = np.linalg.pinv(tau_cov)
    Omega_inv = np.linalg.pinv(Omega)
    
    # 5. Calcul du terme central M^{-1}
    M_inv = np.linalg.pinv(tau_cov_inv + P.T @ Omega_inv @ P)
    
    # 6. Calcul des nouveaux rendements (mu_bl) et de la covariance (cov_bl)
    mu_bl = M_inv @ (tau_cov_inv @ Pi + P.T @ Omega_inv @ Q)
    cov_bl = cov_hist + M_inv
    
    # 7. Sécurité algébrique : Validation et correction de cov_bl
    if not is_pos_def(cov_bl, threshold=1e-8):
        cov_bl = fix_cov_matrix(cov_bl, threshold=1e-5)
        
    return mu_bl, cov_bl