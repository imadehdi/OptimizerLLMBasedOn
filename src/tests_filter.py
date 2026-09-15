import pandas as pd
from src.universe_funnel import UniverseFunnel

# 1. Le JSON que l'interface graphique t'envoie
ui_config = {
    "identifier_col": "Ticker",
    "force_exclude": ["SAP.DE"],                # On déteste le management
    "force_include": ["DSY.PA"],                # Notre conviction forte (repêchage)
    "exclusions": {
        "Sector": ["Energy", "Tobacco"]         # Pas d'énergie, pas de tabac
    },
    "inclusions": {
        "Market_Cap": ["Large", "Mid"]          # Uniquement des grosses ou moyennes
    },
    "thresholds": {
        "ESG_Score": {"min": 60, "max": None, "keep_na": True}  # Bénéfice du doute pour les non-notés
    },
    "thematics": [
        {"Region": ["Europe"], "Sector": ["Tech"]},         # Thème 1
        {"Region": ["US"], "Sector": ["Healthcare"]}        # OU Thème 2
    ]
}

# 2. Ton DataFrame (simulé ici)
df_raw = pd.DataFrame({
    "Ticker": ["AAPL.O", "SAP.DE", "DSY.PA", "SANOFI.PA", "TOTAL.PA", "LVMH.PA", "PFE.N"],
    "Region": ["US", "Europe", "Europe", "Europe", "Europe", "Europe", "US"],
    "Sector": ["Tech", "Tech", "Tech", "Healthcare", "Energy", "Consumer", "Healthcare"],
    "Market_Cap": ["Large", "Large", "Mid", "Large", "Large", "Large", "Large"],
    "ESG_Score": [80, 90, 40, 75, 55, None, 65]
})

# 3. L'exécution
funnel = UniverseFunnel(df_data=df_raw, config=ui_config)
df_filtered, audit = funnel.apply()

# 4. Affichage du Reporting d'Audit
print("=== RAPPORT D'ENTONNOIR ===")
for step in audit["steps"]:
    print(f"{step['step'].upper():<15} : {step['impact']:<15} (Reste: {step['remaining']})")
print(f"Univers final : {audit['final_universe']} / {audit['initial_universe']}")

print("\n=== ACTIFS ÉLIGIBLES ===")
print(df_filtered[["Ticker", "Sector", "ESG_Score"]])
