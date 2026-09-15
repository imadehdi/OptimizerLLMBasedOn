import pandas as pd
import numpy as np
import logging

class UniverseFunnel:
    """
    Moteur de filtrage déterministe pour restreindre l'univers d'investissement.
    Applique séquentiellement : Exclusions, Inclusions, Seuils, Thématiques, et Overrides.
    """
    
    def __init__(self, df_data: pd.DataFrame, config: dict):
        # On travaille sur une copie pour ne pas altérer la donnée d'origine
        self.df = df_data.copy()
        self.config = config.get("universe_filter", config) # Tolérance sur la racine JSON
        self.id_col = self.config.get("identifier_col", "Ticker")
        
        # Initialisation du rapport d'audit
        self.audit_report = {
            "initial_universe": len(self.df),
            "steps": [],
            "final_universe": 0
        }
        
    def _validate_columns(self, columns: list):
        """Vérifie que les colonnes demandées existent bien dans le DataFrame."""
        missing = [col for col in columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Colonnes introuvables dans les données : {missing}")

    def apply(self) -> tuple[pd.DataFrame, dict]:
        """
        Exécute le pipeline de filtrage complet.
        Retourne le DataFrame filtré et le rapport d'audit.
        """
        # Le masque global démarre avec tout l'univers à True
        global_mask = pd.Series(True, index=self.df.index)
        current_count = len(self.df)
        
        # ---------------------------------------------------------
        # 1. FORCE EXCLUDE (Blacklist absolue)
        # ---------------------------------------------------------
        force_exclude = self.config.get("force_exclude", [])
        if force_exclude:
            self._validate_columns([self.id_col])
            mask = ~self.df[self.id_col].isin(force_exclude)
            global_mask = global_mask & mask
            self._log_audit("force_exclude", current_count, global_mask)
            current_count = global_mask.sum()

        # ---------------------------------------------------------
        # 2. EXCLUSIONS (Le Veto)
        # ---------------------------------------------------------
        exclusions = self.config.get("exclusions", {})
        if exclusions:
            self._validate_columns(exclusions.keys())
            for col, values in exclusions.items():
                mask = ~self.df[col].isin(values)
                global_mask = global_mask & mask
            self._log_audit("exclusions", current_count, global_mask)
            current_count = global_mask.sum()

        # ---------------------------------------------------------
        # 3. INCLUSIONS (Le Filtre Global)
        # ---------------------------------------------------------
        inclusions = self.config.get("inclusions", {})
        if inclusions:
            self._validate_columns(inclusions.keys())
            for col, values in inclusions.items():
                mask = self.df[col].isin(values)
                global_mask = global_mask & mask
            self._log_audit("inclusions", current_count, global_mask)
            current_count = global_mask.sum()

        # ---------------------------------------------------------
        # 4. THRESHOLDS (Les Seuils Quantitatifs)
        # ---------------------------------------------------------
        thresholds = self.config.get("thresholds", {})
        if thresholds:
            self._validate_columns(thresholds.keys())
            for col, rules in thresholds.items():
                col_data = pd.to_numeric(self.df[col], errors='coerce')
                
                min_val = rules.get("min")
                max_val = rules.get("max")
                keep_na = rules.get("keep_na", False)
                
                # Initialise un sous-masque à True
                thresh_mask = pd.Series(True, index=self.df.index)
                
                if min_val is not None:
                    thresh_mask = thresh_mask & (col_data >= float(min_val))
                if max_val is not None:
                    thresh_mask = thresh_mask & (col_data <= float(max_val))
                    
                # Gestion des données manquantes
                if keep_na:
                    thresh_mask = thresh_mask | col_data.isna()
                else:
                    # Si keep_na est False, on s'assure d'exclure les NaN explicitement
                    thresh_mask = thresh_mask & col_data.notna()
                    
                global_mask = global_mask & thresh_mask
                
            self._log_audit("thresholds", current_count, global_mask)
            current_count = global_mask.sum()

        # ---------------------------------------------------------
        # 5. THEMATICS (Les Profils Alternatifs / Logique OR)
        # ---------------------------------------------------------
        thematics = self.config.get("thematics", [])
        if thematics:
            # Masque thématique global initialisé à False (car c'est une liste de OR)
            theme_global_mask = pd.Series(False, index=self.df.index)
            
            for theme_dict in thematics:
                self._validate_columns(theme_dict.keys())
                # Masque de CE thème initialisé à True (car c'est un dictionnaire de AND)
                current_theme_mask = pd.Series(True, index=self.df.index)
                for col, values in theme_dict.items():
                    current_theme_mask = current_theme_mask & self.df[col].isin(values)
                
                # On ajoute ce thème aux thèmes acceptés (OR)
                theme_global_mask = theme_global_mask | current_theme_mask
                
            global_mask = global_mask & theme_global_mask
            self._log_audit("thematics", current_count, global_mask)
            current_count = global_mask.sum()

        # ---------------------------------------------------------
        # 6. FORCE INCLUDE (La Whitelist - Rédemption Finale)
        # ---------------------------------------------------------
        # Le Force Include agit en dernier : il utilise un OR (|) pour repêcher 
        # une valeur qui aurait été éliminée par les étapes précédentes.
        force_include = self.config.get("force_include", [])
        if force_include:
            self._validate_columns([self.id_col])
            mask_revive = self.df[self.id_col].isin(force_include)
            global_mask = global_mask | mask_revive
            self._log_audit("force_include", current_count, global_mask)
            
        # Extraction du DataFrame final
        df_filtered = self.df[global_mask].reset_index(drop=True)
        self.audit_report["final_universe"] = len(df_filtered)
        
        return df_filtered, self.audit_report

    def _log_audit(self, step_name: str, previous_count: int, current_mask: pd.Series):
        """Enregistre l'impact de chaque étape pour le reporting."""
        new_count = current_mask.sum()
        dropped = previous_count - new_count
        
        # Particularité pour le "force_include" qui RAJOUTE des valeurs
        if dropped < 0 and step_name == "force_include":
            impact = f"+ {abs(dropped)} revived"
        else:
            impact = f"- {dropped} dropped"
            
        self.audit_report["steps"].append({
            "step": step_name,
            "impact": impact,
            "remaining": int(new_count)
        })
