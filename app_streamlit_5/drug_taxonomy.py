"""
drug_taxonomy.py
=================
Double classification des substances (Pharmacologique et Chimique).

Ce module sépare rigoureusement les deux logiques :
1. PHARMACOLOGIQUE : L'effet principal de la substance sur le système nerveux 
   (Hallucinogène, Stimulant, Empathogène, Dissociatif, Délirant, Dépresseur, etc.)
2. CHIMIQUE : La structure moléculaire de la substance 
   (Tryptamine, Phénéthylamine, Arylcyclohexylamine, Benzodiazépine, etc.)

Module partagé entre app.py et precompute_drug_vectors.py.
"""

# ==============================================================================
# LOGIQUE 1 : CLASSIFICATION PHARMACOLOGIQUE (Effets sur le SNC)
# ==============================================================================

PHARMACOLOGICAL_FAMILY_MAP = {
    **{d: "Hallucinogènes" for d in [
        # Lysergamides
        "1B-LSD", "1cP-LSD", "1F-LSD", "1P-LSD", "AL-LAD", "ALD-52", "ETH-LAD", "LSA", "LSD",
        # Tryptamines
        "4-AcO-DET", "4-AcO-DMT", "4-AcO-DPT", "4-AcO-MET", "4-AcO-MiPT", "4-AcO-MPT",
        "4-HO-DET", "4-HO-DiPT", "4-HO-DPT", "4-HO-EPT", "4-HO-MALT", "4-HO-MET", "4-HO-MiPT",
        "4-MeO-MiPT", "4-PrO-DMT", "5-MeO-DALT", "5-MeO-MALT", "5-MeO-MET", "5-MeO-MiPT",
        "AMT", "DALT", "DiPT", "DMT", "DPT", "MiPT", "Mushrooms", "Psilocybin Mushrooms",
        "Ayahuasca", "Syrian Rue",
        # Phénéthylamines psychédéliques
        "2C-B", "2C-B-FLY", "2C-C", "2C-D", "2C-E", "2C-I", "2C-iP", "2C-P", "2C-T-7",
        "25C-NBOMe", "25i-NBOH", "25I-NBOMe", "DOB", "DOC", "DOM", 
        "Mescaline / Cactuses", "Allylescaline", "Escaline", "Methallylescaline", 
        "3C-E", "BOD", "βk-2C-B", "Phenethylamine",
    ]},
    **{d: "Empathogènes" for d in [
        # Sérotoninergiques principaux
        "MDMA", "MDA", "5-APDB", "6-APB",
    ]},
    **{d: "Stimulants" for d in [
        # Dopaminergiques / Noradrénergiques principaux
        "Amphetamine", "Methylphenidate", "Lisdexamfetamine", "Armodafinil",
        "Bupropion", "2-FMA", "3-FEA", "3-FPM", "A-PHP", "NM-2-AI", "Isopropylphenidate",
        "MEAI", "Difluoromethyl-ALEPH",
    ]},
    **{d: "Dissociatifs" for d in [
        # Antagonistes NMDA + Kappa-opioïdes (Salvia)
        "Ketamine", "MXE", "2F-DCK", "DCK", "3-HO-PCE", "3-HO-PCP", "3-Me-PCP", "3-MeO-PCE",
        "3-MeO-PCP", "3F-PCP", "Diphenidine", "Ephenidine", "Methoxphenidine", "O-PCE",
        "DXM", "Nitrous Oxide", "MXPr", "MXM", "Memantine", "Salvia divinorum",
    ]},
    **{d: "Délirants" for d in [
        # Anticholinergiques à haute dose
        "DPH", "Benzydamine",
    ]},
    **{d: "Dépresseurs" for d in [
        # GABAergiques, sédatifs, hypnotiques
        "Alprazolam", "Pregabalin", "Zaleplon", "Zopiclone", "Pyrazolam", "F-Phenibut",
        "Gaboxadol", "Mirtazapine",
    ]},
    **{d: "Opioïdes" for d in [
        "U-47700",
    ]},
    **{d: "Cannabinoïdes" for d in [
        "Cannabis", "THC",
    ]},
}

PHARM_COLORS = {
    "Hallucinogènes": "#A855F7",  # Violet
    "Empathogènes": "#F472B6",    # Rose bonbon
    "Stimulants": "#FB7185",      # Rose / Rouge clair
    "Dissociatifs": "#34D399",    # Vert
    "Délirants": "#B91C1C",       # Rouge sombre (Avertissement)
    "Dépresseurs": "#64748B",     # Gris bleu
    "Opioïdes": "#EF4444",        # Rouge vif
    "Cannabinoïdes": "#65A30D",   # Vert olive
    "Autres / divers": "#9089A8", # Gris violet
}


# ==============================================================================
# LOGIQUE 2 : CLASSIFICATION CHIMIQUE (Structure moléculaire)
# ==============================================================================

CHEMICAL_FAMILY_MAP = {
    **{d: "Lysergamides" for d in [
        "1B-LSD", "1cP-LSD", "1F-LSD", "1P-LSD", "AL-LAD", "ALD-52", "ETH-LAD", "LSA", "LSD",
    ]},
    **{d: "Tryptamines" for d in [
        "4-AcO-DET", "4-AcO-DMT", "4-AcO-DPT", "4-AcO-MET", "4-AcO-MiPT", "4-AcO-MPT",
        "4-HO-DET", "4-HO-DiPT", "4-HO-DPT", "4-HO-EPT", "4-HO-MALT", "4-HO-MET", "4-HO-MiPT",
        "4-MeO-MiPT", "4-PrO-DMT", "5-MeO-DALT", "5-MeO-MALT", "5-MeO-MET", "5-MeO-MiPT",
        "AMT", "DALT", "DiPT", "DMT", "DPT", "MiPT", "Mushrooms", "Psilocybin Mushrooms",
        "Ayahuasca", "Syrian Rue", # Syrian Rue est riche en β-carbolines (proches des tryptamines)
    ]},
    **{d: "Phénéthylamines" for d in [
        # Psychédéliques
        "2C-B", "2C-B-FLY", "2C-C", "2C-D", "2C-E", "2C-I", "2C-iP", "2C-P", "2C-T-7",
        "25C-NBOMe", "25i-NBOH", "25I-NBOMe", "DOB", "DOC", "DOM", "Mescaline / Cactuses",
        "Allylescaline", "Escaline", "Methallylescaline", "3C-E", "BOD", "βk-2C-B", "Phenethylamine",
        # Empathogènes & Stimulants (chimiquement dérivés de la phénéthylamine)
        "MDMA", "MDA", "5-APDB", "6-APB",
        "Amphetamine", "Methylphenidate", "Lisdexamfetamine", "Armodafinil",
        "Bupropion", "2-FMA", "3-FEA", "3-FPM", "A-PHP", "NM-2-AI", "Isopropylphenidate",
        "MEAI", "Difluoromethyl-ALEPH",
    ]},
    **{d: "Arylcyclohexylamines" for d in [
        # Famille de la Kétamine / PCP
        "Ketamine", "MXE", "2F-DCK", "DCK", "3-HO-PCE", "3-HO-PCP", "3-Me-PCP", "3-MeO-PCE",
        "3-MeO-PCP", "3F-PCP", "O-PCE", "MXPr", "MXM",
    ]},
    **{d: "Diaryléthylamines" for d in [
        # Proches des arylcyclohexylamines (Diphenidine, etc.)
        "Diphenidine", "Ephenidine", "Methoxphenidine",
    ]},
    **{d: "Morphinanes" for d in [
        # Dérivés de la morphine (bien que le DXM soit dissociatif et pas opioïde)
        "DXM",
    ]},
    **{d: "Benzodiazépines" for d in [
        "Alprazolam", "Pyrazolam",
    ]},
    **{d: "Terpènes" for d in [
        # La Salvinorine A est un diterpène, pas un alcaloïde
        "Salvia divinorum",
    ]},
    **{d: "Anticholinergiques" for d in [
        # DPH et apparentés
        "DPH", "Benzydamine", "Mirtazapine",
    ]},
    **{d: "Opioïdes" for d in [
        "U-47700",
    ]},
    **{d: "Cannabinoïdes" for d in [
        "Cannabis", "THC",
    ]},
    **{d: "Autres / divers" for d in [
        # Gaz, acides aminés, Adamantanes, etc.
        "Nitrous Oxide", "Pregabalin", "F-Phenibut", "Gaboxadol", "Zaleplon", "Zopiclone", "Memantine",
    ]},
}

CHEM_COLORS = {
    "Lysergamides": "#A855F7",           # Violet
    "Tryptamines": "#22D3EE",            # Cyan
    "Phénéthylamines": "#F59E0B",        # Orange / Ambre
    "Arylcyclohexylamines": "#10B981",   # Émeraude
    "Diaryléthylamines": "#059669",      # Vert plus foncé
    "Morphinanes": "#8B5CF6",            # Violet bleuté
    "Benzodiazépines": "#60A5FA",        # Bleu ciel
    "Terpènes": "#4ADE80",              # Vert clair
    "Anticholinergiques": "#DC2626",     # Rouge alarme
    "Opioïdes": "#EF4444",              # Rouge vif
    "Cannabinoïdes": "#65A30D",          # Vert olive
    "Autres / divers": "#9089A8",        # Gris
}


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def pharm_family(drug: str) -> str:
    """Retourne la famille pharmacologique d'une substance."""
    return PHARMACOLOGICAL_FAMILY_MAP.get(drug, "Autres / divers")

def chem_family(drug: str) -> str:
    """Retourne la famille chimique d'une substance."""
    return CHEMICAL_FAMILY_MAP.get(drug, "Autres / divers")

def get_pharm_color(family: str) -> str:
    """Retourne la couleur associée à une famille pharmacologique."""
    return PHARM_COLORS.get(family, PHARM_COLORS["Autres / divers"])

def get_chem_color(family: str) -> str:
    """Retourne la couleur associée à une famille chimique."""
    return CHEM_COLORS.get(family, CHEM_COLORS["Autres / divers"])
