'''
    General functions and global parameters, that are used in different scripts
'''
import os

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import logging

from typing import Tuple
from Bio import SeqIO

# take logger from main
logger = logging.getLogger(__name__)

### STATIC VALUES ###
# load config and assign values to global variables
DATAPATH = os.path.join(os.path.dirname(__file__), "srr_data")
RESULTSPATH = os.path.join(os.path.dirname(__file__), "results")

# segments, nuclotides, and strains
CMAP = "Accent"
CUTOFF = 15
N_SAMPLES = 35000
RESULTSPATH = os.path.join(RESULTSPATH, f"cutoff_{CUTOFF}")
SEGMENTS = list(["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"])
NUCLEOTIDES = dict({"A": "Adenine", "C": "Cytosin", "G": "Guanine", "U": "Uracil"})

DATASET_STRAIN_DICT = dict({
    # H1N1
    "Alnaji2021": "PR8",
    "Pelz2021": "PR8",
    "Wang2023": "PR8",
    "Wang2020": "PR8",
    "Zhuravlev2020": "PR8",
    "Kupke2020": "PR8",
    "VdHoecke2015": "PR8",
    "Alnaji2019_Cal07": "Cal07",
    "Alnaji2019_NC": "NC",
    "Mendes2021": "WSN_Mendes_rev",
    "Boussier2020": "WSN",
    # H3N2
    "Alnaji2019_Perth": "Perth",
    "Berry2021_A": "Connecticut",
    # H5N1
    "Penn2022": "Turkey",
    # H7N9
    "Lui2019": "Anhui",
    # B
    "Alnaji2019_BLEE": "BLEE",
    "Berry2021_B": "Victoria",
    "Valesano2020_Vic": "Victoria",
    "Sheng2018": "Brisbane",
    "Berry2021_B_Yam": "Yamagata",
    "Southgate2019": "Yamagata",
    "Valesano2020_Yam": "Yamagata"
})

ACCNUMDICT = dict({
    "Wang2023": dict({
        "SRR16770171": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770172": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770173": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770174": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770175": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770181": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770182": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770183": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770184": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770185": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770186": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770191": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770192": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770193": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "1", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770197": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770198": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770201": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770200": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770199": dict({"IFNAR": "1", "IFNLR": "0", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770207": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770208": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770209": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770210": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770211": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770212": dict({"IFNAR": "0", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770219": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770218": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"}),
        "SRR16770217": dict({"IFNAR": "1", "IFNLR": "1", "Cells": "Mouse", "Replicate": "2", "Resolution": "bulk", "Context": "in vivo", "Compartment": "unknown"})
    }),
    "Wang2020": dict({
        "SRR7722028": dict({"Cells": "A549", "Time": "6", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722030": dict({"Cells": "A549", "Time": "12", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722032": dict({"Cells": "A549", "Time": "24", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722029": dict({"Cells": "A549", "Time": "6", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722031": dict({"Cells": "A549", "Time": "12", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722033": dict({"Cells": "A549", "Time": "24", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),

        "SRR7722036": dict({"Cells": "HBEpC", "Time": "6", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722038": dict({"Cells": "HBEpC", "Time": "12", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722040": dict({"Cells": "HBEpC", "Time": "24", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722037": dict({"Cells": "HBEpC", "Time": "6", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722039": dict({"Cells": "HBEpC", "Time": "12", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5}),
        "SRR7722041": dict({"Cells": "HBEpC", "Time": "24", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell", "MOI": 5})
    }),
    "Mendes2021": dict({
        "SRR15720520": dict({"Replicate": "e11", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"}),
        "SRR15720521": dict({"Replicate": "e12", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"}),
        "SRR15720522": dict({"Replicate": "e21", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"}),
        "SRR15720523": dict({"Replicate": "e12", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"}),
        "SRR15720524": dict({"Replicate": "d11", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"}),
        "SRR15720525": dict({"Replicate": "d12", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"}),
        "SRR15720526": dict({"Replicate": "d21", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"}),
        "SRR15720527": dict({"Replicate": "d22", "Cells": "A549", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "48hpi"})
    }),
    "Pelz2021": dict({
        "SRR15084925": dict({"Time": "seed"}),
        "SRR15084924": dict({"Time": "0.5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084913": dict({"Time": "1dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084908": dict({"Time": "1.4dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084907": dict({"Time": "3.5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084906": dict({"Time": "4dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084905": dict({"Time": "4.5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084904": dict({"Time": "5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084903": dict({"Time": "5.5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084902": dict({"Time": "8dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084923": dict({"Time": "9dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084922": dict({"Time": "9.4dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084921": dict({"Time": "12.4dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084919": dict({"Time": "13dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084918": dict({"Time": "13.5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084917": dict({"Time": "16dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084916": dict({"Time": "17dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084915": dict({"Time": "17.5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084914": dict({"Time": "18dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084912": dict({"Time": "19.5dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084911": dict({"Time": "20dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084910": dict({"Time": "20.4dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True}),
        "SRR15084909": dict({"Time": "21dpi", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Multi-timepoint": True})
    }),
    "Alnaji2019_Cal07": dict({
        "SRR8754522": dict({"Replicate": "1", "Passage": "6", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754523": dict({"Replicate": "2", "Passage": "6", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754531": dict({"Replicate": "1", "Passage": "6_t", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754532": dict({"Replicate": "1", "Passage": "3_t", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754533": dict({"Replicate": "1", "Passage": "1_t", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"})
    }),
    "Alnaji2019_NC": dict({
        "SRR8754513": dict({"Replicate": "2", "Passage": "1", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754514": dict({"Replicate": "1", "Passage": "1", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754527": dict({"Replicate": "1", "Passage": "6", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754538": dict({"Replicate": "2", "Passage": "6", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"})
    }),
    "Alnaji2019_Perth": dict({
        "SRR8754517": dict({"Replicate": "2", "Passage": "8", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754524": dict({"Replicate": "1", "Passage": "4", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754525": dict({"Replicate": "2", "Passage": "4", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754526": dict({"Replicate": "1", "Passage": "8", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"})
    }),
    "Alnaji2019_BLEE": dict({
        "SRR8754507": dict({"Replicate": "1", "Passage": "8", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754508": dict({"Replicate": "2", "Passage": "7", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754509": dict({"Replicate": "1", "Passage": "7", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"}),
        "SRR8754516": dict({"Replicate": "2", "Passage": "8", "Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "Time": "24hpi"})
    }),
    "Lui2019": dict({
        "SRR8949705": dict({"Cells": "Mouse", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "MOI": 0.5, "Time": "24hpi"}),
        "SRR8945328": dict({"Cells": "Mouse", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "MOI": 0.5, "Time": "24hpi"}),
    }),
    "Penn2022": dict({
        "ERR10231074": dict({"Time": "24hpi", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H1"}),
        "ERR10231075": dict({"Time": "48hpi", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H1"}),
        "ERR10231076": dict({"Time": "6hpi", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H1"}),
        "ERR10231077": dict({"Time": "96hpi", "Mode": "High", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H1"}),
        "ERR10231078": dict({"Time": "24hpi", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H2"}),
        "ERR10231079": dict({"Time": "48hpi", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H2"}),
        "ERR10231080": dict({"Time": "6hpi", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H2"}),
        "ERR10231081": dict({"Time": "96hpi", "Mode": "High", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "H2"}),
        "ERR10231089": dict({"Time": "96hpi", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L2"}),
        "ERR10231082": dict({"Time": "24hpi", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L1"}),
        "ERR10231085": dict({"Time": "96hpi", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L1"}),
        "ERR10231083": dict({"Time": "48hpi", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L1"}),
        "ERR10231084": dict({"Time": "6hpi", "Mode": "Low", "Lineage": "1", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L1"}),
        "ERR10231086": dict({"Time": "24hpi", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L2"}),
        "ERR10231087": dict({"Time": "48hpi", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L2"}),
        "ERR10231088": dict({"Time": "6hpi", "Mode": "Low", "Lineage": "2", "Cells": "Mouse", "Context": "in vivo", "Replicate": "L2"})
    }),
    "Alnaji2021": dict({
        "SRR14352106": dict({"Replicate": "C", "Cells": "MDCK-SIAT1", "Time": "24hpi", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352107": dict({"Replicate": "B", "Cells": "MDCK-SIAT1", "Time": "24hpi", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352108": dict({"Replicate": "A", "Cells": "MDCK-SIAT1", "Time": "24hpi", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352109": dict({"Replicate": "C", "Cells": "MDCK-SIAT1", "Time": "6hpi", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352110": dict({"Replicate": "B", "Cells": "MDCK-SIAT1", "Time": "6hpi", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352111": dict({"Replicate": "A", "Cells": "MDCK-SIAT1", "Time": "6hpi", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352112": dict({"Replicate": "C", "Cells": "MDCK-SIAT1", "Time": "3hpi", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352113": dict({"Replicate": "X", "Time": "0hpi", "Resolution": "bulk"}),
        "SRR14352116": dict({"Replicate": "B", "Cells": "MDCK-SIAT1", "Time": "3hpi", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 10}),
        "SRR14352117": dict({"Replicate": "A", "Cells": "MDCK-SIAT1", "Time": "3hpi", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 10})
    }),
    "Kupke2020": dict({
        "SRR10489473": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "0"}),
		"SRR10489474": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU1low_0"}),
		"SRR10489475": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU640high_0"}),
		"SRR10489476": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU24low_0"}),
		"SRR10489477": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU11low_0"}),
		"SRR10489478": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU470high_0"}),
		"SRR10489479": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU450high_0"}),
		"SRR10489480": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU23low_0"}),
		"SRR10489481": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU430high_0"}),
		"SRR10489482": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU570high_0"}),
		"SRR10489483": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU670high_0"}),
		"SRR10489484": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU720high_0"}),
		"SRR10489485": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU380high_0"}),
		"SRR10489486": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU680high_0"}),
		"SRR10489487": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU23low_1"}),
		"SRR10489488": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU390high_0"}),
		"SRR10489489": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU38low_0"}),
		"SRR10489490": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU380high_1"}),
		"SRR10489491": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU22low_0"}),
		"SRR10489492": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU650high_0"}),
		"SRR10489493": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU34low_0"}),
		"SRR10489494": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU19low_0"}),
		"SRR10489495": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU18low_0"}),
		"SRR10489496": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU390high_1"}),
		"SRR10489497": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU27low_0"}),
		"SRR10489498": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU410high_0"}),
		"SRR10489499": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU40low_0"}),
		"SRR10489500": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU900high_0"}),
		"SRR10489501": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU3low_0"}),
		"SRR10489502": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU450high_1"}),
		"SRR10489503": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU470high_1"}),
		"SRR10489504": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU1100high_0"}),
		"SRR10489505": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU360high_0"}),
		"SRR10489506": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU5low_0"}),
		"SRR10489507": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU690high_0"}),
		"SRR10489508": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU490high_0"}),
		"SRR10489509": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU580high_0"}),
		"SRR10489510": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU400high_0"}),
		"SRR10489511": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU380high_2"}),
		"SRR10489512": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU16low_0"}),
		"SRR10489513": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU360high_1"}),
		"SRR10489514": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU360high_2"}),
		"SRR10489515": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU360high_3"}),
		"SRR10489516": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU37low_0"}),
		"SRR10489517": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU360high_4"}),
		"SRR10489518": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU460high_0"}),
		"SRR10489519": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU330high_0"}),
		"SRR10489520": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU25low_0"}),
		"SRR10489521": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU650high_1"}),
		"SRR10489522": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU890high_0"}),
		"SRR10489523": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU1low_1"}),
		"SRR10489524": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU17low_0"}),
		"SRR10489525": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU460high_1"}),
		"SRR10489526": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU3low_1"}),
		"SRR10489527": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU430high_1"}),
		"SRR10489528": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU40low_1"}),
		"SRR10489529": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU21low_0"}),
		"SRR10489530": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU420high_0"}),
		"SRR10489531": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU350high_0"}),
		"SRR10489532": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU28low_0"}),
		"SRR10489533": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU29low_0"}),
		"SRR10489534": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU22low_1"}),
		"SRR10489535": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU11low_1"}),
		"SRR10489536": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU510high_0"}),
		"SRR10489537": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU11low_2"}),
		"SRR10489538": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU19low_1"}),
		"SRR10489539": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU6low_0"}),
		"SRR10489540": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU16low_1"}),
		"SRR10489541": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU400high_1"}),
		"SRR10489542": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU4low_0"}),
		"SRR10489543": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU370high_0"}),
		"SRR10489544": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU12low_0"}),
		"SRR10489545": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU31low_0"}),
		"SRR10489546": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU560high_0"}),
		"SRR10489547": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU14low_0"}),
		"SRR10489548": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU810high_0"}),
		"SRR10489549": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU9low_0"}),
		"SRR10489550": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU26low_0"}),
		"SRR10489551": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU400high_2"}),
		"SRR10489552": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU34low_1"}),
		"SRR10489553": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU24low_1"}),
		"SRR10489554": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU20low_0"}),
		"SRR10489555": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU2low_0"}),
		"SRR10489556": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU24low_2"}),
		"SRR10489557": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU37low_1"}),
		"SRR10489558": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU430high_2"}),
		"SRR10489559": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU360high_5"}),
		"SRR10489560": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU550high_0"}),
		"SRR10489561": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU31low_1"}),
		"SRR10489562": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU14low_1"}),
		"SRR10489563": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU450high_0"}),
		"SRR10489564": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU14low_2"}),
		"SRR10489565": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU960high_0"}),
		"SRR10489566": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU28low_1"}),
		"SRR10489567": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU510high_1"}),
		"SRR10489568": dict({"Resolution": "singlecell", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "intracellular", "Replicate": "PFU4low_1"}),
        "SRR10530642": dict({"Resolution": "bulk", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "extracellular", "Replicate": "1"}),
        "SRR10530643": dict({"Resolution": "bulk", "Cells": "MDCK", "Context": "in vitro", "MOI": 10, "Time": "12hpi", "Compartment": "extracellular", "Replicate": "2"})
    }),
    "Sheng2018": dict({
        "SRR3211978": dict({"Cells": "A549", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 1}),
        "SRR3211980": dict({"Cells": "A549", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 1}),
        "SRR3211976": dict({"Cells": "A549", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 1}),
        "SRR3211977": dict({"Cells": "A549", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 1}),
        "SRR3211974": dict({"Cells": "A549", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 1}),
        "SRR3211975": dict({"Cells": "A549", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 1}),
        "SRR3211972": dict({"Cells": "A549", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "bulk", "MOI": 1})
    }),
    "Zhuravlev2020": dict({
        "ERR4566024": dict({"Cells": "A549", "Time": "48hpi", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"}),
        "ERR4566025": dict({"Cells": "A549", "Time": "48hpi", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"}),
        "ERR4566028": dict({"Cells": "HEK293FT", "Time": "48hpi", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"}),
        "ERR4566029": dict({"Cells": "HEK293FT", "Time": "48hpi", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"}),
        "ERR4566032": dict({"Cells": "MRC5", "Time": "48hpi", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"}),
        "ERR4566033": dict({"Cells": "MRC5", "Time": "48hpi", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"}),
        "ERR4566036": dict({"Cells": "WI38", "Time": "48hpi", "Replicate": "1", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"}),
        "ERR4566037": dict({"Cells": "WI38", "Time": "48hpi", "Replicate": "2", "Context": "in vitro", "Compartment": "intracellular", "Resolution": "singlecell"})
    }),
    "Berry2021_A": dict({
        "SRR15182178": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "4-2"}),
        "SRR15182177": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "4-1"}),
        "SRR15182176": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "3-2"}),
        "SRR15182175": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "3-1"}),
        "SRR15182174": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "2-2"}),
        "SRR15182173": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "2-1"}),
        "SRR15182172": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "1-2"}),
        "SRR15182171": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "1-1"})
    }),
    "Berry2021_B": dict({
        "SRR15183345": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "1-2"}),
        "SRR15183344": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "1-1"}),
        "SRR15183352": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "2-1"}),
        "SRR15183353": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "2-2"}),
        "SRR15196408": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "3-1"}),
        "SRR15196409": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "3-2"}),
        "SRR15196410": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "4-1"}),
        "SRR15196411": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "5-1"}),
        "SRR15196412": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "5-2"}),
        "SRR15196413": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "4-2"}),
        "SRR15196414": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "6-1"}),
        "SRR15196415": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "6-2"}),
        "SRR15196416": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "7-1"}),
        "SRR15196417": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "7-2"}),
        "SRR15196419": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "9-1"}),
        "SRR15196418": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "8-1"}),
        "SRR15196420": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "9-2"}),
        "SRR15196421": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "8-2"}),
        "SRR15196422": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "10-1"}),
        "SRR15196423": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "10-2"}),
        "SRR15196424": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "11-1"}),
        "SRR15196425": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "11-2"})
    }),
    "Berry2021_B_Yam": dict({
        "SRR15183338": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "1-1"}),
        "SRR15183343": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "3-2"}),
        "SRR15183342": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "3-1"}),
        "SRR15183341": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "1-2"}),
        "SRR15183340": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "2-2"}),
        "SRR15183339": dict({"Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk", "Cells": "Human", "Replicate": "2-1"})
    }),
    "Valesano2020_Vic": dict({
        "SRR10013092": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013237": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013181": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013242": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013050": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013272": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013047": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013239": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013071": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013201": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013072": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013200": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013108": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013256": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013037": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013254": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013279": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013219": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013221": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"})
    }),
    "Valesano2020_Yam": dict({
        "SRR10013243": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013084": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013188": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013094": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013178": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013236": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013063": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013209": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013241": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013240": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013229": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013068": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013205": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013067": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013206": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013062": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013210": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013070": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013203": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013103": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013170": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013223": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013244": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "SRR10013275": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"})
    }),
    "Southgate2019": dict({
        "ERR3474616": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474621": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474642": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474643": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474658": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474661": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474662": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474663": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474664": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474666": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474671": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474674": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474675": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474676": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474679": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474684": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474685": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474686": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474687": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474689": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474692": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474693": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474694": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474695": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474697": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474698": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474699": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474701": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474702": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474703": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474704": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474705": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474706": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474707": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474709": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474710": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474712": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474713": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474714": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474715": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474716": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474717": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474718": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474719": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474720": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474721": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474722": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474723": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474724": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474725": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474726": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474728": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474729": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474750": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474751": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474781": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474796": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"}),
        "ERR3474809": dict({"Cells": "human", "Context": "in vivo", "Compartment": "extracellular", "Resolution": "bulk"})
    }),
    "VdHoecke2015": dict({
        "SRR1757953": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "MOI": 0.01}),
        "SRR1758027": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk", "MOI": 0.01})
    }),
    "Boussier2020": dict({
        "180628A_rec_A-P1p_S218": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628A_rec_B-P1p_S219": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628A_rec_C-P1p_S219": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628A_rec_D-P1p_S221": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628A_rec_WT1p6-1213_S242": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628B_rec_A-P1p-PCR_S213": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628B_rec_B-P1p-PCR_S214": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628B_rec_C-P1p-PCR_S215": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628B_rec_D-P1p-PCR_S216": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180628B_rec_WT-P1p-PCR_S217": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705A_rec_AP1pb_S294": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705A_rec_BP1pb_S295": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705A_rec_CP1pb_S296": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705A_rec_DP1pb_S297": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705A_rec_WTP1pb_S298": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705B_rec_AP1pPCRb_S289": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705B_rec_BP1pPCRb_S290": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705B_rec_CP1pPCRb_S291": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705B_rec_DP1pPCRb_S292": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180705B_rec_WTP1pPCRb_S293": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180706A_rec_AP1pc_S10": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180706A_rec_BP1pc_S11": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"}),
        "180706A_rec_DP1pc_S12": dict({"Cells": "MDCK", "Context": "in vitro", "Compartment": "extracellular", "Resolution": "bulk"})
    })
})

SEGMENT_DICTS = dict({
    "PR8": dict({
        "AF389115.1": "PB2",
        "AF389116.1": "PB1",
        "AF389117.1": "PA",
        "AF389118.1": "HA",
        "AF389119.1": "NP",
        "AF389120.1": "NA",
        "AF389121.1": "M",
        "AF389122.1": "NS"
    }),
    "Cal07": dict({
        "CY121687.1": "PB2",
        "CY121686.1": "PB1",
        "CY121685.1": "PA",
        "CY121680.1": "HA",
        "CY121683.1": "NP",
        "CY121682.1": "NA",
        "CY121681.1": "M",
        "CY121684.1": "NS"
    }),
    "NC": dict({
        "CY147325.1": "PB2",
        "CY147324.1": "PB1",
        "CY147323.1": "PA",
        "CY147318.1": "HA",
        "CY147321.1": "NP",
        "CY147320.1": "NA",
        "CY147319.1": "M",
        "CY147322.1": "NS"
    }),
    "Perth": dict({
        "KJ609203.1": "PB2",
        "KJ609204.1": "PB1",
        "KJ609205.1": "PA",
        "KJ609206.1": "HA",
        "KJ609207.1": "NP",
        "KJ609208.1": "NA",
        "KJ609209.1": "M",
        "KJ609210.1": "NS"
    }),
    "BLEE": dict({
        "CY115118.1": "PB2",
        "CY115117.1": "PB1",
        "CY115116.1": "PA",
        "CY115111.1": "HA",
        "CY115114.1": "NP",
        "CY115113.1": "NA",
        "CY115112.1": "M",
        "CY115115.1": "NS"
    }),
    "WSN_Mendes_rev": dict({
        "PB2_vRNA": "PB2",
        "PB1_vRNA": "PB1",
        "PA_vRNA": "PA",
        "HA_vRNA": "HA",
        "NP_vRNA": "NP",
        "NA_vRNA": "NA",
        "M_vRNA": "M",
        "NS_vRNA": "NS"
    }),
    "WSN": dict({
        "LC333182.1": "PB2",
        "LC333183.1": "PB1",
        "LC333184.1": "PA",
        "LC333185.1": "HA",
        "LC333186.1": "NP",
        "LC333187.1": "NA",
        "LC333188.1": "M",
        "LC333189.1": "NS"
    }),
    "Anhui": dict({
        "439504": "PB2",
        "439508": "PB1",
        "439503": "PA",
        "439507": "HA",
        "439505": "NP",
        "439509": "NA",
        "439506": "M",
        "439510": "NS"
    }),
    "Turkey": dict({
        "EF619975.1": "PB2",
        "EF619976.1": "PB1",
        "EF619979.1": "PA",
        "AF389118.1": "HA",
        "EF619977.1": "NP",
        "EF619973.1": "NA",
        "EF619978.1": "M",
        "EF619974.1": "NS"
    }),
    "Brisbane": dict({
        "CY115158.1": "PB2",
        "CY115157.1": "PB1",
        "CY115156.1": "PA",
        "CY115151.1": "HA",
        "CY115154.1": "NP",
        "CY115153.1": "NA",
        "CY115152.1": "M",
        "CY115155.1": "NS"
    }),
    "swine": dict({
        "KR701038.1": "PB2",
        "KR701039.1": "PB1",
        "KR701040.1": "PA",
        "KR701041.1": "HA",
        "KR701042.1": "NP",
        "KR701043.1": "NA",
        "KR701044.1": "M",
        "KR701045.1": "NS"
    }),
    "Cal09": dict({
        "JF915190.1": "PB2",
        "JF915189.1": "PB1",
        "JF915188.1": "PA",
        "JF915184.1": "HA",
        "JF915187.1": "NP",
        "JF915186.1": "NA",
        "JF915185.1": "M",
        "JF915191.1": "NS"
    }),
    "Greninger_cons": dict({
        "PB2": "PB2",
        "PB1": "PB1",
        "PA": "PA",
        "HA": "HA",
        "NP": "NP",
        "NA": "NA",
        "M": "M",
        "NS": "NS"
    }),
    "Connecticut": dict({
        "KM654658.1": "PB2",
        "KM654706.1": "PB1",
        "KM654754.1": "PA",
        "KM654822.1": "HA",
        "KM654847.1": "NP",
        "KM654920.1": "NA",
        "KM654969.1": "M",
        "KM654612.1": "NS"
    }),
    "Victoria": dict({
        "CY018660.1": "PB2",
        "CY018659.1": "PB1",
        "CY018658.1": "PA",
        "CY018653.1": "HA",
        "CY018656.1": "NP",
        "CY018655.1": "NA",
        "CY018654.1": "M",
        "CY018657.1": "NS"
    }),
    "H3N2_Thailand": dict({
        "KP335735.1": "PB2",
        "KP335793.1": "PB1",
        "KP335851.1": "PA",
        "KP335964.1": "HA",
        "KP336026.1": "NP",
        "KP336139.1": "NA",
        "KP336201.1": "M",
        "KP336259.1": "NS"
    }),
    "Yamagata": ({
        "OQ034430.1": "PB2",
        "OQ034429.1": "PB1",
        "OQ034431.1": "PA",
        "OQ034432.1": "HA",
        "OQ034433.1": "NP",
        "OQ034434.1": "NA",
        "OQ034435.1": "M",
        "OQ034436.1": "NS"
    }),
    "H1N1_Thailand": ({
        "KU051428.1": "PB2",
        "KU051429.1": "PB1",
        "KU051430.1": "PA",
        "KU051431.1": "HA",
        "KU051432.1": "NP",
        "KU051433.1": "NA",
        "KU051434.1": "M",
        "KU051435.1": "NS"
    }),
    "Malaysia": ({
        "CY040456.1": "PB2",
        "CY040455.1": "PB1",
        "CY040454.1": "PA",
        "CY040449.1": "HA",
        "CY040452.1": "NP",
        "CY040451.1": "NA",
        "CY040450.1": "M",
        "CY040453.1": "NS"
    })
})


### FUNCTIONS ###
def get_dataset_names(cutoff: int = 0, selection: str = "") -> list:
    '''
        Allows to select dataset names based on their cultivation type.
        :param cutoff: Threshold for min number of DelVGs in each dataset
        :param selection: cultivation type either 'in vivo mouse', 'in vitro'
                         or 'in vivo human'

        :return: list of dataset names
    '''
    if cutoff == 0 and selection == "":
        return list(DATASET_STRAIN_DICT.keys())

    path = os.path.join(RESULTSPATH, "metadata", f"dataset_stats_{CUTOFF}.csv")
    df = pd.read_csv(path)
    names = df[df["Size"] >= cutoff]["Dataset"].to_list()

    # make selection based on in vivo/cells etc.
    if selection == "in vivo mouse":
        select_names = ["Wang2023", "Penn2022", "Lui2019"]
    elif selection == "in vitro":
        select_names = ["Alnaji2021", "Pelz2021", "Wang2020", "Kupke2020", "Zhuravlev2020", "VdHoecke2015",
                        "Alnaji2019_Cal07", "Alnaji2019_NC", "Mendes2021", "Boussier2020", "Alnaji2019_Perth",
                        "Alnaji2019_BLEE", "Sheng2018"]
    elif selection == "in vivo human":
        select_names = ["Berry2021_A", "Berry2021_B", "Berry2021_B_Yam", "Southgate2019", "Valesano2020_Yam",
                        "Valesano2020_Vic"]
    elif selection == "IAV":
        select_names = ["Alnaji2021", "Pelz2021", "Wang2023", "Wang2020", "Kupke2020", "Zhuravlev2020", "VdHoecke2015",
                        "Alnaji2019_Cal07", "Alnaji2019_NC", "Mendes2021", "Boussier2020", "Alnaji2019_Perth",
                        "Berry2021_A", "Penn2022", "Lui2019"]
    elif selection == "IBV":
        select_names = ["Alnaji2019_BLEE", "Berry2021_B", "Valesano2020_Vic", "Sheng2018", "Berry2021_B_Yam",
                        "Southgate2019", "Valesano2020_Yam"]
    else:
        select_names = names

    names = [name for name in names if name in select_names]
    return names


def load_single_dataset(exp: str, acc: str, segment_dict: dict) -> pd.DataFrame:
    '''
        Load a single dataset, defined by one SRA accession number.
        :param exp: name of the experiment (is also folder name)
        :param acc: SRA accession number
        :param segment_dict: dictionary that maps the ids of the reference
                            fastas to the segment names

        :return: Pandas Dataframe with one DelVG population
    '''
    path = os.path.join(DATAPATH, exp, f"{exp}_{acc}.csv")
    logging.debug(f'loading accession number {acc}\nFrom path {path}')
    df = pd.read_csv(path,
                     dtype={"Segment": "string", "Start": "int64", "End": "int64", "NGS_read_count": "int64",
                            "ACC_num": "string", "Replicate": "string", "Passage": "string", "Cells": "string",
                            "Context": "string", "Compartment": "string", "Resolution": "string", "Time": "string",
                            "MOI": "float64", "Multi-timepoint": "string", "Mode": "string", "Lineage": "string",
                            "IFNAR": "int64", "IFNLR": "int64"},
                     na_values=["", "None"],
                     keep_default_na=False)
    if df.empty:
        logging.warning(f'No data found in file for {exp} and {acc}. Dataframe empty.')
    else:
        logging.debug(f'Found data from {exp} and {acc}.')
    df["Segment"] = df["Segment"].replace(segment_dict)
    df["ACC_num"] = acc

    return df


def load_dataset(dataset: str) -> pd.DataFrame:
    '''
        Load a full dataset, defined by multiple SRA accession numbers.
        :param exp: name of the experiment (is also folder name)

        :return: Pandas Dataframe with one DelVG population of whole experiment
    '''
    acc_nums = ACCNUMDICT[dataset]
    strain = DATASET_STRAIN_DICT[dataset]
    dfs = list()
    logging.info(f'Found accession numbers for {dataset}: {acc_nums.keys()}')
    for acc_num, meta in acc_nums.items():
        df = load_single_dataset(dataset, acc_num, SEGMENT_DICTS[strain])
        for key in meta.keys():
            df[key] = meta[key]
        dfs.append(df)
    concat_df = pd.concat(dfs)
    logging.debug(f'Loaded data from {dataset}:\ncols: {set(concat_df.columns)}\n{concat_df.head()}')

    return concat_df


def load_all(dfnames: list, expected: str = False) -> Tuple[list, list]:
    '''
        Load a list of datasets.
        :param dfnames: list of dataset names, each is one experiment
        :param expected: if True, expected data is loaded additionally

        :return: Tuple
            List of Pandas Dataframes each containing one experiment
            List of dataset names in same order as first list
    '''
    dfs = list()
    expected_dfs = list()
    for dfname in dfnames:
        strain = DATASET_STRAIN_DICT[dfname]
        df = join_data(load_dataset(dfname))
        dfs.append(preprocess(strain, df, CUTOFF))
        if expected:
            f = os.path.join(DATAPATH, "random_sampled", f"{dfname}_{CUTOFF}.csv")
            if os.path.exists(f):
                dtypes = {"Start": int, "End": int, "Segment": str, "NGS_read_count": int,
                          "key": str, "Strain": str, "isize": int, "full_seq": str,
                          "deleted_sequence": str, "seq_around_deletion_junction": str}
                exp_df = pd.read_csv(f, dtype=dtypes)
            else:
                df = df[df["NGS_read_count"] >= CUTOFF].copy()
                exp_df = preprocess(strain, generate_expected_data(strain, df), 1)
                exp_df.to_csv(f, index=False)
            expected_dfs.append(exp_df)
    return dfs, expected_dfs


def sort_datasets_by_type(dfs: list, dfnames: list, cutoff: int) -> Tuple[list, list]:
    '''
        Sorts a given name of experiments by cultivation type.
        :param dfs: list of datasets, ordered as in dfnames
        :param dfnames: list of dataset names, each is one experiment
        :param cutoff: Threshold for min number of DelVGs in each dataset

        :return: Tuple
            List of Pandas Dataframes each containing one experiment
            List of dataset names in same order as first list
    '''
    vitro = get_dataset_names(cutoff=cutoff, selection="in vitro")
    vivo = get_dataset_names(cutoff=cutoff, selection="in vivo mouse")
    patients = get_dataset_names(cutoff=cutoff, selection="in vivo human")
    dfnames_new_order = vitro + vivo + patients
    combined_data = list(zip(dfnames, dfs))

    def custom_sort(item):
        return dfnames_new_order.index(item[0])

    sorted_data = sorted(combined_data, key=custom_sort)
    dfnames_sorted, dfs_sorted = zip(*sorted_data)

    return dfs_sorted, dfnames_sorted


def join_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Combine duplicate DelVGs and sum their NGS count.
        :param df: Pandas DataFrame with DelVG data

        :return: Pandas DataFrame without duplicate DelVGs
    '''
    return df.groupby(["Segment", "Start", "End"]).sum(["NGS_read_count"]).reset_index()


def load_mapped_reads(experiment: str) -> pd.DataFrame:
    '''
        Loads data about the reads that were mapped to each segment.
        :param experiment: name of the experiment (is also folder name)

        :return: Pandas DataFrame with mapped reads per segment
    '''
    acc_nums = ACCNUMDICT[experiment]

    dfs = list()
    for acc_num, meta in acc_nums.items():
        path = os.path.join(DATAPATH, experiment, f"{acc_num}_mapped_reads_per_segment.csv")
        if not os.path.exists(path):
            path = os.path.join(DATAPATH, experiment, f"{acc_num}both_mapped_reads_per_segment.csv")
        df = pd.read_csv(path, dtype={"counts": "int64", "segment": "string"}, na_values=["", "None"],
                         keep_default_na=False)
        for m in meta.keys():
            df[m] = meta[m]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    return concat_df


def load_all_mapped_reads(dfnames: list) -> list:
    '''
        Loads data about the mapped reads for all given experiments.
        :param dfnames: list of dataset names, each is one experiment

        :return: List of Pandas Dataframes each containing mapped reads for one
                experiment
    '''
    mr_dfs = list()
    for experiment in dfnames:
        df = load_mapped_reads(experiment)
        mr_dfs.append(df)
    return mr_dfs


def get_sequence(strain: str, seg: str, full: bool = False) -> object:
    '''
        Loads a DNA sequence given the strain and segment.
        :param strain: name of the strain
        :param seg: name of the segment
        :param full: if True the whole Biopython Seq Object is returned
                    if False a string object is returned

        :return: Biopython Seq Object or str() of the sequence
    '''
    fasta_file = os.path.join(DATAPATH, "strain_segment_fastas", strain, f"{seg}.fasta")
    seq_obj = SeqIO.read(fasta_file, "fasta")
    if full:
        return seq_obj
    else:
        return str(seq_obj.seq.transcribe())


def get_seq_len(strain: str, seg: str) -> int:
    '''
        Calculates the length of a specific sequence given the strain and
        segment.
        :param strain: name of the strain
        :param seg: name of the segment

        :return: length of the sequence as int
    '''
    return len(get_sequence(strain, seg))


def get_p_value_symbol(p: float) -> str:
    '''
        Indicates the statistical significance by strings. Is used for plots.
        :param p: p-value of the test

        :return: string indicating the significance level
    '''
    if p < 0.00001:
        return "***"
    elif p < 0.001:
        return "** "
    elif p < 0.05:
        return " * "
    else:
        return "ns."


def calc_cliffs_d(d1: list, d2: list) -> float:
    '''
        Cliff, Norman (1993). Dominance statistics: Ordinal analyses to answer
        ordinal questions (eq. 3)
        Cliffs d ranges from -1 (max effect of group 2) to 0 (no effect) to
        1 (max effect of group 1) Meissel K. and Yao E. (2024)
        :param d1: dataset 1
        :param d2: dataset 2

        :return: cliff's d
    '''
    U, _ = stats.mannwhitneyu(d1, d2)
    cliffs_d = 2 * U / (len(d1) * len(d2)) - 1
    return cliffs_d


######################
### DIRECT REPEATS ###
######################
def calculate_direct_repeat(seq: str, s: int, e: int, w_len: int) -> Tuple[int, str]:
    '''
        Counts the number of overlapping nucleotides directly before start and
        end of junction site --> direct repeats
        :param seq: nucleotide sequence
        :param s: start point
        :param e: end point
        :param w_len: length of window to be searched

        :return: Tuple
            Integer giving the number of overlapping nucleotides
            String of the overlapping nucleotides
    '''
    counter = 0
    start_window = seq[s - w_len: s]
    end_window = seq[e - 1 - w_len: e - 1]

    # if they are the same return directly to avoid off-by-one error
    if start_window == end_window:
        return len(start_window), start_window

    if len(seq) < e:
        return 0, "_"

    for i in range(len(end_window) - 1, -1, -1):
        if start_window[i] == end_window[i]:
            counter += 1
        else:
            break
    overlap_seq = str(start_window[i + 1:w_len])

    assert counter == len(overlap_seq), f"{counter=}, {len(overlap_seq)}"
    if len(overlap_seq) == 0:
        overlap_seq = "_"

    return counter, overlap_seq


def count_direct_repeats_overall(df: pd.DataFrame, seq: str) -> Tuple[dict, dict]:
    '''
        Calculates the number of direct repeats for each data point.
        :param df: dataframe with sequence and junction site data
        :param seq: RNA sequence of the given segement and strain

        :return: Tuple
            Dict with the count of the direct repeat lengths
            Dict with the overlapping sequences and their count
    '''
    w_len = 5
    nuc_overlap_dict = dict({i: 0 for i in range(0, w_len + 1)})
    overlap_seq_dict = dict()

    for _, row in df.iterrows():
        s = row["Start"]
        e = row["End"]
        idx, overlap_seq = calculate_direct_repeat(seq, s, e, w_len)
        nuc_overlap_dict[idx] += 1
        if overlap_seq in overlap_seq_dict:
            overlap_seq_dict[overlap_seq] += 1
        else:
            overlap_seq_dict[overlap_seq] = 1

    return nuc_overlap_dict, overlap_seq_dict


#############################
### NUCLEOTIDE ENRICHMENT ###
#############################
def count_nucleotide_occurrence(seq: str, p: int) -> dict:
    '''
        Counts the number of nucleotides next to a given point.
        Goes 5 steps in both directions.
        :param seq: whole RNA sequence
        :param p: point on the sequence where to count

        :return: Counter dict with an entry for each nucleotide. In each entry
                the counter for each position is given.
    '''
    window = seq[p - 5:p + 5]
    r_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})

    for i, char in enumerate(window):
        r_dict[char][i] = 1
    return r_dict


def count_nucleotide_occurrence_overall(df: pd.DataFrame, seq: str) -> Tuple[dict, dict]:
    '''
        Counts the occurrence of each nucleotide at different positions around
        the junction site
        :param df: dataframe with sequence and junction site data
        :param seq: rna sequence where to count the occurrence

        :return: Tuple
            Dict with nucleotide count for start of deletion site
            Dict with nucleotide count for end of deletion site
    '''

    count_start_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    count_end_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    normalize = 0

    for _, row in df.iterrows():
        seq_start_dict = count_nucleotide_occurrence(seq, row["Start"])
        seq_end_dict = count_nucleotide_occurrence(seq, row["End"] - 1)
        normalize += 1
        for nuc in count_start_dict.keys():
            count_start_dict[nuc] += seq_start_dict[nuc]
            count_end_dict[nuc] += seq_end_dict[nuc]

    return count_start_dict, count_end_dict


#####################
### expected data ###
#####################
def generate_expected_data(strain: str, df: pd.DataFrame) -> pd.DataFrame:
    '''
        Randomly samples deletion sites for a given dataset which can be used
        to compare the results of the real dataset.
        :param strain: name of the strain
        :param df: DelVG dataset

        :return: artifical dataset that includes random deletion sites
    '''
    for seg in SEGMENTS:
        df_s = df.loc[df["Segment"] == seg]
        if len(df_s) == 0:
            continue
        seq = get_sequence(strain, seg)
        start = int(df_s["Start"].mean())
        end = int(df_s["End"].mean())
        s = (max(start - 200, 50), start + 200)
        e = (end - 200, min(end + 200, len(seq) - 50))

        # skip if there is no range given this would lead to oversampling of a single position
        if s[0] == s[1] or e[0] == e[1]:
            continue
        # positions are overlapping
        if s[1] > e[0]:
            continue
        if "samp_df" in locals():
            temp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            temp_df["Segment"] = seg
            samp_df = pd.concat([samp_df, temp_df], ignore_index=True)
        else:
            samp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            samp_df["Segment"] = seg

    samp_df["NGS_read_count"] = 1
    return samp_df.reset_index()


def generate_sampling_data(seq: str, s: Tuple[int, int], e: Tuple[int, int], n: int) -> pd.DataFrame:
    '''
        Generates sampling data by creating random start and end points for
        artificial deletion sites. Generated data is used to calculate the
        expected values.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion sites
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion sites
        :param n: number of samples to generate

        :return: Pandas DataFrame of the artifical data set
    '''
    df_no_duplicates = create_sampling_space(seq, s, e)
    return df_no_duplicates.sample(n)


def create_sampling_space(seq: str, s: Tuple[int, int], e: Tuple[int, int]) -> pd.DataFrame:
    '''
        Creates all possible candidates that would be expected.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion sites
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion sites

        :return: dataframe with possible DelVG candidates
    '''
    # create all combinations of start and end positions that are possible
    combinations = [(x, y) for x in range(s[0], s[1] + 1) for y in range(e[0], e[1] + 1)]

    # create for each the DelVG Sequence
    sequences = [seq[:start] + seq[end - 1:] for (start, end) in combinations]

    # filter out duplicate DelVG sequences while keeping the ones with highest start number
    start, end = zip(*combinations)
    temp_df = pd.DataFrame(data=dict({"Start": start, "End": end, "Sequence": sequences}))

    # Find the index of the row with the maximum value in the 'Start' column for each 'Sequence'
    max_start_index = temp_df.groupby('Sequence')['Start'].idxmax()
    result_df = temp_df.loc[max_start_index]
    # Replicate each row by the number of times it was found in the group
    result_df = result_df.loc[result_df.index.repeat(temp_df.groupby('Sequence').size())]
    df_no_duplicates = result_df.reset_index(drop=True).drop("Sequence", axis=1)

    return df_no_duplicates


#######################
### Data processing ###
#######################
def create_nucleotide_ratio_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    '''
        Counts nucleotides around the deletion site. Used to create heatmaps.
        :param df: Pandas DataFrame that was created using sequence_df()
        :param col: column name which sequence to use

        :return: Pandas DataFrame with probabilites for the nucleotides
    '''
    probability_matrix = pd.DataFrame(columns=NUCLEOTIDES.keys())
    seq_matrix = df.filter([col], axis=1)
    seq_matrix = seq_matrix[col].str.split("", expand=True)
    # drop first and last column
    seq_matrix = seq_matrix.drop([0, len(seq_matrix.columns) - 1], axis=1)

    for n in NUCLEOTIDES.keys():
        probability_matrix[n] = seq_matrix.apply(lambda x: dict(x.value_counts()).get(n, 0) / len(x), axis=0)

    return probability_matrix


def plot_heatmap(y: list, x: list, vals: list, ax: object,
                 format=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=False, cbar_ax=None, cbar_kws=None) -> object:
    '''
        Helper function to plot heatmap.
        :param y: columns of heatmap
        :param x: rows of heatmap
        :param vals: values for heatmap
        :param ax: matplotlib.axes object
        :param: additional parameters check sns.heatmap() for more information

        :return: generated heatmap on matplotlib.axes object
    '''
    df = pd.DataFrame({"x": x, "y": y, "vals": vals})
    df = pd.pivot_table(df, index="x", columns="y", values="vals", sort=False)
    ax = sns.heatmap(df, fmt=format, annot=True, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar, cmap=cmap, cbar_ax=cbar_ax,
                     cbar_kws=cbar_kws)
    return ax


def sequence_df(df: pd.DataFrame, strain: str, isize: int = 5) -> pd.DataFrame:
    '''
        Generate a DataFrame with sequence information.
        :param df: Pandas DataFrame containing the DelVGs in the "key" column
            Nomenclature: {seg}_{start}_{end}
        :param strain: name of the strain
        :param isize: the size of the sequence before and after the start and
            end positions. Default is 5.

    :return: Pandas DataFrame with the following columns:
            - "key": The original key from the input DataFrame.
            - "Segment": The segment
            - "Start": The start position of the deletion site
            - "End": The end position of the deletion site
            - "seq": The dip sequence
            - "deleted_sequence": The deleted sequence
            - "isize": The specified size for the before and after sequences
            - "full_seq": full sequence of the wild type virus
            - "Strain": strain used in the experiment
            - "seq_around_deletion_junction": sequence around deletion sites
            - "NGS_read_count": NGS count measured in the experiment

    '''
    df["Strain"] = strain
    df["Start"] = df.apply(lambda row: int(row["key"].split("_")[1]), axis=1)
    df["End"] = df.apply(lambda row: int(row["key"].split("_")[2]), axis=1)
    df["Segment"] = df.apply(lambda row: row["key"].split("_")[0], axis=1)
    df["isize"] = isize

    def wrap_get_sequence(row):
        return get_sequence(row["Strain"], row["Segment"])

    df["full_seq"] = df.apply(wrap_get_sequence, axis=1)

    def wrap_get_deleted_sequence(row):
        return get_deleted_sequence(row["key"], row["Strain"])

    df["deleted_sequence"] = df.apply(wrap_get_deleted_sequence, axis=1)

    def get_seq_around_del(row):
        seq_head = get_dip_sequence(row["key"], row["Strain"])[1]
        seq_foot = get_dip_sequence(row["key"], row["Strain"])[2]

        seq_before_start = seq_head[-row["isize"]:]
        seq_after_start = row["deleted_sequence"][:row["isize"]]
        seq_before_end = row["deleted_sequence"][-row["isize"]:]
        seq_after_end = seq_foot[:row["isize"]]
        return seq_before_start + seq_after_start + seq_before_end + seq_after_end

    df["seq_around_deletion_junction"] = df.apply(get_seq_around_del, axis=1)
    return df


def preprocess(strain: str, df: pd.DataFrame, thresh: int) -> pd.DataFrame:
    '''
        Excluding DelVGs with to low NGS count and running sequence_df().
        :param strain: name of the strain
        :param df: Pandas DataFrame with DelVG data
        :param thresh: Threshold for min number of count for each DelVG

        :return: resulting df of sequence_df() function
    '''
    if thresh > 1:
        df = df[df["NGS_read_count"] >= thresh].copy()
    df["key"] = df["Segment"] + "_" + df["Start"].map(str) + "_" + df["End"].map(str)
    return sequence_df(df, strain)


def get_deleted_sequence(delvg_id: str, strain: str) -> str:
    '''
        Return the sequence of the deletion site.
        :param delvg_id: the id of the DelVG ({seg}_{start}_{end})
        :param strain: name of the strain

        :return: the sequence that is deleted in a DelVG
    '''
    seg, start, end = delvg_id.split("_")
    seq = get_sequence(strain, seg)
    return seq[int(start):int(end) - 1]


def get_dip_sequence(delvg_id: str, strain: str) -> Tuple[str, str, str]:
    '''
        Return the remaining sequence of a DelVG. Deletion is filled with "*".
        :param delvg_id: the id of the DelVG ({seg}_{start}_{end})
        :param strain: name of the strain

        :return: Tuple
            the remaining sequence of a DelVG
            the sequence before the deletion site
            the sequence after the deletion site
    '''
    seg, start, end = delvg_id.split("_")
    fl_seq = get_sequence(strain, seg)
    seq_head = fl_seq[:int(start)]
    seq_foot = fl_seq[int(end) - 1:]
    del_length = int(end) - int(start)
    return seq_head + "*" * del_length + seq_foot, seq_head, seq_foot