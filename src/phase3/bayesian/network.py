"""Bayesian network structure definition."""

from __future__ import annotations

from pgmpy.models import BayesianNetwork

EDGES = [
    ("Has_Payment_Clause", "Payment_Or_Termination_Risky"),
    ("Has_Termination_Clause", "Payment_Or_Termination_Risky"),
    ("Has_Liability_Clause", "Liability_Or_Confidentiality_Risky"),
    ("Has_Confidentiality_Clause", "Liability_Or_Confidentiality_Risky"),
    ("Payment_Or_Termination_Risky", "Cross_Clause_Conflict"),
    ("Liability_Or_Confidentiality_Risky", "Cross_Clause_Conflict"),
    ("Has_Dispute_Resolution_Clause", "Cross_Clause_Conflict"),
    ("Cross_Clause_Conflict", "Contract_Risk_Level"),
]


def build_network() -> BayesianNetwork:
    return BayesianNetwork(EDGES)

