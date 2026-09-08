SHARED_TASK = """
You are advising the risk function of {bank_name}.

The bank currently uses the Federal Reserve supervisory severely adverse
scenario as its primary internal stress scenario.

Using only the information provided in the context:

(a) Identify assumptions that the 2023 severely adverse scenario shares
with the 2015-2022 severely adverse scenarios.

(b) Generate alternative stress scenarios that could either:

1. deplete the bank's CET1 ratio by at least 300 basis points, or
2. force the bank to sell securities or other assets to meet liquidity needs,

through channels that are not adequately covered by the supervisory scenario.

For each proposed scenario, provide:

- Trigger
- Transmission channel
- Key quantitative magnitudes
- Supporting evidence from the provided documents

Do not use information or events occurring after March 7, 2023.
"""

def prompt_arm_a(bank_label, fixed_context):
    return f"""
{SHARED_TASK.format(bank_name=bank_label)}

FIXED CONTEXT
-------------
{fixed_context}
"""

def prompt_arm_b(bank_label, fixed_context, retrieved_context):
    return f"""
{SHARED_TASK.format(bank_name=bank_label)}

FIXED CONTEXT
-------------
{fixed_context}

RETRIEVED EVIDENCE
------------------
{retrieved_context}

Approach the task systematically.

Compare the historical supervisory scenarios before proposing alternatives.
Identify recurring assumptions, examine the bank's balance-sheet exposures,
and use the retrieved evidence to develop plausible and well-supported
alternative scenarios.

Avoid merely rephrasing the Federal Reserve scenarios. Consider multiple
possible transmission mechanisms and ensure that each proposed scenario is
grounded in the supplied evidence.
"""

def prompt_arm_c(bank_label, fixed_context, retrieved_context, convergence_analysis, lenses):
    return f"""
{SHARED_TASK.format(bank_name=bank_label)}

FIXED CONTEXT
-------------
{fixed_context}

RETRIEVED EVIDENCE
------------------
{retrieved_context}

CONVERGENCE ANALYSIS
--------------------
{convergence_analysis}

DIVERSITY LENSES
----------------
{lenses}

The initial analyses show recurring assumptions and areas of conceptual
overlap.

Use the convergence analysis and diversity lenses to deliberately search
outside those dominant assumptions.

Generate alternative scenarios that:
- break or reverse important shared assumptions,
- use transmission mechanisms underrepresented in the initial analyses,
- remain plausible for the supplied bank profile,
- remain grounded in the supplied evidence,
- satisfy the same stress objective defined in the task.

Do not introduce post-cutoff information.
"""

def build_retrieval(bank_profile):
    return f"""
Identify pre-March-7-2023 evidence relevant to stress transmission
mechanisms, interest-rate risk, liquidity risk, funding risk,
securities valuation risk, and credit risk for a bank with the
following balance-sheet characteristics:

{bank_profile}
"""