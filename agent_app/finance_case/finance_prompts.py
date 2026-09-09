SHARED_TASK = """
You are advising the risk function of {bank_name}.

The bank currently uses the Federal Reserve supervisory severely adverse
scenario as its primary internal stress scenario.

Using ONLY the information provided in the context:

(a) Compare the 2023 severely adverse scenario specifically against
the 2015-2022 scenarios.

For each candidate shared assumption:

1. State the 2023 value or pattern first.
2. List which 2015-2022 years show the same pattern.
3. List important exceptions.
4. Conclude whether the assumption is:
   - strongly shared,
   - partially shared,
   - or not shared.

Do not summarize the historical scenarios year-by-year without
explicitly comparing them to 2023.


(b) Generate alternative stress scenarios that could either:

1. deplete the bank's CET1 ratio by at least 300 basis points, or
2. force the bank to sell securities or other assets to meet liquidity needs,

through channels that are not adequately covered by the supervisory scenario.

For each proposed scenario, provide:

- Trigger
- Transmission channel
- Key quantitative magnitudes, if supported
- Supporting evidence from the provided context

SCENARIO ADMISSION RULE
-----------------------
Before outputting a scenario, verify that it has sufficient grounding.

If your reasoning would require phrases such as:
- "the context does not specify..."
- "it is reasonable to assume..."
- "such exposures are plausible..."
- "Bank A may have..."
- "potentially..."
to establish a required Bank A exposure, DISCARD the scenario.
Do not output it.
A diversity lens does not override the evidence requirement.

QUANTITATIVE DISCIPLINE
-----------------------
Do not invent CET1 impacts, loss amounts, deposit outflows, asset-sale
amounts, or other quantitative magnitudes.

Only provide a quantitative estimate when it can be reasonably derived
from the supplied data.

If the supplied information is insufficient to calculate whether the
300-basis-point CET1 threshold would be reached, state:

"Requires quantitative calibration."

If you state "Requires quantitative calibration", DO NOT provide
an unsupported numerical range afterward.

Either:
A. provide a number derived from supplied data and explain the derivation,
or
B. state "Requires quantitative calibration" with no invented estimate.

A scenario may still satisfy the task if it provides a plausible
liquidity mechanism that could force asset sales.


EVIDENCE DISCIPLINE
-------------------
Do not use information that is not contained in the supplied context.

Do not use information or events occurring after March 7, 2023.

Do not infer the identity of the anonymized bank.

Regional-bank 10-K evidence in the retrieved corpus is comparative
industry evidence only. Do not attribute another bank's balance-sheet
figures or exposures to {bank_name}.

For numerical cross-year comparisons, treat the
STRUCTURED CROSS-YEAR VARIABLE TABLE as authoritative.
Do not reconstruct or regroup numerical values from memory.

EVIDENCE PROVENANCE RULES
-------------------------
TARGET BANK PROFILE:
May be used to establish Bank A's actual exposures and characteristics.

MACRO_CONTEXT:
May support macroeconomic triggers or mechanisms.
It does not establish a Bank A-specific exposure.

INDUSTRY_CONTEXT:
May support general banking-sector mechanisms or conditions.
Industry aggregates must NOT be treated as Bank A figures.

COMPARATOR_BANK:
May support analogies or demonstrate that a mechanism exists at another bank.
It must NEVER be used to claim that Bank A has the same exposure,
portfolio composition, funding structure, or customer concentration.


ASSET-CLASS DISCIPLINE
----------------------
Do not infer the composition of a bank asset category unless it is
explicitly stated in the supplied bank profile.

In particular:

- total loans must not be treated as CRE loans;
- AFS or HTM securities must not be assumed to be equities,
  corporate bonds, technology securities, or CRE-linked securities
  unless the context explicitly states this;
- an equity-market drawdown must not be directly applied as a loss
  rate to a debt-securities portfolio;
- unrealized HTM losses do not automatically reduce CET1 unless a
  relevant realization, sale, or accounting/regulatory transmission
  mechanism is specified.

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

Do not combine values from different years.
Before citing a numerical value, verify that the value belongs to
the scenario year being discussed.

"""

DIVERSITY_LENSES = """
1. Counter-mainstream perspective
2. Overlooked stakeholder or depositor behavior
3. Historical analogy
4. Failure-mode analysis
5. Institutional or market-structure difference
6. Resilience versus efficiency trade-off
7. Coordination or correlated-behavior risk
8. Long-term systemic risk
"""

def prompt_arm_c(bank_label, fixed_context, retrieved_context, baseline_response, strong_response, supported_directions):
    return f"""
{SHARED_TASK.format(bank_name=bank_label)}

FIXED CONTEXT
-------------
{fixed_context}

RETRIEVED EVIDENCE
-------------
{retrieved_context}

ARM A OUTPUT
-------------
{baseline_response}

ARM B OUTPUT
-------------
{strong_response}

SUPPORTED DIRECTIONS
-------------
{supported_directions}

DIVERSITY LENSES
-------------
{DIVERSITY_LENSES}

TASK
-------------
Generate alternative stress scenarios that deliberately expand beyond
the dominant assumptions identified in Arm A and Arm B.

The diversity lenses are  SEARCH HEURISTICS, not evidence.

A lens may suggest where to look, but a scenario must still be supported
by the supplied fixed context or retrieved evidence.

For each scenario:
1. State which dominant assumption or mechanism it challenges.
2. Give a clear trigger.
3. Give a causal transmission channel.
4. Cite supporting retrieved evidence using the exact CHUNK_ID.
5. Explain why the scenario is distinct from Arm A and Arm B.
6. Give quantitative magnitudes only when supported by supplied data.
7. Otherwise state "Requires quantitative calibration."

GROUNDING RULES
-------------
Do not invent:
- counterparties
- hedge-fund exposures
- derivative exposures
- depositor types
- securities holdings
- credit concentrations
- operational dependencies
- regulatory actions
- market events
unless they are supported by the supplied context.

Do not use a diversity lens merely to create a novel story.

If a potentially diversi direction is not supported by the supplied evidence, do not generate it as a scenario.

Regional-bank 10-K evidence is comparative evidence only.
Do not attribute Regions or Huntington characteristics directly to {bank_label},

Do not use post-March-7-2023 information.
Do not infer the bank's real-world identity.
"""

def prompt_homogeneity(
    bank_label,
        fixed_context,
        retrieved_context,
        baseline_response,
        strong_response,
):
    return f"""
You are evaluating conceptual convergence between two bank
stress-testing analyses.

BANK
----
{bank_label}

FIXED CONTEXT
-------------
{fixed_context}

RETRIEVED EVIDENCE
------------------
{retrieved_context}

ARM A — NAIVE ANALYSIS
----------------------
{baseline_response}

ARM B — STRONG-PROMPT ANALYSIS
------------------------------
{strong_response}

TASK
----
Analyze Arm A and Arm B for conceptual homogeneity.

Identify:

1. SHARED ASSUMPTIONS
   - Which assumptions recur across both outputs?
   - Check those assumptions against the supplied context.
   - Flag assumptions that appear inconsistent with the context.

2. OVERLAPPING MECHANISMS
   - Which apparently different scenarios rely on essentially
     the same causal pathway?

3. DOMINANT FRAMING
   - Which risk perspectives dominate both analyses?

4. EVIDENCE-SUPPORTED UNDEREXPLORED CHANNELS
   - Identify risk channels that receive little attention but
     have support in the supplied fixed context or retrieved evidence.
   - Cite the relevant CHUNK_ID whenever the support comes from
     retrieved evidence.

5. UNSUPPORTED POSSIBILITIES
   - Separately identify potentially interesting directions that
     are NOT supported by the supplied evidence.
   - These must NOT be recommended to the diversity-expansion agent.

6. CONCEPTUAL REDUNDANCIES
   - Identify surface-level variations of the same underlying idea.

7. DIVERSITY GAPS
   - Recommend only evidence-supported directions that a subsequent
     diversity-expansion agent could explore.

IMPORTANT
---------
Both outputs analyze the SAME target bank: {bank_label}.

"Naive Strategy" and "Strong-Prompt Strategy" are generation
strategies. They do NOT refer to different banks.

Do not generate final alternative stress scenarios.

Do not use post-March-7-2023 knowledge.

Do not infer the identity of the anonymized bank.

Do not treat different wording as conceptual diversity.

Do not convert unsupported possibilities into recommended directions.

Before recommending any underexplored channel, perform a grounding check:
GROUNDING CHECK
- Is this exposure explicitly present in the target bank profile?
- Is this mechanism explicitly supported by retrieved evidence?
- If the evidence describes Huntington or Regions, do not treat that
  exposure as belonging to the target bank.
- General discussion of a risk does not establish that Bank A has
  that exposure.
If no explicit support exists, place the idea under
UNSUPPORTED POSSIBILITIES and do NOT recommend it to the Lens Agent.


OUTPUT FORMAT
-------------
{{
  "shared_assumptions": [
    {{
      "assumption": "...",
      "assessment": "..."
    }}
  ],

  "factual_issues": [
    {{
      "strategy": "naive or strong",
      "issue": "...",
      "reason": "..."
    }}
  ],

  "overlapping_mechanisms": [
    "..."
  ],

  "dominant_framing": [
    "..."
  ],

  "supported_directions": [
    {{
      "direction": "...",
      "target_bank_support": "...",
      "external_chunk_ids": ["..."],
      "why_underexplored": "..."
    }}
  ],

  "unsupported_possibilities": [
    {{
      "direction": "...",
      "reason": "..."
    }}
  ]
}}
"""


def build_retrieval(bank_profile):
    return f"""
Identify pre-March-7-2023 evidence relevant to stress transmission
mechanisms, interest-rate risk, liquidity risk, funding risk,
securities valuation risk, and credit risk for a bank with the
following balance-sheet characteristics:

{bank_profile}
"""