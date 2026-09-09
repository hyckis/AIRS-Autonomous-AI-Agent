# ------------------------------------------------------------
# 1. BASE TASK
#    Shared by Arm A and Arm B.
#    Keep this focused on the actual experimental task.
# ------------------------------------------------------------

BASE_TASK = """
You are advising the risk function of {bank_name}.

The bank currently uses the Federal Reserve supervisory severely adverse
scenario as its primary internal stress scenario.

Using only the supplied context, complete BOTH parts below.

A. 2023 VS. 2015-2022 COMPARISON

Compare the 2023 severely adverse scenario specifically against
the 2015-2022 severely adverse scenarios.

For each candidate shared assumption:

1. State the 2023 value or pattern.
2. Identify which 2015-2022 years show the same or similar pattern.
3. Identify important exceptions.
4. Classify the assumption as:
   - Strongly shared
   - Partially shared
   - Not shared

Do not summarize the historical scenarios year-by-year without
explicitly comparing them with 2023.


B. ALTERNATIVE STRESS SCENARIOS

Generate alternative stress scenarios that could either:

1. deplete the bank's CET1 ratio by at least 300 basis points, or
2. force the bank to sell securities or other assets to meet
   liquidity needs,

through channels not adequately represented in the supervisory scenario.

For each scenario provide:

- Trigger
- Transmission channel
- Supporting evidence
- Quantitative magnitude, if supported
"""


# ------------------------------------------------------------
# 2. COMMON INTEGRITY RULES
#    Minimal rules used across A/B/C.
#    These are experimental hygiene, not a "strong prompt treatment".
# ------------------------------------------------------------

COMMON_INTEGRITY_RULES = """
GENERAL RULES
-------------
- Use only information supplied in the prompt.
- Do not use or refer to events occurring after March 7, 2023.
- Do not infer the real-world identity of the anonymized bank.

OUTPUT STYLE
------------
- Begin directly with the requested output.
- Do not include introductory filler, acknowledgments, meta-commentary,
  disclaimers, concluding summaries, closing remarks, or offers for
  further help.
- Output only the requested sections and fields.
"""


# ------------------------------------------------------------
# 3. STRONG GROUNDING RULES
#    Used by Arm B and Arm C, NOT Arm A.
# ------------------------------------------------------------

GROUNDING_RULES = """
EVIDENCE AND GROUNDING RULES
----------------------------

A scenario may be output only when the target-bank exposures required
by its transmission mechanism are supported by the supplied context.

TARGET-BANK EXPOSURE
- Customer concentration identifies who the bank serves.
  It does not establish the composition of the bank's loans,
  securities, CRE exposure, derivatives, or fee income.
- Total loans must not be treated as CRE loans or as loans to a
  particular customer sector unless explicitly stated.
- AFS or HTM securities must not be assigned a particular asset
  composition unless explicitly stated.
- If a required target-bank exposure is not supplied, do not assume it.

EVIDENCE PROVENANCE
- TARGET BANK PROFILE may establish actual target-bank characteristics.
- MACRO_CONTEXT may support macroeconomic triggers or mechanisms,
  but not target-bank-specific exposures.
- INDUSTRY_CONTEXT may support banking-sector mechanisms or conditions,
  but industry aggregates are not target-bank figures.
- COMPARATOR_BANK evidence may demonstrate that a mechanism exists
  elsewhere, but must not be used to claim that the target bank has
  the same exposure, portfolio composition, funding structure,
  or customer concentration.

ASSET-CLASS DISCIPLINE
- Do not apply an equity-market drawdown directly as a loss rate to
  a debt-securities portfolio.
- Unrealized HTM losses do not automatically reduce CET1.
  A realization or sale mechanism must be established.

QUANTITATIVE DISCIPLINE
- Use a numerical magnitude only when it can be derived from supplied data.
- Otherwise write exactly:
  "Requires quantitative calibration."
- Do not provide a speculative numerical value after that statement.

EXTERNAL TRIGGERS
- Do not invent regulatory actions, rating-agency actions, policy changes,
  institutional interventions, or other external events unless supported
  by the supplied evidence.

CITATION DISCIPLINE
- When retrieved evidence is used, cite only exact supplied CHUNK_IDs.
- Never invent a CHUNK_ID.

For numerical comparison across Federal Reserve scenarios, treat the
STRUCTURED CROSS-YEAR VARIABLE TABLE as authoritative.
"""


# ------------------------------------------------------------
# 4. ARM A — NAIVE
#    Fixed context only.
#    Important: do NOT add the strong grounding treatment here.
# ------------------------------------------------------------

def prompt_arm_a(bank_label, fixed_context):
    return f"""
{BASE_TASK.format(bank_name=bank_label)}

{COMMON_INTEGRITY_RULES}

FIXED CONTEXT
-------------
{fixed_context}
"""


# ------------------------------------------------------------
# 5. ARM B — STRONG PROMPT
#    Fixed context + retrieval + systematic grounding.
# ------------------------------------------------------------

def prompt_arm_b(bank_label, fixed_context, retrieved_context):
    return f"""
{BASE_TASK.format(bank_name=bank_label)}

{COMMON_INTEGRITY_RULES}

{GROUNDING_RULES}

FIXED CONTEXT
-------------
{fixed_context}

RETRIEVED EVIDENCE
------------------
{retrieved_context}

STRONG-PROMPT INSTRUCTIONS
--------------------------
Analyze systematically:

1. Identify recurring assumptions in the supervisory scenarios.
2. Identify target-bank vulnerabilities explicitly supported by evidence.
3. Use retrieved evidence to identify underrepresented stress-transmission
   mechanisms.
4. Generate alternative scenarios that are meaningfully distinct from
   the supervisory scenario rather than merely rephrasing it.
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

def prompt_arm_c(
  bank_label, 
  fixed_context, 
  retrieved_context, 
  supported_directions,
  dominant_framing,
  overlapping_mechanisms
  ):
    return f"""
You are generating evidence-grounded alternative stress scenarios
for {bank_label}.

{COMMON_INTEGRITY_RULES}

{GROUNDING_RULES}

FIXED CONTEXT
-------------
{fixed_context}

RETRIEVED EVIDENCE
-------------
{retrieved_context}

VALIDATED SUPPORTED DIRECTIONS
-------------
{supported_directions}

DOMINANT FRAMING
-------------
{dominant_framing}

OVERLAPPING MECHANISMS
-------------
{overlapping_mechanisms}

DIVERSITY LENSES
-------------
{DIVERSITY_LENSES}

TASK
-------------
Generate alternative stress scenarios that expand beyond the dominant
framing and overlapping mechanisms above.

Use the diversity lenses as SEARCH HEURISTICS, not as evidence.

Develop scenarios only from the VALIDATED SUPPORTED DIRECTIONS.
A diversity lens does not permit unsupported exposures, triggers,
or transmission mechanisms.

For each scenario provide:

- Dominant assumption or mechanism challenged
- Trigger
- Transmission channel
- Supporting evidence using exact CHUNK_ID(s)
- Why the scenario is distinct from the prior analyses
- Quantitative magnitude, or exactly:
  "Requires quantitative calibration."


GROUNDING RULES
-------------
Do not introduce any target-bank exposure, dependency, actor,
or external event that is not supported by the supplied context.

A diversity lens is a search heuristic, not evidence.
Unsupported directions must be discarded.

Do not use post-March-7-2023 information.

Do not infer the bank's real-world identity.

Every scenario MUST be derived from exactly one VALIDATED SUPPORTED DIRECTION.
Do not create a scenario that cannot be traced to a validated direction.

A diversity lens may change the framing or emphasis of a validated direction, but may not introduce a new unsupported trigger, exposure, actor, or institutional action.
"""

def prompt_homogeneity(
    bank_label,
    fixed_context,
    retrieved_context,
    baseline_response,
    strong_response,
    allowed_exposure_keys,
):
    return f"""
You are evaluating conceptual convergence between two stress-testing
analyses of the SAME target bank: {bank_label}.

FIXED CONTEXT
-------------
{fixed_context}

RETRIEVED EVIDENCE
------------------
{retrieved_context}

NAIVE STRATEGY OUTPUT
---------------------
{baseline_response}

STRONG-PROMPT STRATEGY OUTPUT
-----------------------------
{strong_response}

TASK
----
Identify:

1. Shared assumptions
   - recurring assumptions across both outputs
   - factual inconsistencies with supplied context

2. Overlapping mechanisms
   - scenarios that use different wording but essentially the same
     causal pathway

3. Dominant framing
   - risk perspectives that dominate both outputs

4. Supported directions
   - underexplored directions that are supported by supplied evidence

5. Unsupported possibilities
   - potentially interesting directions that lack sufficient evidence

6. Factual issues
   - identify factual, numerical, provenance, or logical problems in either strategy.


GROUNDING RULE FOR SUPPORTED DIRECTIONS
---------------------------------------
A direction is supported only if:

1. every target-bank exposure required by the direction is explicitly
   supported by the TARGET BANK PROFILE; and

2. its proposed trigger or transmission mechanism is supported by the
   supplied fixed or retrieved evidence.

Comparator-bank evidence may support a mechanism but cannot establish
a target-bank exposure.

General discussion of a risk does not establish that the target bank
has that exposure.

Evidence supporting one component of a scenario does not automatically
support an invented trigger or transmission mechanism.

If either condition fails, place the direction under
"unsupported_possibilities" instead of "supported_directions".

CHUNK-ID RULE
-------------
"external_chunk_ids" may contain ONLY exact CHUNK_IDs appearing in
RETRIEVED EVIDENCE.

Do NOT put:
- "TARGET BANK PROFILE"
- "FIXED CONTEXT"
- section labels
- source names

inside "external_chunk_ids".

Describe target-bank-profile support only in "target_bank_support".

If a supported direction does not require retrieved evidence, use:
"external_chunk_ids": []


ADDITIONAL RULES
----------------
- Do not generate final alternative stress scenarios.
- Do not use information or events after March 7, 2023.
- Do not infer the anonymized bank's identity.
- Do not treat different wording as conceptual diversity.
- When citing retrieved evidence, use exact supplied CHUNK_IDs only.


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
      "required_target_bank_exposure": "...",
      "required_exposure_keys": [
        "customer_concentration",
        "uninsured_deposits",
        "htm_securities"
      ]
      "exposure_explicitly_supported": true,
      "mechanism_support": "...",
      "external_chunk_ids": ["..."],
      "unsupported_assumptions_required": [],
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

REQUIRED EXPOSURE KEYS
----------------------
List EVERY target-bank exposure required for the direction's
trigger and transmission mechanism.

Do not select only one representative exposure.

Every required target-bank exposure must be mapped to an allowed key.

If ANY required exposure cannot be represented by an allowed key,
include:
"unsupported"

Allowed keys for this target bank:
{allowed_exposure_keys}

Example:
A scenario requiring technology-sector loan defaults and HTM sales
requires BOTH the sector-specific loan exposure and HTM exposure.

If technology-sector loans are not explicitly supported, return:
"required_exposure_keys": ["unsupported", "htm_securities"]

exposure_keys = item.get(
    "required_exposure_keys",
    ["unsupported"],
)



SUPPORTED-DIRECTION ADMISSION TEST
----------------------------------
Before placing a direction in supported_directions:

1. State the exact target-bank exposure required.
2. Determine whether that exposure is explicitly present in
   the TARGET BANK PROFILE.
3. Identify the evidence supporting the transmission mechanism.
4. List every additional assumption required by the direction.

A direction may be placed in supported_directions ONLY when:

- exposure_explicitly_supported = true
- unsupported_assumptions_required = []

Customer concentration alone does NOT support sector-specific loans,
CRE exposure, securities exposure, geographic exposure, derivatives,
credit lines, or fee-income exposure.

If the direction requires any such unsupported inference, place it in
unsupported_possibilities instead.

IMPORTANT:
"exposure_explicitly_supported" refers to the REQUIRED exposure,
not merely to a related fact.

Example:
If the required exposure is "technology-sector loan exposure",
a statement that technology is a customer concentration does NOT
make exposure_explicitly_supported true.

The exact asset/funding exposure required by the causal mechanism
must itself be explicitly stated.

"""

# ------------------------------------------------------------
# 9. RETRIEVAL QUERY
# ------------------------------------------------------------

def build_retrieval(bank_profile):
    return f"""
Retrieve pre-March-7-2023 evidence relevant to stress-transmission
mechanisms for a bank with the following profile.

Focus on:
- interest-rate risk
- liquidity and funding risk
- securities valuation risk
- credit conditions
- mechanisms through which losses or liquidity pressure could
  lead to asset sales or capital erosion

BANK PROFILE
------------
{bank_profile}
"""

