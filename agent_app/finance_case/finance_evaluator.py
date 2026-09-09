# agent_app/finance_case/finance_evaluator.py

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional



# ============================================================
# Imports
# ============================================================

from .finance_llm_backend import call_llm
#from ..util import extract_json

def extract_json(text):
    text = text.strip()
    # text = normalize_quotes(text)

    # case 1: ```json
    text = re.sub(r"^```json\s", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # case 2: direct parsing
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # case 3: extract first json obj inside the txt
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    
    raise ValueError("No json obj found")

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CORPUS_DIR = BASE_DIR / "corpus"
PROCESSED_DIR = CORPUS_DIR / "processed"
INDEX_DIR = CORPUS_DIR / "index"

BANK_PROFILE_DIR = PROCESSED_DIR / "bank_profiles"
FED_SCENARIO_DIR = PROCESSED_DIR / "fed_scenarios"

CHUNK_INDEX_PATH = INDEX_DIR / "finance_chunk.json"

DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_EVAL_DIR = DEFAULT_RESULTS_DIR / "evaluations"


# ============================================================
# Bank A realized 2023 mechanism answer key
#
# EVALUATOR ONLY.
# NEVER feed this into generation.
# ============================================================

BANK_A_REALIZED_CHANNELS = {
    "interest_rate_stress": (
        "Interest rates rise or remain high, creating duration / "
        "valuation stress on securities."
    ),
    "securities_unrealized_losses": (
        "Securities experience material unrealized valuation losses."
    ),
    "uninsured_deposit_vulnerability": (
        "A large uninsured-deposit base creates funding vulnerability."
    ),
    "deposit_outflow_liquidity_stress": (
        "Deposit withdrawals create acute liquidity pressure."
    ),
    "forced_securities_sale": (
        "Liquidity needs force securities sales."
    ),
    "loss_realization_capital_impact": (
        "Previously unrealized securities losses become realized "
        "and affect earnings or capital."
    ),
}


MECHANISM_FAMILIES = [
    "deposit_liquidity_run",
    "securities_interest_rate",
    "cre_credit",
    "sector_credit",
    "operational_cyber",
    "regulatory_rating",
    "market_funding",
    "other",
]


# ============================================================
# Temporal leakage patterns
#
# explicit_temporal_leakage:
#     strong indication of post-cutoff knowledge
#
# hindsight_risk:
#     suspicious hindsight-style wording, but not automatically
#     counted as definite leakage
# ============================================================

EXPLICIT_LEAKAGE_PATTERNS = {
    "svb_name": re.compile(
        r"\bSilicon Valley Bank\b|\bSVB\b",
        re.IGNORECASE,
    ),

    "post_cutoff_march_date": re.compile(
        r"\bMarch\s+"
        r"(?:[89]|1\d|2\d|3[01])"
        r"(?:st|nd|rd|th)?[,]?\s+2023\b",
        re.IGNORECASE,
    ),

    "march_2023_failure_event": re.compile(
        r"(?:failure|collapse|bank run|banking crisis)"
        r".{0,40}\bMarch\s+2023\b"
        r"|"
        r"\bMarch\s+2023\b.{0,40}"
        r"(?:failure|collapse|bank run|banking crisis)",
        re.IGNORECASE,
    ),
}


HINDSIGHT_RISK_PATTERNS = {
    "regional_banking_crisis": re.compile(
        r"\bregional banking crisis\b",
        re.IGNORECASE,
    ),

    "recent_bank_failures": re.compile(
        r"\brecent bank failures?\b",
        re.IGNORECASE,
    ),

    "events_in_march_2023": re.compile(
        r"\bevents? in March 2023\b",
        re.IGNORECASE,
    ),
}


# Exact corpus CHUNK_ID references such as:
# [fdic_qbp_2022_q4_overview_000]
# [CHUNK_ID: fdic_qbp_2022_q4_overview_000]
CHUNK_REFERENCE_RE = re.compile(
    r"\[(?:CHUNK_ID:\s*)?"
    r"([A-Za-z0-9][A-Za-z0-9_.\-]*_\d{3})"
    r"\]",
    re.IGNORECASE,
)


# ============================================================
# Basic I/O
# ============================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def normalize_bank_id(bank_id: Any) -> str:
    value = str(bank_id or "").strip().upper()

    if value in {"A", "BANK_A", "BANK A"}:
        return "A"

    if value in {"B", "BANK_B", "BANK B"}:
        return "B"

    raise ValueError(
        f"Unknown bank_id: {bank_id}"
    )


# ============================================================
# Infer bank/run from filename
#
# Recommended filenames:
# bank_a_run_01.txt
# bank_a_run_02.txt
# ...
# bank_b_run_05.txt
# ============================================================

def infer_bank_run_from_filename(
    path: Path,
) -> tuple[Optional[str], Optional[int]]:

    name = path.stem.lower()

    match = re.search(
        r"bank[_\-\s]*([ab]).*?"
        r"run[_\-\s]*0*(\d+)",
        name,
    )

    if not match:
        return None, None

    bank_id = match.group(1).upper()
    run_id = int(match.group(2))

    return bank_id, run_id


# ============================================================
# Legacy TXT / Markdown parser
# ============================================================

def extract_section(
    text: str,
    start_label: str,
    end_labels: List[str],
) -> str:

    start_match = re.search(
        rf"(?im)^\s*{re.escape(start_label)}\s*$",
        text,
    )

    if not start_match:
        return ""

    start = start_match.end()
    end = len(text)

    for label in end_labels:
        match = re.search(
            rf"(?im)^\s*{re.escape(label)}\s*$",
            text[start:],
        )

        if match:
            candidate_end = start + match.start()
            end = min(
                end,
                candidate_end,
            )

    return text[start:end].strip()


def parse_critique_text(
    critique_text: str,
) -> Dict[str, Any]:

    critique_text = (
        critique_text or ""
    ).strip()

    if not critique_text:
        return {}

    # Try strict JSON first
    try:
        parsed = json.loads(
            critique_text
        )

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    # Existing logs are usually Python dict repr:
    # {'shared_assumptions': ..., ...}
    try:
        parsed = ast.literal_eval(
            critique_text
        )

        if isinstance(parsed, dict):
            return parsed

    except Exception as exc:
        print(
            "WARNING: Failed to parse Critique:",
            exc,
        )

    return {}


def parse_finance_text_result(
    path: Path,
    bank_id: str,
    run_id: int,
) -> Dict[str, Any]:

    text = read_text(path)

    arm_a = extract_section(
        text=text,
        start_label="Arm A:",
        end_labels=[
            "Arm B:",
            "Critique:",
            "Arm C:",
        ],
    )

    arm_b = extract_section(
        text=text,
        start_label="Arm B:",
        end_labels=[
            "Critique:",
            "Arm C:",
        ],
    )

    critique_raw = extract_section(
        text=text,
        start_label="Critique:",
        end_labels=[
            "Arm C:",
        ],
    )

    arm_c = extract_section(
        text=text,
        start_label="Arm C:",
        end_labels=[],
    )

    critique = parse_critique_text(
        critique_raw
    )

    if not arm_a:
        print(
            f"WARNING: Arm A not found in {path.name}"
        )

    if not arm_b:
        print(
            f"WARNING: Arm B not found in {path.name}"
        )

    if not arm_c:
        print(
            f"WARNING: Arm C not found in {path.name}"
        )

    return {
        "bank_id": normalize_bank_id(
            bank_id
        ),
        "run_id": run_id,

        "outputs": {
            "baseline": arm_a,
            "strong": arm_b,
            "lens": arm_c,
        },

        "convergence_analysis": critique,

        "source_file": str(path),
        "source_format": "legacy_txt",

        # Legacy logs did not store exact top-k retrieval IDs.
        "strict_same_run_retrieval_provenance_available": False,
    }


# ============================================================
# Load evidence
# ============================================================

def load_bank_profile(
    bank_id: str,
) -> Dict[str, Any]:

    bank_id = normalize_bank_id(
        bank_id
    )

    path = (
        BANK_PROFILE_DIR
        / f"bank_{bank_id.lower()}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Bank profile not found: {path}"
        )

    return load_json(path)


def load_fed_scenarios() -> Dict[str, Any]:

    scenarios = {}

    if not FED_SCENARIO_DIR.exists():
        return scenarios

    for path in sorted(
        FED_SCENARIO_DIR.glob("*.json")
    ):
        try:
            scenarios[path.stem] = (
                load_json(path)
            )
        except Exception as exc:
            print(
                f"WARNING: Could not load {path}: {exc}"
            )

    return scenarios

def normalize_scenario_header(line: str) -> str:
    """
    Remove Markdown/list decoration for header detection only.

    Examples:
        **1. Scenario: ABC**
        1. **Scenario:** ABC
        ### Scenario D1: ABC
        **Scenario 2: ABC**
        ### Scenario 1: D1 - ABC
    """
    s = line.strip()

    # Markdown headings
    s = re.sub(r"^#{1,6}\s*", "", s)

    # Markdown bold markers
    s = s.replace("**", "").strip()

    # Numbered list prefix
    s = re.sub(r"^\d+\.\s*", "", s)

    return s.strip()


def is_scenario_header(line: str) -> bool:
    s = normalize_scenario_header(line)

    patterns = [
        # Scenario:
        # Scenario 1:
        # Scenario D1:
        r"^Scenario(?:\s+D?\d+)?\s*:",

        # D1:
        # D1: Scenario 1:
        r"^D\d+\s*:",

        # Scenario 1: D1 - ...
        r"^Scenario\s+\d+\s*:\s*D\d+",
    ]

    return any(
        re.search(pattern, s, re.IGNORECASE)
        for pattern in patterns
    )


def scenario_id_from_line(
    line: str,
    fallback_index: int,
) -> str:
    s = normalize_scenario_header(line)

    # Prefer D1 / D2 if present
    d_match = re.search(
        r"\bD(\d+)\b",
        s,
        re.IGNORECASE,
    )

    if d_match:
        return f"D{d_match.group(1)}"

    scenario_match = re.search(
        r"\bScenario\s+(\d+)\b",
        s,
        re.IGNORECASE,
    )

    if scenario_match:
        return f"S{scenario_match.group(1)}"

    return f"S{fallback_index}"


def split_finance_scenarios(
    text: str,
    arm: str,
) -> List[Dict[str, str]]:

    if not text:
        return []

    if re.search(
        r"No evidence-supported alternative direction",
        text,
        re.IGNORECASE,
    ):
        return []

    if arm in {"baseline", "strong"}:
        working_text = extract_alternative_section(text)
    else:
        working_text = text.strip()

    lines = working_text.splitlines()

    header_indices = []

    for i, line in enumerate(lines):
        if is_scenario_header(line):
            header_indices.append(i)

    if not header_indices:
        print(
            f"WARNING: No scenario headers detected "
            f"for arm={arm}"
        )
        return []

    scenarios = []

    for pos, start_idx in enumerate(header_indices):
        if pos + 1 < len(header_indices):
            end_idx = header_indices[pos + 1]
        else:
            end_idx = len(lines)

        block_lines = lines[start_idx:end_idx]
        block = "\n".join(block_lines).strip()

        scenario_id = scenario_id_from_line(
            lines[start_idx],
            pos + 1,
        )

        scenarios.append({
            "scenario_id": scenario_id,
            "text": block,
        })

    return scenarios




def extract_chunk_id_from_item(
    item: Dict[str, Any],
) -> Optional[str]:

    # Top-level
    for key in [
        "chunk_id",
        "id",
        "CHUNK_ID",
    ]:
        value = item.get(key)

        if value:
            return str(value)

    # Common nested metadata format
    metadata = item.get("metadata")

    if isinstance(metadata, dict):
        for key in [
            "chunk_id",
            "id",
            "CHUNK_ID",
        ]:
            value = metadata.get(key)

            if value:
                return str(value)

    return None


def load_chunk_index() -> Dict[str, Dict[str, Any]]:

    if not CHUNK_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Chunk index not found: {CHUNK_INDEX_PATH}"
        )

    raw = load_json(CHUNK_INDEX_PATH)

    # {"chunks": [...]}
    if (
        isinstance(raw, dict)
        and isinstance(raw.get("chunks"), list)
    ):
        items = raw["chunks"]

    # {"documents": [...]}
    elif (
        isinstance(raw, dict)
        and isinstance(raw.get("documents"), list)
    ):
        items = raw["documents"]

    # {"some_chunk_id": {...}}
    elif isinstance(raw, dict):
        result = {}

        for key, value in raw.items():
            if not isinstance(value, dict):
                continue

            embedded_id = extract_chunk_id_from_item(
                value
            )

            chunk_id = embedded_id or str(key)

            result[chunk_id] = value

        if result:
            return result

        items = []

    elif isinstance(raw, list):
        items = raw

    else:
        raise ValueError(
            "Unsupported finance_chunks.json structure"
        )

    result = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        chunk_id = extract_chunk_id_from_item(
            item
        )

        if chunk_id:
            result[chunk_id] = item

    if not result:
        raise ValueError(
            "Loaded finance_chunks.json but found "
            "zero chunk IDs."
        )

    return result




def get_chunk_text(
    chunk: Dict[str, Any],
) -> str:

    for key in [
        "text",
        "content",
        "chunk_text",
        "document",
        "body",
    ]:
        value = chunk.get(key)

        if value:
            return str(value)

    return json.dumps(
        chunk,
        ensure_ascii=False,
    )


# ============================================================
# Scenario parsing
# ============================================================

def extract_alternative_section(
    text: str,
) -> str:

    if not text:
        return ""

    patterns = [
        r"(?im)^\s*#{0,6}\s*"
        r"(?:\*\*)?"
        r"B\.\s*ALTERNATIVE\s+STRESS\s+SCENARIOS"
        r"(?:\*\*)?\s*$",

        r"(?im)^\s*#{0,6}\s*"
        r"(?:\*\*)?"
        r"ALTERNATIVE\s+STRESS\s+SCENARIOS"
        r"(?:\*\*)?\s*$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
        )

        if match:
            return text[
                match.end():
            ].strip()

    return text.strip()


SCENARIO_HEADER_RE = re.compile(
    r""" ^\s* # Optional numbered list prefix: 
    # 1. Scenario... 
    (?:\d+\.\s*)? # Optional Markdown heading: # ### Scenario... 
    (?:\#{1,6}\s*)? # Optional Markdown bold: # **Scenario... 
    \*{0,2} 
    (?: # Scenario 1: # Scenario D1: # Scenario: Scenario 
    (?:\s+D?\d+)? \s*: | # D1: # D1: Scenario 1: 
    D\d+ \s*: \s* (?: Scenario (?:\s+\d+)? \s*:? )? ) .*$ """, 
    flags=( re.IGNORECASE | re.MULTILINE | re.VERBOSE ), 
)


def scenario_id_from_header(
    header: str,
    fallback_index: int,
) -> str:

    d_match = re.search(
        r"\bD(\d+)\b",
        header,
        re.IGNORECASE,
    )

    if d_match:
        return (
            f"D{d_match.group(1)}"
        )

    number_match = re.search(
        r"\bScenario\s+(\d+)\b",
        header,
        re.IGNORECASE,
    )

    if number_match:
        return (
            f"S{number_match.group(1)}"
        )

    return f"S{fallback_index}"



# ============================================================
# Exact citation ID audit
#
# Important:
# Legacy TXT does NOT let us prove that a valid chunk ID
# was one of that run's exact top-5 retrieved chunks.
#
# We therefore check:
# - exact ID exists in frozen finance index
# - malformed / wrong-case IDs fail
#
# ============================================================

def extract_chunk_references(
    text: str,
) -> List[str]:

    refs = CHUNK_REFERENCE_RE.findall(
        text or ""
    )

    # preserve order + unique
    return list(
        dict.fromkeys(refs)
    )


def audit_scenario_citations(
    scenario_text: str,
    chunk_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    refs = extract_chunk_references(
        scenario_text
    )

    valid = [
        ref
        for ref in refs
        if ref in chunk_index
    ]

    invalid = [
        ref
        for ref in refs
        if ref not in chunk_index
    ]

    return {
        "cited_chunk_ids": refs,

        "valid_exact_chunk_ids": valid,

        "invalid_exact_chunk_ids": invalid,

        "all_cited_chunk_ids_exist": (
            len(invalid) == 0
        ),

        "strict_same_run_retrieval_check":
            "unavailable_for_legacy_txt",
    }


# ============================================================
# Critique / admission audit
# ============================================================

def audit_direction_provenance(
    critique: Dict[str, Any],
    chunk_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    supported_directions = (
        critique.get(
            "supported_directions",
            [],
        )
    )

    if not isinstance(
        supported_directions,
        list,
    ):
        supported_directions = []

    audits = []

    for i, item in enumerate(
        supported_directions,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        external_ids = item.get(
            "external_chunk_ids",
            [],
        )

        if not isinstance(
            external_ids,
            list,
        ):
            external_ids = [
                str(external_ids)
            ]

        external_ids = [
            str(x)
            for x in external_ids
        ]

        valid_ids = [
            chunk_id
            for chunk_id in external_ids
            if chunk_id in chunk_index
        ]

        invalid_ids = [
            chunk_id
            for chunk_id in external_ids
            if chunk_id not in chunk_index
        ]

        unsupported = item.get(
            "unsupported_assumptions_required",
            [],
        )

        if not isinstance(
            unsupported,
            list,
        ):
            unsupported = [
                str(unsupported)
            ]

        direction_id = item.get(
            "direction_id"
        )

        audits.append({
            "candidate_index": i,

            "direction": item.get(
                "direction",
                "",
            ),

            "direction_id":
                direction_id,

            "required_exposure_keys":
                item.get(
                    "required_exposure_keys",
                    [],
                ),

            "exposure_explicitly_supported":
                item.get(
                    "exposure_explicitly_supported"
                ),

            "external_chunk_ids":
                external_ids,

            "valid_exact_chunk_ids":
                valid_ids,

            "invalid_exact_chunk_ids":
                invalid_ids,

            "unsupported_assumptions_required":
                unsupported,

            "admitted_to_lens":
                bool(direction_id),

            # Useful for B1/B2/B4-type cases
            "possible_provenance_rejection": (
                not direction_id
                and bool(invalid_ids)
            ),
        })

    return {
        "strict_same_run_retrieval_provenance_available":
            False,

        "num_candidates":
            len(audits),

        "num_admitted":
            sum(
                1
                for x in audits
                if x["admitted_to_lens"]
            ),

        "num_with_invalid_chunk_id":
            sum(
                1
                for x in audits
                if x[
                    "invalid_exact_chunk_ids"
                ]
            ),

        "directions": audits,
    }


# ============================================================
# Temporal / hindsight leakage audit
# ============================================================

def audit_temporal_leakage(
    text: str,
) -> Dict[str, Any]:

    explicit_flags = []
    hindsight_flags = []

    for (
        name,
        pattern,
    ) in EXPLICIT_LEAKAGE_PATTERNS.items():

        if pattern.search(
            text or ""
        ):
            explicit_flags.append(
                name
            )

    for (
        name,
        pattern,
    ) in HINDSIGHT_RISK_PATTERNS.items():

        if pattern.search(
            text or ""
        ):
            hindsight_flags.append(
                name
            )

    return {
        "explicit_temporal_leakage":
            bool(explicit_flags),

        "explicit_leakage_flags":
            explicit_flags,

        "hindsight_risk":
            bool(hindsight_flags),

        "hindsight_risk_flags":
            hindsight_flags,
    }


# ============================================================
# Evidence context for semantic judge
#
# For legacy TXT, only give the judge:
# - target bank profile
# - fixed Fed scenario context
# - retrieved chunks ACTUALLY CITED by this scenario
#
# This avoids giving the evaluator evidence that the
# generation itself did not reference.
# ============================================================

def build_cited_evidence_context(
    scenario_text: str,
    chunk_index: Dict[str, Dict[str, Any]],
) -> str:

    refs = extract_chunk_references(
        scenario_text
    )

    blocks = []

    for ref in refs:
        chunk = chunk_index.get(
            ref
        )

        if chunk is None:
            blocks.append(
                f"[CHUNK_ID: {ref}]\n"
                "[INVALID OR UNKNOWN CHUNK_ID]"
            )
            continue

        blocks.append(
            f"[CHUNK_ID: {ref}]\n"
            f"{get_chunk_text(chunk)}"
        )

    if not blocks:
        return (
            "[No valid retrieved CHUNK_ID "
            "was explicitly cited in this scenario.]"
        )

    return "\n\n".join(
        blocks
    )


def build_fixed_context_text() -> str:

    scenarios = load_fed_scenarios()

    return json.dumps(
        scenarios,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Semantic finance judge prompt
# ============================================================

def build_finance_judge_prompt(
    bank_id: str,
    arm: str,
    scenario_id: str,
    scenario_text: str,
    bank_profile: Dict[str, Any],
    fixed_context: str,
    cited_evidence_context: str,
) -> str:

    bank_id = normalize_bank_id(
        bank_id
    )

    if bank_id == "A":
        mechanism_instructions = f"""
BANK A REALIZED-MECHANISM COVERAGE

This section evaluates mechanism recovery only.

Use the following evaluator-only answer key:

{json.dumps(
    BANK_A_REALIZED_CHANNELS,
    ensure_ascii=False,
    indent=2,
)}

For EACH channel return:
0 = absent
1 = partially present / indirectly implied
2 = clearly and causally present

IMPORTANT:
Matching the realized mechanism does NOT mean the
scenario is evidence-grounded.

A scenario may have high mechanism coverage while
also containing unsupported triggers or details.
"""

    else:
        mechanism_instructions = """
BANK B REALIZED-MECHANISM COVERAGE

Do NOT evaluate Bank B against Bank A's realized
2023 failure mechanism.

Return:
"mechanism_coverage": {}
"""

    return f"""
You are a strict evaluator of a bank reverse-stress-test scenario.

Evaluate ONLY from the supplied evidence below.

DO NOT use outside knowledge.

DO NOT infer the real-world identity of Bank A or Bank B.

============================================================
GROUNDING RULES
============================================================

1. TARGET BANK PROFILE is the only source that can establish
   target-bank-specific balance-sheet exposures.

2. Industry evidence may establish that a mechanism is possible
   in the banking industry, but does NOT establish that the target
   bank has the same exposure.

3. Comparator-bank evidence may establish that a mechanism exists
   elsewhere, but does NOT establish target-bank exposure.

4. Customer concentration does NOT prove:
   - sector-specific loans
   - sector-specific securities
   - derivatives exposure
   - fee-income exposure
   - sector-specific uninsured-deposit composition

5. Total loans do NOT prove CRE loans or sector-specific loans.

6. Investment securities do NOT prove corporate-bond,
   municipal, CMBS, or other unsupported composition.

7. HTM unrealized losses do NOT automatically reduce CET1.
   A supported realization / sale / accounting mechanism is needed.

8. Plausible is NOT the same as supported.

9. A trigger such as:
   - cyberattack
   - fraud allegation
   - rating downgrade
   - regulatory intervention
   - social-media panic
   - natural disaster
   - specific company failure
   is unsupported unless supplied evidence directly supports it.

10. Do not reward an invented event simply because it is realistic.

11. The core exposure may be supported while the trigger
    or later transmission steps are unsupported.


CRITICAL CLAIM-EXTRACTION RULE:

Only flag an unsupported exposure, trigger, transmission step,
or quantitative detail if that claim is ACTUALLY ASSERTED in
SCENARIO TO EVALUATE.

Do NOT list:
- hypothetical examples from this rubric,
- exposure categories merely mentioned in these instructions,
- claims the scenario never made,
- missing information.

If the scenario does not assert it, it MUST NOT appear in any
unsupported_* list.




============================================================
SCORING
============================================================

target_exposure_support:
2 = The exact target-bank exposure used by the scenario is explicitly
stated in the TARGET BANK PROFILE.

1 = The profile supports a related exposure, but the scenario makes an
additional inference about its composition or form.

0 = The required target-bank exposure is absent.

Important:
- customer concentration in technology does NOT establish technology loans;
- customer concentration in life science does NOT establish life-science loans;
- total loans do NOT establish sector-specific loans;
- Bank A has no explicit CRE exposure;
- HTM holdings and uninsured deposits ARE explicit Bank A exposures.



trigger_support:
2 = The specific trigger event is explicitly supported by the supplied
evidence.

1 = The same trigger EVENT TYPE is explicitly supported by the supplied
evidence, but the scenario adds unsupported details such as actor,
timing, severity, or magnitude.

0 = The trigger event/type itself is not present in the supplied evidence.

Plausibility is NOT evidence.

Therefore:
- technology customer concentration does NOT support a technology-company default;
- uninsured deposits do NOT support a whistleblower/audit event;
- HTM losses do NOT support a sovereign downgrade;
- sector concentration does NOT support a new regulatory shock.

If the rationale describes the trigger as "unsupported", "invented",
"not explicitly supported", or merely "plausible", the score MUST be 0,
unless the underlying event type itself appears in the supplied evidence.



transmission_support:
2 = causal pathway is fully supported by target-bank exposure and supplied evidence
1 = core causal pathway is supported, but one or more steps/actors/dependencies are invented
0 = main causal pathway is unsupported or conceptually incorrect

quantitative_support:

2 = The scenario either:
    (a) uses only quantities explicitly present in the TARGET BANK PROFILE
        or supplied evidence, OR
    (b) states "Requires quantitative calibration." and does NOT assert
        an assumed stress percentage, loss amount, withdrawal rate,
        CET1 impact, or other invented quantitative magnitude.

1 = The scenario mixes grounded quantities with one or more
    assumed/calibrated quantitative claims.

0 = The scenario asserts unsupported, mathematically incorrect,
    or financially invalid quantitative magnitudes.

IMPORTANT:
- Do NOT penalize a scenario for omitting a quantitative magnitude.
- Missing calibration is not an unsupported quantitative claim.
- Do NOT put a missing number in unsupported_quantitative_details.
- Quantities copied from the TARGET BANK PROFILE are evidence,
  not speculative scenario magnitudes.


============================================================
MECHANISM FAMILY
============================================================

Choose one or more only from:

{json.dumps(
    MECHANISM_FAMILIES,
    ensure_ascii=False,
)}

{mechanism_instructions}

EXPOSURE VS. MECHANISM RULE:

target_exposure_support evaluates ONLY whether the target bank
actually has the balance-sheet exposure used by the scenario.

Do NOT lower target_exposure_support merely because the trigger
or causal mechanism is unsupported.

Conversely, a supported exposure does NOT make the transmission
mechanism supported.

For HTM securities:
- HTM holdings and unrealized losses may be fully supported exposures.
- Unrealized HTM losses do NOT by themselves force a sale.
- A forced-sale transmission requires a separately supported
  liquidity/funding need or other valid realization mechanism.
- "Sell HTM to maintain regulatory capital because HTM market value fell"
  is not fully supported merely because HTM unrealized losses exist.



============================================================
OUTPUT
============================================================

Return JSON ONLY:

{{
  "target_exposure_support": 0,
  "trigger_support": 0,
  "transmission_support": 0,
  "quantitative_support": 0,
  "bank_specificity": 0,

  "unsupported_exposures": [],
  "unsupported_trigger_details": [],
  "unsupported_transmission_details": [],
  "unsupported_quantitative_details": [],

  "mechanism_families": [],
  "mechanism_summary": "",

  "mechanism_coverage": {{}},

  "grounding_rationale": ""
}}

============================================================
TARGET BANK
============================================================

Bank {bank_id}

============================================================
ARM
============================================================

{arm}

============================================================
SCENARIO ID
============================================================

{scenario_id}

============================================================
TARGET BANK PROFILE
============================================================

{json.dumps(
    bank_profile,
    ensure_ascii=False,
    indent=2,
)}

============================================================
FIXED FED SUPERVISORY CONTEXT
============================================================

{fixed_context}

============================================================
RETRIEVED EVIDENCE ACTUALLY CITED BY THIS SCENARIO
============================================================

{cited_evidence_context}

============================================================
SCENARIO TO EVALUATE
============================================================

{scenario_text}
""".strip()


# ============================================================
# Semantic evaluation
# ============================================================

def clamp_score(
    value: Any,
    minimum: int = 0,
    maximum: int = 2,
) -> Optional[int]:

    try:
        value = int(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def evaluate_finance_scenario_semantic(
    bank_id: str,
    arm: str,
    scenario_id: str,
    scenario_text: str,
    bank_profile: Dict[str, Any],
    fixed_context: str,
    cited_evidence_context: str,
    backend: str = "local_ollama",
    model: Optional[str] = None,
) -> Dict[str, Any]:

    prompt = build_finance_judge_prompt(
        bank_id=bank_id,
        arm=arm,
        scenario_id=scenario_id,
        scenario_text=scenario_text,
        bank_profile=bank_profile,
        fixed_context=fixed_context,
        cited_evidence_context=
            cited_evidence_context,
    )

    raw = call_llm(
        prompt,
        backend=backend,
        model=model,
        temperature=0.1,
    )

    try:
        parsed = extract_json(
            raw
        )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "LLM judge did not return "
                "a JSON object."
            )

        score_fields = [
            "target_exposure_support",
            "trigger_support",
            "transmission_support",
            "quantitative_support",
            "bank_specificity",
        ]

        for key in score_fields:
            parsed[key] = clamp_score(
                parsed.get(key)
            )

        list_fields = [
            "unsupported_exposures",
            "unsupported_trigger_details",
            "unsupported_transmission_details",
            "unsupported_quantitative_details",
            "mechanism_families",
        ]

        for key in list_fields:
            value = parsed.get(
                key,
                [],
            )

            if not isinstance(
                value,
                list,
            ):
                value = [
                    str(value)
                ]

            parsed[key] = value

        parsed[
            "mechanism_families"
        ] = [
            family
            for family
            in parsed[
                "mechanism_families"
            ]
            if family
            in MECHANISM_FAMILIES
        ]

        coverage = parsed.get(
            "mechanism_coverage",
            {},
        )

        if not isinstance(
            coverage,
            dict,
        ):
            coverage = {}

        if normalize_bank_id(
            bank_id
        ) == "A":

            cleaned = {}

            for channel in (
                BANK_A_REALIZED_CHANNELS
            ):
                value = clamp_score(
                    coverage.get(
                        channel,
                        0,
                    )
                )

                cleaned[channel] = (
                    value
                    if value is not None
                    else 0
                )

            parsed[
                "mechanism_coverage"
            ] = cleaned

        else:
            parsed[
                "mechanism_coverage"
            ] = {}

        grounding_values = [
            parsed.get(
                "target_exposure_support"
            ),
            parsed.get(
                "trigger_support"
            ),
            parsed.get(
                "transmission_support"
            ),
        ]

        grounding_values = [
            x
            for x in grounding_values
            if x is not None
        ]

        parsed[
            "evidence_grounding_score"
        ] = (
            round(
                sum(
                    grounding_values
                )
                / len(
                    grounding_values
                ),
                3,
            )
            if grounding_values
            else None
        )

        parsed[
            "parse_failed"
        ] = False

        return parsed

    except Exception as exc:

        return {
            "target_exposure_support":
                None,

            "trigger_support":
                None,

            "transmission_support":
                None,

            "quantitative_support":
                None,

            "bank_specificity":
                None,

            "evidence_grounding_score":
                None,

            "unsupported_exposures":
                [],

            "unsupported_trigger_details":
                [],

            "unsupported_transmission_details":
                [],

            "unsupported_quantitative_details":
                [],

            "mechanism_families":
                [],

            "mechanism_summary":
                "",

            "mechanism_coverage":
                {},

            "grounding_rationale":
                f"Parsing failed: {exc}",

            "parse_failed":
                True,

            "raw_judge_output":
                (
                    raw[:4000]
                    if raw
                    else ""
                ),
        }


# ============================================================
# Helpers for aggregation
# ============================================================

def average_numeric(
    values: List[Any],
) -> Optional[float]:

    clean = []

    for value in values:
        if value is None:
            continue

        try:
            clean.append(
                float(value)
            )
        except Exception:
            continue

    if not clean:
        return None

    return round(
        sum(clean)
        / len(clean),
        3,
    )


def aggregate_arm_metrics(
    bank_id: str,
    evaluated_scenarios:
        List[Dict[str, Any]],
    arm_text: str,
) -> Dict[str, Any]:

    semantics = [
        item.get(
            "semantic",
            {},
        )
        for item in
        evaluated_scenarios
    ]

    family_counter = Counter()

    for semantic in semantics:
        for family in semantic.get(
            "mechanism_families",
            [],
        ):
            family_counter[
                family
            ] += 1

    full_grounding_count = 0

    for semantic in semantics:
        if (
            semantic.get(
                "target_exposure_support"
            ) == 2
            and semantic.get(
                "trigger_support"
            ) == 2
            and semantic.get(
                "transmission_support"
            ) == 2
        ):
            full_grounding_count += 1

    result = {
        "num_scenarios":
            len(
                evaluated_scenarios
            ),

        "avg_target_exposure_support":
            average_numeric([
                x.get(
                    "target_exposure_support"
                )
                for x in semantics
            ]),

        "avg_trigger_support":
            average_numeric([
                x.get(
                    "trigger_support"
                )
                for x in semantics
            ]),

        "avg_transmission_support":
            average_numeric([
                x.get(
                    "transmission_support"
                )
                for x in semantics
            ]),

        "avg_quantitative_support":
            average_numeric([
                x.get(
                    "quantitative_support"
                )
                for x in semantics
            ]),

        "avg_bank_specificity":
            average_numeric([
                x.get(
                    "bank_specificity"
                )
                for x in semantics
            ]),

        "avg_evidence_grounding_score":
            average_numeric([
                x.get(
                    "evidence_grounding_score"
                )
                for x in semantics
            ]),

        "fully_grounded_scenarios":
            full_grounding_count,

        "fully_grounded_rate": (
            round(
                full_grounding_count
                / len(
                    evaluated_scenarios
                ),
                3,
            )
            if evaluated_scenarios
            else None
        ),

        "mechanism_family_counts":
            dict(
                family_counter
            ),

        "num_distinct_mechanism_families":
            len(
                family_counter
            ),

        "temporal_audit":
            audit_temporal_leakage(
                arm_text
            ),
    }

    # Bank A set-level realized mechanism coverage
    if normalize_bank_id(
        bank_id
    ) == "A":

        channel_max = {
            channel: 0
            for channel
            in BANK_A_REALIZED_CHANNELS
        }

        # Use max across scenarios because one run can
        # split the realized chain across D1 and D2.
        for semantic in semantics:
            coverage = semantic.get(
                "mechanism_coverage",
                {},
            )

            for channel in channel_max:
                try:
                    value = int(
                        coverage.get(
                            channel,
                            0,
                        )
                    )
                except Exception:
                    value = 0

                channel_max[
                    channel
                ] = max(
                    channel_max[
                        channel
                    ],
                    value,
                )

        total = sum(
            channel_max.values()
        )

        max_possible = (
            len(
                BANK_A_REALIZED_CHANNELS
            )
            * 2
        )

        result[
            "bank_a_realized_channel_coverage"
        ] = channel_max

        result[
            "bank_a_realized_coverage_total"
        ] = total

        result[
            "bank_a_realized_coverage_rate"
        ] = round(
            total
            / max_possible,
            3,
        )

        result[
            "bank_a_full_channel_hits"
        ] = sum(
            1
            for value
            in channel_max.values()
            if value == 2
        )

        result[
            "bank_a_partial_or_full_channel_hits"
        ] = sum(
            1
            for value
            in channel_max.values()
            if value >= 1
        )

    return result


# ============================================================
# Evaluate one parsed TXT result
# ============================================================
def apply_finance_score_corrections(
    scenario_text: str,
    semantic: Dict[str, Any],
    bank_id: str,
) -> Dict[str, Any]:

    text = scenario_text.lower()

    # ========================================================
    # 1. Quantitative support
    # ========================================================

    calibration_only = bool(
    re.search(
        r"\*\*Quantitative\s+Magnitude:\*\*\s*"
        r"Requires quantitative calibration\.?\s*$",
        scenario_text,
        re.IGNORECASE,
    ))

    if calibration_only:
        semantic["quantitative_support"] = 2
        semantic["unsupported_quantitative_details"] = []

    # These indicate that the model actually introduced
    # an assumed scenario magnitude after saying calibration
    # is required.
    speculative_quant_patterns = [
        r"\b(?:assume|consider)\b",
        r"\b\d+(?:\.\d+)?%\s+"
        r"(?:decline|drop|loss|outflow|withdrawal|markdown)",
        r"\b(?:cet1)\b.*\b\d+(?:\.\d+)?\s*bps\b",
    ]

    has_speculative_quant = any(
        re.search(
            pattern,
            scenario_text,
            re.IGNORECASE,
        )
        for pattern in speculative_quant_patterns
    )

    # "Requires quantitative calibration." by itself
    # is NOT an unsupported quantitative claim.
    if (
        calibration_only
        and not has_speculative_quant
    ):
        semantic["quantitative_support"] = 2
        semantic[
            "unsupported_quantitative_details"
        ] = []

    # ========================================================
    # 2. Bank specificity
    # ========================================================

    bank_specific_markers = {
        "A": [
            "87.5%",
            "91.327",
            "76.169",
            "15.16",
            "technology",
            "life science",
            "healthcare",
            "venture capital",
            "private equity",
        ],
        "B": [
            "45.4%",
            "45.444",
            "13.53",
            "commercial real estate",
            "cre",
        ],
    }

    hits = sum(
        marker in text
        for marker in bank_specific_markers.get(
            bank_id,
            [],
        )
    )

    if hits >= 2:
        semantic["bank_specificity"] = 2

    elif hits == 1:
        semantic["bank_specificity"] = max(
            1,
            semantic.get(
                "bank_specificity",
                0,
            ),
        )

    # ========================================================
    # 3. Bank A:
    #    customer concentration != sector loan exposure
    # ========================================================

    if bank_id == "A":

        sector_loan_patterns = [
            r"\btechnology\b.*\b(?:loan|lending|borrower|default)",
            r"\b(?:loan|lending|borrower|default).*\btechnology\b",

            r"\blife science\b.*\b(?:loan|lending|borrower|default)",
            r"\b(?:loan|lending|borrower|default).*\blife science\b",

            r"\bhealthcare\b.*\b(?:loan|lending|borrower|default)",
            r"\b(?:loan|lending|borrower|default).*\bhealthcare\b",

            r"\bventure capital\b.*\b(?:loan|lending|borrower|default)",
            r"\b(?:loan|lending|borrower|default).*\bventure capital\b",
        ]

        inferred_sector_loan = any(
            re.search(
                pattern,
                scenario_text,
                re.IGNORECASE | re.DOTALL,
            )
            for pattern in sector_loan_patterns
        )

        if inferred_sector_loan:
            semantic[
                "target_exposure_support"
            ] = min(
                semantic.get(
                    "target_exposure_support",
                    0,
                ),
                1,
            )

        # Bank A has no explicit CRE loan exposure.
        if re.search(
            r"\b(?:commercial real estate|cre)\b"
            r".*\b(?:loan|lending|exposure|portfolio)\b",
            scenario_text,
            re.IGNORECASE | re.DOTALL,
        ):
            semantic[
                "target_exposure_support"
            ] = 0

    return semantic


def evaluate_finance_result(
    result: Dict[str, Any],
    backend: str = "local_ollama",
    model: Optional[str] = None,
) -> Dict[str, Any]:

    bank_id = normalize_bank_id(
        result.get(
            "bank_id"
        )
    )

    run_id = result.get(
        "run_id"
    )

    bank_profile = (
        load_bank_profile(
            bank_id
        )
    )

    fixed_context = (
        build_fixed_context_text()
    )

    chunk_index = (
        load_chunk_index()
    )

    critique = result.get(
        "convergence_analysis",
        {},
    )

    if not isinstance(
        critique,
        dict,
    ):
        critique = {}

    outputs = result.get(
        "outputs",
        {},
    )

    evaluation = {
        "bank_id":
            bank_id,

        "run_id":
            run_id,

        "source_file":
            result.get(
                "source_file"
            ),

        "source_format":
            result.get(
                "source_format"
            ),

        "judge_backend":
            backend,

        "judge_model":
            model,

        "strict_same_run_retrieval_provenance_available":
            False,

        "direction_provenance_audit":
            audit_direction_provenance(
                critique,
                chunk_index,
            ),

        "arms": {},
    }

    arm_map = {
        "baseline": outputs.get(
            "baseline",
            "",
        ),

        "strong": outputs.get(
            "strong",
            "",
        ),

        "lens": outputs.get(
            "lens",
            "",
        ),
    }

    for (
        arm,
        arm_text,
    ) in arm_map.items():

        scenarios = (
            split_finance_scenarios(
                arm_text,
                arm,
            )
        )

        evaluated_scenarios = []

        for scenario in scenarios:

            scenario_id = (
                scenario[
                    "scenario_id"
                ]
            )

            scenario_text = (
                scenario[
                    "text"
                ]
            )

            citation_audit = (
                audit_scenario_citations(
                    scenario_text,
                    chunk_index,
                )
            )

            temporal_audit = (
                audit_temporal_leakage(
                    scenario_text
                )
            )

            cited_context = (
                build_cited_evidence_context(
                    scenario_text,
                    chunk_index,
                )
            )

            semantic = (
                evaluate_finance_scenario_semantic(
                    bank_id=bank_id,
                    arm=arm,
                    scenario_id=scenario_id,
                    scenario_text=scenario_text,
                    bank_profile=bank_profile,
                    fixed_context=fixed_context,
                    cited_evidence_context=cited_context,
                    backend=backend,
                    model=model,
                )
            )

            # NEW: deterministic post-processing
            semantic = apply_finance_score_corrections(
                scenario_text=scenario_text,
                semantic=semantic,
                bank_id=bank_id,
            )

            # Recompute this AFTER corrections
            semantic["evidence_grounding_score"] = round(
                (
                    semantic.get("target_exposure_support", 0)
                    + semantic.get("trigger_support", 0)
                    + semantic.get("transmission_support", 0)
                ) / 3.0,
                3,
            )


            evaluated_scenarios.append({
                "scenario_id":
                    scenario_id,

                "scenario_text":
                    scenario_text,

                "citation_audit":
                    citation_audit,

                "temporal_audit":
                    temporal_audit,

                "semantic":
                    semantic,
            })

        abstained = (
            arm == "lens"
            and not scenarios
            and bool(
                re.search(
                    r"No evidence-supported "
                    r"alternative direction",
                    arm_text or "",
                    re.IGNORECASE,
                )
            )
        )

        evaluation[
            "arms"
        ][arm] = {
            "num_scenarios":
                len(
                    scenarios
                ),

            "abstained":
                abstained,

            "scenarios":
                evaluated_scenarios,

            "aggregate":
                aggregate_arm_metrics(
                    bank_id=
                        bank_id,

                    evaluated_scenarios=
                        evaluated_scenarios,

                    arm_text=
                        arm_text,
                ),
        }

    return evaluation


# ============================================================
# Flatten scenario-level CSV
# ============================================================

def flatten_scenario_rows(
    evaluation: Dict[str, Any],
) -> List[Dict[str, Any]]:

    rows = []

    bank_id = evaluation.get(
        "bank_id"
    )

    run_id = evaluation.get(
        "run_id"
    )

    for (
        arm,
        arm_data,
    ) in evaluation.get(
        "arms",
        {},
    ).items():

        scenarios = arm_data.get(
            "scenarios",
            [],
        )

        if not scenarios:
            rows.append({
                "bank_id":
                    bank_id,

                "run_id":
                    run_id,

                "arm":
                    arm,

                "scenario_id":
                    (
                        "ABSTAIN"
                        if arm_data.get(
                            "abstained"
                        )
                        else "NO_SCENARIO"
                    ),

                "abstained":
                    arm_data.get(
                        "abstained",
                        False,
                    ),
            })

            continue

        for scenario in scenarios:

            semantic = scenario.get(
                "semantic",{},
            )
            citation = scenario.get(
                "citation_audit",
                {},
            )
            temporal = scenario.get(
                "temporal_audit",
                {},
            )
            row = {
                "bank_id":
                    bank_id,

                "run_id":
                    run_id,

                "arm":
                    arm,

                "scenario_id":
                    scenario.get(
                        "scenario_id"
                    ),

                "target_exposure_support":
                    semantic.get(
                        "target_exposure_support"
                    ),

                "trigger_support":
                    semantic.get(
                        "trigger_support"
                    ),

                "transmission_support":
                    semantic.get(
                        "transmission_support"
                    ),

                "quantitative_support":
                    semantic.get(
                        "quantitative_support"
                    ),

                "bank_specificity":
                    semantic.get(
                        "bank_specificity"
                    ),

                "evidence_grounding_score":
                    semantic.get(
                        "evidence_grounding_score"
                    ),

                "mechanism_families":
                    json.dumps(
                        semantic.get(
                            "mechanism_families",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                "mechanism_summary":
                    semantic.get(
                        "mechanism_summary",
                        "",
                    ),

                "unsupported_exposures":
                    json.dumps(
                        semantic.get(
                            "unsupported_exposures",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                "unsupported_trigger_details":
                    json.dumps(
                        semantic.get(
                            "unsupported_trigger_details",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                "unsupported_transmission_details":
                    json.dumps(
                        semantic.get(
                            "unsupported_transmission_details",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                "unsupported_quantitative_details":
                    json.dumps(
                        semantic.get(
                            "unsupported_quantitative_details",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                "all_cited_chunk_ids_exist":
                    citation.get(
                        "all_cited_chunk_ids_exist"
                    ),

                "valid_exact_chunk_ids":
                    json.dumps(
                        citation.get(
                            "valid_exact_chunk_ids",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                "invalid_exact_chunk_ids":
                    json.dumps(
                        citation.get(
                            "invalid_exact_chunk_ids",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                "explicit_temporal_leakage":
                    temporal.get(
                        "explicit_temporal_leakage"
                    ),

                "hindsight_risk":
                    temporal.get(
                        "hindsight_risk"
                    ),

                "parse_failed":
                    semantic.get(
                        "parse_failed",
                        False,
                    ),

                "abstained":
                    False,
            }

            if bank_id == "A":
                coverage = semantic.get(
                    "mechanism_coverage",
                    {},
                )

                for channel in (
                    BANK_A_REALIZED_CHANNELS
                ):
                    row[
                        f"coverage_{channel}"
                    ] = coverage.get(
                        channel,
                        0,
                    )

            rows.append(
                row
            )

    return rows


# ============================================================
# Flatten arm-level summary CSV
# ============================================================

def flatten_arm_summary_rows(
    evaluation: Dict[str, Any],
) -> List[Dict[str, Any]]:

    rows = []

    bank_id = evaluation.get(
        "bank_id"
    )

    for (
        arm,
        arm_data,
    ) in evaluation.get(
        "arms",
        {},
    ).items():

        aggregate = arm_data.get(
            "aggregate",
            {},
        )

        row = {
            "bank_id":
                bank_id,

            "run_id":
                evaluation.get(
                    "run_id"
                ),

            "arm":
                arm,

            "num_scenarios":
                aggregate.get(
                    "num_scenarios"
                ),

            "abstained":
                arm_data.get(
                    "abstained",
                    False,
                ),

            "avg_target_exposure_support":
                aggregate.get(
                    "avg_target_exposure_support"
                ),

            "avg_trigger_support":
                aggregate.get(
                    "avg_trigger_support"
                ),

            "avg_transmission_support":
                aggregate.get(
                    "avg_transmission_support"
                ),

            "avg_quantitative_support":
                aggregate.get(
                    "avg_quantitative_support"
                ),

            "avg_bank_specificity":
                aggregate.get(
                    "avg_bank_specificity"
                ),

            "avg_evidence_grounding_score":
                aggregate.get(
                    "avg_evidence_grounding_score"
                ),

            "fully_grounded_scenarios":
                aggregate.get(
                    "fully_grounded_scenarios"
                ),

            "fully_grounded_rate":
                aggregate.get(
                    "fully_grounded_rate"
                ),

            "num_distinct_mechanism_families":
                aggregate.get(
                    "num_distinct_mechanism_families"
                ),

            "mechanism_family_counts":
                json.dumps(
                    aggregate.get(
                        "mechanism_family_counts",
                        {},
                    ),
                    ensure_ascii=False,
                ),

            "explicit_temporal_leakage":
                aggregate.get(
                    "temporal_audit",
                    {},
                ).get(
                    "explicit_temporal_leakage"
                ),

            "hindsight_risk":
                aggregate.get(
                    "temporal_audit",
                    {},
                ).get(
                    "hindsight_risk"
                ),
        }

        if bank_id == "A":

            row[
                "bank_a_realized_coverage_rate"
            ] = aggregate.get(
                "bank_a_realized_coverage_rate"
            )

            row[
                "bank_a_full_channel_hits"
            ] = aggregate.get(
                "bank_a_full_channel_hits"
            )

            row[
                "bank_a_partial_or_full_channel_hits"
            ] = aggregate.get(
                "bank_a_partial_or_full_channel_hits"
            )

        rows.append(
            row
        )

    return rows


# ============================================================
# CSV writer
# ============================================================

def write_csv(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


# ============================================================
# Load one TXT / MD
# ============================================================

def load_legacy_file(
    path: Path,
    bank_id: Optional[str] = None,
    run_id: Optional[int] = None,
) -> Dict[str, Any]:

    if path.suffix.lower() not in {
        ".txt",
        ".md",
    }:
        raise ValueError(
            "This version expects .txt or .md files."
        )

    inferred_bank, inferred_run = (
        infer_bank_run_from_filename(
            path
        )
    )

    resolved_bank = (
        bank_id
        or inferred_bank
    )

    resolved_run = (
        run_id
        if run_id is not None
        else inferred_run
    )

    if resolved_bank is None:
        raise ValueError(
            f"Could not infer bank from filename "
            f"{path.name}. "
            f"Use --bank-id A or --bank-id B, "
            f"or rename file like bank_a_run_01.txt."
        )

    if resolved_run is None:
        raise ValueError(
            f"Could not infer run_id from filename "
            f"{path.name}. "
            f"Use --run-id N, "
            f"or rename file like bank_a_run_01.txt."
        )

    return parse_finance_text_result(
        path=path,
        bank_id=resolved_bank,
        run_id=int(
            resolved_run
        ),
    )


# ============================================================
# Evaluate one file
# ============================================================

def evaluate_file(
    input_path: Path,
    output_dir: Path,
    backend: str,
    model: Optional[str],
    bank_id: Optional[str] = None,
    run_id: Optional[int] = None,
) -> Dict[str, Any]:

    print(
        "\n========================================"
    )
    print(
        f"Evaluating: {input_path.name}"
    )
    print(
        "========================================"
    )

    result = load_legacy_file(
        path=input_path,
        bank_id=bank_id,
        run_id=run_id,
    )

    evaluation = (
        evaluate_finance_result(
            result=result,
            backend=backend,
            model=model,
        )
    )

    output_path = (
        output_dir
        / f"{input_path.stem}_evaluation.json"
    )

    save_json(
        evaluation,
        output_path,
    )

    print(
        f"Saved evaluation JSON: "
        f"{output_path}"
    )

    return evaluation


# ============================================================
# Batch runner
# ============================================================

def run_batch(
    input_paths: List[Path],
    output_dir: Path,
    backend: str,
    model: Optional[str],
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_scenario_rows = []
    all_summary_rows = []

    for input_path in input_paths:

        evaluation = evaluate_file(
            input_path=input_path,
            output_dir=output_dir,
            backend=backend,
            model=model,
        )

        all_scenario_rows.extend(
            flatten_scenario_rows(
                evaluation
            )
        )

        all_summary_rows.extend(
            flatten_arm_summary_rows(
                evaluation
            )
        )

    scenario_csv = (
        output_dir
        / "finance_evaluation_scenarios.csv"
    )

    summary_csv = (
        output_dir
        / "finance_evaluation_arm_summary.csv"
    )

    write_csv(
        all_scenario_rows,
        scenario_csv,
    )

    write_csv(
        all_summary_rows,
        summary_csv,
    )

    print(
        "\n========================================"
    )
    print(
        "FINANCE EVALUATION COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        f"Scenario-level CSV: {scenario_csv}"
    )

    print(
        f"Arm-level summary CSV: {summary_csv}"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Finance evaluator for legacy "
            "FINANCE CASE COMPLETE TXT/MD logs."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Single .txt/.md result file."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing result TXT/MD files."
        ),
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="bank_*_run_*.txt",
        help=(
            "Glob pattern for --input-dir. "
            "Default: bank_*_run_*.txt"
        ),
    )

    parser.add_argument(
        "--bank-id",
        type=str,
        choices=[
            "A",
            "B",
        ],
        default=None,
        help=(
            "For a single legacy file if bank "
            "cannot be inferred from filename."
        ),
    )

    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help=(
            "For a single legacy file if run "
            "cannot be inferred from filename."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="local_ollama",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:12b",
    )

    args = parser.parse_args()

    if (
        args.input is None
        and args.input_dir is None
    ):
        parser.error(
            "Provide --input or --input-dir."
        )

    if (
        args.input is not None
        and args.input_dir is not None
    ):
        parser.error(
            "Use either --input or --input-dir, "
            "not both."
        )

    # --------------------------
    # Single file
    # --------------------------

    if args.input is not None:

        evaluation = evaluate_file(
            input_path=args.input,
            output_dir=args.output_dir,
            backend=args.backend,
            model=args.model,
            bank_id=args.bank_id,
            run_id=args.run_id,
        )

        scenario_rows = (
            flatten_scenario_rows(
                evaluation
            )
        )

        summary_rows = (
            flatten_arm_summary_rows(
                evaluation
            )
        )

        write_csv(
            scenario_rows,
            args.output_dir
            / "finance_evaluation_scenarios.csv",
        )

        write_csv(
            summary_rows,
            args.output_dir
            / "finance_evaluation_arm_summary.csv",
        )

        return

    # --------------------------
    # Batch
    # --------------------------

    input_paths = sorted(
        args.input_dir.glob(
            args.pattern
        )
    )

    # Also allow md if default txt pattern
    if (
        not input_paths
        and args.pattern
        == "bank_*_run_*.txt"
    ):
        input_paths = sorted(
            list(
                args.input_dir.glob(
                    "bank_*_run_*.txt"
                )
            )
            +
            list(
                args.input_dir.glob(
                    "bank_*_run_*.md"
                )
            )
        )

    if not input_paths:
        raise FileNotFoundError(
            f"No matching TXT/MD files found "
            f"in {args.input_dir}"
        )

    run_batch(
        input_paths=input_paths,
        output_dir=args.output_dir,
        backend=args.backend,
        model=args.model,
    )


if __name__ == "__main__":
    main()
