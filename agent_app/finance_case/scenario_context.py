from pathlib import Path
import json
from .finance_corpus import load_bank_profile

# paths
BASE_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = BASE_DIR / "finance_corpus" / "processed" / "fed_scenarios"
EXPECTED_YEARS = list(range(2015, 2024))

# scenario loading
def load_scenario(year):
    year = int(year)
    path = SCENARIO_DIR / f"{year}.json"
    if not path.exists(): raise FileNotFoundError(f"Scenario file not found: {path}")
    with open(path, "r", encoding="utf-8") as f: scenario = json.load(f)
    return scenario

def load_fed_scenarios(years=None):
    if years is None: years = EXPECTED_YEARS
    scenarios = []
    for year in sorted(years):
        scenario = load_scenario(year)
        scenarios.append(scenario)
    return scenarios


# Formatting helpers
def _format_value(value):
    if isinstance(value, list): return "\n".join(f"- {item}" for item in value)
    if isinstance(value, dict): return "\n".join(f"- {key}: {val}" for key, val in value.items())
    return str(value)

def format_bank_profile(profile):
    lines = []
    bank_id = profile.get("bank_id", "Unknown Bank")
    as_of_date = profile.get("as_of_date", "Unknown")
    lines.append(f"Bank: {bank_id}")
    lines.append(f"As of date: {as_of_date}")

    # Balance sheet
    balance_sheet = profile.get("balance_sheet", {})
    if balance_sheet: 
        lines.append("\nBalance Sheet:")
        for key, value in balance_sheet.items(): lines.append(f"- {key}: {value}")

    # Securities
    securities = profile.get("securities", {})
    if securities:
        lines.append("\nSecurities:")
        for category, values in securities.items():
            lines.append(f"- {category}:")
            if isinstance(values, dict):
                for key, value in values.items(): lines.append(f"  - {key}: {value}")
            else: lines.append(f"  - {values}")

    # Deposits
    deposits = profile.get("deposits", {})
    if deposits:
        lines.append("\nDeposits:")
        for key, value in deposits.items():
            if isinstance(value, list):
                lines.append(f"- {key}:")
                for item in value: lines.append(f"  - {item}")
            else: lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def format_scenario(scenario):
    year = scenario.get("year", "Unknown")
    scenario_type = scenario.get("scenario_type", "Severely Adverse")
    scenario_period = scenario.get("scenario_period", "Unknown")

    # Allow either naming convention
    narrative = (
        scenario.get("core_narrative")
        or scenario.get("narrative")
        or scenario.get("scenario_narrative", "")
    )
    summary = scenario.get("summary", {})
    additional_context = scenario.get("additional_context", [])
    lines = [
        f"### {year} {scenario_type} Scenario",
        f"Scenario period: {scenario_period}",
    ]

    if narrative: lines.extend(["", "Narrative:", narrative.strip(),])
    if summary: 
        lines.extend(["", "Key Variables:"])
        # Preferred ordering
        preferred_keys = [
            "unemployment_peak_pct",
            "real_gdp_trough_pct",
            "real_gdp_trough_reference",
            "treasury_3m_start_pct",
            "treasury_3m_end_pct",
            "treasury_10y_start_pct",
            "treasury_10y_end_pct",
            "equity_drawdown_pct",
            "house_price_drawdown_pct",
            "cre_price_drawdown_pct",
        ]
        used_keys = set()

        for key in preferred_keys:
            if key in summary:
                lines.append(f"- {key}: {summary[key]}")
                used_keys.add(key)

        # Keep any additional summary variables
        for key, value in summary.items():
            if key not in used_keys: lines.append(f"- {key}: {value}")

    if additional_context:
        lines.extend(["", "Additional Context:"])

        if isinstance(additional_context, list):
            for item in additional_context: lines.append(f"- {item}")
        elif isinstance(additional_context, dict):
            for key, value in (additional_context.items()): lines.append(f"- {key}: {value}")
        else:
            lines.append(str(additional_context))

    return "\n".join(lines)


def format_scenarios(scenarios):
    return "\n\n".join(format_scenario(scenario) for scenario in scenarios)


# Build fixed context
def build_fixed_context(bank_id, years=None):
    bank_id = bank_id.upper()
    if bank_id not in {"A", "B"}: raise ValueError("bank_id must be 'A' or 'B'")
    profile = load_bank_profile(bank_id)
    scenarios = load_fed_scenarios(years=years)
    bank_text = format_bank_profile(profile)
    scenario_text = format_scenarios(scenarios)

    fixed_context = f"""
==================================================
BANK PROFILE
==================================================

{bank_text}


==================================================
FEDERAL RESERVE SUPERVISORY STRESS SCENARIOS
2015-2023 SEVERELY ADVERSE SCENARIOS
==================================================

{scenario_text}
""".strip()

    return fixed_context


# structured version for auditing
def get_fixed_context_components(bank_id, years=None):
    profile = load_bank_profile(bank_id)
    scenarios = load_fed_scenarios(years=years)
    return {
        "bank_id": bank_id.upper(),
        "bank_profile": profile,
        "scenario_years": [scenario.get("year") for scenario in scenarios],
        "scenarios": scenarios,
        "formatted_context": build_fixed_context(bank_id, years=years),
    }


# Manual test
if __name__ == "__main__":
    context = build_fixed_context("A")
    print(context)