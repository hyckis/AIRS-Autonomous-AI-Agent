import json
import re

def extract_json(text):
    text = text.strip()

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
        except json.JSONDecodeError: "No json obj found"
    
    raise ValueError("No json obj found")