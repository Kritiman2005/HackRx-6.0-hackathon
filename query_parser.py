import re

def parse_query(query: str):
    age_match = re.search(r'(\d+)[ -]?year[- ]?old|(\d+)[ ]?M|(\d+)[ ]?F', query)
    procedure_match = re.search(r'(?i)(surgery|operation|treatment|therapy|hospitalization|procedure)', query)
    location_match = re.search(r'(?i)(in\s+)?([A-Z][a-z]+)', query)
    duration_match = re.search(r'(\d+)[ -]?(month|year)[- ]?(old|policy)?', query)

    age = next((m for m in age_match.groups() if m), None) if age_match else None
    procedure = procedure_match.group(0) if procedure_match else None
    location = location_match.group(2) if location_match else None
    duration = duration_match.group(0) if duration_match else None

    return {
        "age": age,
        "procedure": procedure,
        "location": location,
        "policy_duration": duration
    }