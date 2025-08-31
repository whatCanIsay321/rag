def choose_strategy(question_type: str, query: str) -> str:
    if question_type == "Bridge":
        return "Sequential-Bridge"
    elif question_type == "Comparison":
        return "Decompose"
    elif question_type == "Aggregation":
        if "列出" in query or "几个" in query or any(c.isdigit() for c in query):
            return "Decompose"
        else:
            return "Sequential-Context"
    elif question_type == "Constraint":
        return "Decompose+Filter"
    elif question_type == "Hybrid":
        return "Decompose+Dispatch"
    else:
        return "Unknown"
