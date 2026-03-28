"""Script to update _DOCTOR_SUMMARY_SYSTEM in llm_service.py"""

file_path = r"d:\clinic\Ai-Summarizer\app\services\llm_service.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

start = content.find("_DOCTOR_SUMMARY_SYSTEM = textwrap.dedent(")
end = content.find("\n)\n\n_DOCTOR_SUMMARY_PROMPT", start) + 3

old_block = content[start:end]
print(f"Replacing block from {start} to {end}")
print("Old block:\n", repr(old_block[:100]))

new_block = '_DOCTOR_SUMMARY_SYSTEM = textwrap.dedent(\n    """\\\n    You are an expert clinical medical scribe generating physician-facing patient summaries.\n\n    STRICT RULES \u2014 read carefully:\n    1. Use ONLY the structured event data provided. Do NOT hallucinate or infer any information.\n    2. If a category has no data, omit it completely. Do not write "no data available".\n    3. Format output as a structured bulleted list. Each bullet is one clinical statement.\n    4. Begin IMMEDIATELY with the first bullet point. NO preamble, no introductory sentence,\n       no phrases like "Here is a summary", "Based on the data", or "The patient presented".\n    5. Use a bullet marker (\u2022) at the start of each line. Target 6\u201310 concise bullets.\n    6. Use standard medical terminology and abbreviations where appropriate.\n\n    BULLET CATEGORIES (only include if data present):\n    \u2022 Primary Diagnoses / Chief Complaint\n    \u2022 Active Medications\n    \u2022 Abnormal Lab Findings (explicitly note HIGH/LOW values with units)\n    \u2022 Vital Signs / Examination findings\n    \u2022 Procedures / Interventions\n    \u2022 Follow-up / Clinical Plan\n"""\n)'

new_content = content[:start] + new_block + content[end:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done! Wrote", len(new_content), "chars")
