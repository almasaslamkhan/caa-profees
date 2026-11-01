# Time Statement Extraction

Your task is to:
1. Detect whether any clinician time statement is present.
2. Extract:
- The **exact time reference string(s)** with surrounding context as written in the note
- **The final total E/M time in minutes**

## Core Rules for Time Selection

1) Prefer explicitly stated totals
- If the note contains a phrase such as “A total of X minutes was spent on this visit”, “Total time spent”, or equivalent,
always use that value as the total E/M time unless it is clearly labeled as preventive-only time.
- Do not override an explicit total with a smaller component (e.g., “time spent with patient”).

2) Handling breakdowns
- If multiple time components are listed:
- **Sum all E/M-related components** (e.g., face-to-face time, documentation, chart review, care coordination).
- Do not use only face-to-face time unless it is explicitly stated to be the total.
- If a breakdown includes preventive medicine time, exclude preventive time from the total.

3) Preventive vs E/M
- If both preventive care time and E/M office visit time are mentioned:
- **ONLY extract the E/M office visit time.**
- If a global total includes preventive time, subtract the preventive portion.

4) Priority order for determining total time
Use the first applicable rule:
1. Explicit total E/M time statement
2. Explicit total visit time minus preventive time
3. Sum of all E/M-related components
4. Explicit E/M-only time statement

5) Ignore non-clinical time references
- Do NOT extract:
- E/M billing codes or descriptors (e.g., “99213 xx–xx minutes”)
- Appointment length or scheduled time
- Generic statements without numbers (e.g., “lengthy discussion”)

6) De-identified time
- If the time is de-identified, return the exact text string as written and set total minutes to 0.

7) No valid time
- If no valid clinician time is present, return:
- Empty string for time reference
- 0 minutes

## Output Requirements
- Return only:
- Extracted time reference string(s) with context
- Final total time in minutes
- Do not include explanations, or formatting artifacts.
- Output must be clear, human-readable, and contain no invalid characters (e.g., <br>).
- Each justification should be very concise and limited to 1 sentence. You only need to mention the total time.
- Examples of concise justifications: A total of 32 minutes was spent on office visit.
