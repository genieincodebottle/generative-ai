# Prompt Chaining related Prompt
MEDICAL_REPORT_STEPS = [
    """Extract key clinical findings from the medical report.
    Format as:
    - Vital Signs: List all vital measurements
    - Symptoms: List reported symptoms
    - Observations: List clinical observations
    - Tests: List any tests mentioned
    - Medications: List any medications discussed""",

    """Analyze the findings and provide a structured summary:
    1. Primary Concerns
    2. Risk Factors
    3. Recommended Actions
    4. Follow-up Requirements""",

    """Generate a standardized report format with clear section spacing:

# Medical Report Analysis

## Clinical Summary
[Provide a concise summary of the key findings and primary medical concerns]

## Risk Assessment
[Detailed evaluation of patient risks and contributing factors]

## Recommendations
[Clear list of recommended actions and interventions]

## Follow-up Plan
[Specific follow-up steps and timeline]

Note: Ensure each section has proper spacing and formatting."""
]

# Parallelization related Prompt
MEDICAL_REVIEW_PROMPTS = [
    """Review for Patient Safety:
    Evaluate this medical content specifically for patient safety concerns.
    Consider potential risks, contraindications, and safety protocols.
    Flag any issues that could compromise patient safety.
    
    Respond with either SAFE or FLAG_SAFETY followed by specific concerns.""",
    
    """Review for Clinical Accuracy:
    Evaluate this medical content for clinical accuracy and evidence-based practice.
    Check alignment with current medical guidelines and standards of care.
    Identify any outdated or incorrect medical information.
    
    Respond with either ACCURATE or FLAG_ACCURACY followed by specific issues.""",
    
    """Review for Regulatory Compliance:
    Evaluate this medical content for compliance with healthcare regulations.
    Consider HIPAA, documentation requirements, and care standards.
    Check for proper role delegation and scope of practice.
    
    Respond with either COMPLIANT or FLAG_COMPLIANCE followed by specific violations.""",
    
    """Review for Implementation Risk:
    Evaluate this medical content for practical implementation risks.
    Consider workflow integration, resource requirements, and staff training needs.
    Identify potential operational challenges or system constraints.
    
    Respond with either LOW_RISK or FLAG_RISK followed by specific concerns.""",
    
    """Review for Ethical Considerations:
    Evaluate this medical content for ethical implications.
    Consider patient autonomy, informed consent, and equity of care.
    Check for potential ethical dilemmas or conflicts.
    
    Respond with either ETHICAL or FLAG_ETHICS followed by specific issues."""
]

# Evaluation and Optimizer related Prompt
MEDICAL_EVALUATOR_PROMPT = """
Evaluate this medical analysis for:
1. Clinical accuracy and completeness
2. Evidence-based reasoning
3. Clear communication and documentation
4. Patient safety considerations
5. Appropriate recommendations

You should be evaluating only and not attempting to provide medical advice.
Only output "PASS" if all criteria are met and you have no further suggestions for improvements.
Output your evaluation concisely in the following format:

<evaluation>PASS, NEEDS_IMPROVEMENT, or FAIL</evaluation>
<feedback>
What aspects need improvement and why, focusing on clinical relevance and patient care.
</feedback>
"""

MEDICAL_GENERATOR_PROMPT = """
Your goal is to provide a comprehensive medical analysis based on the given task. 
If there is feedback from previous analyses, incorporate it to improve patient care and clinical accuracy.

Output your analysis in the following format:

<thoughts>
[Your clinical reasoning process and how you plan to address any feedback]
</thoughts>

<response>
[Your structured medical analysis and recommendations]
</response>
"""

# Orchestrator related Prompt
MEDICAL_ORCHESTRATOR_PROMPT = """
    You are a medical orchestrator analyzing a patient case to determine necessary specialized analyses.
    Break down this case into specific analytical tasks that will provide comprehensive evaluation.

    Medical Case: {case}
    Clinical Context: {context}

    For each aspect of the case that needs analysis, create a specialized task following these guidelines:
    1. Make tasks specific to this case's unique needs
    2. Ensure tasks are independent enough to be processed separately
    3. Each task should contribute to overall case understanding

    Return your response in this EXACT format:

    <analysis>
    [Provide your understanding of the case and rationale for the tasks you've chosen.
    Explain why these specific analyses are needed for this case.]
    </analysis>

    <tasks>
        <task>
        <type>cardiac_evaluation</type>
        <description>Evaluate cardiovascular symptoms and findings</description>
        <requirements>Cardiology expertise, ECG interpretation skills</requirements>
        <priority>high</priority>
        </task>
        [Additional tasks as needed...]
    </tasks>
    """

MEDICAL_WORKER_PROMPT = """
Generate specialized medical analysis based on:
Case: {original_task}
Analysis Type: {task_type}
Focus Area: {task_description}

Consider all relevant clinical guidelines and best practices for {task_type}.
Ensure recommendations are evidence-based and patient-centered.

Return your response in this format:

<response>
Your detailed medical analysis here, maintaining clinical accuracy and addressing specific requirements.
Include evidence-based recommendations where appropriate.
</response>
"""

# Tool Calling related Prompt
MEDICAL_TOOL_PROMPT = """
Based on the patient case, determine which medical tool function to execute.

Available functions:
1. update_patient_record(patient_id: str, data: dict)
   - Updates patient record with new clinical data
   - Example: {{"patient_id": "P123", "data": {{"symptoms": ["fever"], "vitals": {{"bp": "120/80"}}}}}}

2. schedule_appointment(patient_id: str, department: str, urgency: str)
   - Schedules patient appointment
   - Example: {{"patient_id": "P123", "department": "cardiology", "urgency": "urgent"}}

3. order_lab_test(patient_id: str, test_type: str)
   - Orders specific laboratory test
   - Example: {{"patient_id": "P123", "test_type": "complete_blood_count"}}

4. refer_specialist(patient_id: str, specialty: str)
   - Refers patient to a specialist
   - Example: {{"patient_id": "P123", "specialty": "cardiology"}}

Analyze the case and select ONE appropriate function. Return EXACTLY in this format:

<analysis>
Brief explanation of your selection based on the patient's needs.
</analysis>

<tool_selection>
{{"function": "function_name", "args": {{"patient_id": "{patient_id}", "param_name": "value"}}}}
</tool_selection>

IMPORTANT: The tool_selection must be valid JSON with NO line breaks.

Patient Case: {case}
"""