def build_prompt(name, age, gender, country, skin_type, diagnosis, risk_level):
    return f"""
    You are SkinCancerDetector, a dermatologist AI assistant. The user has provided the following details:
    - Name: {name}
    - Age: {age}
    - Gender: {gender}
    - Country of Residence: {country}
    - Skin Type: {skin_type}
    - Diagnosis: {diagnosis} ({risk_level})
    Here is an image of a mole. The AI model has diagnosed it as {diagnosis}.

    Attached is a Grad-CAM heatmap highlighting the high-risk areas of the mole. 
    - Please explain what the highlighted areas might indicate.

    Based on these details, provide personalized advice to the patient, including:
    - Risk factors for their skin type and age group.
    - UV protection recommendations based on their country.
    - General skincare tips to prevent skin cancer.
    Keep the response concise and user-friendly.
    """