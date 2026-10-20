MEDICAL_SYSTEM_PROMPT = """You are a careful assistant that provides general health information.
You are not a doctor. Never diagnose a disease, predict that the user has a disease, or
claim that a symptom pattern confirms a condition. When helpful, describe a short,
clearly labeled list of possible causes as general information, explain that symptoms
overlap and a clinician must assess the user, and give low-risk precautions and clear
red flags for seeking care. Do not prescribe medication or recommend changing a
prescription. Do not claim that a medical reference was consulted unless its verified
content was supplied to you. Encourage the user to consult a qualified healthcare
professional for personal medical decisions. If the user describes a possible medical
emergency, tell them to contact their local emergency services immediately. Be
empathetic, clear, and concise, and ask for relevant context when needed. For images,
describe only visible, non-diagnostic details. Never diagnose or predict disease from
a photo, radiograph, scan, or other medical image; say that such images need review by
a qualified clinician. You may read visible labels or document text, noting uncertainty."""

EMERGENCY_KEYWORDS = (
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "shortness of breath",
    "chest pain",
    "chest pressure",
    "stroke",
    "face drooping",
    "severe bleeding",
    "unconscious",
    "overdose",
    "suicidal",
    "anaphylaxis",
)

OFFLINE_GUIDANCE = {
    "headache": (
        "Headaches can have many causes, and I can’t determine the cause for you. "
        "Resting, drinking water, and noting possible triggers may help. Seek urgent "
        "care for a sudden severe headache, head injury, confusion, weakness, vision "
        "changes, or fever with a stiff neck. A clinician or pharmacist can advise "
        "whether a pain reliever is appropriate for you."
    ),
    "fever": (
        "A fever can occur with many illnesses. Rest, drink fluids, and monitor your "
        "temperature. Contact a clinician if symptoms are severe, worsening, persistent, "
        "or you are concerned; seek urgent care for trouble breathing, confusion, "
        "dehydration, a stiff neck, or a seizure. Advice differs for infants and "
        "people with significant medical conditions."
    ),
    "cough": (
        "Coughs have many possible causes, including infections and allergies. Rest, "
        "fluids, and avoiding smoke may help. Seek urgent care for difficulty breathing, "
        "blue lips, chest pain, confusion, or coughing blood. Contact a clinician if "
        "the cough is worsening or not improving."
    ),
    "sore throat": (
        "A sore throat can have several causes and this chat cannot diagnose it. Fluids "
        "and rest may ease discomfort. Seek urgent care for difficulty breathing or "
        "swallowing, drooling, severe dehydration, or rapid worsening. Ask a clinician "
        "if symptoms are severe or persist."
    ),
    "rash": (
        "Rashes have many possible causes, so I can’t identify one from a description. "
        "Avoid suspected irritants and note when it started. Seek emergency care for "
        "trouble breathing, facial or tongue swelling, blistering or peeling skin, or "
        "a rapidly spreading rash with fever. A clinician can assess a new, painful, "
        "or worsening rash."
    ),
    "stomach": (
        "Stomach discomfort has many possible causes. Sip fluids and rest if symptoms "
        "are mild. Seek urgent care for severe or worsening pain, fainting, blood in "
        "vomit or stool, a rigid abdomen, or signs of dehydration. Contact a clinician "
        "if symptoms persist or you are concerned."
    ),
    "sleep": (
        "A regular sleep and wake schedule, a relaxing bedtime routine, and reducing "
        "caffeine late in the day may support sleep. Persistent sleep problems can "
        "have medical causes; a healthcare professional can help assess them."
    ),
}


def offline_reply(message: str, has_upload: bool = False) -> str:
    text = message.casefold()
    if any(keyword in text for keyword in EMERGENCY_KEYWORDS):
        return (
            "This could be an emergency. Please contact your local emergency services "
            "now or go to the nearest emergency department. Do not wait for an online "
            "reply. If possible, ask someone nearby to stay with you."
        )

    for topic, guidance in OFFLINE_GUIDANCE.items():
        if topic in text:
            return f"General information (offline): {guidance}\n\nThis is not a diagnosis or a substitute for medical care."

    if has_upload:
        return (
            "I can extract text offline, but cannot visually interpret an uploaded "
            "image or assess symptoms from it. OCR text is an unverified reading; it "
            "does not confirm what the picture shows. X-rays and other medical scans "
            "need review by a qualified clinician or radiologist."
        )

    return (
        "I can share general health information, but offline mode has limited guidance "
        "and can’t assess individual symptoms or diagnose conditions. Try asking about "
        "a headache, fever, cough, sore throat, rash, stomach discomfort, or sleep. "
        "For personal medical advice, contact a qualified healthcare professional. "
        "For a possible emergency, contact your local emergency services."
    )