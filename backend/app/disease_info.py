"""Disease metadata, agronomic notes, and classification labels.

Provides structured disease information including explicit disease name, pathogen,
crop identification, symptoms, severity, and agronomic management advice.
"""

LABELS = {
    "Blight": "Northern Leaf Blight",
    "Common_Rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Healthy": "Healthy",
    "Not_Leaf": "Not a corn leaf",
    "Not_Corn": "Not a corn leaf",
}

NON_MAIZE_CLASSES = {"Not_Leaf", "Not_Corn", "Non_Corn", "Non_Maize", "Other", "Background"}


def canon_class(raw_name: str) -> str:
    """Normalize dataset or model class strings to standard canonical keys."""
    k = raw_name.strip()
    low = k.lower().replace("-", "_")
    if any(t in low for t in ["not_corn", "not_leaf", "non_corn", "non_maize", "background", "other", "negative"]):
        return "Not_Leaf"
    if "rust" in low:
        return "Common_Rust"
    if "blight" in low:
        return "Blight"
    if any(t in low for t in ["gray", "grey", "cercospora", "spot"]):
        return "Gray_Leaf_Spot"
    if "healthy" in low:
        return "Healthy"
    return k


INFO = {
    "Blight": {
        "title": "Northern Leaf Blight",
        "disease_name": "Northern Leaf Blight",
        "crop_name": "Maize (Corn)",
        "pathogen": "Exserohilum turcicum (Fungus)",
        "severity": "Moderate to High",
        "status": "Diseased",
        "summary": "A major fungal foliar disease of maize that produces long, elliptical grayish-green or tan lesions, leading to extensive canopy necrosis and potential yield loss.",
        "symptoms": [
            "Long, elliptical, cigar-shaped gray-green to tan lesions (2.5 to 15 cm)",
            "Lesions develop initially on lower leaves and spread upward to upper canopy",
            "Under humid conditions, dark fungal sporulation appears inside the lesions",
            "Multiple lesions can coalesce, resulting in complete blighting of leaf tissue",
        ],
        "advice": [
            "Scout adjacent corn rows to evaluate incidence and vertical spread toward the ear leaf",
            "Consult local agronomic extension guidelines for economic threshold fungicide sprays (e.g. strobilurins / triazoles)",
            "Select resistant maize hybrids for subsequent planting cycles",
            "Incorporate crop rotation and deep tillage of infected crop residues to break pathogen overwintering",
        ],
    },
    "Common_Rust": {
        "title": "Common Rust",
        "disease_name": "Common Rust",
        "crop_name": "Maize (Corn)",
        "pathogen": "Puccinia sorghi (Fungus)",
        "severity": "Moderate",
        "status": "Diseased",
        "summary": "A widespread fungal rust disease characterized by prominent reddish-brown powdery pustules on both upper and lower leaf surfaces.",
        "symptoms": [
            "Small, circular to elongated cinnamon-brown pustules scattered across both leaf surfaces",
            "Pustules rupture the leaf epidermis, releasing powdery reddish-brown urediniospores",
            "Late in the season, pustules turn dark brownish-black as overwintering teliospores develop",
            "Severe infections cause chlorosis, premature leaf desiccation, and reduced grain fill",
        ],
        "advice": [
            "Monitor weather forecasts; disease spreads rapidly under cool (16-25°C), high-humidity conditions",
            "Assess whether pustules are reaching the ear leaf before tassel emergence",
            "Fungicide treatment may be warranted on high-value seed corn or susceptible sweet corn hybrids",
            "Plant resistant maize cultivars featuring specific Rp resistance genes in rust-prone areas",
        ],
    },
    "Gray_Leaf_Spot": {
        "title": "Gray Leaf Spot",
        "disease_name": "Gray Leaf Spot",
        "crop_name": "Maize (Corn)",
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "severity": "High",
        "status": "Diseased",
        "summary": "One of the most destructive foliar diseases in continuous-corn fields, producing distinct rectangular lesions strictly bounded by parallel leaf veins.",
        "symptoms": [
            "Small, tan spots with yellow halos that elongate into narrow, rectangular lesions",
            "Lesions are sharply delimited by parallel leaf veins, creating a blocky rectangular shape",
            "Mature lesions turn grayish-tan as fungal spores are produced under prolonged dew or high humidity",
            "Extensive leaf blighting leads to stalk lodging, stalk rot, and severe grain weight reduction",
        ],
        "advice": [
            "Avoid continuous corn monoculture; rotate with non-host crops like soybeans for at least 1-2 seasons",
            "Evaluate fungicide application between VT (tasseling) and R2 (blister stage) if lesions reach the ear leaf",
            "Utilize conservation tillage practices that promote residue breakdown while controlling erosion",
            "Choose hybrids with high multi-genic tolerance to Cercospora zeae-maydis",
        ],
    },
    "Healthy": {
        "title": "Healthy Maize Leaf",
        "disease_name": "None (Healthy)",
        "crop_name": "Maize (Corn)",
        "pathogen": "None detected",
        "severity": "None",
        "status": "Healthy",
        "summary": "No signs of target foliar fungal diseases (Northern Leaf Blight, Common Rust, or Gray Leaf Spot) were detected in this maize leaf photo.",
        "symptoms": [
            "Uniform green leaf pigmentation without necrotic lesions, pustules, or vein-delimited spots",
            "Normal photosynthetic leaf blade architecture",
        ],
        "advice": [
            "Maintain routine field scouting at weekly intervals, especially during vegetative V6 through reproductive R3 stages",
            "Ensure balanced nitrogen and potassium fertilization to support plant disease defense",
            "Re-check with a new close-up photo if symptoms or discoloration develop later in the season",
        ],
    },
    "Not_Corn": {
        "title": "Not a Maize Leaf",
        "disease_name": "Not a Maize Leaf",
        "crop_name": "Non-Maize / Unrecognized",
        "pathogen": "N/A",
        "severity": "N/A",
        "status": "Not Maize",
        "summary": "The uploaded photo does not exhibit the characteristic visual patterns or foliage features of a maize (corn) leaf. Corn disease diagnosis cannot be reliably performed on non-maize imagery.",
        "symptoms": [],
        "advice": [
            "Ensure the photograph captures a single maize (corn) leaf clearly in natural daytime lighting",
            "Fill the camera frame primarily with the corn leaf blade, avoiding fingers, shadows, soil, or background crops",
            "This model is specifically trained for maize (Zea mays); photos of weeds, other crops, or non-plant objects are rejected",
        ],
    },
}

# Alias Not_Leaf to Not_Corn
INFO["Not_Leaf"] = INFO["Not_Corn"]


def get_disease_info(class_name: str) -> dict:
    """Retrieve full structured disease metadata with guaranteed safe fallbacks."""
    key = canon_class(class_name)
    if key in INFO:
        return INFO[key]
    # Fallback for unrecognized classes
    return {
        "title": LABELS.get(key, key),
        "disease_name": LABELS.get(key, key),
        "crop_name": "Maize (Corn)" if key not in NON_MAIZE_CLASSES else "Non-Maize / Unrecognized",
        "pathogen": "Unknown / Under evaluation",
        "severity": "Under evaluation",
        "status": "Not Maize" if key in NON_MAIZE_CLASSES else "Diseased",
        "summary": f"Diagnosis result: {key}.",
        "symptoms": [],
        "advice": [
            "Confirm diagnosis with a local agricultural extension specialist",
            "Ensure image is sharp and well-lit",
        ],
    }
