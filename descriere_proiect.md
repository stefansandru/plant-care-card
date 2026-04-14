# Identificare și Îngrijire Plante API (Plant Care RAG)

## Ce problemă rezolvă?
Deseori, oamenii au plante de apartament sau de grădină pentru care nu știu cum să le ofere îngrijirea adecvată, sau chiar le uită numele. 
Această aplicație rezolvă problema extrem de simplu: **faci o poză plantei**, iar sistemul o identifică automat și îți generează un card complet de îngrijire, oferindu-ți indicații clare despre câtă lumină, apă sau umiditate are nevoie, precum și detalii despre toxicitate sau boli frecvente.

---

## 📥 Input (Intrarea datelor)
Endpoint-ul `/api/v1/plant-care` așteaptă un request de tip `POST` care să conțină o imagine (trimisă ca `multipart/form-data`).

**Tipuri de imagini suportate:**
- `.jpg` / `.jpeg` (`image/jpeg`)
- `.png` (`image/png`)
- `.webp` (`image/webp`)
- `.bmp` (`image/bmp`)
- `.tiff` (`image/tiff`)

---

## 📤 Output (Ieșirea datelor)
Dacă imaginea este procesată cu succes, aplicația returnează un **JSON** complex care conține atât predicția modelului de imagine, cât și cardul detaliat de îngrijire (generat de LLM-ul Mistral AI).

### Structura JSON:
```json
{
  "error": false,
  "classification": {
    "label": "soybeans",  // Numele plantei identificat în poză
    "confidence": 0.87,   // Precizia predicției
    "top_labels": ["soybeans", "orange", "ginger"],
    "top_confidences": [0.87, 0.08, 0.05]
  },
  "plant_care_card": {
    "common_name": "Soybean",
    "scientific_name": "Glycine max",
    "plant_family": "Fabaceae",
    "native_habitat": "Asia de Est",
    "care_requirements": {
      "light": {
        "level": "BRIGHT_DIRECT",
        "description": "Necesită soare plin, 6-8 ore pe zi pentru o creștere optimă."
      },
      "water": {
        "frequency": "MODERATE",
        "instructions": "Udați constant, lăsând primii 2-3 cm de sol să se usuce între udări."
      },
      "soil": {
        "type": "WELL_DRAINING",
        "ph_range": "6.0 - 6.8"
      },
      "temperature_celsius": {
        "min": 15,
        "max": 30,
        "ideal": 25
      },
      "humidity": "MODERATE"
    },
    "health_and_safety": {
      "toxicity": "NON_TOXIC",
      "pests_and_diseases": ["Afide", "Acarieni", "Boli fungice foliare"]
    },
    "growth_characteristics": {
      "mature_size": "Micuță până la medie, în funcție de varietate.",
      "growth_rate": "FAST"
    },
    "additional_tips": "Asigurați-vă că nu udati frunzele direct în soare puternic pentru a evita arsurile."
  }
}
```

---

## 🧠 De ce am ales modelul EfficientNet-B1 (CNN)?
Aplicația folosește modelul pre-antrenat **EfficientNet-B1** pentru recunoașterea inițială a plantei din imagine, înainte de a rula asistentul inteligent LLM (Mistral). Această decizie a fost bazată pe un compromis optim între timpul de rulare (memoria consumată) și acuratețea obținută. 

Așa cum se vede în tabelul comparativ de mai jos în urma testelor, **EfficientNet-B1** a obținut o acuratețe excelentă (peste 90%) având o rețea mult mai mică din punct de vedere arhitectural (număr de parametri) comparativ cu modele grele precum ResNet18 sau EfficientNet B3:

| Model | Număr Parametri (Milioane) | Acuratețe |
| :--- | :---: | :---: |
| MobileNetV3 | 5.5 | 85.4% |
| EfficientNet B0 | 4.0 | 85.6% |
| **EfficientNet B1** | **6.6** | **90.2%** |
| EfficientNet B2 | 7.7 | 91.7% |
| EfficientNet B3 | 10.7 | 86.5% |
| ResNet18 | 11.2 | 84.2% |

Deși **B2** are o acuratețe marginal mai bună cu ~~1.5%~~, am ales **B1** pentru că folosește cu aproximativ *1.1M* mai puțini parametri, facilitând generarea de predicții extrem de rapide pentru request-urile pe API fără pierderi semnificative de calitate.
