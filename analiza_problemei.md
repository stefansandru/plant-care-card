# Analiza Problemei: Identificare și Îngrijire Plante (Plant Care RAG API)

## 1. Analiza Literaturii (Articole Reprezentative)

Pentru a fundamenta abordarea aleasă (combinarea unei rețele neurale pentru clasificare vizuală cu un sistem RAG bazat pe LLM pentru generarea de conținut aplicat), au fost investigate articole academice și pachete de cercetare de pe arhiva **arXiv** și platforme precum **Papers with Code**:

### Articolul 1: "Plant leaf disease classification using EfficientNet deep learning model" (Atila et al., 2021)
* **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1574954120301321](https://www.sciencedirect.com/science/article/abs/pii/S1574954120301321)
* **Contribuția principală:** Evaluează și implementează exhaustiv arhitectura EfficientNet (versiunile B0-B7) special pentru ecosistemul botanic, folosind setul de date notoriu PlantVillage. Demonstrează superioritatea clară a acestui model în fața arhitecturilor tradiționale mari (precum VGG, ResNet) privind eficiența calculării.
* **Sumar al rezultatelor:** Modelul derivat (EfficientNet-B5/B4) a atins o acuratețe de peste 99% pe setul de antrenament și păstrează scoruri imense (98%+) chiar și adăugând noise pe imagini (situații din viața reală), având un număr mult redus de parametri.
* **Limitări ale abordării:** Fiind un model pur de viziune computerizată, depinde de o bază de date precis etichetată. Mai mult, algoritmul livrează către utilizator doar o simplă "etichetă" (ex: numele plantei sau bolii), lăsând utilizatorul să caute singur ce presupune acel diagnostic sau cum trebuie procedat în continuare.

### Articolul 2: "AgroLLM: A specialized LLM framework for Agricultural applications using RAG" (arXiv:2503.04788)
* **Link / Papers with Code:** [https://arxiv.org/abs/2503.04788](https://arxiv.org/abs/2503.04788) 
* **Contribuția principală:** Propune un cadru de asistență bazat pe inteligență artificială generativă care rezolvă problema "halucinațiilor" (răspunsuri false și încrezătoare din partea LLM-urilor generale). Abordarea implică arhitectura **RAG** (Retrieval-Augmented Generation) operată pe o bază de PDF-uri, manuale de botanică și articole de cercetare (prin Vector Databases precum FAISS).
* **Sumar al rezultatelor:** Sistemul (testat pe modele Mistral, Gemini și GPT-4) a demonstrat o rată ridicată de acuratețe, ajungând la o fidelitate factuală de 95%, răspunzând sigur și argumentat la comenzi complexe legate de fitotehnie sau îngrijirea culturilor.
* **Limitări ale abordării:** Interacțiunea este strict de natură text-chatbot. O persoană neexperimentată se poate lupta să descrie corect tehnic în limbaj text problema sau trăsăturile plantei pentru ca RAG-ul să funcționeze adecvat ("User-input error"). În plus, procesul complex de query vectorial crește simțitor timpii de latență.

### Articolul 3: "Farmer.Chat: A Generative AI-powered Advisory System" (arXiv:2409.08916)
* **Link:** [https://arxiv.org/abs/2409.08916](https://arxiv.org/abs/2409.08916)
* **Contribuția principală:** Un asistent inteligent care înglobează tehnologia RAG contextualizată. Conectează extragerea de informații din documente cu date în timp real (real-time data API), oferind o abordare multi-modală pentru o interacțiune îmbunătățită cu mediul limitat tehnologic.
* **Sumar al rezultatelor:** A redus mult bariera de acces la sfaturi de specialitate. S-a demonstrat că utilizatorii preferă sfaturi direcționate, ce țin cont de constrângeri reale, și s-a demonstrat valoarea deciziilor dirijate de o interogare RAG pe surse actualizate vs modele închise pre-antrenate.
* **Limitări ale abordării:** Complexitatea foarte mare de mentenanță și implementare și vulnerabilitatea imensă la conexiuni externe (în momentul în care pica vreun API terț de integrare – sistemul devenea inoperabil tehnic). 

---

## 2. Soluția Propusă: Aplicația `plant-care-card`

### Scurtă descriere / Schema
Aplicația dezvoltată reprezintă un API modern care combină flexibilitatea rețelelor neurale (de la Articolul 1) cu inteligența aplicată RAG (Articolul 2). Fluxul aplicației are loc în 2 pași majori interconectați:

1. **Modulul de Clasificare Vizuală:** Endpoint-ul API preia o imagine (upload poză cu planta de la utilizator). Aceasta trece prin modelul **PyTorch EfficientNet-B1**, care identifică și "pune eticheta" rapid pe plantă și calculează procentul de încredere (confidence).
2. **Modulul de Îngrijire & RAG (LangGraph Pipeline):** Numele plantei predicționate pornește automat un flux cu 3 noduri dirijat de baza pe grafe (`LangGraph`):
   - *Agent Researcher:* Caută web în timp real (prin `Tavily API`) cerințe detaliate despre acea plantă.
   - *Agent Generator:* Analizează rezultatele căutării via LLM (`Mistral AI`) și construiește cu precizie un dicționar de îngrijire dictat de un model strict `Pydantic` (`PlantCareCard` – necesar de udare, boli, soare etc.). 
   - *Agent Validator:* AI-ul se auto-verifică; dacă cardul generat are lipsuri critice, comandă rescrierea datelor, dacă este valid se returnează tot JSON-ul pe HTTP ca produs finit.

### Motivație - De ce este o soluție potrivită pentru problema abordată?
* **Zero efort descriptiv pentru utilizator:** Utilizatorul nu trebuie să cunoască nici măcar detalii vizuale, este suficientă **doar trimiterea unei simple poze**. Rețeaua CNN EfficientNet previne erorile de tipificare text prezente în soluțiile LLM clasice.
* **Tehnologie Dinamică, fără baze de date gigantice:** Din cauza cantității imense de specii de plante, mentenanța de către dezvoltatori a unui DataBase (ex. MongoDB, SQL) cu miile profiluri complete statice este o muncă asiduă. Sistemul nostru RAG generează *on-the-fly* aceste documente cu expertiză reală de pe web (Tavily + Mistral), fișele rămânând veșnic actualizate fără mentenanța unui backend clasic.
* **Eficiență în producție:** Folosirea unui model cu puțini parametri precum `EfficientNet-B1` (pentru care latența este infimă spre deosebire de B7) lăsând partea grea logică procesată Cloud (Mistral API API-calls). Schema formatată strict impusă Pydantic face output-urile JSON standardizate și imbatabile la afișarea într-un portal front-end din start.

---

## 3. Plan de lucru

### Ce seturi de date există? Cum se pot obține?
- Pentru antrenarea rețelei neurale am selectat un set de date vast și diversificat disponibil pe platforma Kaggle: **"yudhaislamisulistya/plants-type-datasets"**. Acesta conține mii de imagini structurate pe tipologii și specii variate de plante, ideal pentru un model general de recunoaștere.
- O altă alternativă consacrată în industrie — ideală dacă se dorește extinderea proiectului pe identificative de patologii (boli) — este **PlantVillage**, care include aproximativ 50.000 de fotografii detaliate ale frunzelor.
- **Obținere:** Seturile de imagini sunt descărcate automatizat prin API-ul `Kaggle` și stocate local pentru antrenamentul PyTorch. Partea RAG din aplicație nu cere un set de pre-antrenament descărcabil, bazându-se pe search engine-uri web dinamice (Tavily).

### Ce limbaje / framework-uri / tehnologii intenționez să folosesc?
- **Limbaje / Structură:** `Python 3` - limbaj imperios în mediul Machine Learning și web backend rapid asincron.
- **Tehnologii ML / Viziune computațională:** Cadrul de tensor calcul `PyTorch` (torch, torchvision) pentru inferența pre-antrenată de `EfficientNet`, ajutat de `Pillow`/`NumPy` pentru maparea pixelilor (preprocessing-ul imaginii).
- **Arhitectură RAG / LLM:** Librăriile `LangChain` și framework-ul `LangGraph` pentru managerierea stărilor și a ciclurilor de verificare automată "Agent-based". Integrarea va fi făcută cu **API Mistral AI** ca model de core reasoning, și **Tavily API** pentru RAG/Research Web online complet. De asemenea modulul de validare se bazează pe abstractizarea oferită de `Pydantic`.
- **Infrastructură Endpoint:** `FastAPI` montat pe un server de producție `Uvicorn`/`Gunicorn` pentru returnarea responsivă asincronă. Tot spațiul funcțional va fi izolat standardizat via `Docker`, conținând și permisiile necesare asamblării pe instanțe CPU+GPU.

### Cum evaluez performanța modelului?
- **Componenta Computer Vision (EfficientNet Model):**
  Alegând un sample Test set, se vizează calcularea metricilor esențiale: `Acuratețea/Accuracy globală` ratei de detecție din inferențe izolate, cuplat cu extragerea `Matricei de Confuzie` (Confusion matrix) pt a monitoriza precis predicțiile eronate între specii de plante asemănătoare vizual. De asemenea, măsurarea performanței pur computațională (Latency per Inference testat sub 100ms vizat).
- **Componenta Text Generation (RAG Pipeline):**
  Monitorizarea parametrului abstract de *Success Retry Rate* – cât de des pipeline-ul are un flow *First-Try* (JSON produs impecabil care trece instant de Validator) față de cât de des este aruncat în ciclu repetitiv (până atinge Max Revisions dictat din `config.py`). În plus, se consideră evaluarea stabilității la fail-through-uri (capacitatea aplicației de a trimite totuși o alertă predictibilă atunci când LLM-ul ratează procesarea).
