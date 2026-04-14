# Sumar al Raportului de Cercetare

**Titlul propus pentru raport:** 
Asistenți Agricoli Multi-Modali: Integrarea Vizualizării Computaționale Avanste cu Modele Lingvistice Mari Augmentate RAG pentru Îngrijirea Plantelor

**Descrierea tematicii abordate și a importanței acesteia:**

Identificarea precisă a speciilor de plante și diagnosticarea afecțiunilor acestora au reprezentat mult timp provocări considerabile, atât pentru amatorii de botanică, cât și pentru fermierii din sectorul agricol. În mod istoric, aceste sarcini necesitau consultanță umană specializată sau accesul la literatură de specialitate complexă. Odată cu dezvoltarea masivă a tehnicilor de Învățare Profundă (Deep Learning) în agricultură [6], în special a Rețelelor Neurale Convoluționale (CNN), recunoașterea automată a formelor și bolilor plantelor și-a câștigat locul pe piață. Abordări care utilizează arhitecturi eficiente din punct de vedere computațional, precum cele din familia EfficientNet, au demonstrat o eficacitate masivă în literatura de specialitate pe seturi de date extinse (precum arhiva diagnostică **PlantVillage**), deschizând calea recunoașterii instantanee folosind doar o fotografie [1, 5]. Succesul acestor abordări teoretice a permis recent preluarea arhitecturilor și antrenarea lor de succes independent pe colecții de imagini moderne, vizând un număr enorm de specii de apartament sau de grădină.

Cu toate acestea, recunoașterea strict vizuală rezolvă doar jumătate din problemă. Deși modelele predictive indică cu o acuratețe tehnică ridicată clasa sau anomalia identificată, o simplă "etichetă" nu le este suficientă utilizatorilor finali (gospodari, fermieri) pentru a implementa măsuri agronomice. Aici intervine o nouă eră marcată de Modelele Lingvistice Mari (Large Language Models - LLMs). Un asistent inteligent trebuie nu doar să vadă și să recunoască, ci și să poată raționa și oferi prescripții contextuale de tratament, cerințe de umiditate, și expunere solară, transformând datele brute într-un act de comunicare util [3]. 

Provocarea majoră a modelelor LLM standard în domeniile specializate (precum botanica medicală sau agricultura) este fenomenul de "halucinație" – tendința AI-ului de a genera sfaturi plauzibile, dar factual complet greșite, care ar putea duce la ofilirea sau distrugerea unei culturi. Soluția de pionierat la această problemă este utilizarea arhitecturii Retrieval-Augmented Generation (RAG) [4]. RAG obligă modelul ca, înainte de a construi un răspuns, să interogheze o bază de cunoștințe externă sau motoare de căutare web, preluând exclusiv informație documentată. Lucrări recente demonstrează cum asistenții bazați pe RAG obțin rate formidabile de corectitudine și încredere directă [2]. 

Tematica acestui proiect își propune fuziunea celor două tehnologii într-un flux de lucru unitar, multi-modal. Studiul vizează construirea și evaluarea unui ecosistem software integrat în care identificarea curată, non-dependentă de descrieri textuale (via EfficientNet) devine elementul declanșator pentru un pipeline RAG guvernat de LLM-uri [2, 3]. Importanța teoretică și practică a raportului provine din soluționarea scalabilității mentenanței bazelor de date: în loc de fișe fixe cu milioane de profiluri de plate scrise manual, asistentul hibrid generează "carduri de îngrijire" structurate, sigure, adaptate la cerere din oceanul de literatură disponibilă online azi.

**Referințe bibliografice:**

[1] Atila, U., Uçar, M., Akyol, K., & Uçar, E. (2021). Plant leaf disease classification using EfficientNet deep learning model. *Ecological Informatics*, 61, 101182. https://www.sciencedirect.com/science/article/abs/pii/S1574954120301321

[2] Samuel, D. J., Skarga-Bandurova, I., Sikolia, D., & Awais, M. (2025). AgroLLM: Connecting Farmers and Agricultural Practices through Large Language Models for Enhanced Knowledge Transfer and Practical Application. *arXiv preprint arXiv:2503.04788*.

[3] Singh, N., Wang'ombe, J., Okanga, N., Zelenska, T., Repishti, J., G K, J., Mishra, S., Manokaran, R., Singh, V., Rafiq, M. I., Gandhi, R., & Nambi, A. (2024). Farmer.Chat: Scaling AI-Powered Agricultural Services for Smallholder Farmers. *arXiv preprint arXiv:2409.08916*.

[4] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

[5] Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in plant science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

[6] Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and electronics in agriculture*, 147, 70-90. https://doi.org/10.1016/j.compag.2018.02.016
