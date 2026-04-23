# Rezultate experimentale

## 1. Seturi de date

- **Tip date**: date reale (imagini de plante), nu artificiale.
- **Sursă**: dataset public Kaggle: [Plants Type Datasets](https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets).
- **Încărcare**: prin `kagglehub`, apoi utilizare foldere dedicate:
  - `Train_Set_Folder`
  - `Validation_Set_Folder`
  - `Test_Set_Folder`
- **Detalii observabile în notebook**:
  - problemă de clasificare multiclasă;
  - modelul are strat final cu `out_features=30` (30 clase);
  - `batch_size=64`;
  - număr batch-uri afișat în notebook:
    - train: **375**
    - validation: **48**
    - test: **47**

### Scurtă descriere a datelor

Datele sunt imagini RGB de plante, etichetate pe clase botanice/tipuri de plante, folosite pentru clasificare supervizată.

### Preprocesări

- citire imagine și scalare în intervalul `[0,1]` (`read_image(...).float() / 255.0`);
- redimensionare la `32x32` pentru train/validation/test;
- augmentări doar pe train:
  - `RandomHorizontalFlip()`
  - `RandomRotation(10)`
- normalizare cu statisticile ImageNet:
  - `mean=[0.485, 0.456, 0.406]`
  - `std=[0.229, 0.224, 0.225]`

## 2. Metricile de evaluare folosite

### Metrici urmărite

- în training/validation: `train_loss`, `train_acc`, `val_loss`, `val_acc`;
- în test: `test_loss`, `test_acc`, `test_precision` (macro), `test_recall` (macro), `test_f1` (macro);
- funcție de cost: `CrossEntropyLoss`.

### Formule

- **Accuracy**:

$$
Acc = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i = y_i)
$$

- **Precision macro**:

$$
P_{macro} = \frac{1}{C}\sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c}
$$

- **Recall macro**:

$$
R_{macro} = \frac{1}{C}\sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}
$$

- **F1 macro**:

$$
F1_{macro} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 P_c R_c}{P_c + R_c}
$$

- **Cross-Entropy Loss**:

$$
L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i,y_i}
$$

## 3. Rezultate efective + analiză

### Rezultate numerice găsite în output-ul notebook-ului

- **Epoch 0**: `val_acc = 0.71155`, `val_loss = 0.896`
- **Epoch 1**: `val_acc = 0.79769`, `val_loss = 0.632`
- **Epoch 2**: `val_acc = 0.87327`, `val_loss = 0.375`
- snapshot de progres (din log): `train_acc ≈ 0.917`, `train_loss ≈ 0.328`, `val_acc ≈ 0.849`, `val_loss ≈ 0.459`

### Tabele / grafice

- notebook-ul conține output-uri de progres și loguri, dar **nu include explicit grafice matplotlib** pentru curbele de învățare;
- **nu este vizibil în fișierul salvat un tabel final explicit cu valorile test** (`test_acc`, `test_precision`, `test_recall`, `test_f1`), deși testarea este rulată în cod.

### Analiză scurtă

- performanța pe validare crește consistent în primele epoci (`val_acc` în creștere, `val_loss` în scădere), ceea ce indică învățare eficientă;
- modelul pare să convergă rapid în regim de transfer learning cu EfficientNet-B1;
- pentru concluzii complete privind generalizarea, este recomandată salvarea explicită a metricilor de test într-un tabel final.

## 4. Metodologie de antrenare și evaluare

### Împărțirea datelor

- împărțire deja disponibilă la nivel de dataset în:
  - train
  - validation
  - test

### Paradigma de învățare

- învățare **supervizată** pentru clasificare multiclasă.

### Descrierea modelului

- backbone: `EfficientNet-B1` preantrenat (`EfficientNet_B1_Weights.DEFAULT`);
- cap de clasificare înlocuit cu strat liniar:
  - `Linear(in_features=1280, out_features=num_classes)`;
- funcție de cost: `CrossEntropyLoss`;
- metrici implementate cu `torchmetrics` (`Accuracy`, `Precision`, `Recall`, `F1`, macro pentru cele per-clasă).

### Detalii de antrenare (hyper-parameters)

- optimizer: `Adam`;
- learning rate: `0.001`;
- `max_epochs=10`;
- `batch_size=64`;
- `EarlyStopping` pe `val_loss`, `patience=3`;
- `ModelCheckpoint` pe `val_acc` (mod `max`, `save_top_k=1`);
- monitorizare LR (`LearningRateMonitor`) și logger CSV (`CSVLogger`);
- rulare pe accelerator `auto`, `devices=1` (în metadata notebook: GPU Kaggle T4).

## Limitări observate

- nu există în notebook un sumar tabelar final al metricilor de test în format explicit;
- există neconcordanță de denumire logger (`name="efficientnet_b0"`) deși experimentul este pentru EfficientNet-B1 (afectează doar etichetarea logurilor, nu antrenarea în sine).
