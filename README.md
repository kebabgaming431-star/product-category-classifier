# Product Category Classifier — Task 3

Autor: Andrei Motau

Clasificarea produselor în 10 categorii folosind titlul. Proiect educațional LINK Academy; modelul oferă sugestii, nu garantează clasificarea corectă.

## Pornire rapidă

Necesită Python 3.12. Din terminal, în folderul proiectului:

```bash
python -m pip install -r requirements.txt
python predict_category.py
```

Introdu un titlu, de exemplu `bosch wap28390gb 8kg 1400 spin`. Scrie `exit` pentru închidere. Se poate face și o singură predicție:

```bash
python predict_category.py --title "kenwood k20mss15 solo"
```

Modelul deja antrenat este inclus. Încarcă numai fișiere pickle de încredere și păstrează aceeași versiune scikit-learn; alte versiuni pot fi incompatibile.

## Reproducerea antrenării

```bash
python train_model.py
```

Comanda citește `products.csv`, curăță datele, compară cinci configurații pe validare, evaluează câștigătorul pe test și salvează modelul final și rapoartele. Rularea durează în funcție de calculator câteva minute. Fișierele de rezultate existente sunt regenerate. Parametri opționali: `--data cale.csv --output folder`.

## Notebook

```bash
jupyter notebook product_classification.ipynb
```

Alege Restart Kernel and Run All Cells. Notebook-ul include rezultatele executate și graficele; folosește logica din `train_model.py`, evitând două implementări divergente. Pentru Colab încarcă și dezarhivează proiectul complet și schimbă directorul curent în acel folder înainte de rulare. Nu este suficient doar notebook-ul.

## Fișiere

- `products.csv`: datele originale primite, nemodificate.
- `train_model.py`: curățare, caracteristici, comparație, evaluare, salvare.
- `predict_category.py`: încărcarea modelului și predicții interactive.
- `product_model.pkl`: pipeline complet TF-IDF + clasificator, antrenat pe toate datele curate.
- `product_classification.ipynb`: analiză completă, executată.
- `metrics.json`: statistici, împărțire, versiuni, hash date și scoruri test.
- `validation_comparison.csv`: comparația configurațiilor pe validare.
- `classification_report.csv`, `confusion_matrix.csv`, `test_errors.csv`: evaluarea finală.
- `example_predictions.csv`: cele șase exemple din cerință și predicțiile reale.
- `requirements.txt`: biblioteci și versiuni.

## Decizii și rezultate

Coloanele au spații în nume. Etichetele CPU/Mobile Phone/fridge sunt uniformizate. Din 35.311 rânduri eliminăm 215 fără titlu/categorie, 4 cu titluri contradictorii și 4.270 duplicate; rămân 30.822 titluri unice. Nu imputăm coloanele neutilizate. Intrarea este numai titlul: ID-urile, ratingul, vizualizările și data nu sunt disponibile în programul interactiv.

Split stratificat cu seed 42: 18.492 train, 6.165 validare, 6.165 test. Vectorizarea și scalarea sunt învățate doar pe train la comparație. Criteriul de selecție este F1 macro pe validare.

| Configurație | Accuracy validare | F1 macro validare |
|---|---:|---:|
| Caractere + LinearSVC | 98,73% | 98,71% |
| Cuvinte + LinearSVC | 95,94% | 96,05% |
| Cuvinte + statistici + LinearSVC | 95,93% | 96,03% |
| Cuvinte + LogisticRegression | 95,02% | 95,13% |
| Cuvinte + MultinomialNB | 94,29% | 94,39% |

Fragmentele de caractere au câștigat. Acestea pot valorifica părți ale codurilor comerciale și variațiile ortografice. Statisticile numerice nu au îmbunătățit baseline-ul SVC în această comparație; nu le-am păstrat în modelul final.

Configurația aleasă, reantrenată pe 80% din date, obține **98,99% accuracy și 99,01% F1 macro** pe test, față de 15,60% accuracy pentru clasa majoritară. Sunt 62 de erori din 6.165 produse. Modelul livrat este ulterior reantrenat pe toate datele curate; nu pretindem că scorurile testului sunt o evaluare independentă a acestui refit.

## Limite și dezvoltare ulterioară

Titlurile foarte apropiate ale aceluiași model comercial pot rămâne în partiții diferite, chiar după deduplicare exactă. Scorurile pot fi optimiste pentru familii de produse complet noi. Sunt necesare evaluări grupate pe modele și în timp. Numai 10 categorii sunt cunoscute; pentru titluri în afara domeniului se va alege tot una dintre ele. Nu se afișează o probabilitate fictivă: scorurile LinearSVC nu sunt probabilități calibrate.

Următorii pași: cross-validation pe development, evaluare pe date ulterioare, prag de abținere calibrat și verificare umană. Exemplele din enunț sunt teste funcționale, nu un set independent.

## Predare

Cerința este **un repository public GitHub și linkul său în formularul LINK Academy**. Acest pachet conține fișierele necesare, dar nu este deja publicat.

Dezarhivează, creează un repository public numit `product-category-classifier`, apoi încarcă fișierele și salvează modificările. Verifică existența modelului, datelor, notebook-ului și README-ului. Copiază adresa repository-ului în formularul temei. Dacă lucrezi cu Git local, fă commit-uri reale pe măsură ce modifici proiectul; nu inventa un istoric al experimentelor.
