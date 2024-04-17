## Dataset(s)

### Welke gegevens ik ga gebruiken
Voor dit project zal ik gebruikmaken van het dataset "Student Alcohol Consumption" gevonden op *[Kaggle: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption/data], api command: "kaggle datasets download -d uciml/student-alcohol-consumption"*. De dataset bevat informatie over de alcoholconsumptie van studenten, en veel specifieke variabelen zoals leeftijd, geslacht, familierelaties en hun prestaties of school.

### Data verdeling
- **Training data**: De trainingsdata zal 70% van het dataset zijn en zal worden gebruikt om het voorspellende model te trainen.
- **Validation data**: 15% van het dataset zal worden gereserveerd voor validatie om de modelhyperparameters af te stemmen.
- **Test data**: De overige 15% zal dienen als testdataset om het model te evalueren.

### Nieuwe gegevens krijgen
Om de service te verbeteren en bij te werken, kunnen nieuwe gegevens worden verkregen via enquêtes.

## Projectuitleg
Het project heeft als doel een voorspellend model te ontwikkelen dat de alcoholconsumptiepatronen van studenten beoordeelt op basis van verschillende factoren. De service zal inzichten geven in mogelijke risicofactoren die verband houden met overmatige alcoholconsumptie onder studenten.

### Toepassing
De toepassing zal gericht zijn naar ouders of gewoon de jeugd die zich afvraagt hoeveel iemand in hun situatie vaak drinkt.

## Tasks & Actions
### Voorbreiding
1. **Endpoint verbinding** Verbinden/ pullen van de endpoint.
2. **Kaggle account verbinden** Kaggle account verbinden met code.
### Datavoorbewerking
1. **Data cleaning**: Verwijder eventuele ontbrekende of inconsistente gegevenspunten.
2. **Data selectie**: Relevante en nuttige kolommen bijhouden of degene die onnodig zijn droppen.
3. **Categorische waarden omzetten**: Categorische waarden omzetten naar numerieke variabelen.
4. **Dataverdeling**: Verdeel het dataset in trainings-, validatie- en testsets.

### Modelontwikkeling
1. **Model selectie**: Kies geschikte machine learning-algoritmen (bijv. regressie, classificatie) voor het voorspellen van alcoholconsumptiepatronen.
2. **Model training**: Train het geselecteerde model met behulp van de trainingsdata.
3. **Model evaluation**: Evalueer de prestaties van het model met behulp van de validatieset en stem de hyperparameters indien nodig af.
