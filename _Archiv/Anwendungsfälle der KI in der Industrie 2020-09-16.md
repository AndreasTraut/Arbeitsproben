Autor: Andreas Traut  
Datum : 06.07.2020

[Download als PDF](https://github.com/AndreasTraut/Arbeitsproben/raw/master/Anwendungsf%C3%A4lle%20der%20KI%20in%20der%20Industrie.pdf)

[TOC]


# Anwendungsfälle der künstlichen Intelligenz (KI) in der Industrie

Es gibt bereits viele exzellente Artikel zum Thema „künstliche Intelligenz“ (KI). In meinem vorliegenden Text möchte ich KI leicht verständlich erklären, Anwendungsbereiche der KI beschreiben und mit praktischen Beispielen und Progammieranwendungen veranschaulichen. 

Im *ersten Teil* werde ich einige Bereiche beschreiben, in denen KI bereits erfolgreich angewendet wird. 

Im *zweiten Teil* werde ich KI leicht verständliche erklären und gehe dabei auch auf einige Besonderheiten (wie z.B. Big Data, Deep-Learning und Process Mining) ein.

Im *dritten Teil* zeige ich an einem Beispiel, wie der Automobilhersteller BMW von KI-Techniken profitiert hat. 

Im *vierten Teil* zeige ich, wie KI-Techniken in der Programmiersprache Python umgesetzt werden können. 

Im *fünften Teil* gebe ich einige Empfehlungen, auf was geachtet werden muss, wenn KI-Techniken in einem Unternehmen eingeführt werden sollen.

Ich denke, die Auswahl an weiteren Artikeln zur Vertiefung der Thematik „künstliche Intelligenz in der Industrie“ ist riesig und ich hoffe, dass diese kurze Einführung für den Einstieg hilfreich ist.

## 1. In welchen Bereichen wird KI bereits angewendet?

Wir wissen, dass die Anwendungsfälle der *„künstlichen Intelligenz in der Industrie“* zahlreich sind. Ich möchte nur einige wichtige Beispiele nennen, die in der Praxis erfolgreich angewendet werden:

  - **Absatzprognosen**: Die künstliche Intelligenz berechnet den zu erwartenden Absatz von Produkten auf Basis sehr vieler Eingangsdaten (z.B. Börsendaten, Wetter, Rohstoffpreise, Zollbeschränkungen, Preisentwicklung an den Absatzmärkten, Inflation, Zinsen oder Social Media Trends). Damit kann der erwartetet Absatz besser bestimmt und die Produktion optimal gesteuert werden.
  - **Automatische Bestellungen**: Die Bestellmengen und Bestellzeitpunkte für Rohmaterialien werden von der künstlichen Intelligenz automatisch ermittelt und optimiert. Damit soll verhindert werden, dass die Lagerkapazitäten überschritten werden oder dass Lieferengpässe entstehen. Außerdem sollen möglichst viele Rabattangebote der Zulieferer optimal genutzt werden.
  - **Produktentwicklung für Serienproduktion**: Es werden automatisierte Tests an den Produkten durchgeführt und von der künstlichen Intelligenz validiert, so dass diese aufzeigen kann, wo an den Produkten noch Anpassungen vorgenommen werden müssen, damit sie kosteneffizient und ohne Fehler in Serie hergestellt werden können.
  - **Qualitätskontrolle**: Es werden mittels Sensoren, Röntgenstrahlen oder hochauflösenden Kameras Bilder von den Produkten erzeugt. Die künstliche Intelligenz kann dann mit Bilderkennungsalgorithmen Fehler in den Produkten erkennen und diese aussortieren. 

In welchem Bereich denken Sie ist in Ihrem Unternehmen Potential für die Anwendung von "künstlicher Intelligenz"? Behalten Sie diese Bereiche im Hinterkopf: ich komme in Kürze wieder darauf zurück. Zunächst möchte ich im nächsten Abschnitt KI erklären. 

## 2. KI leicht verständlich erklärt

Mein Schaubild zeigt auf der einen Seite die **„Eingangsdaten“** (oder Rohmaterial) und auf der anderen **„Ausgangsdaten“** (oder Endprodukte). Dazwischen laufen **Prozesse** ab (dargestellt durch die beiden orange/grauen Pfeile). Während der Laufzeit dieser Prozesse werden die Zwischenergebnisse üblicherweise mittels log-Dateien protokolliert und gespeichert (was ebenfalls Daten sind). 

![](./media/KuenstlicheIntelligenz.jpg)

##### Was für "Prozesse" sind das beispielsweise? 

Diese "Prozesse" könnten eines der oben gezeigten Beispiele sein, also ein *„Bestellprozess“* oder ein *„Qualitätssicherungsprozess“* , usw. Diese Prozesse könnten beispielsweise Materialen verarbeiten, die während der Verarbeitung einer gewissen Temperatur / einem gewissen Druck ausgesetzt sind. Temperatur und Druck werden während der Verarbeitung mit einem Zeitstempel historisiert und in log-Dateien gespeichert. Dieser Prozess kann auch regional an verschiedenen Orten ablaufen oder organisatorisch in verschiedenen Unternehmenseinheiten.

##### Auf welche Daten hat die künstliche Intelligenz Zugriff? 

Die künstliche Intelligenz hat nun Zugriff auf alle Daten: 

- alle Eingangsdaten: beinhaltet auch alle Daten, die die Materialbeschaffenheit (Länge, Breite, Gewicht) beschreiben
- alle log-Daten: beinhaltet auch alle Daten, die sich aus den Verarbeitungsschritten ergeben, wie z.B. Temperatur, Druck mit denen Materialien verarbeitet werden
- alle Ausgangsdaten: beinhaltet auch Daten, die das Produkt bewerten. Beispielsweise könnte ein Mensch das Produkt als *"nicht ok"* bewerten, weil es defekt ist oder weil eine wichtige KPI-Kennzahl nicht zufriedenstellend ist. 

Die künstliche Intelligenz weiß alles und kann somit jederzeit einen **Zusammenhang** zwischen „Eingangsdaten“ und „Ausgangsdaten“ (bzw. „Rohmaterial“ und „Endprodukt“) herstellen und hat dabei stets auch den Prozess (die log-Dateien) im Blick. Zum Beispiel: sobald ein Mensch die „Ausgangsdaten“ oder das „Endprodukt“ als *„nicht ok“* bewertet, kann die künstliche Intelligenz einen Rückschluss ziehen, welcher Eingangsparameter oder welcher Prozessschritt am relevantesten für die Anomalie war und kann einen Vorschlag machen, was geändert werden müsste. 

##### Ist die künstliche Intelligenz wirklich intelligent? 

Die künstliche Intelligenz ist dabei gar nicht „intelligent“ wie wir Menschen es landläufig verstehen: KI ist nur ein Algorithmus, der diese Zusammenhänge mit Modellen darstellen kann. Hierbei gibt es unterschiedliche Ansätze, je nachdem, was gerade relevant ist: 

- Man spricht man von einem ["Big Data"](https://de.wikipedia.org/wiki/Big_Data) Problem, wenn man sehr viele Eingangsdaten betrachtet, so dass die herkömmlichen Methoden der Datenverarbeitung scheitern. Das ist z.B. der Fall, wenn Sensordaten (Temperatur, Druck, Bewegung) von Maschinen gesammelt werden. Entsprechend wendet man dann auch "Big Data" Ansätze an, die sich dann teilweise recht stark von den herkömmlichen Ansätzen der Datenverarbeitung unterscheiden. Mit dieser Thematik habe ich mich [hier](https://github.com/AndreasTraut/Machine-Learning-with-Python) beschäftigt und die unterschiedlichen Vorgehensweisen gegenübergestellt. 
- Treten hingegen die Eingangsdaten etwas in den Hintergrund (also kein Big Data), aber dafür die Prozesse in den Vordergrund, spricht man von ["Process Mining"](https://de.wikipedia.org/wiki/Process-Mining). Hier werden dann häufig die Log-Dateien, die ja die Prozesse protokollieren, in Modelle transformiert und dann ausgewertet. 
- Beispielsweise spricht man von einem ["Deep Learning"](https://github.com/AndreasTraut/Deep-Learning) Problem, wenn neuronale Netze zum Einsatz kommen, was häufig der Fall ist, wenn die Eingangsdaten Bilder oder Dokumente sind und die Aufgabe ist, diese zu gruppieren. Mehrere Schichten von "Neuronen" erhalten aus der Schicht darunter Daten, auf die sie dann Rechenoperationen anwenden und an die nächste Schicht weitergeben. So entsteht ein mehrschichtiges Netz. Auch damit habe ich mich beschäftigt und für den interessierten Leser [hier](https://github.com/AndreasTraut/Deep-Learning) meine Erfahrungen dokumentiert.

Es geht nicht darum, diese Ansätze alle im Detail zu verstehen. Es geht darum, dass man versteht, dass mit einem strukturierten, logischen Ansatz diese **Zusammenhänge** der Eingangsdaten, der log-Dateien und der Ausgansdaten mit KI durchdrungen werden können, um damit einen Profit für Ihr Unternehmen herauszuholen - wie erkläre ich im nächsten Abschnitt. 

## 3. Wie hat BMW von KI profitiert?

Das „Capgemini Research Institut“ hat im Dezember 2019 [hier](https://www.capgemini.com/de-de/news/ki-in-der-industrie/) eine interessante Studie veröffentlicht: darin wurden 300 Unternehmen aus den Sektoren industrielle Fertigung, Automobil, Konsumgüter, Luftfahrt und Verteidigung untersucht und es wurde dabei festgestellt, dass Unternehmen in Deutschland im Vergleich zu anderen Ländern bereits sehr viel künstliche Intelligenz in ihren Wertschöpfungsketten /  Produktionsprozessen einsetzen, diese jedoch noch vertiefen sollten.

Ich möchte kurz ein konkretes Beispiel herausgreifen, auf das sich diese Studie unter anderem stützt:
Siehe <https://www.cbronline.com/big-data/analytics/bmw-optimised-supply-chain-teradata-big-data/>

![](./media/KuenstlicheIntelligenz_BMW.jpg)

Quelle: BMW Group

Der Automobilhersteller BMW arbeitet in 31 Produktionsstandorten in denen sehr komplexen Prozesse ablaufen. Im Bild oben habe ich versucht, die unzähligen verschiedenen Rohmaterialen rot darzustellen. Der **Prozess von den Rohmaterialen bis zum Endprodukt** (dem Auto) ist sehr lange, kompliziert und unübersichtlich und läuft teilweise über verschiedene Kontinente verstreut ab. Dabei wird an vielen Stellen im Prozessverlauf Inventar zwischengelagert. Eine große Herausforderung war es für BMW, die vielen Daten, die dabei entstehen in einer sinnvollen Form abzuspeichern (Stichwort: Data Warehouse, Data Lake). 

Einen Erfolg hat BMW im Jahr 2016 feiern können, als es bei der Analyse seines Inventars wertvolle Erkenntnisse gewinnen konnte: die Teams haben die Inventarkosten um 70% reduzieren können, weil sie mehr Transparenz über ihre vielen Produktionsorte gewonnen haben und die Prozesse optimieren konnten.

![](./media/KuenstlicheIntelligenz_BMW_KI.jpg)

Im Jahr 2016 beschreibt Klaus Straub, CIO bei BMW in [diesem Artikel](https://www.i-cio.com/profession/cio-profiles/item/powering-digital-disruption-at-bmw>), die Ideen zur digitalen Transformation des Unternehmens. Schon damals sah er das große Potential, das durch Künstliche Intelligenz entstehen würde, um beispielsweise die **Qualität zu verbessern** oder die **Prozesse effizienter zu gestalten**, wobei die Verknüpfung der IT mit den realen Produktionsabläufen eine große Herausforderung sein würde. Doch wie lässt sich das konkret umsetzen? Darüber möchte ich im folgenden Abschnitt einen Einblick geben. 

## 4. Wie können KI-Techniken konkret umgesetzt werden?

Es gibt viele frei verfügbare (Open-Source) Tools, die man je nach Fragestellung nur noch auf die eigenen Bedürfnisse anpassen muss. Wie genau das geht zeige ich im Folgenden jeweils in meinem Programmcode. 

##### Was ist auf dem Bild zusehen? 

Wenn beispielsweise ein Bild von einem Bauteil vorliegt und sich die Frage stellt, was in diesem Bild abgebildet ist, könnte ein Mensch das durch bloßes draufschauen schnell herausfinden. Der Mensch könnte auch sehen, ob das Bauteil defekt ist oder nicht. Die KI kann das auch und für diese Fragestellung wähle ich einen Deep-Learning Ansatz. Es sind nur 10 Zeilen Programmcode notwendig, damit mir das Modell sagt, dass es auf dem folgenden Bild mit 86% Wahrscheinlichkeit eine *"Tasse"* sieht und mit 6.8% eine *"Kaffeetasse (coffepot)"*. Das ist beeindruckend, unter anderem auch, weil die Open-Source Tools, die jedem frei und kostenlos zur Verfügung stehen, vom Anwender nicht erwarten, dass er die umfangreiche Berechnungen dahinter im Detail versteht. Meinen Programmcode können Sie [hier](https://github.com/AndreasTraut/Deep-Learning/blob/master/Image_classification/Image_classifier_example_2_transfer_learning_ResNet52-German.ipynb) einsehen. 

![](./media/9145.jpg)

![](./media/transfer_learning.jpg)

##### Welche Gruppen können gebildet werden? 

Angenommen wir haben ein Bild, über das wir noch nicht viel wissen und welches wir in eine uns bekannte Gruppierung einordnen möchten. Beispielweise ein Röntgenbild, bei dem wir uns fragen, ob die ein oder andere Krankheit darauf zu sehen ist. Oder ein Bild einer Pflanze für das wir den Namen und Pflegehinweise wissen wollen. Auch Texte können gruppiert werden. Bei einem Dokument oder Vertrag können wir auf der Suche nach ähnlichen Texten sein. Das Gruppieren von ähnlichen Dingen ist ein häufig diskutiertes Problem. Wir kennen alle die nützliche Google-Funktion, mit der sich ähnliche Bilder anzeigen lassen. Man gibt einen Begriff (z.B. Espressotasse) in die Suchleiste ein und bekommt ähnliche Bilder angezeigt: 

![](./media/Google_aehnliche_Bilder.jpg)

Bei der Umsetzung des Programmcodes stellen sich zwei Fragen. Die erste Frage ist: 

Wie vergleicht man zwei Bilder miteinander? Oder zwei Texte? Das ist nicht einfach, aber dieses Problem ist schon vielfach analysiert worden und es gibt Vorgehensweisen, die man nur kopieren muss und ich werde Ihnen gleich zeigen wie. 

Die zweite Frage ist: Wie geht man vor, um alle Dinge (Bilder oder Texte) paarweise miteinander zu vergleichen? Nehmen wir 1 Million Dinge, die wir paarweise miteinander vergleichen, um damit die Gruppen bilden zu können. Dann haben wir schon 1 Million mal 999 999 / 2, also etwa 500 Milliarden Rechenoperationen. Das kann sehr lange dauern. Da es sich bei dieser Fragestellung um sehr viele Eingangsdaten handelt, wähle ich einen Big Data Ansatz, nämlich das "Local-Sensitive-Hashing (LSH)": LSH ist eine Technik, um ähnliche Dinge mit einer hohen Wahrscheinlichkeit in Gruppen einzuteilen. Man verzichtet also auf absolut exakte Ergebnisse und nimmt eine kleine Fehlerwahrscheinlichkeit in Kauf. Diese Wahrscheinlichkeit kann mit Steuerungsparametern eingestellt werden (je nach Bedarf). Sind diese Parameter einmal eingestellt, kann der KI-Algorithmus sehr schnell neue Bilder in Gruppen einordnen. Der Vorteil auf eine absolut exakte 100%-ige Gruppierung zu verzichten, liegt auf der Hand: LSH läuft viel schneller als 100% exakte Algorithmen. 

Das Ergebnis meiner Arbeit war: ich habe auf über 9000 Bildern (je etwa 300*300 Pixel) den LSH Algorithmus angewendet, unter anderem auch meine Sammlung an Kirchenfenstern und Espressotassen (ich trinke gerne Espresso und habe irgendwann angefangen, die Tassen zu fotografieren). Auf meinem eigenen Computer hat diese Gruppierung ein paar Minuten gedauert und war nur einmalig notwendig. Danach habe ich ein völlig neues Bild einer Espressotasse aus dem Internet heruntergeladen: 

![](./media/testpicture_1.png)

Dieses Bild habe ich dem Programm gegeben und damit dann ähnliche Bilder aus meiner Bildersammlung suchen lassen. Nach wenigen Sekunden hat mir das Programm dann diese 15 Bilder angezeigt: 

![](./media/testpicture_1similar15.png)

Dasselbe habe ich dann auch für meine Kirchenfenster Bilder getestet mit diesem Testbild: 

![](./media/testpicture_4.png)

Ergebnis: 

![](./media/testpicture_4similar.png)

Wer einen kurzen Blick auf die Programmzeilen werfen möchte, um sich zu vergewissern, dass nur wenige Codezeilen für dieses Problem nötig sind, kann das gerne [hier](https://github.com/AndreasTraut/Deep_learning_explorations/blob/master/8_Image_similarity_search/Beispiel_aehnliche_Bilder_finden.ipynb) tun. Für Programmierer, denen dieser kurze Eindruck noch nicht genügt, habe ich [hier etwas ausführlicher](https://github.com/AndreasTraut/Deep_learning_explorations/blob/master/README.md) die Materie zum "Local-Sensitive-Hashing" dokumentiert. Und da sich die Deep-Learning Technik nicht in wenigen Worten erklären lässt, habe ich sie [hier](https://github.com/AndreasTraut/Deep-Learning) vertieft erklärt. 

Im nächsten Abschnitt gebe ich Ihnen ein erstes Konzept an die Hand, um KI Techniken in Ihrem Unternehmen einzuführen. 

## 5. Welche Empfehlungen zur Umsetzung von KI Techniken würde ich Ihnen geben?

Falls Sie nun Interesse haben, auch in Ihrem Unternehmen künstliche Intelligenz einzusetzen, ist die Empfehlung, dass Sie sich folgende Gedanken machen: 

- Was erwarten Sie, dass Ihnen die Analyse Ihrer Daten konkret für Vorteile bringen sollte? Sammeln und strukturieren sie zunächst Hintergrundwissen über Ihr Unternehmen: welche Unternehmenseinheiten sind betroffen, wer sind die Schlüsselpersonen, wer ist „Sponsor“ des Projekts. 
- Beschreiben Sie das Problem und die Motivation für das Datenanalyse-Projekt. Machen Sie sich auch Gedanken über die derzeitige Situation, deren Vor- und Nachteile. Dies benötigen Sie zum Abgleich mit dem neuen Datenanalyse-Projekt. 
- Beschreiben Sie, wie ein erfolgreich umgesetztes Datenanalyse-Projekt aussehen würde: gibt es Erfolgskennzahlen (objektive Ziele) oder subjektive Ziele, die Sie definieren bzw. beschreiben können, um den „Erfolg“ des Datenanalyse-Projekts für Ihr Unternehmen zu messen? Es ist wichtig messbare Unternehmens-Ziele zu definieren, damit sich daraus weitere messbare Ziele zur weiteren Umsetzung des Datenanalyse-Projekts ableiten lassen: welche Art von Datenanalyse soll für das Problem angestrebt werden (reicht der oben erwähnte binäre Klassifikator oder werden komplexere Modelle benötigt?). Welche Daten werden für diese Modelle konkret benötigt und welche technischen und organisatorischen Schritte sind nötig, um diese Daten aus verschiedenen Quellen zu extrahieren, zu transformieren, modellieren und evaluieren? 
- Stellen Sie sich früh auch die Fragen, wie das "Deployment" ablaufen soll, also wie die Programme, die in einer Testumgebung entwickelt wurden, im täglichen produktiven Betrieb zum Laufen gebracht werden sollen. Sollen eigene Computer verwendet werden oder die Cloud? Gelten in Ihrem Unternehmen hohe Datenschutz-Anforderungen ist eine Cloud-Lösung vielleicht nicht die erste Wahl und zu hinterfragen. Eine teure Investition in eigene Hardware könnte dann der nächste Schritt für Sie sein. Möchten Sie hingegen erst einmal verschiedene Dinge ausprobieren und sind Sie noch nicht bereit, massiv Kapital in neue Hardware zu investieren, dann könnte eine Cloud-Lösung ideal für Sie sein. Meine Erfahrungen, die ich mit der Microsoft Azure Cloud Plattform gesammelt habe, können Sie [hier](https://github.com/AndreasTraut/Experiences-with-MicrosoftAzure) nachlesen. 

Es gibt methodische Ansätze, die sich bei der Transformation in ein solches Datenanalyse-Projekt anwenden lassen. Da die Kosten, für Big Data Systeme enorm hoch sind (finanzielle Kosten aber auch die Zeit, die Ihre Mitarbeiter gebunden sind) und in der Regel viele Unternehmensbereiche betroffen sind, empfiehlt es sich, einen strukturierten Ansatz zu gehen. 

Ich hoffe, dass meine kurze Einführung in die Thematik *„künstliche Intelligenz in der Industrie“* für den Einstieg hilfreich war und denke, dass die Auswahl an weiteren Artikeln zur Vertiefung der Thematik riesig ist. Viel Spaß und Erfolg bei der weiteren Recherche und Umsetzung wünscht Ihnen

*Andreas Traut*



