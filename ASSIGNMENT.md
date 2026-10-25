# Úvod

Cieľom projektu je osvojiť si **prehľad fungovania v dátovej vede**, základné koncepty a techniky analýzy dát, pochopia, ako fungujú a získajú intuíciu pre ich vhodnú aplikáciu za účelom objavovania znalostí v dátach. Taktiež získajú predstavu, aké otázky vieme pomocou analýzy dát zodpovedať a aplikovať **základné prístupy strojového učenia**. Dôraz je kladený na analýzu a predspracovanie dát, použitie metód strojového učenia, spôsoby ich vyhodnotenia a porovnania.

Projekt sa vypracúva **v dvojiciach** v akceptovateľnej kvalite. Pri riešení sa používa programovací jazyk **Python** a dostupné knižnice pre dátovú vedu ako pandas, numpy, scipy, statsmodels, scikit-learn**,** atd**.**. V každej fáze a aktivite sa odovzdáva vykonateľný **Jupyter Notebook** do UIS, ktorý obsahuje všetky vykonané transformácie nad dátami s vhodnou dokumentáciou. Odovzdaný notebook musí obsahovať nielen kód, ale aj jeho výsledky (vypočítané hodnoty, výpisy, vizualizácie a pod.) spolu s komentárom k získaným výsledkom a z toho plynúce rozhodnutia pre ďalšie kroky dátového procesu. Schopnosť dobre komunikovať a prezentovať relevantné výsledky predstavuje významnú zložku hodnotenia.

Pri každej fáze v odovzdanom notebooku uveďte **percentuálny podiel práce** členov dvojice.

# 

# Dáta

[https://drive.google.com/drive/folders/1qI7dqHpDH7lbdgbB\_9i3ZSsoCuhR766F?usp=sharing](https://drive.google.com/drive/folders/1qI7dqHpDH7lbdgbB_9i3ZSsoCuhR766F?usp=sharing)    
***(každá dvojica má jeden dataset pod číslom, ktoré máte na cvičení)***

Prediktívna údržba (angl. **Predictive Maintenance**, PdM) predstavuje moderný prístup k monitorovaniu technických zariadení, ktorého cieľom je včasná identifikácia vznikajúcich porúch ešte pred ich úplným prejavením. Výskyt porúch vo vozidlách verejnej dopravy počas bežnej prevádzky môže viesť k prerušeniu jazdy, neplánovaným odstávkam a významným ekonomickým stratám. Okrem negatívneho dopadu na prevádzkovateľa spôsobujú takéto situácie aj zníženie spoľahlivosti dopravy a spokojnosti cestujúcich. Tradičné prístupy založené na pravidelnej údržbe alebo pevných prahových hodnotách senzorov často nedokážu zachytiť vznikajúce poruchy dostatočne skoro.

Moderné metódy založené na strojovom učení umožňujú analyzovať kontinuálne zbierané údaje zo senzorov a identifikovať anomálie, ktoré môžu signalizovať blížiace sa zlyhanie zariadenia. Modely využívajú dáta z rôznych senzorov monitorujúcich stav komponentov, pričom dokážu rozpoznať skryté vzory správania vedúce k poruche. Okrem samotnej detekcie anomálií môžu tieto prístupy podporovať aj predikciu budúcich porúch, odhad zostávajúcej životnosti komponentov či analýzu príčin zlyhania. Vďaka tomu je možné plánovať údržbu pred vznikom kritických porúch, minimalizovať neplánované odstávky a znižovať náklady na opravy.

Cieľom tohto zadania je oboznámiť sa s problematikou prediktívnej údržby, pochopiť možnosti využitia umelej inteligencie v podobe strojového učenia pri detekcii kritického zlyhania systému v senzorových dátach, ktoré umožní včasnú údržbu. 

# Zadanie projektu 

**Rail Air-Production-Unit (APU) Predictive Maintenance**

## The QUEST

Každá dvojica bude pracovať s pridelenou dátovou sadou od 2\. týždňa. **Vašou úlohou** je predikovať závislé hodnoty premennej “**failure**” **(predikovaná premenna)** pomocou metód strojového učenia. Budete sa musieť pritom vysporiadať s viacerými problémami, ktoré sa v dátach nachádzajú ako formáty dát, chýbajúce, vychýlené hodnoty a mnohé ďalšie. 

Očakavaným **výstupom** projektu  je:

1. **najlepší model** strojového učenia;   
2. **data pipeline** pre jeho vybudovanie na základe vstupných dát.

## Fáza 1 \- Prieskumná analýza: 15 bodov

#### 1.1 Základný opis dát spolu s ich charakteristikami (5b)

EDA s vizualizáciou

* (A-1b) Analýza štruktúr dát ako súbory (štruktúry a vzťahy, počet, typy, …), záznamy (štruktúry, počet záznamov, počet atribútov, typy, …)  
* (B-1b) Analýza jednotlivých atribútov: pre zvolené významné atribúty (min 10\) analyzujte ich distribúcie a základné deskriptívne štatistiky a či spĺňa predpísané podmienky a rozsah meraných hodnôt.   
* (C-1b) Párová analýza dát: Identifikujte vzťahy a závislostí medzi dvojicami atribútov.  
* (D-1b) Párová analýza dát: Identifikujte závislosti medzi **predikovanou** premennou a ostatnými premennými (potenciálnymi prediktormi).  
* (E-1b) Dokumentujte Vaše prvotné zamyslenie k riešeniu zadania projektu, napr. sú niektoré atribúty medzi sebou závislé? od ktorých atribútov závisí predikovaná premenná? či je potrebné kombinovať záznamy z viacerých súborov? 

#### 1.2 Identifikácia problémov, integrácia a čistenie dát (5b)

* (A-2b) Identifikujte aj prvotne riešte problémy v dátach napr.: nevhodná štruktúra dát, duplicitné záznamy, ktoré môžu vznikať po určitých dátových transformáciach, nejednotné formáty, chýbajúce hodnoty, vychýlené hodnoty. V dátach sa môžu nachádzať aj iné, tu nevymenované problémy, resp. menej problémov ako bolo uvedených.   
* (B-2b) Kontrola správnosť v dátach  
  * či obsahujú abnormálne hodnoty  
  * či obsahujú nelogické dátové vzťahy, ktoré sú následkom dátovej kolekcie a anotovania dát     
* (C-1b) Vychýlené hodnoty (outlier detection), vyskúšajte riešiť problém min. 2 technikami  
  * odstránenie vychýlených alebo odľahlých pozorovaní  
  * nahradenie vychýlenej hodnoty hraničnými hodnotami rozdelenia (napr. 5%, 95%)

#### 1.3 Formulácia a štatistické overenie hypotéz o dátach (5b)

* (A-4b) Sformulujte **dve hypotézy** o dátach v kontexte zadanej predikčnej úlohy. Formulované hypotézy overte vhodne zvolenými štatistickými testami.

Príklad formulovania:   
***(SK) flowmeter** má v priemere vyššiu hodnotu v stave **s failure** ako bez nej**.***

* (B-1b) Overte či Vaše štatistické testy majú dostatok podpory z dát, teda či majú dostatočne silnú štatistickú silu.

**V odovzdanej správe (Jupyter notebook) by ste tak mali odpovedať na otázky:**  
Majú dáta vhodný formát pre ďalšie spracovanie? Aké problémy sa v nich vyskytujú? Nadobúdajú niektoré atribúty nekonzistentné hodnoty? Ako riešíte tieto Vami identifikované problémy?

**Správa sa odovzdáva v 5\. týždni semestra**. Dvojica svojmu cvičiacemu odprezentuje vykonanú fázu v Jupyter Notebooku podľa potreby na cvičení. V notebooku uveďte **percentuálny podiel práce** členov dvojice. Správu elektronicky odovzdá **jeden člen z dvojice** do systému **UIS** do nedele **18.10.2026 23:59**.

## Fáza 2 \- Predspracovanie údajov: 15 bodov

**Správa sa odovzdáva v 7\. týždni semestra**. Dvojica svojmu cvičiacemu odprezentuje vykonanú fázu v notebooku podľa potreby na cvičení. Uveďte percentuálny podiel práce členov dvojice. Správu elektronicky odovzdá **jeden člen z dvojice** do systému **UIS** do nedele **01.11.2026 23:59**.

## Fáza 3 \- Strojové učenie: 20 bodov

**Správa sa odovzdáva v 10\. týždni semestra.** Dvojica svojmu cvičiacemu odprezentuje vykonanú fázu v Jupyter Notebooku na cvičení. V notebooku uveďte percentuálny podiel práce členov dvojice. Následne správu elektronicky odovzdá **jeden člen z dvojice** do systému **UIS** do nedele **22.11.2026 23:59**.​​

# Aktivity na cvičení: 10 bodov

**Správa sa odovzdáva v 12\. týždni semestra.** Dvojica svojmu cvičiacemu odprezentuje vykonanú fázu v Jupyter Notebooku na cvičení. V notebooku uveďte percentuálny podiel práce členov dvojice. Následne správu elektronicky odovzdá **jeden člen z dvojice** do systému **UIS** do nedele **06.12.2025 23:59**.​​  
