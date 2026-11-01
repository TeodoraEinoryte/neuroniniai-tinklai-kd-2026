# Klaidų auditas (S7, ataskaitos 4.10 skyrius)

## Kas čia yra

- `audit_sample.csv` – 100 sunkiausių tekstų ir jų duomenys. Tai tekstai iš išplėstinės testavimo imties, kuriuos klaidingai atpažįsta bent 80 % visų S1 paleidimų (tokių yra 702, iš jų atsitiktinai atrinkta 100, sėkla 2026). Oficialioje testavimo imtyje tokių tekstų tik 33, todėl imta išplėstinė.
- `audit_A.csv` – žymi **Teodora (A)**, `audit_B.csv` – žymi **Edgaras (B)**. Abiejuose tie patys 100 tekstų.

Tikslas – įvertinti, kiek likusių klaidų yra ne modelių kaltė (nepatikima žymė, kalbos reiškinys, duomenų trūkumas), o kiek – tikros modelio klaidos.

## Kas sprendžia žmogus, o kas – programa

Žmogus sprendžia tik tai, ko programa nustatyti negali: ar kalta žymė, ar yra neigimas, ar sarkazmas. Visa kita programa nustato pagal teksto raktinio žodžio grupę (`data.keyword_groups`, tos pačios grupės kaip ataskaitos 4.8 skyriuje ir S13 lentelėse).

**Žmogaus kodai** (stulpelis „kategorija (0–3)“, vienas skaičius tekstui, **pirmas tinkantis** šia tvarka):

| Kodas | Kategorija | Kada |
|---|---|---|
| 1 | Žymė nepatikima (klaidinga, dviprasmiška arba aiškios emocijos nėra) | **Klaidinga** – perskaitęs tekstą aiškiai pasirinktum kitą emociją. **Dviprasmiška** – tekstas vienodai tinka dviem emocijoms (žymei ir prognozei). **Aiškios emocijos nėra** – negalėtum pasirinkti nė vienos iš 6 emocijų (netinka nei žymė, nei prognozė, nei kita); jei emocija silpna, bet žymė tinka – tai 0, ne 1. Žymės duomenų rinkinyje sudarytos automatiškai pagal emocijos žodį, todėl tokių atvejų gali būti nemažai. |
| 2 | Neigimas | Žymė tinka, bet emocija tekste paneigta („i dont feel …“, „not …“, „never …“), ir modelis klysta dėl to. |
| 3 | Sarkazmas ar ironija | Žymė tinka, bet tekstas sako viena, o reiškia priešingai. |
| 0 | Nė viena | Žymė tinka, neigimo ir sarkazmo nėra. |

**Galutinė kategorija.** Jei kodas 1–3 – jis ir lieka. Jei kodas 0, programa priskiria pagal grupę:

| Grupė | Galutinė kategorija |
|---|---|
| be emocijos žodžio | Nėra emocijos žodžio |
| daugiareikšmis žodis | Daugiareikšmis emocijos žodis |
| mokymo imtyje nematytas žodis | Mokymo imtyje nematytas emocijos žodis (modelis negali žinoti žodžio, kurio nematė) |
| vienareikšmis žodis | Tikra modelio klaida |

„Be emocijos žodžio“ reiškia tiksliai tai, ką rodo mūsų detektorius (`data.emotion_word`): jis ieško emocijos žodžio iškart po „feel / feeling / felt“, praleisdamas „so“, „very“ ir pan. Todėl „i feel like i have been around some bitter people“ jam yra be emocijos žodžio, nors „bitter“ tekste yra. Ataskaitoje rašome būtent taip.

## Pavyzdžiai (sugalvoti, ne iš tų 100 tekstų)

| Tekstas | Žymė | Prognozė | Kodas | Kodėl |
|---|---|---|---|---|
| i feel so lucky to have such great friends around me | baimė | džiaugsmas | 1 | aiškiai džiaugsmas – žymė klaidinga |
| i feel so blessed to have you in my life | džiaugsmas | meilė | 1 | tinka abi vienodai – dviprasmiška |
| i feel like i should go to the store before it closes | baimė | džiaugsmas | 1 | netinka nė viena iš 6 emocijų – aiškios emocijos nėra |
| i dont feel happy anymore since she left | liūdesys | džiaugsmas | 2 | žymė tinka, modelį suklaidino paneigtas „happy“ |
| oh great i feel so thrilled to be stuck in traffic again | pyktis | džiaugsmas | 3 | žymė tinka, „thrilled“ ironiškas |
| i feel like crying every time i hear that song | liūdesys | baimė | 0 | žymė tinka, nėra nei neigimo, nei ironijos |

## Kaip žymėti

1. Atsidaryk **savo** failą. Į kito žmogaus failą ir į `audit_sample.csv` stulpelius „klaidingų paleidimų dalis“ ir „raktinio žodžio grupė“ nežiūrėk, kol abu nebaigsite – jie gali paveikti sprendimą. Žymime nepriklausomai.
2. Kiekvienam tekstui į stulpelį **„kategorija (0–3)“** įrašyk vieną skaičių pagal lentelę viršuje.
3. Stulpelis „pastaba“ – neprivalomas, trumpai, jei kažkas neaišku.
4. Nekeisk stulpelių pavadinimų, eilučių tvarkos ir teksto.

## Failo formatas

CSV, skyriklis „;“, koduotė UTF-8. Galima atidaryti:
- **Excel** – tiesiog atidaryti failą; išsaugoti per „Save As“ → „CSV UTF-8“;
- **Google Sheets** – „Failas → Importuoti“, skyriklis „Kabliataškis“; baigus „Failas → Atsisiųsti → CSV“ ir perrašyti savo failą.

## Kai abu baigsite

Įkelkite savo failus į repozitoriją ir paleiskite:

```
python compare.py S7
```

Jis suskaičiuos galutinių kategorijų pasiskirstymą (A, B ir sutarti tekstai, `tables/s7_audit.csv`), mūsų sutapimą žmogaus sprendimuose (kodai 0–3): sutapusių tekstų dalį ir Cohen κ, bei sutartus žmogaus sprendimus pagal raktinio žodžio grupes (`tables/s7_audit_groups.csv`). Rodomi abu sutapimo skaičiai, nes κ gali būti žemas, kai beveik visi tekstai gauna tą patį kodą.
