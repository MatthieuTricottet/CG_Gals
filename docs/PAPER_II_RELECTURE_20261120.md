# Paper II — relecture éditoriale du 20 novembre 2026

Version préparée pour la relecture des auteurs à partir de `fe3cc7e`, avec `b3d2910` et le relevé Gary Gate C comme références rédactionnelles. Aucun commit ni push. « Data availability » a été supprimé conformément à la dernière instruction.

Les PDF livrés sont `output/paper/paper.pdf` et `output/paper/online_supplement.pdf`. Les sources générées sont régénérées par `PYTHONPATH=src .venv_clean/bin/python -m src.main`, en mode `RENDER_PAPER_ONLY` ; les analyses ne sont pas relancées.

## Rédaction et résultats

L’abstract, la Sect. 3.1 et la discussion distinguent l’excès morphologique sur les populations complètes de l’incertitude sur un écart résiduel à séparation similaire. Le lecteur retrouve : +14,4 points sur les échantillons complets CG4–Control4C ; +3,4 [−7,7, +14,4] **à 175 kpc**, avec pentes séparées dans la fenêtre 95–216 kpc ; −0,4 [−12,0, +11,3] sur **40 paires de quartets** appariés en compacité. La comparaison ne couvre pas les CG4 les plus compacts : elle ne tranche pas leur écart résiduel et n’établit ni équivalence ni causalité.

La validation DS18 suit directement la définition de la classe E (990 galaxies appariées, 92 % avec TType ≤ 0). La récupération des 17 CG4 et l’accord DR7–DR16 restent dans le corps ; les ventilations par contrôle sont dans la note de la Table 3. Les votes manquants, les votes incertains et les classes DR16 conservées sans contrepartie vérifiée restent distincts.

La médiane des six séparations est notée `\langle R_{ij}\rangle`, comme dans Paper I ; leur moyenne arithmétique est notée `\overline{R}_{ij}`. Les quantités calculées ont été vérifiées avant de changer les symboles. Les valeurs des séparations sont inchangées. La légende de la Fig. 3 distingue points à masse fixée, courbes, recouvrement et résumés globaux/appariés. La légende de la figure des autres propriétés décrit désormais ses huit panneaux ; cette grande figure apparaît près de son commentaire.

La Sect. 3.5 conserve dans l’article la sensibilité du quenching intermédiaire : +3,9 [−11,8, +19,6] avec pente commune, contre +18,2 [+1,4, +35,1] avec pentes séparées à 175 kpc. Le déficit Hα dans la fibre, son atténuation après exclusions BPT et le changement concomitant d’échantillon sont distingués du quenching par sSFR totale et du résultat global à morphologie fixée. Les variantes complètes restent dans la Table S.1.

Les 46 identités de cœurs parmi 56 CG4 non isolés expliquent l’intérêt de Control4C dans la discussion ; les six hôtes offrant du recouvrement radial expliquent la fragilité de l’ajustement en rayon dans la Sect. 3.1. Les répétitions ont été coupées. Deux figures de distributions couleur déjà résumées par les coefficients ont été retirées du supplément.

Deux alimentations héritées ont été réconciliées avec les CSV gelés, sans nouveau calcul : intervalles et p-values morphologiques de la Table 4, et ligne regroupée de la Table 5 (intervalle fractionnaire [−0,009, 0,212], p = 0,0967). Les fichiers du gel, les JSON numériques et `additions_macros.tex` restent identiques octet pour octet à HEAD.

## Mesures avant/après

| Mesure | Avant | Après | Marge après révision |
|---|---:|---:|---|
| Mots de l’abstract, compteur A&A | 299 | 264 | 36 mots sur 300 |
| Pages du corps, titre à références incluses | 12 | 11 | 1 page sur 12 |
| Pages des annexes intégrées | 7 | 7 | 1 page sur 8 |
| Pages du supplément séparé | 8 | 7 | Canal distinct à confirmer |

Appendix A commence page **13 avant / 12 après** ; la dernière page est **19 avant / 18 après**. Les comptes sont mesurés sur les PDF compilés.

Le corps passe de **5 507 à 4 517 mots**, soit **−990 mots (−18,0 %)**. Même méthode avant/après : `texcount -merge`, rubrique « Words in text », sur la source rendue entre `\maketitle` et `\begin{acknowledgements}`, avant les annexes. Ce compteur exclut l’abstract, les légendes et les références ; la suppression de Data availability contribue à la baisse. Le compteur de la classe A&A est instrumenté uniquement pour imprimer son résultat dans le journal, sans modifier son algorithme. Le compte initial de l’abstract a été obtenu en compilant le préambule et l’abstract sauvegardés avant révision.

## A&A et décision restante

La [politique officielle sur la longueur des articles](https://www.aanda.org/2025-press-releases/3045-shaping-a-sustainable-future-for-a-a-new-policy-on-paper-length), publiée le 16 janvier 2025 et mise à jour le 12 mars 2026, a été consultée directement le 20 novembre 2026. Pour un regular paper dont le premier auteur est en France, les plafonds sans dépassement sont 12 pages de corps et 8 pages d’annexes ; le supplément de longueur est de 150 € par page. La mise à jour précise que le compte est établi à la soumission. Le [guide A&A](https://www.aanda.org/doc_journal/instructions/aadoc.pdf) impose un abstract de 300 mots au plus et décrit les annexes après les références, publiées en copie prête à publier.

**La gratuité de l’ensemble article + supplément ne peut pas encore être affirmée.** Le guide décrit les annexes pour le texte, les figures et les tables ; ses possibilités de données au CDS ou de multimédia ne confirment pas un canal séparé pour ce PDF composite. Le statut de `online_supplement.pdf` doit être confirmé avec la rédaction. Son ajout sans nouvelle composition aux annexes représente **14 pages (7 + 7), soit 6 au-delà du plafond**, potentiellement 900 € de dépassement au tarif indiqué. Une intégration avec recomposition pourrait modifier ce compte. Les auteurs doivent choisir entre un canal séparé explicitement accepté, une archive externe durable pour les diagnostics secondaires, ou une nouvelle réduction/intégration. Les résultats indispensables, dont la dépendance du quenching aux pentes, sont déjà dans l’article.

## Vérifications et fichiers

- Génération normale : pdflatex, BibTeX, deux passes supplémentaires ; supplément compilé deux fois.
- **16 tests ciblés réussis** : `test_generate_report.py`, `test_submission_qc.py`, `test_gz1_recovery.py`, `test_consistency_gate.py` (`PYTHONPATH=src`).
- Validation Jinja/LaTeX des deux templates et contrôles de style Gary réussis ; `git diff --check` réussi.
- **81 vérifications indépendantes réussies** : effectifs et dénominateurs contre les CSV gelés, résultats imprimés, Table 4, notations, renvois, intégrité des données, pagination et abstract. Journaux finaux sans référence/citation indéfinie, label multiple ni débordement signalé.
- Les pages des deux PDF ont été rendues et inspectées. Relecture continue de l’abstract, de la Sect. 3.1, de la légende de compacité, de la discussion et de la conclusion ; vérification des placements des figures et tables dans leurs annexes respectives.
- Le contrôle historique `phase4_checks.py` a aussi été exécuté : il **n’est pas déclaré réussi**. Il lit le cache antérieur à la récupération DR7, attend l’ancien ordre/format des tables et examine des tables désactivées. Les effectifs effectivement publiés ont été contrôlés directement contre le gel à sa place ; son journal est conservé.

Sources modifiées :

- `src/paper_template/paper_template.tex` : révision du manuscrit, notations, tables gelées et placements ; suppression de Data availability.
- `src/paper_template/online_supplement_template.tex` : déduplication, renvois et regroupement des diagnostics.
- `src/freeze_values.py` : lecture directe des intervalles morphologiques gelés de la Table 4.
- `src/second_order_battery.py` : libellé de la médiane conforme à Paper I, sans changement du calcul.
- `analysis/gary_r2/style_checks.py` : contrôle du corps compatible avec l’absence de Data availability et excluant les remerciements.

Les `.tex`, PDF et fichiers auxiliaires correspondants sont régénérés dans `output/paper/`. Les mesures et contrôles sont conservés dans `output/editorial_review_20261120/metrics.json`, `checks.json` et les journaux voisins. Le diff reste disponible pour relecture.
