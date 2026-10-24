# -*- coding: utf-8 -*-
"""v10 POUR BRINDA : lecture de chaque figure trait par trait, après inspection des images."""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "OXYFIX_Presentation_POUR_MOI_v11.pptx"
FIG = ROOT / "results" / "figures"

CREAM = RGBColor(0xFA, 0xF3, 0xE7)
CARD = RGBColor(0xF1, 0xE4, 0xCE)
BROWN = RGBColor(0x4A, 0x32, 0x22)
TEXT = RGBColor(0x2E, 0x1F, 0x14)
TERRA = RGBColor(0xB0, 0x3A, 0x08)
OLIVE = RGBColor(0x44, 0x54, 0x14)
SAND = RGBColor(0x7A, 0x5A, 0x28)

prs = Presentation()
prs.slide_width = Cm(33.87)
prs.slide_height = Cm(19.05)
BLANK = prs.slide_layouts[6]


def add_txt(slide, x, y, w, h, text, size=14, bold=False, color=TEXT, align=PP_ALIGN.LEFT, italic=False):
    tx = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color
    p.alignment = align
    return tx


def add_lines(slide, x, y, w, h, lines, size=11.5, color=TEXT, bold_first=False):
    tx = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tx.text_frame
    tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold_first and i == 0
        p.space_after = Pt(1)
    return tx


def base(slide, title):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = CREAM
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(0), Cm(0), Cm(33.87), Cm(0.28))
    bar.shadow.inherit = False
    bar.fill.solid(); bar.fill.fore_color.rgb = TERRA
    bar.line.fill.background()
    add_txt(slide, 1.2, 0.65, 31.5, 1.3, title, size=25, bold=True, color=BROWN)


def figure_slide(title, fname, phrase, lecture, conclusion):
    path = FIG / fname
    s = prs.slides.add_slide(BLANK)
    base(s, title)
    if path.exists():
        with Image.open(path) as im:
            iw, ih = im.size
        max_w, max_h = 28.8, 7.9
        w_cm = max_w
        h_cm = max_w * ih / iw
        if h_cm > max_h:
            h_cm = max_h
            w_cm = max_h * iw / ih
        x = (33.87 - w_cm) / 2
        s.shapes.add_picture(str(path), Cm(x), Cm(1.95), width=Cm(w_cm), height=Cm(h_cm))
    add_txt(s, 1.2, 10.05, 31.5, 1.0, phrase, size=16.5, bold=True, color=TERRA, align=PP_ALIGN.CENTER)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(1.2), Cm(11.15), Cm(31.5), Cm(5.75))
    box.shadow.inherit = False
    box.fill.solid(); box.fill.fore_color.rgb = CARD
    box.line.fill.background()
    add_lines(s, 1.9, 11.35, 30.1, 5.4, ["Lecture de la figure, trait par trait :"] + ["• " + l for l in lecture], size=10)
    add_txt(s, 1.2, 17.05, 31.5, 1.8, "Conclusion à dire : " + conclusion, size=12, bold=True, color=OLIVE)
    return s


def figure_slide_split(title, fname, phrase, lecture, conclusion):
    """Figures en 4 panneaux : explications sur une page, figure plein ecran sur la suivante."""
    path = FIG / fname
    # Page 1 : explications
    s = prs.slides.add_slide(BLANK)
    base(s, title)
    add_txt(s, 1.2, 3.1, 31.5, 1.4, phrase, size=19, bold=True, color=TERRA, align=PP_ALIGN.CENTER)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(1.2), Cm(4.9), Cm(31.5), Cm(10.6))
    box.shadow.inherit = False
    box.fill.solid(); box.fill.fore_color.rgb = CARD
    box.line.fill.background()
    add_lines(s, 1.9, 5.3, 30.1, 9.9, ["Lecture de la figure, trait par trait :"] + ["• " + l for l in lecture], size=13)
    add_txt(s, 1.2, 15.9, 31.5, 2.2, "Conclusion à dire : " + conclusion, size=14, bold=True, color=OLIVE)
    # Page 2 : figure plein ecran
    s2 = prs.slides.add_slide(BLANK)
    base(s2, title + " — la figure en grand")
    if path.exists():
        with Image.open(path) as im:
            iw, ih = im.size
        max_w, max_h = 31.9, 16.6
        w_cm = max_w
        h_cm = max_w * ih / iw
        if h_cm > max_h:
            h_cm = max_h
            w_cm = max_h * iw / ih
        x = (33.87 - w_cm) / 2
        y = 2.1 + (16.6 - h_cm) / 2
        s2.shapes.add_picture(str(path), Cm(x), Cm(y), width=Cm(w_cm), height=Cm(h_cm))
    return s2


def text_slide(title, lines, size=16.5):
    s = prs.slides.add_slide(BLANK)
    base(s, title)
    y = 3.2
    for head, rest in lines:
        add_txt(s, 1.6, y, 30.6, 2.6, head, size=size + 1, bold=True, color=TERRA)
        if rest:
            add_txt(s, 1.6, y + 0.92, 30.6, 2.3, rest, size=size - 1.5, color=TEXT)
        y += 3.05
    return s


# 1 titre
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = CREAM
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(0), Cm(0), Cm(33.87), Cm(0.55))
bar.shadow.inherit = False
bar.fill.solid(); bar.fill.fore_color.rgb = TERRA
bar.line.fill.background()
add_txt(s, 1.6, 2.4, 30.5, 3.4, "Apprentissage automatique pour la prédiction de traits fonctionnels microbiens", size=31, bold=True, color=BROWN)
add_txt(s, 1.6, 5.6, 30.5, 1.6, "Le cas de la tolérance à l'oxygène de la fixation biologique d'azote", size=23, bold=True, color=TERRA)
add_txt(s, 1.6, 7.8, 30.5, 1.2, "Projet OXYFIX — version pour présenter (chaque figure lue trait par trait)", size=17, color=SAND, italic=True)
card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(1.6), Cm(10.2), Cm(30.5), Cm(4.4))
card.shadow.inherit = False
card.fill.solid(); card.fill.fore_color.rgb = CARD
card.line.fill.background()
add_txt(s, 2.4, 10.8, 29.0, 3.4, "Question centrale : un modèle qui prédit « aérobie / anaérobie » à partir des protéines de la nitrogénase lit-il un signal propre à l'enzyme — ou simplement l'identité de la lignée bactérienne ?", size=16.5, color=TEXT)
add_txt(s, 1.6, 15.6, 30.5, 2.2, "LEMOGUE JIOFACK Brinda — Master 2 Recherche, Intelligence Artificielle\nURIFIA, Université de Dschang — Encadreur : Prof. BOMGNI Alain Bertrand", size=14.5, color=BROWN)

text_slide("Le point de départ", [
    ("Le travail antérieur", "L'étude menée avant nous dans ce laboratoire a publié un score élevé : accuracy 0,9225 et AUC 0,9772, avec le descripteur CKSAAP (1 600 nombres qui décrivent la composition des protéines) calculé sur NifH, NifD et NifK de 787 souches, en découpage aléatoire uniquement."),
    ("Son interprétation", "Ces descripteurs y ont été lus comme des déterminants de la tolérance à l'oxygène, c'est-à-dire comme la cause biologique du comportement des souches face à l'oxygène."),
    ("Notre travail", "Reprendre exactement ce même jeu de données et demander, avec des contrôles : que mesure réellement ce score ?"),
])
text_slide("Les mots du projet, en français simple (1/2)", [
    ("Diazotrophe / nitrogénase / NifH, NifD, NifK", "Un diazotrophe est une bactérie capable de fixer l'azote de l'air. La nitrogénase est l'enzyme qui fait ce travail ; elle est faite de plusieurs pièces, dont NifH, NifD et NifK. Problème : l'oxygène casse cette enzyme en quelques minutes."),
    ("Descripteur / CKSAAP / découpage", "Un descripteur transforme une séquence (la suite des acides aminés) en nombres comparables par l'ordinateur ; CKSAAP en produit 1 600. Le découpage (split) partage les souches entre apprentissage et test : aléatoire = on mélange tout ; par genre disjoint = aucun genre des deux côtés à la fois."),
    ("AUC / accuracy / rappel", "L'AUC note le classement : 1,00 = parfait, 0,50 = pile ou face. L'accuracy est la part de bonnes réponses. Le rappel, par classe, dit quelle part des aérobies (ou des anaérobies) le modèle retrouve : il montre quelle classe souffre quand le test durcit."),
    ("Pointillé, dans presque toutes les figures", "Un trait en pointillés est presque toujours une ligne de repère, pas un résultat : 0,50 = le hasard, 0,90 ou 0,70 = des seuils qu'on s'est fixés pour juger. Les traits pleins avec des points ou des barres, eux, sont les résultats."),
])
text_slide("Les mots du projet, en français simple (2/2)", [
    ("Genre mixte / mono-classe / LOGO", "Un genre mixte contient des aérobies ET des anaérobies ; mono-classe = une seule sorte. LOGO (leave-one-genus-out) : on retire un genre entier et on teste dessus. Sans genre mixte, ce test ne peut pas exister."),
    ("Contrôles RecA, GyrB, RpoB / famille, ordre, clade", "Ce sont des protéines de ménage (elles entretiennent la cellule), sans rapport avec l'oxygène, soumises au même test que Nif. La famille regroupe plusieurs genres, l'ordre plusieurs familles ; un clade est une branche de l'arbre de parenté (la phylogénie)."),
    ("Permutation / calibration / SMOTE / embedding", "Permutation : mélanger les étiquettes 1 000 fois pour fabriquer le score du hasard. Calibration : vérifier que « 90 % sûr » veut bien dire vrai 90 fois sur 100. SMOTE : fabriquer de faux exemples pour équilibrer les classes. Embedding (ESM-2) : résumer chaque séquence en 1 280 nombres appris par une grande IA."),
    ("Barres d'erreur (les petites « moustaches »)", "Le petit trait noir au bout d'une barre ou d'un point montre de combien le résultat bouge quand on refait le tirage (l'écart-type). Moustache courte = résultat stable ; moustache longue = résultat fragile."),
])

figure_slide("Phase 0 et 3 · D'où viennent les étiquettes ?", "phase3_corpus_and_eval.png",
             "787 étiquettes confrontées à BacDive : 489 documentées, 221 à vérifier, 77 manquantes.",
             ["À gauche (panneau A) : trois barres = le tri après vérification BacDive. La grande barre verte = 489 souches retenues (Strict, étiquettes prouvées). La petite barre bleue = 70 « candidats » (plusieurs fiches qui disent la même chose). La barre rouge = 228 exclues (étiquettes floues ou contredites).",
              "À droite (panneau B) : pour chaque corpus, deux barres. La bleue = découpage aléatoire, la marron = découpage par genre, plus dur. La hauteur se lit sur l'axe de gauche (AUC, de 0,5 à 1,0).",
              "Les pointillés horizontaux sont des repères : rouge à 0,90 (« très bon »), orange à 0,70 (« moyen »). Exemple de lecture : sur Full 787, la bleue touche 0,97 et la marron 0,86 — même corpus, seul le test change.",
              "Le texte en haut le rappelle : les étiquettes héritées n'ont pas été modifiées ; on a filtré, pas réécrit."],
             "avant de juger un modèle, on a d'abord jugé les étiquettes ; et on garde deux corpus, le complet (787) et le propre (489), pour vérifier que le diagnostic ne dépend pas du bruit.")
figure_slide("Phase 1 · Nif contre les protéines de contrôle", "phase1_nif_vs_controls.png",
             "RecA, GyrB, RpoB égalent Nif : le score lit l'organisme, pas la nitrogénase.",
             ["À gauche : quatre barres horizontales, une par protéine. La bleue, tout en bas, c'est Nif, notre cible. Les trois marron au-dessus sont les contrôles (RecA, GyrB, RpoB). Toutes les barres arrivent au même endroit, entre 0,96 et 0,98 sur l'axe du bas (AUC).",
              "Deux pointillés verticaux servent de repères : l'orange à 0,70 (seuil « mixte ») et le rouge à 0,90 (seuil « taxonomique ») ; toutes les barres les dépassent largement.",
              "À droite, les courbes ROC : en bas (FPR) la part de fausses alertes, à gauche (TPR) la part de vraies détections. Le pointillé noir en diagonale = le hasard. Les quatre courbes colorées montent presque tout droit dans le coin en haut à gauche : c'est la forme d'un très bon classement.",
              "Mais les quatre courbes sont collées les unes aux autres, et la légende donne leurs AUC : 0,971 ; 0,969 ; 0,965 ; 0,974 — la rouge, tout en haut, est un contrôle (RpoB)."],
             "si trois protéines sans rapport avec l'oxygène font aussi bien que la nitrogénase, le modèle reconnaît la bactérie, pas l'enzyme.")
figure_slide("Phase 1 · Le rappel par classe", "phase1_per_class_recall.png",
             "Quand le découpage durcit, ce sont les aérobies qu'on perd.",
             ["Deux panneaux jumeaux : à gauche Nif, à droite RecA. Dans chaque panneau, quatre barres : deux foncées à gauche (découpage aléatoire), deux plus claires à droite (découpage par genre). La hauteur se lit sur l'axe (rappel, de 0 à 1) ; le chiffre est écrit au-dessus de chaque barre.",
              "Exemple de lecture, panneau Nif : première barre 0,93 (aérobies en aléatoire), troisième barre 0,68 (aérobies en genre) : c'est elle qui s'effondre. La quatrième barre, anaérobies en genre, reste à 0,83.",
              "Le pointillé horizontal à 0,5 sert de repère bas. Dans le panneau RecA, la chute des aérobies est encore plus forte : 0,92 tombe à 0,45.",
              "Pourquoi : les anaérobies vivent dans quelques grandes familles faciles à reconnaître ; les aérobies sont apparus plusieurs fois, dans des familles différentes. Le modèle perd donc d'abord ceux qu'il reconnaissait par la famille."],
             "la classe qui souffre révèle ce que le modèle avait appris : la parenté, pas un mécanisme de protection, sinon ce sont les aérobies — ceux qui ont un mécanisme — qui tiendraient le mieux.")
figure_slide("Phase 2 · La courbe de décroissance", "phase2_decay_curve.png",
             "0,971 -> 0,856 -> 0,549 : le même modèle, du presque parfait au pile ou face.",
             ["La courbe bleue avec des ronds, c'est Nif (la cible) ; la courbe marron avec des carrés, c'est RecA (le contrôle). Chaque point porte son chiffre écrit juste au-dessus.",
              "De gauche à droite, le test devient plus dur : aléatoire, puis genre disjoint (aucun genre des deux côtés), puis LOGO (un genre entier jamais vu). Les deux courbes descendent en pente régulière.",
              "Trois pointillés horizontaux sont des repères pour l'oeil : rouge à 0,90 (très bon), orange à 0,70 (moyen), gris à 0,50 (le hasard, tout en bas). À droite, les deux courbes se rejoignent presque sur la ligne du hasard : 0,549 et 0,560.",
              "Au milieu, la bleue est au-dessus de la marron (0,856 contre 0,723) : c'est le seul endroit où Nif se détache — sur un seul tirage, donc à confirmer (ce sera la Phase 5)."],
             "le même modèle vaut 0,97 ou 0,55 selon la règle du test ; un score ne veut rien dire sans son protocole, et sur un genre jamais vu, cible et contrôle valent le hasard.")
figure_slide("Phase 3b · Le corpus propre ne change pas le diagnostic", "phase3b_strict_controls_random.png",
             "Sur les 489 étiquettes documentées, les contrôles restent devant Nif.",
             ["Une seule barre bleue : Nif, à 0,955. Trois barres marron : les contrôles, à 0,969, 0,958 et 0,970. Le chiffre est écrit au-dessus de chaque barre.",
              "L'axe de gauche commence à 0,5 (le hasard) et monte à 1,0 ; attention, il grossit les différences : visuellement les barres semblent très différentes, en réalité elles tiennent entre 0,955 et 0,970.",
              "Le pointillé rouge à 0,90 est le repère « très bon » : tout le monde le dépasse, y compris les contrôles.",
              "Ce test répond à l'objection « vos contrôles gagnent parce que vos étiquettes sont fausses » : ici, chaque étiquette est prouvée par BacDive, et les contrôles restent devant."],
             "nettoyer les étiquettes ne sauve pas la nitrogénase ; la parité n'était pas un bruit d'étiquetage.")
figure_slide("Phase 3b · Décroissance sur le corpus propre", "phase3b_strict_decay.png",
             "Sur Strict, le LOGO devient indéfini : les genres assez grands sont tous mono-classe.",
             ["Deux panneaux avec les mêmes courbes que la Phase 2 : bleue = Nif, marron = RecA, pointillés = repères 0,90, 0,70 et hasard 0,50.",
              "À gauche (corpus complet 787) : la courbe complète en trois marches, qui finit au hasard (0,55 et 0,56). C'est notre référence.",
              "À droite (corpus propre 489) : il n'y a que deux points, et les courbes restent tout en haut (0,96 à 0,98). Le troisième point, LOGO, n'existe pas : le titre du panneau le dit, « LOGO absent : impossible (genres mono-classe) ».",
              "Pourquoi la droite reste haute : après nettoyage, les genres assez grands ne contiennent qu'une seule classe chacun. Retirer un tel genre n'enlève rien au modèle, qui retrouve la réponse grâce aux genres voisins restés dans l'apprentissage. Un test facile peut donc avoir l'air excellent."],
             "sur le corpus propre, le test le plus dur disparaît silencieusement ; des scores élevés là-dessus ne rassurent pas, ils signalent que le test a cessé de tester.")
figure_slide("Phase 4 · Changer de classifieur ne change rien", "phase4_models_strict.png",
             "Cinq algorithmes, même diagnostic : le facteur limitant est le protocole, pas le modèle.",
             ["À gauche : cinq barres d'accuracy en découpage aléatoire (ExtraTrees en vert, les autres en bleu). Elles sont toutes à 0,929 — le chiffre est écrit en haut de chacune. Le pointillé rouge à 0,9225 est la valeur publiée par l'étude précédente : on est au même niveau.",
              "À droite : les mêmes cinq modèles, mais en AUC de découpage par genre (barres marron, de 0,961 à 0,978). Le pointillé rouge à 0,9772 est l'AUC publiée par l'étude précédente — mais attention, elle avait été obtenue en aléatoire, pas en genre : la figure le dit elle-même, « PAS genre ».",
              "Le piège que cette figure nous apprend à éviter : ne jamais comparer l'accuracy d'un côté avec l'AUC de l'autre, ni un protocole avec un autre. Le titre en haut le rappelle en toutes lettres.",
              "Conclusion visuelle : les barres d'un même panneau sont interchangeables ; ce qui fait bouger les chiffres, c'est le panneau (le protocole), pas la barre (le modèle)."],
             "changer d'algorithme déplace le score de quelques millièmes ; changer de protocole le fait passer de 0,97 au hasard. Le problème n'est pas l'outil, c'est la règle du test.")
figure_slide("Phase 4b · Le protocole d'origine, rejoué tel quel", "phase4b_alida_combo_full_vs_strict.png",
             "CKSAAP + XGBoost reproduits exactement : le score est vrai, c'est son interprétation qui change.",
             ["À gauche, l'accuracy en aléatoire : barre bleue = corpus complet (0,899), barre marron = corpus propre (0,908). Le pointillé rouge à 0,9225 est la valeur publiée : nos deux barres sont juste en dessous, à moins de deux souches de test près.",
              "À droite, l'AUC en découpage par genre : le complet tombe à 0,858, le propre reste à 0,970. Le pointillé rouge à 0,9772 est la valeur publiée en aléatoire — on ne doit pas la comparer directement à ces barres, qui sont en genre.",
              "Pourquoi le propre (0,970) dépasse le complet (0,858) en genre : après nettoyage, les genres sont presque tous mono-classe, donc le retrait d'un genre n'enlève presque rien (même piège qu'en Phase 3b).",
              "Ce test utilise exactement la recette de départ (CKSAAP + XGBoost, graine 42) : il montre que le score publié se reproduit chez nous, sans erreur de calcul de part ni d'autre."],
             "le score d'origine est vrai dans son protocole ; notre travail ne conteste pas son calcul, il conteste ce qu'on peut conclure à partir de lui.")
figure_slide_split("Phase 5 · Vingt tirages au lieu d'un", "phase5_repeated_auc.png",
             "Des moyennes avec écarts-types : l'écart Nif/contrôles en genre tombe à environ +0,03.",
             ["Quatre panneaux : en haut l'aléatoire, en bas le genre ; à gauche le complet, à droite le propre. Dans chaque panneau, un point par protéine, avec le chiffre « moyenne +/- écart-type » écrit à côté.",
              "Le point est la moyenne des 20 tirages ; la petite ligne horizontale qui le traverse (la moustache) montre de combien ça bouge d'un tirage à l'autre. En haut, les moustaches sont minuscules : tout est stable entre 0,97 et 0,99.",
              "En bas à gauche (le panneau qui compte) : le point bleu de Nif est à 0,945, ceux des contrôles à 0,911-0,916, avec des moustaches plus longues. L'écart réel est donc d'environ +0,03 — pas les +0,116 qu'un seul tirage avait montrés.",
              "Le pointillé vertical à 0,5, à gauche de chaque panneau, rappelle où est le hasard : tout le monde en est loin ici ; ce n'est pas là que se joue la question, mais dans l'écart entre le point bleu et les autres."],
             "répété 20 fois, l'avantage de Nif en genre devient petit (+0,03) mais stable ; c'est ce chiffre honnête qu'il faut présenter, pas le spectaculaire d'un seul tirage.")
figure_slide("Phase 5b · Un genre jamais vu, genre par genre", "phase5b_logo_auc_distribution.png",
             "8 genres mixtes seulement sur 328 : la généralisation à un genre inconnu n'est pas démontrée.",
             ["Ce sont des histogrammes : des barres collées les unes aux autres. En bas, l'AUC obtenue pour un genre ; à gauche, combien de genres sont tombés dans chaque tranche. Les barres bleu-gris sont pour Nif, les beiges pour RecA.",
              "Exemple de lecture : la grande barre bleue entre 0,3 et 0,4, qui monte jusqu'à 3, veut dire que pour 3 genres, Nif a obtenu une AUC entre 0,3 et 0,4 — pire que le hasard. À l'inverse, les barres tout à droite, vers 0,95-1,0, sont les genres parfaitement réussis.",
              "Le pointillé rouge vertical à 0,5 est le hasard : les barres sont étalées des deux côtés, de 0,0 à 1,0. C'est l'image même de « on ne sait pas » : tout dépend du genre tiré.",
              "Le panneau de droite est entièrement vide : sur le corpus propre, il n'existe aucun genre mixte, donc aucune barre à dessiner. Le vide est lui-même le résultat. Au total : 8 genres seulement, moyenne 0,549, intervalle [0,367 ; 0,753]."],
             "avec 8 genres étalés de 0 à 1, on ne peut ni déclarer la réussite ni l'échec ; et il ne faut jamais présenter l'AUC poolée (0,969), qui mélange les genres et réintroduit la reconnaissance des familles.")
figure_slide_split("Phase 5b · Le modèle est-il trop sûr de lui ?", "phase5b_calibration.png",
             "En découpage par genre, les probabilités annoncées s'écartent de la réalité.",
             ["Quatre panneaux : en haut l'aléatoire, en bas le genre ; à gauche le complet, à droite le propre. Trois traits dans chaque panneau : le pointillé gris en diagonale, marqué « Parfaite » ; la courbe bleue = Nif ; la courbe marron = RecA. Chaque point est un paquet de prédictions semblables.",
              "Comment lire un point : en bas, la probabilité annoncée par le modèle ; à gauche, la part réelle d'aérobies dans ce paquet. Exemple, en bas à gauche (Full, genre), point marron vers 0,1 en bas et 0,48 en hauteur : le modèle annonce « 10 % aérobie » alors qu'en réalité 48 % le sont.",
              "Si le point est sur la diagonale, le modèle dit vrai. Au-dessus de la diagonale, il annonce moins que la réalité (trop prudent) ; en dessous, il annonce plus (trop sûr de lui).",
              "En haut (aléatoire), les courbes collent à la diagonale : le modèle est honnête. En bas (genre), elles s'en écartent : dès qu'on s'éloigne des familles connues, la confiance affichée ne vaut plus rien, même si le classement reste à peu près correct."],
             "un modèle peut bien classer et mal annoncer sa confiance ; en genre inconnu, ses « 90 % sûr » ne doivent pas être crus.")
figure_slide("Phase 6 · Prédire par la parenté seule (GTDB)", "phase6_gtdb_nn_vs_hgb.png",
             "0,972 sans modèle de séquence : les plus proches parents dans l'arbre suffisent presque.",
             ["Deux panneaux : à gauche le corpus complet, à droite le propre. Dans chaque panneau, deux paires de barres : à gauche « genre », à droite « aléatoire ».",
              "Dans chaque paire : la barre marron = la prédiction par la parenté seule, écrite « GTDB plus proche voisin (k=5) » dans la légende ; la barre bleue = le vrai modèle (Nif CKSAAP + HGB). Le k=5 veut dire : pour deviner l'étiquette d'une souche, on regarde ses 5 plus proches parents dans l'arbre GTDB et on prend l'étiquette majoritaire — comme deviner la langue de quelqu'un en écoutant sa famille.",
              "Exemple de lecture, panneau de gauche, paire « aléatoire » : marron et bleue arrivent à la même hauteur, environ 0,97. La parenté seule fait donc aussi bien que l'analyse des protéines.",
              "Les petits traits noirs en haut des barres sont les barres d'erreur (de combien ça bouge sur 20 tirages) ; la barre marron du « genre » en a une plus longue : la parenté seule est moins régulière quand on retire des genres. Le pointillé en bas à 0,5 rappelle le hasard."],
             "si deviner par la famille donne le même score qu'analyser la nitrogénase, le score ne prouve rien sur la nitrogénase : on a mesuré la part de la parenté au lieu de la soupçonner.")
figure_slide("Phase 6b · Vingt et un nombres simples contre CKSAAP", "phase6b_trivial_vs_cksaap.png",
             "Longueur + composition des protéines atteignent presque les 1 600 dimensions de CKSAAP.",
             ["Deux panneaux : à gauche les résultats sur Nif, à droite sur RecA (le contrôle). Dans chaque panneau, quatre paires de barres : complet-aléatoire, complet-genre, propre-aléatoire, propre-genre.",
              "Dans chaque paire : la barre marron = le descripteur minuscule (longueur + composition en acides aminés, 21 nombres) ; la barre bleue = CKSAAP complet (1 600 nombres). Les petits traits noirs en haut sont les écarts-types sur 20 tirages.",
              "Exemple de lecture, première paire à gauche : marron environ 0,96, bleue environ 0,97. Avec 21 nombres on a presque tout le score des 1 600.",
              "L'axe commence à 0,4 et le pointillé en bas marque 0,5 (hasard) : toutes les barres sont très au-dessus, l'enjeu n'est pas là ; l'enjeu est la petite différence entre marron et bleue dans chaque paire — et elle est petite partout, même pour le contrôle à droite."],
             "si la composition globale, sans aucun ordre des lettres, donne presque tout le score, le modèle lit une propriété de l'organisme entier — exactement ce que la parenté transmet.")
figure_slide("Phase 6d · Le GC du génome ajouté", "phase6d_gc_vs_trivial_cksaap.png",
             "Avec le GC en plus, CKSAAP n'ajoute qu'environ un point d'AUC.",
             ["Deux panneaux (complet à gauche, propre à droite) ; dans chacun, deux groupes : « GC seul » et « GC+longueur+AAC ». Dans chaque groupe, la barre bleue = aléatoire, la barre marron = genre (découpage plus dur).",
              "À gauche de chaque panneau, le GC seul : barres plus basses (environ 0,78 en bleu, 0,62 en marron) avec de longues moustaches noires — en genre, il s'effondre presque et il est instable.",
              "À droite de chaque panneau, le GC ajouté à la longueur et à la composition : les barres montent à 0,96-0,97 et se resserrent. Il ne reste alors qu'environ un point d'écart avec CKSAAP complet (0,974).",
              "Le GC, c'est la part des lettres G et C dans l'ADN du génome ; chaque espèce a son GC typique, comme une empreinte digitale de l'organisme, sans aucun rapport direct avec la nitrogénase."],
             "trois signatures de l'organisme entier (longueur, composition, GC) égalent presque le descripteur de la nitrogénase : la part propre de l'enzyme dans le score tient en environ un point d'AUC.")
figure_slide("Phase 6c · NifH, NifD, NifK une par une", "phase6c_nif_families.png",
             "Chaque sous-unité fait comme les trois réunies : aucune ne porte un signal qui se détache.",
             ["Deux panneaux : corpus complet à gauche, propre à droite. Quatre groupes de barres : « Nif combiné » (les trois pièces ensemble), puis NifH, NifD, NifK séparément.",
              "Dans chaque groupe : la barre bleue = aléatoire (toujours la plus haute), la barre marron = genre (plus dur, toujours plus basse). Les moustaches noires montrent la variation sur 20 tirages.",
              "Exemple de lecture : sur le complet, les quatre barres bleues sont collées entre 0,96 et 0,98 ; les quatre marron entre 0,90 et 0,95. Aucune pièce ne dépasse les autres, ni en bleu ni en marron.",
              "Attention aux effectifs : NifH est identifiable chez 778 souches, NifD chez 385, NifK chez 365. Ce sont les séquences que l'annotation a permis de reconnaître dans les fichiers ; cela ne veut pas dire que les autres souches n'ont pas la pièce dans leur génome."],
             "si le secret de la tolérance était écrit dans une pièce précise de la machine, cette pièce devait dépasser les autres ; aucune ne le fait : le signal est diffus, comme la composition héritée de la lignée.")
figure_slide("Phase 7 · Une IA qui lit les protéines (ESM-2)", "phase7_esm_vs_cksaap.png",
             "Même avec ESM-2, le contrôle RecA reste devant en aléatoire.",
             ["Deux panneaux (complet, propre) ; quatre groupes de barres : Nif-aléatoire, Nif-genre, RecA-aléatoire, RecA-genre. Dans chaque groupe : la barre violette = ESM-2 (la grande IA), la barre bleue = CKSAAP (le descripteur classique).",
              "Exemple de lecture, premier groupe à gauche : violette environ 0,946, bleue environ 0,974 — la classique dépasse même l'IA. Troisième groupe (RecA aléatoire) : les deux montent à environ 0,965-0,974, et l'IA du contrôle dépasse l'IA de la cible (premier groupe).",
              "Les moustaches d'ESM-2 sont plus longues, surtout en genre : la riche représentation est aussi plus variable quand on retire des genres.",
              "Le modèle ESM-2 est « gelé » : entraîné ailleurs, sur des millions de protéines publiques, jamais sur nos étiquettes. Il ne peut donc pas avoir triché ; mais ses nombres contiennent l'histoire évolutive des protéines, donc la parenté."],
             "même la représentation la plus riche lit d'abord l'organisme : le contrôle dépasse la cible en aléatoire, et CKSAAP n'est pas battu. L'objection « votre descripteur est trop pauvre » est fermée.")
figure_slide_split("Phase 8 · Mélanger les étiquettes 1 000 fois", "phase8_permutation_nulls.png",
             "Vraie AUC très loin des nulles (p = 0,001) ; mais la nulle par genres monte à 0,60.",
             ["Quatre panneaux : en haut le complet, en bas le propre ; à gauche Nif, à droite RecA. Dans chaque panneau, deux « montagnes » (histogrammes) : en bas, le score obtenu ; à gauche, combien de mélanges (sur 1 000) sont tombés dans chaque tranche. Plus la montagne est haute, plus le hasard produit souvent ce score-là.",
              "La montagne saumon (centrée sur 0,50) = étiquettes mélangées complètement au hasard. La montagne verte (décalée vers 0,60) = mélangées en respectant les genres. Là où elles se recouvrent, la couleur devient brunâtre : c'est juste leur superposition.",
              "Le grand trait vertical noir, tout à droite (0,95-0,98), c'est le vrai score : en 1 000 mélanges, le hasard n'y arrive jamais (p = 0,001). Le petit pointillé vertical à 0,5 marque le centre du hasard pur.",
              "La comparaison qui compte est entre les montagnes : respecter les genres déplace le hasard de 0,50 à environ 0,60. Cette différence (environ +0,10) est la part de score que la seule structure des familles fabrique. Et regardez les panneaux de droite : RecA, le contrôle, a exactement la même allure."],
             "le signal est réel (le hasard ne l'atteint jamais), mais il est partagé avec le contrôle ; et la structure des familles fabrique à elle seule un dixième du score. Significatif ne veut pas dire propre à la nitrogénase.")
text_slide("Phase 8 · La validation croisée imbriquée", [
    ("C'est quoi, simplement ?", "Imaginez un élève : la boucle intérieure, ce sont ses exercices d'entraînement, qui servent à régler sa méthode (choisir les réglages du modèle). La boucle extérieure, c'est l'examen final : des genres entiers jamais touchés pendant les réglages, notés une seule fois. Si l'élève réglait sa méthode en regardant l'examen, sa note ne vaudrait rien."),
    ("Les taux de l'examen", "Nif : 0,951 (corpus complet) et 0,972 (corpus propre). RecA : 0,891 et 0,914. Les réglages choisis à l'intérieur étaient presque toujours les mêmes (profondeur 3 ou 6, apprentissage 0,1)."),
    ("Ce qu'on peut dire, honnêtement", "Même avec cette procédure propre, Nif garde un petit avantage quand on retire des genres (+0,06 environ). C'est compatible avec un petit signal propre à la cible. Mais cela ne prouve pas un mécanisme : le test décisif reste le genre mixte jamais vu, et il n'y en a presque pas. On présente donc ce résidu comme une question ouverte et mesurée, pas comme une victoire."),
])
figure_slide("Phase 9 · Plus dur que le genre : famille et ordre", "phase9_family_order_decay.png",
             "Les scores tiennent parce que 89 à 95 % des familles et ordres sont mono-classe.",
             ["Deux panneaux (complet, propre) avec des courbes qui descendent doucement de gauche à droite : aléatoire, genre, famille, ordre. La courbe orange avec des ronds = Nif ; la verte avec des carrés = RecA. Les petites barres verticales aux points sont les écarts-types (de combien ça bouge sur 20 tirages).",
              "Exemple de lecture, panneau de gauche : le rond orange reste vers 0,95-0,97 sur les trois premières marches puis 0,91 à « ordre » ; le carré vert descend un peu plus (0,91 ; 0,90 ; 0,86). Ni l'une ni l'autre ne s'effondre comme dans la courbe de décroissance LOGO.",
              "À droite (propre), l'écart se creuse à la dernière marche : Nif 0,925, RecA 0,80, avec de longues moustaches — donc plus d'incertitude.",
              "Pourquoi ça ne s'effondre pas : la famille regroupe plusieurs genres, l'ordre plusieurs familles, mais dans notre corpus 89 à 95 % de ces groupes n'ont qu'une seule classe. Retirer un groupe qui n'a qu'une réponse possible n'enlève presque rien au modèle. Le nom du test est plus dur ; son contenu ne l'est pas."],
             "la dureté d'un test dépend du contenu du corpus, pas du nom du rang taxonomique ; des scores élevés en famille/ordre ne contredisent pas l'effondrement en LOGO, ils montrent que ces groupes-là ne testaient presque rien.")
figure_slide("Phase 10b · Faut-il rééquilibrer les classes (SMOTE) ?", "phase10_smote_variants.png",
             "Les quatre variantes donnent les mêmes taux : SMOTE est abandonné.",
             ["Deux panneaux (complet, propre) ; huit barres dans chacun, groupées par couleur : orange = SMOTE, vert = « none » (ne rien faire), moutarde = class_weight (donner plus de poids à la petite classe), violet = xgb_spw (la même pondération chez XGBoost). Dans chaque couleur, la barre de gauche est l'aléatoire, celle de droite le genre.",
              "Exemple de lecture : toutes les barres de gauche (aléatoire) sont entre 0,97 et 0,98 ; toutes celles de droite (genre) entre 0,94 et 0,97. On peut les permuter, rien ne change au millième près.",
              "Les moustaches noires sont courtes et se recouvrent toutes : aucune variante ne se détache des autres de façon crédible.",
              "Le corpus est presque équilibré (356 aérobies contre 431 anaérobies) : il n'y avait presque rien à rééquilibrer. SMOTE fabriquait de fausses souches pour rien."],
             "quand quatre traitements donnent le même résultat, on garde le plus simple : pas de SMOTE. C'est démontré par la figure, pas affirmé par habitude.")
text_slide("Phase 10 et 12 · Séparer par ressemblance des séquences (MMseqs2)", [
    ("L'idée", "MMseqs2 regroupe les séquences qui se ressemblent beaucoup (à 30 % puis 50 % d'identité). On interdit ensuite qu'un même groupe soit à la fois dans l'apprentissage et dans le test : plus aucune séquence presque jumelle ne peut souffler la réponse au modèle."),
    ("Ce qui s'est passé", "Sur toutes les séquences ensemble, presque toutes les souches (99,9 %) finissent reliées dans un seul groupe géant. Même protéine par protéine, à 30 %, il ne reste qu'un à trois groupes. Ces protéines sont tellement conservées entre bactéries que tout se touche : le découpage devient impossible, il ne reste presque rien pour apprendre."),
    ("La règle qu'on s'impose", "L'évaluation par protéine à 50 % est en train de se terminer. Et surtout : aucune valeur sortie d'un découpage impossible n'est présentée comme un résultat. Un 0,500 obtenu parce que l'apprentissage était vide ne prouve pas que le modèle vaut le hasard ; il prouve que le test était impossible. Le dire clairement nous protège."),
])
text_slide("Phase 11 et 13 · Des données propres et traçables", [
    ("Provenance publiée", "Pour chaque souche des 787 : son identifiant BacDive, la phrase exacte retrouvée sur sa fiche, notre décision et la preuve. Aucune étiquette n'a été réécrite en silence : on a écarté les douteuses, on ne les a pas « corrigées ». C'est ce qui rend le travail vérifiable par quelqu'un d'autre."),
    ("Comptages réconciliés", "En comptant directement dans les fichiers : 6 710 séquences (3 426 aérobies + 3 284 anaérobies), exactement l'ancien comptage principal. Un second ancien comptage, plus petit de 375, n'est expliqué par aucun document : au lieu d'inventer une raison, on le signale comme « à vérifier ». Compter soi-même avant d'écrire un nombre, c'est la base."),
    ("Étiquette à quatre niveaux", "Depuis BacDive : aérobie, facultatif, microaérophile, anaérobie. La limite est écrite noir sur blanc dans nos fichiers : ces niveaux décrivent le rapport de la bactérie à l'oxygène, pas encore les conditions exactes dans lesquelles elle fixe l'azote. Cette preuve-là reste à aller chercher, au moins pour les 489 souches du corpus propre — et on ne l'inventera pas."),
])
text_slide("Ce que les résultats établissent", [
    ("1. Le score est reproductible, mais pas spécifique", "Protéines de ménage, parenté seule, 22 nombres simples ou grande IA ESM-2 : tout le monde atteint le même niveau quand le test est facile. Le score publié est vrai ; simplement, il ne parle pas de la nitrogénase."),
    ("2. Tout dépend de la règle du test", "Du presque parfait (0,97) quand les cousines sont des deux côtés, au pile ou face (0,549) sur un genre mixte jamais vu. Et un test qui a l'air dur — famille, ordre, groupes de séquences — peut cesser de l'être quand presque tous les groupes n'ont qu'une seule réponse."),
    ("3. Un petit résidu reste possible, sans plus", "Quand on répète proprement, Nif garde un petit avantage (+0,03 à +0,06). C'est compatible avec un petit signal propre à la cible. Mais notre dispositif ne permet ni de déclarer que c'est le mécanisme de tolérance, ni de dire qu'il n'y a rien. On présente donc une question ouverte et mesurée, pas une victoire et pas un échec."),
])
text_slide("Ce qui reste à faire", [
    ("Calculs", "Terminer l'évaluation MMseqs par protéine au seuil de 50 % : c'est le dernier calcul. Tout le reste est fini, répété, et chaque phase a son fichier de décision avec ses chiffres et ses limites."),
    ("Données", "Aller chercher la preuve des conditions de fixation à quatre niveaux, au moins pour le corpus propre ; puis déposer données et code en accès public (Zenodo, avec un DOI, et GitHub) dans les prochains jours, pour que n'importe qui puisse refaire nos calculs."),
    ("Manuscrit", "Remplacer les anciens chiffres d'un seul tirage par les chiffres répétés des Phases 5 à 13, mettre toutes les figures dans le même style, et compléter les contributions des auteurs et les remerciements."),
])

prs.save(OUT)
print("OK ->", OUT, "| diapositives :", len(prs.slides._sldIdLst))
