"""Unit tests for heuristic locale-based classifiers.

Validates all locale JSON files by running HeuristicSentenceTypeClassifier and
HeuristicQuestionTypeClassifier against labelled sentence samples.

Coverage:
  - Sentence types (question / statement / exclamation / request / command)
    for all 21 locale languages.
  - Question-type labels (all 47 labels that have rules in en.json) for English.
  - Word-start question triggers (who / where / when / how / why / what) for
    every non-English locale.
"""
import pytest
from little_questions.classifiers import (
    HeuristicSentenceTypeClassifier,
    HeuristicQuestionTypeClassifier,
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Sentence-type test data  (lang, text, expected_label)
# 3 samples per label per language  →  21 langs × 5 labels × 3 = 315 cases
# ─────────────────────────────────────────────────────────────────────────────

SENTENCE_TYPE_CASES = [
    # ── ENGLISH ──────────────────────────────────────────────────────────────
    ("en", "What is the capital of France?", "question"),
    ("en", "Who invented the telephone?", "question"),
    ("en", "When did World War II end?", "question"),

    ("en", "The sky is blue.", "statement"),
    ("en", "Paris is the capital of France.", "statement"),
    ("en", "Water boils at 100 degrees Celsius.", "statement"),

    ("en", "What a beautiful day!", "exclamation"),
    ("en", "Incredible performance!", "exclamation"),
    ("en", "Well done!", "exclamation"),

    ("en", "Could you help me please?", "request"),
    ("en", "Would you open the door?", "request"),
    ("en", "Can you pass the salt?", "request"),

    ("en", "find the nearest hospital", "command"),
    ("en", "play some jazz music", "command"),
    ("en", "search for the missing file", "command"),

    # ── SPANISH ──────────────────────────────────────────────────────────────
    ("es", "¿Cuál es la capital de España?", "question"),
    ("es", "¿Quién inventó el teléfono?", "question"),
    ("es", "¿Cuándo terminó la guerra?", "question"),

    ("es", "La capital de España es Madrid.", "statement"),
    ("es", "El cielo es azul.", "statement"),
    ("es", "El agua hierve a 100 grados.", "statement"),

    ("es", "¡Qué día tan bonito!", "exclamation"),
    ("es", "¡Increíble actuación!", "exclamation"),
    ("es", "¡Bien hecho!", "exclamation"),

    ("es", "¿Podrías tú ayudarme con esto?", "request"),
    ("es", "¿Podría usted abrir la ventana?", "request"),
    ("es", "busca la información por favor", "request"),

    ("es", "busca el hospital más cercano", "command"),
    ("es", "abre la aplicación", "command"),
    ("es", "encuentra el documento", "command"),

    # ── PORTUGUESE ───────────────────────────────────────────────────────────
    ("pt", "Qual é a capital de Portugal?", "question"),
    ("pt", "Quem inventou o telefone?", "question"),
    ("pt", "Quando terminou a guerra?", "question"),

    ("pt", "A capital de Portugal é Lisboa.", "statement"),
    ("pt", "O céu é azul.", "statement"),
    ("pt", "A água ferve a 100 graus.", "statement"),

    ("pt", "Que dia lindo!", "exclamation"),
    ("pt", "Incrível atuação!", "exclamation"),
    ("pt", "Muito bem feito!", "exclamation"),

    ("pt", "Você poderia me ajudar?", "request"),
    ("pt", "Você pode abrir a janela?", "request"),
    ("pt", "encontre o documento por favor", "request"),

    ("pt", "encontre o hospital mais próximo", "command"),
    ("pt", "abra a aplicação", "command"),
    ("pt", "busque o documento", "command"),

    # ── CATALAN ──────────────────────────────────────────────────────────────
    ("ca", "¿Quina és la capital de Catalunya?", "question"),
    ("ca", "¿Qui va inventar el telèfon?", "question"),
    ("ca", "¿Quan va acabar la guerra?", "question"),

    ("ca", "La capital de Catalunya és Barcelona.", "statement"),
    ("ca", "El cel és blau.", "statement"),
    ("ca", "L'aigua bull a 100 graus.", "statement"),

    ("ca", "Quin dia tan bonic!", "exclamation"),
    ("ca", "Increïble actuació!", "exclamation"),
    ("ca", "Molt ben fet!", "exclamation"),

    ("ca", "Podries tu ajudar-me amb això?", "request"),
    ("ca", "Podria vostè obrir la finestra?", "request"),
    ("ca", "cerca la informació si us plau", "request"),

    ("ca", "cerca l'hospital més proper", "command"),
    ("ca", "obre l'aplicació", "command"),
    ("ca", "troba el document", "command"),

    # ── FRENCH ───────────────────────────────────────────────────────────────
    ("fr", "Quelle est la capitale de la France?", "question"),
    ("fr", "Qui a inventé le téléphone?", "question"),
    ("fr", "Quand la guerre a-t-elle pris fin?", "question"),

    ("fr", "La capitale de la France est Paris.", "statement"),
    ("fr", "Le ciel est bleu.", "statement"),
    ("fr", "L'eau bout à 100 degrés.", "statement"),

    ("fr", "Quelle belle journée!", "exclamation"),
    ("fr", "Incroyable performance!", "exclamation"),
    ("fr", "Bien joué!", "exclamation"),

    ("fr", "Pourriez-vous m'aider s'il vous plaît?", "request"),
    ("fr", "Pouvez-vous ouvrir la fenêtre?", "request"),
    ("fr", "cherche le document s'il te plaît", "request"),

    ("fr", "trouve l'hôpital le plus proche", "command"),
    ("fr", "ouvre l'application", "command"),
    ("fr", "cherche le document", "command"),

    # ── GERMAN ───────────────────────────────────────────────────────────────
    ("de", "Was ist die Hauptstadt Deutschlands?", "question"),
    ("de", "Wer hat das Telefon erfunden?", "question"),
    ("de", "Wann endete der Krieg?", "question"),

    ("de", "Die Hauptstadt Deutschlands ist Berlin.", "statement"),
    ("de", "Der Himmel ist blau.", "statement"),
    ("de", "Wasser siedet bei 100 Grad.", "statement"),

    ("de", "Was für ein schöner Tag!", "exclamation"),
    ("de", "Unglaubliche Leistung!", "exclamation"),
    ("de", "Gut gemacht!", "exclamation"),

    ("de", "Könntest du mir bitte helfen?", "request"),
    ("de", "Könnten sie das Fenster öffnen?", "request"),
    ("de", "such das Dokument bitte", "request"),

    ("de", "finde das nächste Krankenhaus", "command"),
    ("de", "öffne die Anwendung", "command"),
    ("de", "zeig mir die Karte", "command"),

    # ── ITALIAN ──────────────────────────────────────────────────────────────
    ("it", "Qual è la capitale dell'Italia?", "question"),
    ("it", "Chi ha inventato il telefono?", "question"),
    ("it", "Quando è finita la guerra?", "question"),

    ("it", "La capitale d'Italia è Roma.", "statement"),
    ("it", "Il cielo è azzurro.", "statement"),
    ("it", "L'acqua bolle a 100 gradi.", "statement"),

    ("it", "Che bella giornata!", "exclamation"),
    ("it", "Prestazione incredibile!", "exclamation"),
    ("it", "Bene fatto!", "exclamation"),

    ("it", "Potresti aiutarmi per favore?", "request"),
    ("it", "Potrebbe aprire la finestra?", "request"),
    ("it", "trova il documento per favore", "request"),

    ("it", "trova l'ospedale più vicino", "command"),
    ("it", "apri l'applicazione", "command"),
    ("it", "cerca il documento", "command"),

    # ── DUTCH ────────────────────────────────────────────────────────────────
    ("nl", "Wat is de hoofdstad van Nederland?", "question"),
    ("nl", "Wie heeft de telefoon uitgevonden?", "question"),
    ("nl", "Wanneer eindigde de oorlog?", "question"),

    ("nl", "De hoofdstad van Nederland is Amsterdam.", "statement"),
    ("nl", "De lucht is blauw.", "statement"),
    ("nl", "Water kookt bij 100 graden.", "statement"),

    ("nl", "Wat een mooie dag!", "exclamation"),
    ("nl", "Ongelooflijke prestatie!", "exclamation"),
    ("nl", "Goed gedaan!", "exclamation"),

    ("nl", "Kun je me helpen?", "request"),
    ("nl", "Kunt u het raam openen?", "request"),
    ("nl", "zoek het document alsjeblieft", "request"),

    ("nl", "vind het dichtstbijzijnde ziekenhuis", "command"),
    ("nl", "open de applicatie", "command"),
    ("nl", "zoek het document", "command"),

    # ── POLISH ───────────────────────────────────────────────────────────────
    ("pl", "Jaka jest stolica Polski?", "question"),
    ("pl", "Kto wynalazł telefon?", "question"),
    ("pl", "Kiedy skończyła się wojna?", "question"),

    ("pl", "Stolica Polski to Warszawa.", "statement"),
    ("pl", "Niebo jest niebieskie.", "statement"),
    ("pl", "Woda wrze w 100 stopniach.", "statement"),

    ("pl", "Co za piękny dzień!", "exclamation"),
    ("pl", "Niesamowity wynik!", "exclamation"),
    ("pl", "Świetna robota!", "exclamation"),

    ("pl", "Mógłbyś mi pomóc?", "request"),
    ("pl", "Czy możesz otworzyć okno?", "request"),
    ("pl", "znajdź dokument proszę", "request"),

    ("pl", "znajdź najbliższy szpital", "command"),
    ("pl", "otwórz aplikację", "command"),
    ("pl", "szukaj dokumentu", "command"),

    # ── ROMANIAN ─────────────────────────────────────────────────────────────
    ("ro", "Care este capitala României?", "question"),
    ("ro", "Cine a inventat telefonul?", "question"),
    ("ro", "Când s-a terminat războiul?", "question"),

    ("ro", "Capitala României este București.", "statement"),
    ("ro", "Cerul este albastru.", "statement"),
    ("ro", "Apa fierbe la 100 de grade.", "statement"),

    ("ro", "Ce zi frumoasă!", "exclamation"),
    ("ro", "Performanță incredibilă!", "exclamation"),
    ("ro", "Bine făcut!", "exclamation"),

    ("ro", "Ai putea să mă ajuți?", "request"),
    ("ro", "Ați putea să deschideți fereastra?", "request"),
    ("ro", "găsește documentul te rog", "request"),

    ("ro", "găsește cel mai apropiat spital", "command"),
    ("ro", "deschide aplicația", "command"),
    ("ro", "caută documentul", "command"),

    # ── SWEDISH ──────────────────────────────────────────────────────────────
    ("sv", "Vad är Sveriges huvudstad?", "question"),
    ("sv", "Vem uppfann telefonen?", "question"),
    ("sv", "När slutade kriget?", "question"),

    ("sv", "Sveriges huvudstad är Stockholm.", "statement"),
    ("sv", "Himlen är blå.", "statement"),
    ("sv", "Vatten kokar vid 100 grader.", "statement"),

    ("sv", "Vilken vacker dag!", "exclamation"),
    ("sv", "Otrolig prestation!", "exclamation"),
    ("sv", "Bra gjort!", "exclamation"),

    ("sv", "Skulle du kunna hjälpa mig?", "request"),
    ("sv", "Kan du öppna fönstret?", "request"),
    ("sv", "hitta dokumentet snälla", "request"),

    ("sv", "hitta närmaste sjukhus", "command"),
    ("sv", "öppna applikationen", "command"),
    ("sv", "sök efter dokumentet", "command"),

    # ── CZECH ────────────────────────────────────────────────────────────────
    ("cs", "Jaké je hlavní město Česka?", "question"),
    ("cs", "Kdo vynalezl telefon?", "question"),
    ("cs", "Kdy skončila válka?", "question"),

    ("cs", "Hlavní město České republiky je Praha.", "statement"),
    ("cs", "Obloha je modrá.", "statement"),
    ("cs", "Voda vře při 100 stupních.", "statement"),

    ("cs", "Jaký krásný den!", "exclamation"),
    ("cs", "Neuvěřitelný výkon!", "exclamation"),
    ("cs", "Skvěle uděláno!", "exclamation"),

    ("cs", "Mohl bys mi pomoci?", "request"),
    ("cs", "Můžeš otevřít okno?", "request"),
    ("cs", "najdi dokument prosím", "request"),

    ("cs", "najdi nejbližší nemocnici", "command"),
    ("cs", "otevři aplikaci", "command"),
    ("cs", "hledej dokument", "command"),

    # ── DANISH ───────────────────────────────────────────────────────────────
    ("da", "Hvad er Danmarks hovedstad?", "question"),
    ("da", "Hvem opfandt telefonen?", "question"),
    ("da", "Hvornår sluttede krigen?", "question"),

    ("da", "Danmarks hovedstad er København.", "statement"),
    ("da", "Himlen er blå.", "statement"),
    ("da", "Vand koger ved 100 grader.", "statement"),

    ("da", "Hvilken smuk dag!", "exclamation"),
    ("da", "Utrolig præstation!", "exclamation"),
    ("da", "Godt klaret!", "exclamation"),

    ("da", "Kunne du hjælpe mig?", "request"),
    ("da", "Vil du åbne vinduet?", "request"),
    ("da", "find dokumentet venligst", "request"),

    ("da", "find det nærmeste hospital", "command"),
    ("da", "åbn applikationen", "command"),
    ("da", "søg efter dokumentet", "command"),

    # ── HUNGARIAN ────────────────────────────────────────────────────────────
    ("hu", "Mi Magyarország fővárosa?", "question"),
    ("hu", "Ki találta fel a telefont?", "question"),
    ("hu", "Mikor ért véget a háború?", "question"),

    ("hu", "Magyarország fővárosa Budapest.", "statement"),
    ("hu", "Az ég kék.", "statement"),
    ("hu", "A víz 100 fokon forr.", "statement"),

    ("hu", "Milyen szép nap!", "exclamation"),
    ("hu", "Hihetetlen teljesítmény!", "exclamation"),
    ("hu", "Jól csináltad!", "exclamation"),

    ("hu", "Tudnál segíteni nekem?", "request"),
    ("hu", "Lennél szíves kinyitni az ablakot?", "request"),
    ("hu", "keress dokumentumot kérem", "request"),

    ("hu", "keress kórházat", "command"),
    ("hu", "nyisd meg az alkalmazást", "command"),
    ("hu", "mutasd a térképet", "command"),

    # ── TURKISH ──────────────────────────────────────────────────────────────
    ("tr", "Türkiye'nin başkenti nedir?", "question"),
    ("tr", "Telefonu kim icat etti?", "question"),
    ("tr", "Savaş ne zaman bitti?", "question"),

    ("tr", "Türkiye'nin başkenti Ankara'dır.", "statement"),
    ("tr", "Gökyüzü mavi.", "statement"),
    ("tr", "Su 100 derecede kaynar.", "statement"),

    ("tr", "Ne güzel bir gün!", "exclamation"),
    ("tr", "İnanılmaz performans!", "exclamation"),
    ("tr", "Aferin!", "exclamation"),

    ("tr", "Yapabilir misin bana yardım etmek?", "request"),
    ("tr", "Yapabilir misiniz pencereyi açmak?", "request"),
    ("tr", "belgeyi bul lütfen", "request"),

    ("tr", "bul en yakın hastane", "command"),
    ("tr", "aç uygulamayı", "command"),
    ("tr", "git markete", "command"),

    # ── RUSSIAN ──────────────────────────────────────────────────────────────
    ("ru", "Какова столица России?", "question"),
    ("ru", "Кто изобрёл телефон?", "question"),
    ("ru", "Когда закончилась война?", "question"),

    ("ru", "Столица России — Москва.", "statement"),
    ("ru", "Небо голубое.", "statement"),
    ("ru", "Вода кипит при 100 градусах.", "statement"),

    ("ru", "Какой прекрасный день!", "exclamation"),
    ("ru", "Невероятное выступление!", "exclamation"),
    ("ru", "Отлично сделано!", "exclamation"),

    ("ru", "Не мог бы ты мне помочь?", "request"),
    ("ru", "Мог бы ты открыть окно?", "request"),
    ("ru", "найди документ пожалуйста", "request"),

    ("ru", "найди ближайшую больницу", "command"),
    ("ru", "открой приложение", "command"),
    ("ru", "ищи документ", "command"),

    # ── UKRAINIAN ────────────────────────────────────────────────────────────
    ("uk", "Яка столиця України?", "question"),
    ("uk", "Хто винайшов телефон?", "question"),
    ("uk", "Коли закінчилась війна?", "question"),

    ("uk", "Столиця України — Київ.", "statement"),
    ("uk", "Небо блакитне.", "statement"),
    ("uk", "Вода кипить при 100 градусах.", "statement"),

    ("uk", "Який прекрасний день!", "exclamation"),
    ("uk", "Неймовірний виступ!", "exclamation"),
    ("uk", "Чудово зроблено!", "exclamation"),

    ("uk", "Чи міг би ти мені допомогти?", "request"),
    ("uk", "Чи можеш ти відкрити вікно?", "request"),
    ("uk", "знайди документ будь ласка", "request"),

    ("uk", "знайди найближчу лікарню", "command"),
    ("uk", "відкрий програму", "command"),
    ("uk", "шукай документ", "command"),

    # ── GREEK ────────────────────────────────────────────────────────────────
    ("el", "Ποια είναι η πρωτεύουσα της Ελλάδας?", "question"),
    ("el", "Ποιος εφηύρε το τηλέφωνο?", "question"),
    ("el", "Πότε τελείωσε ο πόλεμος?", "question"),

    ("el", "Η πρωτεύουσα της Ελλάδας είναι η Αθήνα.", "statement"),
    ("el", "Ο ουρανός είναι μπλε.", "statement"),
    ("el", "Το νερό βράζει στους 100 βαθμούς.", "statement"),

    ("el", "Τι υπέροχη μέρα!", "exclamation"),
    ("el", "Απίστευτη εμφάνιση!", "exclamation"),
    ("el", "Μπράβο!", "exclamation"),

    ("el", "Θα μπορούσες να με βοηθήσεις?", "request"),
    ("el", "Μπορείς να ανοίξεις το παράθυρο?", "request"),
    ("el", "βρες το έγγραφο παρακαλώ", "request"),

    ("el", "βρες το πλησιέστερο νοσοκομείο", "command"),
    ("el", "παίξε ελληνική μουσική", "command"),
    ("el", "δείξε μου τον χάρτη", "command"),

    # ── BASQUE ───────────────────────────────────────────────────────────────
    ("eu", "Zein da Euskal Herriko hiriburua?", "question"),
    ("eu", "Nork asmatu zuen telefonoa?", "question"),
    ("eu", "Noiz amaitu zen gerra?", "question"),

    ("eu", "Euskal Herriko hiriburua Gasteiz da.", "statement"),
    ("eu", "Zerua urdin da.", "statement"),
    ("eu", "Ura 100 gradutan irakiten da.", "statement"),

    ("eu", "Ze egun ederra!", "exclamation"),
    ("eu", "Harrigarria!", "exclamation"),
    ("eu", "Ondo egina!", "exclamation"),

    ("eu", "al zenezake lagundu niri?", "request"),
    ("eu", "egin al dezakezu leihoa ireki?", "request"),
    ("eu", "aurkitu dokumentua mesedez", "request"),

    ("eu", "aurkitu hurbilena ospitalea", "command"),
    ("eu", "ireki aplikazioa", "command"),
    ("eu", "bilatu dokumentua", "command"),

    # ── GALICIAN ─────────────────────────────────────────────────────────────
    ("gl", "¿Cal é a capital de Galicia?", "question"),
    ("gl", "¿Quen inventou o teléfono?", "question"),
    ("gl", "¿Cando rematou a guerra?", "question"),

    ("gl", "A capital de Galicia é Santiago de Compostela.", "statement"),
    ("gl", "O ceo é azul.", "statement"),
    ("gl", "A auga ferve a 100 graos.", "statement"),

    ("gl", "Que día tan bonito!", "exclamation"),
    ("gl", "Actuación increíbel!", "exclamation"),
    ("gl", "Ben feito!", "exclamation"),

    ("gl", "¿Poderías axudarme con isto?", "request"),
    ("gl", "¿Podería vostede abrir a ventá?", "request"),
    ("gl", "atopa o documento por favor", "request"),

    ("gl", "atopa o hospital máis próximo", "command"),
    ("gl", "abre a aplicación", "command"),
    ("gl", "busca o documento", "command"),

    # ── PERSIAN / FARSI ──────────────────────────────────────────────────────
    ("fa", "پایتخت ایران کجاست?", "question"),
    ("fa", "تلفن را چه کسی اختراع کرد?", "question"),
    ("fa", "جنگ کِی پایان یافت?", "question"),

    ("fa", "پایتخت ایران تهران است.", "statement"),
    ("fa", "آسمان آبی است.", "statement"),
    ("fa", "آب در ۱۰۰ درجه می‌جوشد.", "statement"),

    ("fa", "چه روز زیبایی!", "exclamation"),
    ("fa", "عالی!", "exclamation"),
    ("fa", "آفرین!", "exclamation"),

    ("fa", "آیا می‌توانی کمکم کنی?", "request"),
    ("fa", "آیا می‌توانید پنجره را باز کنید?", "request"),
    ("fa", "سند را پیدا کن لطفاً", "request"),

    ("fa", "برو به خانه", "command"),
    ("fa", "بیا اینجا", "command"),
    ("fa", "ببند در را", "command"),
]


class TestHeuristicSentenceType:
    @pytest.mark.parametrize(
        "lang,text,expected",
        SENTENCE_TYPE_CASES,
        ids=[f"{lang}-{exp}-{i}" for i, (lang, _, exp) in enumerate(SENTENCE_TYPE_CASES)],
    )
    def test_predict(self, lang, text, expected):
        clf = HeuristicSentenceTypeClassifier(lang)
        result = clf.predict(text)
        assert result == expected, (
            f"[{lang}] predict({text!r}) → {result!r}, expected {expected!r}"
        )

    @pytest.mark.parametrize(
        "lang,text,expected",
        SENTENCE_TYPE_CASES,
        ids=[f"score-{lang}-{exp}-{i}" for i, (lang, _, exp) in enumerate(SENTENCE_TYPE_CASES)],
    )
    def test_score_top_label(self, lang, text, expected):
        """score() top key must match expected label."""
        clf = HeuristicSentenceTypeClassifier(lang)
        scores = clf.score(text)
        top = max(scores, key=lambda k: scores[k])
        assert top == expected, (
            f"[{lang}] top score for {text!r}: {top!r}, expected {expected!r}. "
            f"scores={scores}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — English question-type tests  (text, expected_label)
# 3 samples per label, covering all 47 labels that have heuristic rules
# ─────────────────────────────────────────────────────────────────────────────

EN_QUESTION_TYPE_CASES = [
    # ── HUM:ind  (who / whose / whom at word_start) ──────────────────────────
    ("Who invented the telephone?", "HUM:ind"),
    ("Who is the president of France?", "HUM:ind"),
    ("Who wrote Romeo and Juliet?", "HUM:ind"),
    ("Whose signature is on the Declaration of Independence?", "HUM:ind"),
    ("Whose house was used as a museum?", "HUM:ind"),
    ("Whom did Shakespeare marry?", "HUM:ind"),

    # ── LOC:other  (where at word_start) ─────────────────────────────────────
    ("Where is the Eiffel Tower located?", "LOC:other"),
    ("Where do polar bears live?", "LOC:other"),
    ("Where is Machu Picchu?", "LOC:other"),

    # ── NUM:date  (when at word_start / born / year / date keyword) ──────────
    ("When did the French Revolution start?", "NUM:date"),
    ("When did the first moon landing occur?", "NUM:date"),
    ("When did the Eiffel Tower open to the public?", "NUM:date"),
    ("What year was the Eiffel Tower constructed?", "NUM:date"),
    ("What is the birthday of Beethoven?", "NUM:date"),
    ("What date did the first moon landing occur?", "NUM:date"),

    # ── DESC:manner  (how at word_start, no "how many/much/far/fast") ─────────
    ("How does photosynthesis work?", "DESC:manner"),
    ("How does a nuclear reactor generate power?", "DESC:manner"),
    ("How is glass manufactured?", "DESC:manner"),

    # ── DESC:reason  (why at word_start) ─────────────────────────────────────
    ("Why is the sky blue?", "DESC:reason"),
    ("Why do we dream during sleep?", "DESC:reason"),
    ("Why does ice float on water?", "DESC:reason"),

    # ── DESC:def  (what / which at word_start, no stronger keyword) ──────────
    ("What is democracy?", "DESC:def"),
    ("What is machine learning?", "DESC:def"),
    ("Which option is simplest to implement?", "DESC:def"),

    # ── NUM:count  (how many / how much anywhere) ─────────────────────────────
    ("How many planets are in the solar system?", "NUM:count"),
    ("How many countries are members of the United Nations?", "NUM:count"),
    ("How much water covers the surface of the Earth?", "NUM:count"),

    # ── NUM:dist  (how far / distance / kilometers anywhere) ─────────────────
    ("What is the distance between the Earth and the Moon?", "NUM:dist"),
    ("How far is Alpha Centauri from the Sun?", "NUM:dist"),
    ("What is the distance from New York to Los Angeles in kilometers?", "NUM:dist"),

    # ── NUM:speed  (how fast / speed / kph anywhere) ──────────────────────────
    ("How fast can a cheetah run?", "NUM:speed"),
    ("What is the speed of light in a vacuum?", "NUM:speed"),
    ("What is the maximum kph a bullet can reach?", "NUM:speed"),

    # ── NUM:money  (price / cost / dollar / euro — no "how many/much" first) ──
    ("What does a haircut cost?", "NUM:money"),
    ("What does a first-class flight to Tokyo cost?", "NUM:money"),
    ("What does a coffee cost in Paris?", "NUM:money"),

    # ── NUM:temp  (temperature / degrees / celsius / fahrenheit) ─────────────
    ("What temperature does water freeze at?", "NUM:temp"),
    ("What is the boiling point in degrees Celsius?", "NUM:temp"),
    ("How hot is the surface of the Sun?", "NUM:temp"),

    # ── NUM:perc  (percent / percentage / ratio / fraction) ──────────────────
    ("What percent of voters participated?", "NUM:perc"),
    ("What fraction of the atmosphere is nitrogen?", "NUM:perc"),
    ("What ratio of men to women exists in Iceland?", "NUM:perc"),

    # ── NUM:weight  (weigh / weight / gram / kg / ounce) ─────────────────────
    ("What does a blue whale weigh?", "NUM:weight"),
    ("What is the weight of the Eiffel Tower?", "NUM:weight"),
    ("What does a baby elephant weigh at birth?", "NUM:weight"),

    # ── NUM:volsize  (volume / liter / gallon) ────────────────────────────────
    ("What is the volume of a standard balloon?", "NUM:volsize"),
    ("What gallon size is a standard tank?", "NUM:volsize"),
    ("What gallon size is a standard Olympic swimming pool?", "NUM:volsize"),

    # ── NUM:period  (how long / how often / duration) ─────────────────────────
    ("How long does it take to fly from London to Sydney?", "NUM:period"),
    ("How often do the summer Olympics occur?", "NUM:period"),
    ("How long is a typical flight?", "NUM:period"),

    # ── NUM:ord  (first / second / third / ranked / order) ───────────────────
    ("What was the first patent ever filed?", "NUM:ord"),
    ("What finished second in the contest?", "NUM:ord"),
    ("What is ranked third in the list?", "NUM:ord"),

    # ── NUM:code  (code / zip / postal / isbn / serial) ──────────────────────
    ("What is the zip code for Los Angeles?", "NUM:code"),
    ("What is the ISBN of this publication?", "NUM:code"),
    ("What is the postal code for central Paris?", "NUM:code"),

    # ── ENTY:animal ───────────────────────────────────────────────────────────
    ("What animal is the fastest on land?", "ENTY:animal"),
    ("What species of bird is unable to fly?", "ENTY:animal"),
    ("What mammal is capable of sustained flight?", "ENTY:animal"),

    # ── ENTY:body ─────────────────────────────────────────────────────────────
    ("What organ filters waste from the blood?", "ENTY:body"),
    ("What muscle is considered the strongest in the human body?", "ENTY:body"),
    ("What bone protects the human brain?", "ENTY:body"),

    # ── ENTY:color ────────────────────────────────────────────────────────────
    ("What color is the sky on a clear day?", "ENTY:color"),
    ("What colour does mixing red and blue produce?", "ENTY:color"),
    ("What color is chlorophyll in plants?", "ENTY:color"),

    # ── ENTY:cremat ───────────────────────────────────────────────────────────
    ("What book did J.R.R. Tolkien publish first?", "ENTY:cremat"),
    ("What film won the Best Picture Academy Award in 2019?", "ENTY:cremat"),
    ("What song is the best-selling single of all time?", "ENTY:cremat"),

    # ── ENTY:currency ─────────────────────────────────────────────────────────
    ("What currency is used in Japan?", "ENTY:currency"),
    ("What coin is worth the least in the US?", "ENTY:currency"),
    ("What exchange rate does the Swiss franc have versus the euro?", "ENTY:currency"),

    # ── ENTY:dismed ───────────────────────────────────────────────────────────
    ("What disease is caused by the HIV virus?", "ENTY:dismed"),
    ("What medicine is most commonly used to treat malaria?", "ENTY:dismed"),
    ("What are the main symptoms of type 2 diabetes?", "ENTY:dismed"),

    # ── ENTY:event ────────────────────────────────────────────────────────────
    ("What war began in Europe in 1939?", "ENTY:event"),
    ("What ceremony opens the Olympic Games?", "ENTY:event"),
    ("What festival is celebrated on 31 October in many countries?", "ENTY:event"),

    # ── ENTY:food ─────────────────────────────────────────────────────────────
    ("What food is sushi traditionally made from?", "ENTY:food"),
    ("What dish originates from Valencia in Spain?", "ENTY:food"),
    ("What beverage is produced by fermenting grapes?", "ENTY:food"),

    # ── ENTY:instru ───────────────────────────────────────────────────────────
    ("What instrument does a violinist play?", "ENTY:instru"),
    ("What musical instrument has 88 keys?", "ENTY:instru"),
    ("What type of drum is central to jazz music?", "ENTY:instru"),

    # ── ENTY:lang ────────────────────────────────────────────────────────────
    ("What language is spoken in Brazil?", "ENTY:lang"),
    ("What dialect is used in the Sicilian region?", "ENTY:lang"),
    ("What tongue do most people in Thailand speak?", "ENTY:lang"),

    # ── ENTY:letter ───────────────────────────────────────────────────────────
    ("What letter comes after Z in the English alphabet?", "ENTY:letter"),
    ("What symbol is used to represent the mathematical constant pi?", "ENTY:letter"),
    ("What character is used to denote copyright?", "ENTY:letter"),

    # ── ENTY:plant ────────────────────────────────────────────────────────────
    ("What plant converts sunlight into glucose?", "ENTY:plant"),
    ("What flower is associated with the Netherlands?", "ENTY:plant"),
    ("What tree has the longest confirmed lifespan?", "ENTY:plant"),

    # ── ENTY:product ──────────────────────────────────────────────────────────
    ("What brand makes the iPhone smartphone?", "ENTY:product"),
    ("What brand of smartphone is the most popular?", "ENTY:product"),
    ("What device is used to measure atmospheric pressure?", "ENTY:product"),

    # ── ENTY:religion ─────────────────────────────────────────────────────────
    ("What religion considers the Quran its holy book?", "ENTY:religion"),
    ("What faith believes in the concept of karma?", "ENTY:religion"),
    ("What church has the largest number of members worldwide?", "ENTY:religion"),

    # ── ENTY:sport ────────────────────────────────────────────────────────────
    ("What sport is played with a puck on ice?", "ENTY:sport"),
    ("What game is contested at Wimbledon?", "ENTY:sport"),
    ("What tournament determines the world's best soccer nation?", "ENTY:sport"),

    # ── ENTY:substance ────────────────────────────────────────────────────────
    ("What substance makes up ordinary table salt?", "ENTY:substance"),
    ("What chemical compound is water?", "ENTY:substance"),
    ("What element is diamond composed of?", "ENTY:substance"),

    # ── ENTY:techmeth ─────────────────────────────────────────────────────────
    ("What technology powers the modern internet?", "ENTY:techmeth"),
    ("What algorithm is used for secure internet encryption?", "ENTY:techmeth"),
    ("What method is used to sort data in O(n log n) time?", "ENTY:techmeth"),

    # ── ENTY:termeq ───────────────────────────────────────────────────────────
    ("What is the synonym for happy?", "ENTY:termeq"),
    ("What does the term quantum entanglement mean?", "ENTY:termeq"),
    ("What is the definition of entropy in thermodynamics?", "ENTY:termeq"),

    # ── ENTY:veh ──────────────────────────────────────────────────────────────
    ("What vehicle travels faster than the speed of sound?", "ENTY:veh"),
    ("What car is considered the most reliable in surveys?", "ENTY:veh"),
    ("What ship sank on its maiden voyage in 1912?", "ENTY:veh"),

    # ── ENTY:word ─────────────────────────────────────────────────────────────
    ("What word means the same as fast?", "ENTY:word"),
    ("What phrase describes extreme happiness?", "ENTY:word"),
    ("What do you say to greet someone formally in Japanese?", "ENTY:word"),

    # ── LOC:city  (city / town / capital keyword) ─────────────────────────────
    ("What city is home to the Colosseum?", "LOC:city"),
    ("What is the capital of Germany?", "LOC:city"),
    ("What town in England is famous for its Shakespeare heritage?", "LOC:city"),

    # ── LOC:country  (country / nation / republic keyword) ───────────────────
    ("What country has the largest land area?", "LOC:country"),
    ("What nation won the most gold medals at the 2020 Olympics?", "LOC:country"),
    ("What republic first declared independence in South America?", "LOC:country"),

    # ── LOC:mount  (mountain / volcano / peak / hill keyword) ────────────────
    ("What mountain is the tallest on Earth above sea level?", "LOC:mount"),
    ("What volcano erupted in 1883?", "LOC:mount"),
    ("What peak on Earth is closest to outer space?", "LOC:mount"),

    # ── LOC:water  (river / lake / ocean / sea / bay keyword) ────────────────
    ("What river is the longest in the world?", "LOC:water"),
    ("What ocean covers the most area on Earth?", "LOC:water"),
    ("What sea separates Europe from Africa?", "LOC:water"),

    # ── LOC:landmass  (continent / island / peninsula keyword) ───────────────
    ("What continent is Australia located on?", "LOC:landmass"),
    ("What island is the largest on Earth?", "LOC:landmass"),
    ("What peninsula does Italy form?", "LOC:landmass"),

    # ── LOC:state  (state / region / district / county keyword) ──────────────
    ("What state is Las Vegas located in?", "LOC:state"),
    ("What region of France is Bordeaux in?", "LOC:state"),
    ("What district of London contains Buckingham Palace?", "LOC:state"),

    # ── HUM:gr  (group / team / organization / company keyword) ──────────────
    ("What group led the civil rights movement?", "HUM:gr"),
    ("What team has won the most league titles?", "HUM:gr"),
    ("What group sets the international aviation standards?", "HUM:gr"),

    # ── HUM:title  (title / role / position / known as keyword) ──────────────
    ("What title does the British monarch hold?", "HUM:title"),
    ("What title does the head of state hold?", "HUM:title"),
    ("What position is known as the head of a university?", "HUM:title"),

    # ── HUM:desc  (describe / description keyword) ────────────────────────────
    ("Can you describe what Albert Einstein looked like?", "HUM:desc"),
    ("What is the description of a neutron star?", "HUM:desc"),
    ("How would you describe the duties of a surgeon?", "HUM:desc"),

    # ── ABBR:abb  (abbreviation / acronym / stand for / initials keyword) ────
    ("What does NASA stand for?", "ABBR:abb"),
    ("What is the abbreviation for the United Nations?", "ABBR:abb"),
    ("What do the initials CEO represent?", "ABBR:abb"),

    # ── ABBR:exp  (abbreviated keyword — must avoid ABBR:abb triggers) ───────
    ("What is the abbreviated form of the United Nations?", "ABBR:exp"),
    ("What is the abbreviated name used for artificial intelligence?", "ABBR:exp"),
    ("What is the abbreviated title used for the word Doctor?", "ABBR:exp"),
]


class TestHeuristicQuestionTypeEN:
    @pytest.mark.parametrize(
        "text,expected",
        EN_QUESTION_TYPE_CASES,
        ids=[f"en-{exp}-{i}" for i, (_, exp) in enumerate(EN_QUESTION_TYPE_CASES)],
    )
    def test_predict_en(self, text, expected):
        clf = HeuristicQuestionTypeClassifier("en")
        result = clf.predict(text)
        assert result == expected, (
            f"[en] predict({text!r}) → {result!r}, expected {expected!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Multilingual question-type word-start trigger tests
# Tests WHO / WHERE / WHEN / HOW / WHY / WHAT equivalents for each locale.
# 3 sentences per trigger × 6 triggers × 20 languages  ≈  360 cases
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
MULTILANG_QUESTION_TYPE_CASES = [
    # ══════════════════════════════════════════════════════════════════════════
    # SPANISH  (strip_leading ¿¡  — accented interrogatives at word_start)
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → quién
    ("es", "¿Quién inventó la radio?", "HUM:ind"),
    ("es", "¿Quién escribió Don Quijote?", "HUM:ind"),
    ("es", "¿Quién fue el primer cosmonauta?", "HUM:ind"),
    # WHERE → dónde
    ("es", "¿Dónde está la Gran Muralla China?", "LOC:other"),
    ("es", "¿Dónde viven los pingüinos?", "LOC:other"),
    ("es", "¿Dónde se encuentra el Amazonas?", "LOC:other"),
    # WHEN → cuándo
    ("es", "¿Cuándo nació Pablo Picasso?", "NUM:date"),
    ("es", "¿Cuándo nació Simón Bolívar?", "NUM:date"),
    ("es", "¿Cuándo llegó el hombre a la luna?", "NUM:date"),
    # HOW → cómo
    ("es", "¿Cómo funciona la fotosíntesis?", "DESC:manner"),
    ("es", "¿Cómo se hace el queso?", "DESC:manner"),
    ("es", "¿Cómo funciona un motor de combustión?", "DESC:manner"),
    # WHY → por qué  (position: start after stripping ¿)
    ("es", "¿Por qué el cielo es azul?", "DESC:reason"),
    ("es", "¿Por qué soñamos?", "DESC:reason"),
    ("es", "¿Por qué flota el hielo?", "DESC:reason"),
    # WHAT → qué / cuál
    ("es", "¿Qué es la democracia?", "DESC:def"),
    ("es", "¿Qué es el aprendizaje automático?", "DESC:def"),
    ("es", "¿Cuál es el concepto de libertad?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # PORTUGUESE
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → quem
    ("pt", "Quem inventou o telefone?", "HUM:ind"),
    ("pt", "Quem escreveu Os Lusíadas?", "HUM:ind"),
    ("pt", "Quem foi o primeiro presidente do Brasil?", "HUM:ind"),
    # WHERE → onde
    ("pt", "Onde fica a Torre de Belém?", "LOC:other"),
    ("pt", "Onde vivem os leões?", "LOC:other"),
    ("pt", "Onde está localizado o Machu Picchu?", "LOC:other"),
    # WHEN → quando
    ("pt", "Quando começou a Primeira Guerra Mundial?", "NUM:date"),
    ("pt", "Quando nasceu Luís de Camões?", "NUM:date"),
    ("pt", "Quando chegou o homem à lua?", "NUM:date"),
    # HOW → como
    ("pt", "Como funciona a fotossíntese?", "DESC:manner"),
    ("pt", "Como se faz o pão?", "DESC:manner"),
    ("pt", "Como funciona um motor elétrico?", "DESC:manner"),
    # WHY → por que
    ("pt", "Por que o céu é azul?", "DESC:reason"),
    ("pt", "Por que sonhamos?", "DESC:reason"),
    ("pt", "Por que o gelo flutua?", "DESC:reason"),
    # WHAT → o que / qual
    ("pt", "O que é democracia?", "DESC:def"),
    ("pt", "O que é inteligência artificial?", "DESC:def"),
    ("pt", "Qual é o sentido da vida?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # CATALAN  (strip_leading ¿¡)
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → qui
    ("ca", "¿Qui va inventar la radio?", "HUM:ind"),
    ("ca", "¿Qui va inventar el telèfon?", "HUM:ind"),
    ("ca", "¿Qui va ser el primer astronauta?", "HUM:ind"),
    # WHERE → on
    ("ca", "¿On és la Gran Muralla Xinesa?", "LOC:other"),
    ("ca", "¿On viuen els pingüins?", "LOC:other"),
    ("ca", "¿On és el Museu del Prado?", "LOC:other"),
    # WHEN → quan
    ("ca", "¿Quan va començar la Segona Guerra Mundial?", "NUM:date"),
    ("ca", "¿Quan va néixer Gaudí?", "NUM:date"),
    ("ca", "¿Quan va arribar l'home a la lluna?", "NUM:date"),
    # HOW → com
    ("ca", "¿Com funciona un reactor nuclear?", "DESC:manner"),
    ("ca", "¿Com es fa el pa?", "DESC:manner"),
    ("ca", "¿Com funciona un motor elèctric?", "DESC:manner"),
    # WHY → per què  (position: start after stripping ¿)
    ("ca", "¿Per què el cel és blau?", "DESC:reason"),
    ("ca", "¿Per què somiem?", "DESC:reason"),
    ("ca", "¿Per què flota el gel?", "DESC:reason"),
    # WHAT → què / quin
    ("ca", "¿Què és la democràcia?", "DESC:def"),
    ("ca", "¿Què és la intel·ligència artificial?", "DESC:def"),
    ("ca", "¿Quin és el sentit de la vida?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # FRENCH
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → qui
    ("fr", "Qui a inventé la radio?", "HUM:ind"),
    ("fr", "Qui a écrit Les Misérables?", "HUM:ind"),
    ("fr", "Qui fut le premier président de la République française?", "HUM:ind"),
    # WHERE → où
    ("fr", "Où se trouve la Tour Eiffel?", "LOC:other"),
    ("fr", "Où vivent les pingouins?", "LOC:other"),
    ("fr", "Où est situé Machu Picchu?", "LOC:other"),
    # WHEN → quand
    ("fr", "Quand est né Victor Hugo?", "NUM:date"),
    ("fr", "Quand est né Napoléon Bonaparte?", "NUM:date"),
    ("fr", "Quand l'homme a-t-il marché sur la lune?", "NUM:date"),
    # HOW → comment
    ("fr", "Comment fonctionne un réacteur nucléaire?", "DESC:manner"),
    ("fr", "Comment fait-on le fromage?", "DESC:manner"),
    ("fr", "Comment fonctionne un panneau solaire?", "DESC:manner"),
    # WHY → pourquoi
    ("fr", "Pourquoi le ciel est-il bleu?", "DESC:reason"),
    ("fr", "Pourquoi rêvons-nous?", "DESC:reason"),
    ("fr", "Pourquoi la glace flotte-t-elle?", "DESC:reason"),
    # WHAT → que / quel / quelle
    ("fr", "Que est la démocratie?", "DESC:def"),
    ("fr", "Quel est le but de l'existence?", "DESC:def"),
    ("fr", "Quelle est la nature de la gravité?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # GERMAN
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → wer
    ("de", "Wer hat das Radio erfunden?", "HUM:ind"),
    ("de", "Wer schrieb Faust?", "HUM:ind"),
    ("de", "Wer war der erste Bundeskanzler?", "HUM:ind"),
    # WHERE → wo
    ("de", "Wo steht der Eiffelturm?", "LOC:other"),
    ("de", "Wo leben Pinguine?", "LOC:other"),
    ("de", "Wo befindet sich Machu Picchu?", "LOC:other"),
    # WHEN → wann
    ("de", "Wann wurde Mozart geboren?", "NUM:date"),
    ("de", "Wann wurde Beethoven geboren?", "NUM:date"),
    ("de", "Wann wurde das Internet erfunden?", "NUM:date"),
    # HOW → wie
    ("de", "Wie funktioniert die Photosynthese?", "DESC:manner"),
    ("de", "Wie wird Käse hergestellt?", "DESC:manner"),
    ("de", "Wie funktioniert ein Elektromotor?", "DESC:manner"),
    # WHY → warum
    ("de", "Warum ist der Himmel blau?", "DESC:reason"),
    ("de", "Warum träumen wir?", "DESC:reason"),
    ("de", "Warum schwimmt Eis auf Wasser?", "DESC:reason"),
    # WHAT → was / welche
    ("de", "Was ist Demokratie?", "DESC:def"),
    ("de", "Was ist maschinelles Lernen?", "DESC:def"),
    ("de", "Welches ist das Richtige hier?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # ITALIAN
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → chi
    ("it", "Chi ha inventato la radio?", "HUM:ind"),
    ("it", "Chi ha scritto la Divina Commedia?", "HUM:ind"),
    ("it", "Chi fu il primo presidente della Repubblica italiana?", "HUM:ind"),
    # WHERE → dove
    ("it", "Dove si trova la Torre Eiffel?", "LOC:other"),
    ("it", "Dove vivono i pinguini?", "LOC:other"),
    ("it", "Dove si trova Machu Picchu?", "LOC:other"),
    # WHEN → quando
    ("it", "Quando è nato Leonardo da Vinci?", "NUM:date"),
    ("it", "Quando è nato Dante Alighieri?", "NUM:date"),
    ("it", "Quando l'uomo è arrivato sulla luna?", "NUM:date"),
    # HOW → come
    ("it", "Come funziona la fotosintesi?", "DESC:manner"),
    ("it", "Come si fa il pane?", "DESC:manner"),
    ("it", "Come funziona un motore elettrico?", "DESC:manner"),
    # WHY → perché
    ("it", "Perché il cielo è blu?", "DESC:reason"),
    ("it", "Perché sogniamo?", "DESC:reason"),
    ("it", "Perché il ghiaccio galleggia?", "DESC:reason"),
    # WHAT → che / cosa / quale
    ("it", "Che cosa è la democrazia?", "DESC:def"),
    ("it", "Cosa è l'intelligenza artificiale?", "DESC:def"),
    ("it", "Quale è il senso della vita?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # DUTCH
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → wie
    ("nl", "Wie heeft de radio uitgevonden?", "HUM:ind"),
    ("nl", "Wie heeft de fiets uitgevonden?", "HUM:ind"),
    ("nl", "Wie was de eerste premier van Nederland?", "HUM:ind"),
    # WHERE → waar
    ("nl", "Waar staat de Eiffeltoren?", "LOC:other"),
    ("nl", "Waar leven pinguïns?", "LOC:other"),
    ("nl", "Waar bevindt Machu Picchu zich?", "LOC:other"),
    # WHEN → wanneer
    ("nl", "Wanneer werd Mozart geboren?", "NUM:date"),
    ("nl", "Wanneer werd Rembrandt geboren?", "NUM:date"),
    ("nl", "Wanneer werd het internet uitgevonden?", "NUM:date"),
    # HOW → hoe
    ("nl", "Hoe werkt fotosynthese?", "DESC:manner"),
    ("nl", "Hoe wordt kaas gemaakt?", "DESC:manner"),
    ("nl", "Hoe werkt een elektromotor?", "DESC:manner"),
    # WHY → waarom
    ("nl", "Waarom is de lucht blauw?", "DESC:reason"),
    ("nl", "Waarom dromen we?", "DESC:reason"),
    ("nl", "Waarom drijft ijs op water?", "DESC:reason"),
    # WHAT → wat / welke
    ("nl", "Wat is democratie?", "DESC:def"),
    ("nl", "Wat is machine learning?", "DESC:def"),
    ("nl", "Welke is de juiste keuze?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # POLISH
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → kto
    ("pl", "Kto wynalazł radio?", "HUM:ind"),
    ("pl", "Kto napisał Pana Tadeusza?", "HUM:ind"),
    ("pl", "Kto był pierwszym prezydentem Polski?", "HUM:ind"),
    # WHERE → gdzie
    ("pl", "Gdzie stoi Wieża Eiffla?", "LOC:other"),
    ("pl", "Gdzie żyją pingwiny?", "LOC:other"),
    ("pl", "Gdzie znajduje się Machu Picchu?", "LOC:other"),
    # WHEN → kiedy
    ("pl", "Kiedy wynalazł Edison żarówkę?", "NUM:date"),
    ("pl", "Kiedy urodził się Fryderyk Chopin?", "NUM:date"),
    ("pl", "Kiedy człowiek wylądował na Księżycu?", "NUM:date"),
    # HOW → jak
    ("pl", "Jak działa fotosynteza?", "DESC:manner"),
    ("pl", "Jak robi się chleb?", "DESC:manner"),
    ("pl", "Jak działa reaktor atomowy?", "DESC:manner"),
    # WHY → dlaczego
    ("pl", "Dlaczego niebo jest niebieskie?", "DESC:reason"),
    ("pl", "Dlaczego śnimy?", "DESC:reason"),
    ("pl", "Dlaczego lód unosi się na wodzie?", "DESC:reason"),
    # WHAT → co / który
    ("pl", "Co to jest demokracja?", "DESC:def"),
    ("pl", "Co to jest uczenie maszynowe?", "DESC:def"),
    ("pl", "Który jest sens życia?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # ROMANIAN
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → cine
    ("ro", "Cine a inventat radioul?", "HUM:ind"),
    ("ro", "Cine a scris Mihai Eminescu?", "HUM:ind"),
    ("ro", "Cine a creat internetul?", "HUM:ind"),
    # WHERE → unde
    ("ro", "Unde se află Turnul Eiffel?", "LOC:other"),
    ("ro", "Unde trăiesc pinguinii?", "LOC:other"),
    ("ro", "Unde este localizat Machu Picchu?", "LOC:other"),
    # WHEN → când
    ("ro", "Când s-a născut Maria Curie?", "NUM:date"),
    ("ro", "Când s-a născut Mihai Eminescu?", "NUM:date"),
    ("ro", "Când a ajuns omul pe Lună?", "NUM:date"),
    # HOW → cum
    ("ro", "Cum funcționează un reactor nuclear?", "DESC:manner"),
    ("ro", "Cum se face pâinea?", "DESC:manner"),
    ("ro", "Cum funcționează un motor electric?", "DESC:manner"),
    # WHY → de ce  (position: start)
    ("ro", "De ce cerul este albastru?", "DESC:reason"),
    ("ro", "De ce visăm?", "DESC:reason"),
    ("ro", "De ce plutește gheața pe apă?", "DESC:reason"),
    # WHAT → ce / care
    ("ro", "Ce este democrația?", "DESC:def"),
    ("ro", "Ce este inteligența artificială?", "DESC:def"),
    ("ro", "Care este sensul vieții?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # SWEDISH
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → vem
    ("sv", "Vem uppfann radion?", "HUM:ind"),
    ("sv", "Vem skrev Röda Rummet?", "HUM:ind"),
    ("sv", "Vem var den förste statsministern?", "HUM:ind"),
    # WHERE → var
    ("sv", "Var ligger Eiffeltornet?", "LOC:other"),
    ("sv", "Var lever pingviner?", "LOC:other"),
    ("sv", "Var finns Machu Picchu?", "LOC:other"),
    # WHEN → när
    ("sv", "När uppfanns datorn?", "NUM:date"),
    ("sv", "När föddes Alfred Nobel?", "NUM:date"),
    ("sv", "När skapades FN?", "NUM:date"),
    # HOW → hur
    ("sv", "Hur fungerar fotosyntes?", "DESC:manner"),
    ("sv", "Hur tillverkas ost?", "DESC:manner"),
    ("sv", "Hur fungerar en elmotor?", "DESC:manner"),
    # WHY → varför
    ("sv", "Varför är himlen blå?", "DESC:reason"),
    ("sv", "Varför drömmer vi?", "DESC:reason"),
    ("sv", "Varför flyter is på vatten?", "DESC:reason"),
    # WHAT → vad / vilken
    ("sv", "Vad är demokrati?", "DESC:def"),
    ("sv", "Vad är maskininlärning?", "DESC:def"),
    ("sv", "Vilken är meningen med livet?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # CZECH
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → kdo
    ("cs", "Kdo vynalezl rádio?", "HUM:ind"),
    ("cs", "Kdo napsal Babičku?", "HUM:ind"),
    ("cs", "Kdo byl první prezident Česka?", "HUM:ind"),
    # WHERE → kde
    ("cs", "Kde leží Londýn?", "LOC:other"),
    ("cs", "Kde žijí tučňáci?", "LOC:other"),
    ("cs", "Kde se nachází Machu Picchu?", "LOC:other"),
    # WHEN → kdy
    ("cs", "Kdy vynalezl Edison žárovku?", "NUM:date"),
    ("cs", "Kdy se narodil Bedřich Smetana?", "NUM:date"),
    ("cs", "Kdy přistál člověk na Měsíci?", "NUM:date"),
    # HOW → jak
    ("cs", "Jak funguje fotosyntéza?", "DESC:manner"),
    ("cs", "Jak se vyrábí sýr?", "DESC:manner"),
    ("cs", "Jak funguje elektromotor?", "DESC:manner"),
    # WHY → proč
    ("cs", "Proč je obloha modrá?", "DESC:reason"),
    ("cs", "Proč sníme?", "DESC:reason"),
    ("cs", "Proč led pluje na vodě?", "DESC:reason"),
    # WHAT → co / který
    ("cs", "Co je demokracie?", "DESC:def"),
    ("cs", "Co je strojové učení?", "DESC:def"),
    ("cs", "Který je smysl života?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # DANISH
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → hvem
    ("da", "Hvem opfandt radioen?", "HUM:ind"),
    ("da", "Hvem skrev H.C. Andersens eventyr?", "HUM:ind"),
    ("da", "Hvem var Danmarks første statsminister?", "HUM:ind"),
    # WHERE → hvor
    ("da", "Hvor ligger Eiffeltårnet?", "LOC:other"),
    ("da", "Hvor lever pingviner?", "LOC:other"),
    ("da", "Hvor er Machu Picchu?", "LOC:other"),
    # WHEN → hvornår
    ("da", "Hvornår begyndte Anden Verdenskrig?", "NUM:date"),
    ("da", "Hvornår blev Søren Kierkegaard født?", "NUM:date"),
    ("da", "Hvornår landede mennesket på månen?", "NUM:date"),
    # HOW → hvordan
    ("da", "Hvordan fungerer fotosyntese?", "DESC:manner"),
    ("da", "Hvordan laves ost?", "DESC:manner"),
    ("da", "Hvordan fungerer en elmotor?", "DESC:manner"),
    # WHY → hvorfor
    ("da", "Hvorfor er himlen blå?", "DESC:reason"),
    ("da", "Hvorfor drømmer vi?", "DESC:reason"),
    ("da", "Hvorfor flyder is på vand?", "DESC:reason"),
    # WHAT → hvad / hvilken
    ("da", "Hvad er demokrati?", "DESC:def"),
    ("da", "Hvad er maskinlæring?", "DESC:def"),
    ("da", "Hvilken mening har livet?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # HUNGARIAN
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → ki
    ("hu", "Ki találta fel a rádiót?", "HUM:ind"),
    ("hu", "Ki írta Arany János balladáit?", "HUM:ind"),
    ("hu", "Ki volt az ország első vezére?", "HUM:ind"),
    # WHERE → hol
    ("hu", "Hol áll az Eiffel-torony?", "LOC:other"),
    ("hu", "Hol élnek a pingvinek?", "LOC:other"),
    ("hu", "Hol van Stonehenge?", "LOC:other"),
    # WHEN → mikor
    ("hu", "Mikor kezdődött a Második Világháború?", "NUM:date"),
    ("hu", "Mikor született Liszt Ferenc?", "NUM:date"),
    ("hu", "Mikor szállt le az ember a Holdra?", "NUM:date"),
    # HOW → hogyan / hogy
    ("hu", "Hogyan működik a fotoszintézis?", "DESC:manner"),
    ("hu", "Hogyan készül a sajt?", "DESC:manner"),
    ("hu", "Hogyan működik az elektromotor?", "DESC:manner"),
    # WHY → miért
    ("hu", "Miért kék az ég?", "DESC:reason"),
    ("hu", "Miért álmodunk?", "DESC:reason"),
    ("hu", "Miért úszik a jég a vízen?", "DESC:reason"),
    # WHAT → mi / melyik / milyen
    ("hu", "Mi a demokrácia?", "DESC:def"),
    ("hu", "Mi a gépi tanulás?", "DESC:def"),
    ("hu", "Melyik az élet értelme?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # TURKISH
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → kim
    ("tr", "Kim radyoyu icat etti?", "HUM:ind"),
    ("tr", "Kim Suç ve Ceza'yı yazdı?", "HUM:ind"),
    ("tr", "Kim icat etti telefonu?", "HUM:ind"),
    # WHERE → nerede
    ("tr", "Nerede bulunuyor Eyfel Kulesi?", "LOC:other"),
    ("tr", "Nerede yaşar penguenler?", "LOC:other"),
    ("tr", "Nerede yer alıyor Machu Picchu?", "LOC:other"),
    # WHEN → ne zaman  (position: start)
    ("tr", "Ne zaman kuruldu bu okul?", "NUM:date"),
    ("tr", "Ne zaman doğdu Mustafa Kemal Atatürk?", "NUM:date"),
    ("tr", "Ne zaman aya ilk ayak basıldı?", "NUM:date"),
    # HOW → nasıl
    ("tr", "Nasıl oluşur kar?", "DESC:manner"),
    ("tr", "Nasıl yapılır peynir?", "DESC:manner"),
    ("tr", "Nasıl üretilir cam?", "DESC:manner"),
    # WHY → neden
    ("tr", "Neden gökyüzü mavi?", "DESC:reason"),
    ("tr", "Neden rüya görürüz?", "DESC:reason"),
    ("tr", "Neden buz suyun üzerinde yüzer?", "DESC:reason"),
    # WHAT → ne / hangi
    ("tr", "Ne demek demokrasi?", "DESC:def"),
    ("tr", "Ne hissettiriyor bu müzik?", "DESC:def"),
    ("tr", "Hangi seçenek doğrudur?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # RUSSIAN
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → кто
    ("ru", "Кто изобрёл радио?", "HUM:ind"),
    ("ru", "Кто написал Преступление и наказание?", "HUM:ind"),
    ("ru", "Кто был первым президентом России?", "HUM:ind"),
    # WHERE → где
    ("ru", "Где находится Эйфелева башня?", "LOC:other"),
    ("ru", "Где живут пингвины?", "LOC:other"),
    ("ru", "Где расположен Мачу-Пикчу?", "LOC:other"),
    # WHEN → когда
    ("ru", "Когда изобрели радио?", "NUM:date"),
    ("ru", "Когда родился Лев Толстой?", "NUM:date"),
    ("ru", "Когда человек впервые высадился на Луну?", "NUM:date"),
    # HOW → как
    ("ru", "Как работает фотосинтез?", "DESC:manner"),
    ("ru", "Как делается сыр?", "DESC:manner"),
    ("ru", "Как работает электродвигатель?", "DESC:manner"),
    # WHY → зачем / почему
    ("ru", "Зачем мы спим?", "DESC:reason"),
    ("ru", "Почему небо голубое?", "DESC:reason"),
    ("ru", "Почему лёд плавает на воде?", "DESC:reason"),
    # WHAT → что / какой
    ("ru", "Что такое демократия?", "DESC:def"),
    ("ru", "Что такое машинное обучение?", "DESC:def"),
    ("ru", "Какой смысл у жизни?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # UKRAINIAN
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → хто
    ("uk", "Хто винайшов радіо?", "HUM:ind"),
    ("uk", "Хто написав Кобзаря?", "HUM:ind"),
    ("uk", "Хто був першим президентом України?", "HUM:ind"),
    # WHERE → де
    ("uk", "Де знаходиться Ейфелева вежа?", "LOC:other"),
    ("uk", "Де живуть пінгвіни?", "LOC:other"),
    ("uk", "Де розташований Мачу-Пікчу?", "LOC:other"),
    # WHEN → коли
    ("uk", "Коли винайшли радіо?", "NUM:date"),
    ("uk", "Коли народився Тарас Шевченко?", "NUM:date"),
    ("uk", "Коли людина вперше висадилась на Місяць?", "NUM:date"),
    # HOW → як
    ("uk", "Як працює фотосинтез?", "DESC:manner"),
    ("uk", "Як роблять сир?", "DESC:manner"),
    ("uk", "Як працює електродвигун?", "DESC:manner"),
    # WHY → навіщо / чому
    ("uk", "Навіщо ми спимо?", "DESC:reason"),
    ("uk", "Чому небо блакитне?", "DESC:reason"),
    ("uk", "Чому лід плаває на воді?", "DESC:reason"),
    # WHAT → що / який
    ("uk", "Що таке демократія?", "DESC:def"),
    ("uk", "Що таке машинне навчання?", "DESC:def"),
    ("uk", "Який сенс має життя?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # GREEK
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → ποιος / ποια
    ("el", "Ποιος εφηύρε το ράδιο?", "HUM:ind"),
    ("el", "Ποιος έγραψε την Οδύσσεια?", "HUM:ind"),
    ("el", "Ποια ήταν η πρώτη πρόεδρος της Ελλάδας?", "HUM:ind"),
    # WHERE → πού
    ("el", "Πού βρίσκεται ο Πύργος του Άιφελ?", "LOC:other"),
    ("el", "Πού ζουν οι πιγκουίνοι?", "LOC:other"),
    ("el", "Πού είναι το Μάτσου Πίτσου?", "LOC:other"),
    # WHEN → πότε
    ("el", "Πότε ανακαλύφθηκε η Αμερική?", "NUM:date"),
    ("el", "Πότε γεννήθηκε ο Αριστοτέλης?", "NUM:date"),
    ("el", "Πότε πάτησε άνθρωπος στη Σελήνη?", "NUM:date"),
    # HOW → πώς
    ("el", "Πώς λειτουργεί ένα μαγνήτης?", "DESC:manner"),
    ("el", "Πώς φτιάχνεται το τυρί?", "DESC:manner"),
    ("el", "Πώς λειτουργεί ένας ηλεκτρικός κινητήρας?", "DESC:manner"),
    # WHY → γιατί
    ("el", "Γιατί ο ουρανός είναι μπλε?", "DESC:reason"),
    ("el", "Γιατί ονειρευόμαστε?", "DESC:reason"),
    ("el", "Γιατί ο πάγος επιπλέει στο νερό?", "DESC:reason"),
    # WHAT → τι / ποιο
    ("el", "Τι είναι η φιλοσοφία?", "DESC:def"),
    ("el", "Τι είναι η τεχνητή νοημοσύνη?", "DESC:def"),
    ("el", "Ποιο είναι το νόημα της ζωής?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # BASQUE
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → nor
    ("eu", "Nor asmatu zuen irratia?", "HUM:ind"),
    ("eu", "Nor zen Euskal Herriko lehen lehendakaria?", "HUM:ind"),
    ("eu", "Nor idatzi zuen Gernika?", "HUM:ind"),
    # WHERE → non
    ("eu", "Non dago Eiffel Dorrea?", "LOC:other"),
    ("eu", "Non bizi dira pinguinoak?", "LOC:other"),
    ("eu", "Non dago Machu Picchu?", "LOC:other"),
    # WHEN → noiz
    ("eu", "Noiz asmatu zen irratia?", "NUM:date"),
    ("eu", "Noiz jaio zen Bilboko lehen alkatea?", "NUM:date"),
    ("eu", "Noiz erori zen Berlineko harresia?", "NUM:date"),
    # HOW → nola
    ("eu", "Nola funtzionatzen du fotosintesiak?", "DESC:manner"),
    ("eu", "Nola egiten da gazta?", "DESC:manner"),
    ("eu", "Nola funtzionatzen du motor elektriko batek?", "DESC:manner"),
    # WHY → zergatik
    ("eu", "Zergatik da zerua urdina?", "DESC:reason"),
    ("eu", "Zergatik amesten dugu?", "DESC:reason"),
    ("eu", "Zergatik igeri egiten du izotzak uretan?", "DESC:reason"),
    # WHAT → zer / zein
    ("eu", "Zer da demokrazia?", "DESC:def"),
    ("eu", "Zer da adimen artifiziala?", "DESC:def"),
    ("eu", "Zein da bizitzaren zentzua?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # GALICIAN  (strip_leading ¿¡)
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → quen
    ("gl", "¿Quen inventou a radio?", "HUM:ind"),
    ("gl", "¿Quen escribiu Rosalía de Castro?", "HUM:ind"),
    ("gl", "¿Quen foi o primeiro presidente da Xunta de Galicia?", "HUM:ind"),
    # WHERE → onde
    ("gl", "¿Onde está a Gran Muralla China?", "LOC:other"),
    ("gl", "¿Onde viven os pingüíns?", "LOC:other"),
    ("gl", "¿Onde se atopa Machu Picchu?", "LOC:other"),
    # WHEN → cando
    ("gl", "¿Cando se inventou o telefono?", "NUM:date"),
    ("gl", "¿Cando naceu Rosalía de Castro?", "NUM:date"),
    ("gl", "¿Cando chegou o home á lúa?", "NUM:date"),
    # HOW → como
    ("gl", "¿Como funciona a fotosíntese?", "DESC:manner"),
    ("gl", "¿Como se fai o pan?", "DESC:manner"),
    ("gl", "¿Como funciona un motor eléctrico?", "DESC:manner"),
    # WHY → por que  (position: start after stripping ¿)
    ("gl", "¿Por que o ceo é azul?", "DESC:reason"),
    ("gl", "¿Por que soñamos?", "DESC:reason"),
    ("gl", "¿Por que o xeo flota?", "DESC:reason"),
    # WHAT → que / cal
    ("gl", "¿Que é a democracia?", "DESC:def"),
    ("gl", "¿Que é a intelixencia artificial?", "DESC:def"),
    ("gl", "¿Cal é o sentido da vida?", "DESC:def"),

    # ══════════════════════════════════════════════════════════════════════════
    # PERSIAN / FARSI
    # ══════════════════════════════════════════════════════════════════════════
    # WHO → چه کسی / کی
    ("fa", "کی اختراع کرد رادیو را?", "HUM:ind"),
    ("fa", "کی شاهنامه را نوشت?", "HUM:ind"),
    ("fa", "کی بود اولین رهبر ایران?", "HUM:ind"),
    # WHERE → کجا
    ("fa", "کجا قرار دارد برج ایفل?", "LOC:other"),
    ("fa", "کجا زندگی می‌کنند پنگوئن‌ها?", "LOC:other"),
    ("fa", "کجا واقع شده است ماچوپیچو?", "LOC:other"),
    # WHEN → چه وقت / کِی
    ("fa", "چه وقت اختراع شد تلفن?", "NUM:date"),
    ("fa", "کِی به دنیا آمد حافظ?", "NUM:date"),
    ("fa", "چه وقت انسان به ماه رفت?", "NUM:date"),
    # HOW → چطور / چگونه
    ("fa", "چطور کار می‌کند آهنربا?", "DESC:manner"),
    ("fa", "چطور درست می‌شود نان?", "DESC:manner"),
    ("fa", "چطور کار می‌کند موتور الکتریکی?", "DESC:manner"),
    # WHY → چرا
    ("fa", "چرا آسمان آبی است?", "DESC:reason"),
    ("fa", "چرا خواب می‌بینیم?", "DESC:reason"),
    ("fa", "چرا یخ روی آب شناور است?", "DESC:reason"),
    # WHAT → چه / کدام
    ("fa", "چه هست دموکراسی?", "DESC:def"),
    ("fa", "چه می‌شود هوش مصنوعی?", "DESC:def"),
    ("fa", "چه می‌باشد فلسفه?", "DESC:def"),
]
# fmt: on


class TestHeuristicQuestionTypeMultilang:
    @pytest.mark.parametrize(
        "lang,text,expected",
        MULTILANG_QUESTION_TYPE_CASES,
        ids=[
            f"{lang}-{exp}-{i}"
            for i, (lang, _, exp) in enumerate(MULTILANG_QUESTION_TYPE_CASES)
        ],
    )
    def test_predict(self, lang, text, expected):
        clf = HeuristicQuestionTypeClassifier(lang)
        result = clf.predict(text)
        assert result == expected, (
            f"[{lang}] predict({text!r}) → {result!r}, expected {expected!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Locale JSON structural sanity checks
# ─────────────────────────────────────────────────────────────────────────────

ALL_LANGS = [
    "en", "es", "pt", "ca", "fr", "de", "it", "nl",
    "pl", "ro", "sv", "cs", "da", "hu", "tr",
    "ru", "uk", "el", "eu", "gl", "fa",
]


class TestLocaleStructure:
    @pytest.mark.parametrize("lang", ALL_LANGS)
    def test_locale_loads(self, lang):
        from little_questions.classifiers import _load_locale
        locale = _load_locale(lang)
        assert isinstance(locale, dict), f"[{lang}] locale is not a dict"
        assert "sentence_type" in locale, f"[{lang}] missing 'sentence_type' key"
        assert "question_type" in locale, f"[{lang}] missing 'question_type' key"

    @pytest.mark.parametrize("lang", ALL_LANGS)
    def test_sentence_type_has_rules(self, lang):
        from little_questions.classifiers import _load_locale
        st = _load_locale(lang)["sentence_type"]
        rules = st.get("rules", [])
        assert len(rules) >= 5, f"[{lang}] sentence_type has fewer than 5 rules"

    @pytest.mark.parametrize("lang", ALL_LANGS)
    def test_question_type_has_rules(self, lang):
        from little_questions.classifiers import _load_locale
        qt = _load_locale(lang)["question_type"]
        rules = qt.get("rules", [])
        assert len(rules) >= 10, f"[{lang}] question_type has fewer than 10 rules"

    @pytest.mark.parametrize("lang", ALL_LANGS)
    def test_regional_variant_resolves_to_same_locale(self, lang):
        """Regional codes like nl-BE and nl should load the same locale."""
        from little_questions.classifiers import _load_locale
        base = _load_locale(lang)
        regional = _load_locale(f"{lang}-XX")  # fake region suffix
        assert base is regional, (
            f"[{lang}] regional variant did not resolve to the cached base locale"
        )

    @pytest.mark.parametrize("lang", ALL_LANGS)
    def test_all_rules_have_required_fields(self, lang):
        from little_questions.classifiers import _load_locale
        locale = _load_locale(lang)
        for section_name in ("sentence_type", "question_type"):
            for i, rule in enumerate(locale[section_name].get("rules", [])):
                assert "match" in rule, (
                    f"[{lang}] {section_name} rule[{i}] missing 'match'"
                )
                assert "scores" in rule, (
                    f"[{lang}] {section_name} rule[{i}] missing 'scores'"
                )
                pos = rule.get("position", "anywhere")
                assert pos in {"start", "end", "word_start", "anywhere"}, (
                    f"[{lang}] {section_name} rule[{i}] unknown position {pos!r}"
                )

    @pytest.mark.parametrize("lang", ALL_LANGS)
    def test_heuristic_sentence_type_returns_valid_label(self, lang):
        from little_questions.constants import SENTENCE_TYPES
        clf = HeuristicSentenceTypeClassifier(lang)
        result = clf.predict("test sentence")
        assert result in SENTENCE_TYPES, (
            f"[{lang}] predict returned unknown label {result!r}"
        )

    @pytest.mark.parametrize("lang", ALL_LANGS)
    def test_heuristic_question_type_returns_valid_label(self, lang):
        from little_questions.constants import EAT_LABELS_53 as QUESTION_TYPES
        clf = HeuristicQuestionTypeClassifier(lang)
        result = clf.predict("what is this")
        assert result in QUESTION_TYPES, (
            f"[{lang}] predict returned unknown label {result!r}"
        )
