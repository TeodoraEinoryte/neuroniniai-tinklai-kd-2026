"""Small web app: an English or Lithuanian sentence in, the emotion probabilities out,
the words the attention model looked at, and a warning when many words are unknown to the model.

Run: python app.py      then open http://localhost:7860
Uses the app's models in results/models/app/ (python predict.py --save-models) through predict.EmotionModel.
"""
import gradio as gr

from predict import UNKNOWN_WARNING, EmotionModel

LANGUAGES = {"en": "anglų kalba", "lt": "lietuvių kalba"}
MODELS = {}
for lang, language in LANGUAGES.items():
    model = EmotionModel(lang)
    MODELS[f"{model.name} ({language})"] = model
CHOICES = list(MODELS)
EXAMPLES = [["I feel so alone tonight", CHOICES[0]],
            ["I didn't feel humiliated at all, I was proud!", CHOICES[0]],
            ["i feel like i have finally found my place in this world", CHOICES[0]],
            ["Šiandien jaučiuosi labai vieniša ir liūdna", CHOICES[1]],
            ["Jaučiuosi labai laiminga, kad tave sutikau", CHOICES[1]]]

VILNIUS_TECH_BLUE = gr.themes.Color(name="vilniustech_blue", c50="#EAF0FC", c100="#D5E1F9", c200="#AEC4F2",
                                    c300="#7FA2E9", c400="#4F7EDD", c500="#0B4DC7", c600="#0A44B0",
                                    c700="#083A96", c800="#062F7A", c900="#05255F", c950="#031A44")
THEME = gr.themes.Default(primary_hue=VILNIUS_TECH_BLUE, font=["Arial", "sans-serif"],
                          font_mono=["Arial", "sans-serif"])   # the probability list uses the "mono" font
CSS = """
#header {background: #0B4DC7; color: white; padding: 18px 24px; border-radius: 8px;}
#header h1, #header p {color: white; margin: 0;}
#header p {margin-top: 6px; opacity: 0.9;}
"""


def analyse(text, choice):
    """Probabilities for gr.Label, (word, weight) pairs for gr.HighlightedText, notes for gr.Markdown."""
    if not text or not text.strip():
        return {}, [], "Įveskite sakinį."
    model = MODELS[choice]
    result = model.predict(text)
    notes = []
    if result["attention"] is not None:
        top = max(result["attention"])               # the most-attended word gets the darkest colour
        words = [(w, float(a / top)) for w, a in zip(result["words"], result["attention"])]
    else:
        words = [(" ".join(result["words"]), None)]  # one plain span: separate unmarked words would lose their spaces
        notes.append(f"{model.name} modelis dėmesio svorių neturi, todėl žodžiai nepažymėti.")
    if result["unknown_share"] >= UNKNOWN_WARNING:
        notes.append(f"⚠️ {result['unknown_share']:.0%} žodžių modeliui nežinomi ({', '.join(result['unknown'])}), "
                     "todėl rezultatas nepatikimas.")
    return result["probabilities"], words, "\n\n".join(notes)


with gr.Blocks(title="Emocijų atpažinimas iš teksto") as demo:
    gr.HTML("<div id='header'><h1>Emocijų atpažinimas iš teksto</h1>"
            "<p>Šešios emocijos: liūdesys, džiaugsmas, meilė, pyktis, baimė, nuostaba. Kursinis darbas "
            "„Neuroninių tinklų modelių palyginimas atpažįstant emocijas iš teksto“ (VILNIUS TECH, 2026).</p></div>")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Sakinys", lines=2, placeholder="Įveskite sakinį angliškai arba lietuviškai")
            choice = gr.Radio(CHOICES, value=CHOICES[0], label="Modelis")
            with gr.Row():
                clear = gr.Button("Išvalyti")
                submit = gr.Button("Atpažinti emociją", variant="primary")
        with gr.Column():
            probabilities = gr.Label(num_top_classes=6, label="Emocijų tikimybės")
            words = gr.HighlightedText(label="Dėmesio svoriai: kuriuos žodžius modelis laikė svarbiausiais")
            notes = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[text, choice], outputs=[probabilities, words, notes], fn=analyse,
                run_on_click=True, cache_examples=False, label="Pavyzdžiai")
    submit.click(analyse, [text, choice], [probabilities, words, notes])
    text.submit(analyse, [text, choice], [probabilities, words, notes])
    clear.click(lambda: ("", {}, [], ""), None, [text, probabilities, words, notes])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, footer_links=[])
