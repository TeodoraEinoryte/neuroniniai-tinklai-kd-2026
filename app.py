"""Small web app (PLAN.md section 7, step 4.4): an English or Lithuanian sentence in, the emotion probabilities out,
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
        words = [(w, None) for w in result["words"]]
        notes.append(f"{model.name} modelis dėmesio svorių neturi, todėl žodžiai nepažymėti.")
    if result["unknown_share"] >= UNKNOWN_WARNING:
        notes.append(f"⚠️ {result['unknown_share']:.0%} žodžių modeliui nežinomi ({', '.join(result['unknown'])}), "
                     "todėl rezultatas nepatikimas.")
    return result["probabilities"], words, "\n\n".join(notes)


demo = gr.Interface(
    fn=analyse,
    inputs=[gr.Textbox(label="Sakinys", lines=2), gr.Radio(CHOICES, value=CHOICES[0], label="Modelis")],
    outputs=[gr.Label(num_top_classes=6, label="Emocijų tikimybės"),
             gr.HighlightedText(label="Į kuriuos žodžius modelis atsižvelgė (dėmesio svoriai)"),
             gr.Markdown()],
    examples=[["I feel so alone tonight", CHOICES[0]],
              ["I didn't feel humiliated at all, I was proud!", CHOICES[0]],
              ["i feel like i have finally found my place in this world", CHOICES[0]],
              ["Šiandien jaučiuosi labai vieniša ir liūdna", CHOICES[1]],
              ["Jaučiuosi labai laiminga, kad tave sutikau", CHOICES[1]]],
    title="Emocijų atpažinimas iš teksto",
    description="Šešios emocijos: liūdesys, džiaugsmas, meilė, pyktis, baimė, nuostaba. Kursinis darbas "
                "„Neuroninių tinklų modelių palyginimas atpažįstant emocijas iš teksto“ (VilniusTECH, 2026).",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
